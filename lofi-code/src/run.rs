use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rquickjs::async_with;
use rquickjs::prelude::*;
use rquickjs::{AsyncContext, AsyncRuntime, Ctx, Function, Object, Promise, Value};
use serde_json::Value as Json;

use lofi_error::{Error, Result};

use crate::bind::bind_tools;
use crate::compile::compile_ts;
use crate::convert::{js_to_json_with_limit, SANDBOX_ERROR_KEY};
use crate::ctx::{
    ExecCtx, ExecImage, ExecOptions, ExecResult, ModelsFn, RecallFn, ResultFn, SubagentFn,
};
use crate::tools::BuiltinTools;

const GUEST_MEMORY_LIMIT: usize = 8 * 1024 * 1024;
/// Maximum native call-stack depth the interpreter may use. `QuickJS` checks
/// this at function-entry granularity, so a deeply recursive guest aborts
/// with a stack-overflow exception rather than segfaulting the host.
const GUEST_MAX_STACK: usize = 1024 * 1024;

/// Interrupt-reason state shared between the host and the `QuickJS` interrupt
/// handler. The handler — called periodically by the interpreter — returns
/// `true` (abort) when this is non-zero, breaking out of synchronous tight
/// loops that the cooperative tokio timeout cannot preempt.
const INTR_RUNNING: u8 = 0;
const INTR_TIMEOUT: u8 = 1;
const INTR_CANCELLED: u8 = 2;
const INTR_WALL: u8 = 3;

/// How long a guest whose wall-clock budget ran out may take to unwind after
/// the interrupt flag is set, before the call is dropped.
const WALL_GRACE: Duration = Duration::from_secs(1);

const SUSPEND_THRESHOLD: Duration = Duration::from_millis(1);

pub(crate) const MAX_LOG_BYTES: usize = 1024 * 1024;
const LOG_TRUNCATION_MARKER: &str = "\n<logs truncated>";

const TRACK_FOREGROUND_TOOLS: &str = r#"
    (lofi, stopped) => {
        const pending = new Set();
        const errors = [];
        const errorText = error =>
            String(error && error.message ? error.message : error);

        // Await-heavy loops (while-true around an await) barely tick the
        // QuickJS interrupt handler, so the CPU budget cannot bound them.
        // Every await on a host tool funnels through this wrapper, so cap
        // the number of tool calls per exec instead. 4096 is far above any
        // legitimate script (paging, batch edits) and trips in about a
        // second for a tight loop.
        const MAX_CALLS = 4096;
        let calls = 0;

        // A guest whose clock ran out is unwinding: the grace it gets is
        // there to let the round it reached settle, not to start new work.
        // The refusal rejects rather than throws, so a script that awaits the
        // tool reads it the same way it reads any other refusal.
        for (const name of Object.keys(lofi)) {
            const tool = lofi[name];
            if (typeof tool !== "function") continue;
            lofi[name] = (...args) => {
                if (stopped()) {
                    return Promise.reject(
                        new Error("the run is stopped: no further tool calls")
                    );
                }
                calls += 1;
                if (calls > MAX_CALLS) {
                    throw new Error(
                        "tool call budget exceeded (" + MAX_CALLS + " calls per exec)"
                    );
                }
                const promise = Promise.resolve().then(() => tool(...args));
                const tracked = promise.then(
                    value => {
                        if (value !== null && typeof value === "object" && value.ok === false) {
                            errors.push({ tool: name, result: value });
                        }
                        return value;
                    },
                    error => {
                        errors.push({ tool: name, error: errorText(error) });
                        throw error;
                    }
                );
                pending.add(tracked);
                tracked.then(
                    () => pending.delete(tracked),
                    () => pending.delete(tracked)
                );
                return tracked;
            };
        }

        return async () => {
            while (pending.size > 0) {
                await Promise.allSettled(Array.from(pending));
            }
            return errors;
        };
    }
"#;

// QuickJS runs the interrupt handler during bytecode execution. Awaited host
// work creates long gaps between ticks, so each gap is capped before it is
// charged. The accumulated budget is preserved across suspensions because
// resetting it would let repeated CPU bursts evade the limit.
async fn install_cpu_guard(
    rt: &AsyncRuntime,
    budget: Duration,
    wall: Option<Instant>,
    cancel: Option<Arc<AtomicBool>>,
) -> Arc<AtomicU8> {
    let intr = Arc::new(AtomicU8::new(INTR_RUNNING));
    let intr_clone = intr.clone();
    let mut last_tick = Instant::now();
    let mut cpu_accumulated = Duration::ZERO;
    rt.set_interrupt_handler(Some(Box::new(move || {
        if intr_clone.load(Ordering::Relaxed) != INTR_RUNNING {
            return true;
        }
        if let Some(c) = &cancel {
            if c.load(Ordering::Relaxed) {
                intr_clone.store(INTR_CANCELLED, Ordering::Relaxed);
                return true;
            }
        }
        let now = Instant::now();
        // The caller's wall-clock budget is absolute, so a slow tool or a
        // long await cannot buy the guest more of it.
        if wall.is_some_and(|end| now >= end) {
            // A bound that already stopped the guest keeps the word it
            // stopped it with: the wall clock passing while a cancel unwinds
            // does not turn a cancellation into a timeout.
            let _ = intr_clone.compare_exchange(
                INTR_RUNNING,
                INTR_WALL,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            return true;
        }
        let delta = now.saturating_duration_since(last_tick);
        last_tick = now;
        cpu_accumulated += delta.min(SUSPEND_THRESHOLD);
        if cpu_accumulated >= budget {
            intr_clone.store(INTR_TIMEOUT, Ordering::Relaxed);
            return true;
        }
        false
    })))
    .await;
    intr
}

/// Record the wall the host saw pass on the interrupt flag: a guest that
/// returns at or after its instant ran past the budget, whether the handler
/// ticked or not. A cancel that landed first keeps its word.
fn note_wall(intr: &Arc<AtomicU8>, end: Instant, cancel: Option<&Arc<AtomicBool>>) {
    if Instant::now() < end {
        return;
    }
    let cancelled = cancel.is_some_and(|flag| flag.load(Ordering::Relaxed));
    let _ = intr.compare_exchange(
        INTR_RUNNING,
        if cancelled { INTR_CANCELLED } else { INTR_WALL },
        Ordering::Relaxed,
        Ordering::Relaxed,
    );
}

/// Run the guest until it finishes or its wall budget ends, whichever is
/// first. A guest awaiting a slow tool ticks rarely, so the interrupt handler
/// alone cannot bound it; setting the flag first keeps a synchronous guest
/// aborting on its next tick, and the grace lets the round it reached unwind.
async fn guest_before_wall<F>(
    guest: F,
    intr: &Arc<AtomicU8>,
    deadline: Option<Instant>,
    cancel: Option<&Arc<AtomicBool>>,
    over: impl Fn() -> Error,
) -> Result<ExecResult>
where
    F: std::future::Future<Output = Result<ExecResult>>,
{
    let Some(end) = deadline else {
        return guest.await;
    };
    tokio::pin!(guest);
    let sleep = tokio::time::sleep_until(tokio::time::Instant::from_std(end));
    tokio::pin!(sleep);
    tokio::select! {
        res = &mut guest => {
            note_wall(intr, end, cancel);
            // A tool the wall cut wakes the guest on the same instant the
            // sleep is due, and a short continuation runs fewer steps than the
            // interrupt handler needs to tick: without this the same cut would
            // report `timed_out` or a value depending on which arm the runtime
            // polled first.
            res
        }
        () = &mut sleep => {
            // A cancel that landed before the deadline keeps its word: the
            // wall passing while the guest unwinds is not a timeout.
            let cancelled = cancel.is_some_and(|flag| flag.load(Ordering::Relaxed));
            let _ = intr.compare_exchange(
                INTR_RUNNING,
                if cancelled { INTR_CANCELLED } else { INTR_WALL },
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            tokio::time::timeout(WALL_GRACE, &mut guest)
                .await
                .unwrap_or_else(|_| Err(over()))
        }
    }
}

/// - **Memory & stack** — `JS_SetMemoryLimit` and `JS_SetMaxStackSize` cap
///   the guest heap and call depth so a runaway allocation or deep recursion
///   aborts with a JS exception instead of exhausting the host.
/// - **CPU budget** — the `QuickJS` interrupt handler (installed via
///   `set_interrupt_handler`) is called on every interpreter tick. It
///   approximates CPU time rather than wall-clock time by capping gaps between
///   handler calls at [`SUSPEND_THRESHOLD`]. When the accumulated time exceeds
///   `opts.timeout` the handler returns
///   *abort*, breaking out of synchronous tight loops (`while (true) {}`)
///   that the cooperative `tokio` runtime cannot preempt. Ticks are rare
///   during awaited host work, so await-heavy loops accumulate budget
///   slowly; the tool-call cap in [`TRACK_FOREGROUND_TOOLS`] bounds those
///   instead.
/// - **Wall budget** — `opts.deadline` bounds the same work by wall clock, so
///   a guest that sits inside an awaited tool is cut at the caller's instant
///   instead of whenever that tool returns. The host owns this value: it is
///   already cut to the run's own cap.
/// - **Cancellation** — if `opts.cancel` is provided, setting it to `true`
///   causes the same interrupt path and surfaces a distinct "cancelled"
///   error. This is how the UI's Ctrl+C breaks a sync loop that `handle
///   .abort()` alone cannot preempt.
///
/// A guest the clock stops is not a guest that failed: once the CPU
/// allowance or the wall instant is spent, or the caller cancels, no further
/// tool call starts, so the grace a cut guest gets to unwind cannot lay down
/// work the result does not mention.
/// # Errors
/// Returns [`Error::SandboxStopped`] with status `"timed_out"` when the CPU
/// allowance or the wall budget ran out, and `"cancelled"` when the caller's
/// flag stopped it; both carry the logs the guest printed. Returns
/// [`Error::Sandbox`] for compile failures, guest exceptions, and tool errors
/// that propagate as thrown exceptions.
#[allow(clippy::too_many_lines)]
pub async fn exec(src: &str, ctx: &ExecCtx, opts: &ExecOptions) -> Result<ExecResult> {
    let js = compile_ts(src)?;

    let rt = AsyncRuntime::new().map_err(|e| Error::Sandbox(format!("runtime: {e}")))?;
    let actx = AsyncContext::full(&rt)
        .await
        .map_err(|e| Error::Sandbox(format!("context: {e}")))?;

    // Bound guest memory and stack so a runaway allocation or deep recursion
    // throws a catchable JS exception instead of exhausting the host.
    rt.set_memory_limit(GUEST_MEMORY_LIMIT).await;
    rt.set_max_stack_size(GUEST_MAX_STACK).await;

    let intr = install_cpu_guard(&rt, opts.timeout, opts.deadline, opts.cancel.clone()).await;

    // The tools this guest starts run under the bound this call was given:
    // the call's own `timeoutMs`, and nothing above it.
    let tool_clock = opts
        .deadline
        .map_or(lofi_types::Deadline::NONE, lofi_types::Deadline::at);
    let tools = Arc::new(
        BuiltinTools::from_session(ctx.session.clone())
            .with_tool_event(ctx.on_tool_event.clone())
            .with_confirm(ctx.confirm.clone())
            .with_intent(ctx.intent.clone())
            .with_approval_timeout(ctx.approval_timeout)
            .with_clock(tool_clock)
            .with_approve_dir(ctx.approve_dir.clone())
            .with_cancel(opts.cancel.clone())
            .with_on_job_acquired(ctx.on_job_acquired.clone()),
    );
    let strings = ctx.strings.clone();
    let recall = ctx.recall.clone();
    let result = ctx.result.clone();
    let skills_dir = ctx.session.skills_dir.clone();
    let subagent = ctx.subagent.clone();
    let models = ctx.session.models.clone();
    let logs = Arc::new(Mutex::new(String::new()));

    // The allowance this call was given, read before the guest starts: the
    // deadline is an instant, so once it has passed it has nothing left to
    // report. What the guest printed before the bound stopped it travels with
    // the stop, so a cut script still shows the progress it reached.
    let wall_ms = opts.deadline.map_or(0, |end| {
        u64::try_from(end.saturating_duration_since(Instant::now()).as_millis()).unwrap_or(u64::MAX)
    });
    let over_logs = logs.clone();
    let guest_intr = intr.clone();
    let guest = async {
        async_with!(&actx => |ctx| {
            // QuickJS can fault while building an error backtrace after its
            // heap limit is exhausted. The harness never exposes guest stacks.
            ctx.eval::<(), _>(
                r#"Error.stackTraceLimit = 0;
                Object.defineProperty(Error, "stackTraceLimit", {
                    value: 0, writable: false, configurable: false
                }); void 0;"#,
            )
            .map_err(|e| Error::Sandbox(format!("context: {e}")))?;
            let drain_tools = install_globals(
                &ctx,
                &tools,
                &strings,
                recall,
                result,
                skills_dir,
                subagent.as_ref(),
                models,
                &logs,
                opts.max_value_bytes,
                opts.deadline,
                &guest_intr,
                opts.cancel.as_ref(),
            )
            .map_err(|e| Error::Sandbox(format!("install: {e}")))?;
            let promise: Promise = ctx
                .eval(js.as_str())
                .map_err(|e| Error::Sandbox(format!("eval: {e}")))?;
            let value: Value = promise
                .into_future()
                .await
                .map_err(|e| Error::Sandbox(format!("guest promise rejected: {e}")))?;
            let errors_promise: Promise = drain_tools
                .call(())
                .map_err(|e| Error::Sandbox(format!("drain tools: {e}")))?;
            let errors: Value = errors_promise
                .into_future()
                .await
                .map_err(|e| Error::Sandbox(format!("drain tools: {e}")))?;
            convert_exec_result(&value, &errors, &logs, &tools, opts.max_value_bytes)
        })
        .await
    };
    let outcome = guest_before_wall(guest, &intr, opts.deadline, opts.cancel.as_ref(), || {
        stopped(
            &over_logs,
            "timed_out",
            format!("guest wall-clock budget exceeded ({wall_ms}ms)"),
        )
    })
    .await;

    let reason = intr.load(Ordering::Relaxed);
    // Conversion runs guest code: a getter can spin until the wall fires and
    // the property it was building is dropped, so a value that came back is
    // no proof that the guest finished inside its bound.
    // A guest blocked in an awaited tool cannot see the interrupt handler
    // tick, so the tool throws into it instead. A caller's own flag is a stop
    // whether or not the guest swallowed that throw and returned a value.
    let interrupted = interrupted(reason, opts, &over_logs, wall_ms).or_else(|| {
        opts.cancel
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
            .then(|| stopped(&over_logs, "cancelled", "guest cancelled".to_string()))
    });
    match interrupted {
        Some(error) => Err(error),
        None => outcome,
    }
}

/// A stop, carrying what the guest printed before it: a cut script still
/// shows the progress it reached.
fn stopped(logs: &Arc<Mutex<String>>, status: &'static str, message: String) -> Error {
    Error::SandboxStopped {
        status,
        message,
        logs: logs
            .lock()
            .ok()
            .map(|logs| logs.clone())
            .filter(|logs| !logs.is_empty()),
    }
}

/// The stop an interrupt flag names, if the guest was cut: the guard that set
/// it threw into the guest, so the flag decides whether the call finished
/// however the value the guest returned looks.
fn interrupted(
    reason: u8,
    opts: &ExecOptions,
    logs: &Arc<Mutex<String>>,
    wall_ms: u64,
) -> Option<Error> {
    match reason {
        INTR_TIMEOUT => Some(stopped(
            logs,
            "timed_out",
            format!("guest CPU budget exceeded ({}ms)", opts.timeout.as_millis()),
        )),
        INTR_CANCELLED => Some(stopped(logs, "cancelled", "guest cancelled".to_string())),
        INTR_WALL => Some(stopped(
            logs,
            "timed_out",
            format!("guest wall-clock budget exceeded ({wall_ms}ms)"),
        )),
        _ => None,
    }
}

fn convert_exec_result(
    value: &Value<'_>,
    errors: &Value<'_>,
    logs: &Arc<Mutex<String>>,
    tools: &BuiltinTools,
    max_value_bytes: usize,
) -> Result<ExecResult> {
    let has_value = !value.is_undefined();
    let (errors_json, errors_truncated) = js_to_json_with_limit(errors, max_value_bytes);
    if let Some(error) = errors_json.get(SANDBOX_ERROR_KEY).and_then(Json::as_str) {
        return Err(Error::Sandbox(format!("nested tool errors {error}")));
    }
    let errors = errors_json.as_array().cloned().unwrap_or_default();
    let errors_bytes = serde_json::to_string(&errors).map_or(0, |json| json.len());
    let logs_budget = max_value_bytes.saturating_sub(errors_bytes);
    // Cloned, not taken: a stop this conversion ends in carries the logs the
    // guest printed before it.
    let mut logs = logs
        .lock()
        .ok()
        .map(|logs| logs.clone())
        .unwrap_or_default();
    let logs_truncated = truncate_logs(&mut logs, logs_budget);
    let value_budget = max_value_bytes
        .saturating_sub(errors_bytes)
        .saturating_sub(logs.len());
    let (mut value, value_truncated) = js_to_json_with_limit(value, value_budget);
    if let Some(error) = value.get(SANDBOX_ERROR_KEY).and_then(Json::as_str) {
        return Err(Error::Sandbox(error.to_string()));
    }
    let image = take_pending_image(&mut value, tools);
    Ok(ExecResult {
        value,
        has_value,
        errors,
        logs,
        value_truncated: value_truncated || errors_truncated || logs_truncated,
        image,
    })
}

fn truncate_logs(logs: &mut String, max_bytes: usize) -> bool {
    if logs.len() <= max_bytes {
        return false;
    }
    let marker_end =
        LOG_TRUNCATION_MARKER.floor_char_boundary(LOG_TRUNCATION_MARKER.len().min(max_bytes));
    let marker = &LOG_TRUNCATION_MARKER[..marker_end];
    let content_end = logs.floor_char_boundary(max_bytes.saturating_sub(marker.len()));
    logs.truncate(content_end);
    logs.push_str(marker);
    true
}

fn take_pending_image(value: &mut Json, tools: &BuiltinTools) -> Option<ExecImage> {
    let object = value.as_object_mut()?;
    if object.get("type").and_then(Json::as_str) != Some("image") {
        return None;
    }
    let token = object.remove("image_token")?.as_u64()?;
    let pending = tools.take_image(token)?;
    Some(ExecImage {
        file: pending.file,
        bytes: pending.bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn install_globals<'js>(
    ctx: &Ctx<'js>,
    tools: &Arc<BuiltinTools>,
    strings: &HashMap<String, String>,
    recall: Option<RecallFn>,
    result: Option<ResultFn>,
    skills_dir: Option<PathBuf>,
    subagent: Option<&SubagentFn>,
    models: Option<ModelsFn>,
    logs: &Arc<Mutex<String>>,
    max_output_bytes: usize,
    deadline: Option<Instant>,
    intr: &Arc<AtomicU8>,
    cancel: Option<&Arc<AtomicBool>>,
) -> rquickjs::Result<Function<'js>> {
    let lofi = Object::new(ctx.clone())?;
    bind_tools(
        ctx, &lofi, tools, recall, result, skills_dir, subagent, models,
    )?;
    lofi.set("tmpDir", tools.tmp_dir().to_string_lossy().to_string())?;
    // What the tracker refuses a call on: the guard, the caller's flag, or a
    // wall that has passed say the guest is stopped, whether or not the
    // interpreter has ticked since. The instant is part of the answer because
    // a tool that ends on its own deadline can wake the guest before the host
    // notices that same deadline.
    let stopped = Function::new(ctx.clone(), {
        let intr = intr.clone();
        let cancel = cancel.cloned();
        move || -> bool {
            intr.load(Ordering::Relaxed) != INTR_RUNNING
                || deadline.is_some_and(|end| Instant::now() >= end)
                || cancel
                    .as_ref()
                    .is_some_and(|flag| flag.load(Ordering::Relaxed))
        }
    })?;
    let install_tracker: Function = ctx.eval(TRACK_FOREGROUND_TOOLS)?;
    let drain_tools = install_tracker.call((lofi.clone(), stopped))?;
    ctx.globals().set("lofi", lofi)?;

    let max_log_bytes = MAX_LOG_BYTES.min(max_output_bytes);
    let print_sink = Function::new(ctx.clone(), {
        let logs = logs.clone();
        move |args: Rest<Coerced<std::string::String>>| -> rquickjs::Result<()> {
            let mut line = String::new();
            // Cap the assembled line at the log budget while building it, so a
            // single huge argument cannot allocate an unbounded `line` before
            // the post-hoc cap runs. (The per-arg `Coerced<String>` is owned
            // by the runtime; this prevents amplifying it via concatenation.)
            for (i, arg) in args.iter().enumerate() {
                if line.len() >= max_log_bytes {
                    break;
                }
                if i > 0 {
                    line.push(' ');
                }
                let room = max_log_bytes.saturating_sub(line.len());
                if arg.len() <= room {
                    line.push_str(arg);
                } else {
                    let take = arg.floor_char_boundary(room);
                    line.push_str(&arg[..take]);
                    break;
                }
            }
            line.push('\n');
            if let Ok(mut buf) = logs.lock() {
                // Bound retained logs so a chatty guest can't exhaust memory.
                // Reserve room for the truncation marker and clamp to a valid
                // char boundary so a multibyte tail doesn't panic.
                let marker_end = LOG_TRUNCATION_MARKER
                    .floor_char_boundary(LOG_TRUNCATION_MARKER.len().min(max_log_bytes));
                let marker = &LOG_TRUNCATION_MARKER[..marker_end];
                if marker.is_empty() || buf.ends_with(marker) {
                    return Ok(());
                }
                let cap = max_log_bytes.saturating_sub(marker.len());
                let room = cap.saturating_sub(buf.len());
                if room == 0 {
                    buf.push_str(marker);
                    return Ok(());
                }
                let take = line.len().min(room);
                let take = line.floor_char_boundary(take);
                buf.push_str(&line[..take]);
                if take < line.len() || buf.len() >= cap {
                    buf.push_str(marker);
                }
            }
            Ok(())
        }
    })?;
    ctx.globals().set("__lofi_print_sink", print_sink)?;
    ctx.eval::<(), _>(format!(
        r"globalThis.print = ((sink, max) => (...args) =>
            sink(...args.map(value => String(value).slice(0, max))))(
                globalThis.__lofi_print_sink, {max_log_bytes}
            );
            delete globalThis.__lofi_print_sink;"
    ))?;

    let strings_obj = Object::new(ctx.clone())?;
    for (k, v) in strings {
        strings_obj.set(k.as_str(), v.as_str())?;
    }
    ctx.globals().set("lofi_strings", strings_obj)?;
    Ok(drain_tools)
}
