use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use lofi_code::{ExecCtx, ExecOptions, ToolEvent};
use lofi_error::{Error, Result};
use lofi_types::{ContentBlock, NativeToolRecord};
use tokio::sync::oneshot;

use super::approvals::{approve_outside_dir, ConfirmRequest, ConfirmResponse};
use super::spend::{clamp_child, ChildSpend};
use super::{ToolCallContext, ToolExecutor};
use crate::agent::{
    cap_exec_envelope, cap_exec_result, cap_tool_result, exec_timeout_ms, lock, parse_exec_input,
    AgentEvent, MAX_EXEC_RESULT_BYTES,
};

/// Report one rejected tool call: emit the error `ToolEnd`, then record the
/// `ToolResult`. Shared by every pre-execution rejection in `execute_tools`.
async fn reject(
    call: &ToolCallContext,
    results: &mut Vec<ContentBlock>,
    id: &str,
    content: impl Into<String>,
    dispatch: std::time::Instant,
) -> Result<()> {
    let content = content.into();
    let elapsed_ms = dispatch.elapsed().as_millis() as u64;
    if !emit(
        call.events.as_ref(),
        AgentEvent::ToolEnd {
            id: id.to_string(),
            result: content.clone(),
            is_error: true,
            elapsed_ms,
        },
    )
    .await
    {
        return Err(Error::Cancelled);
    }
    results.push(ContentBlock::ToolResult {
        tool_use_id: id.to_string(),
        content,
        is_error: true,
        images: Vec::new(),
    });
    Ok(())
}

impl ToolExecutor {
    #[allow(clippy::too_many_lines)]
    pub(in crate::agent) async fn execute_tools(
        &self,
        tool_uses: &[(&str, &str, &serde_json::Value)],
        call: &ToolCallContext,
    ) -> Result<Vec<ContentBlock>> {
        if tool_uses.is_empty() {
            return Ok(Vec::new());
        }
        // The child's idle deadline pauses for the whole run of calls: a long
        // one emits no event until it returns.
        let _running = self.activity.enter();
        // The scratch directory arrives with the first tool run, so a
        // session that never runs one leaves no directory behind.
        if let Some(lease) = &self.tmp_lease {
            lease.ensure()?;
        }
        let native_cursor = call.session.clone();
        let recall = call.recall.clone();
        let result = call.result.clone();
        let cancel = call.cancel.as_ref();
        let on_job_acquired = call.on_job_acquired.clone();
        let mut results: Vec<ContentBlock> = Vec::with_capacity(tool_uses.len());
        // Native tool events are emitted from a *sync* `on_tool_event`
        // callback inside the sandbox, so they can't `await` on the bounded
        // UI channel. `try_send` would drop events when the channel fills
        // (e.g. a `Promise.all` burst of Start/End events), leaving native
        // tiles stuck "running" even after the exec succeeds. Relay them
        // through an unbounded channel whose forwarder drains onto the
        // bounded channel with `send().await` — lossless and order-preserving.
        let (relay_tx, mut relay_rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
        let relay_dst = call.events.clone();
        let _forwarder = tokio::spawn(async move {
            while let Some(ev) = relay_rx.recv().await {
                match &relay_dst {
                    Some(dst) => {
                        if !dst.emit(ev).await {
                            break;
                        }
                    }
                    None => {
                        drop(ev);
                    }
                }
            }
        });
        for (id, name, input) in tool_uses {
            let dispatch = std::time::Instant::now();
            if *name != "exec" {
                reject(
                    call,
                    &mut results,
                    id,
                    format!("unknown tool: {name}"),
                    dispatch,
                )
                .await?;
                continue;
            }
            let (code, strings, description) = parse_exec_input(input);
            if description.is_empty() {
                reject(
                    call,
                    &mut results,
                    id,
                    "exec: missing required 'description' argument",
                    dispatch,
                )
                .await?;
                continue;
            }
            if code.is_empty() {
                reject(
                    call,
                    &mut results,
                    id,
                    "exec: missing required 'code' argument",
                    dispatch,
                )
                .await?;
                continue;
            }
            // The call's own bound: what it asked, as an absolute instant, so
            // the prompt, the compile, and every tool it starts come out of it
            // instead of adding to it. Nothing above the call bounds it.
            if input
                .get("timeoutMs")
                .filter(|asked| !asked.is_null())
                .is_some_and(|asked| asked.as_u64().is_none_or(|ms| ms == 0))
            {
                reject(
                    call,
                    &mut results,
                    id,
                    failed_result(
                        "exec: timeoutMs must be a positive integer",
                        "failed",
                        None,
                        None,
                    ),
                    dispatch,
                )
                .await?;
                continue;
            }
            let exec_deadline = exec_timeout_ms(input).and_then(|ms| {
                std::time::Instant::now().checked_add(std::time::Duration::from_millis(ms))
            });
            // The bound this call's tools run under: the call's own `timeoutMs`.
            let clock = exec_deadline.map_or(lofi_types::Deadline::NONE, lofi_types::Deadline::at);
            // The ToolInput event must land before we invoke the sandbox;
            // if the consumer is gone, stop instead of running tools (which
            // may shell out or mutate files) for a dead receiver.
            if !emit(
                call.events.as_ref(),
                AgentEvent::ToolInput {
                    id: id.to_string(),
                    code: code.clone(),
                    label: Some(description.clone()),
                },
            )
            .await
            {
                return Err(Error::Cancelled);
            }
            let native_tx = relay_tx.clone();
            let parent = id.to_string();
            let native_pending: Arc<Mutex<HashMap<u64, (String, String)>>> =
                Arc::new(Mutex::new(HashMap::new()));
            let native_records: Arc<Mutex<Vec<NativeToolRecord>>> =
                Arc::new(Mutex::new(Vec::new()));
            let on_tool_event: Arc<dyn Fn(ToolEvent) + Send + Sync> = {
                let native_pending = native_pending.clone();
                let native_records = native_records.clone();
                let retain_records = native_cursor.is_some();
                let event_parent = parent.clone();
                Arc::new(move |ev: ToolEvent| match ev {
                    ToolEvent::Start { id, name, args } => {
                        lock(&native_pending).insert(id, (name.clone(), args.clone()));
                        let _ = native_tx.send(AgentEvent::NativeToolStart {
                            parent: event_parent.clone(),
                            id,
                            name,
                            args,
                        });
                    }
                    ToolEvent::End {
                        id,
                        result,
                        is_error,
                    } => {
                        let display_result = cap_tool_result(&result);
                        if let Some((name, args)) = lock(&native_pending).remove(&id) {
                            if retain_records {
                                lock(&native_records).push(NativeToolRecord {
                                    parent: event_parent.clone(),
                                    call_id: id,
                                    name,
                                    args,
                                    result,
                                    is_error,
                                });
                            }
                        }
                        let _ = native_tx.send(AgentEvent::NativeToolEnd {
                            parent: event_parent.clone(),
                            id,
                            result: display_result,
                            is_error,
                        });
                    }
                }) as Arc<dyn Fn(ToolEvent) + Send + Sync>
            };
            let confirm: Option<lofi_code::ConfirmFn> = self.confirm_tx.as_ref().map(|tx| {
                let tx = tx.clone();
                let counter = self.confirm_counter.clone();
                let watch = self.confirm_watch.clone();
                Arc::new(move |prompt: lofi_code::ConfirmPrompt| {
                    let tx = tx.clone();
                    let counter = counter.clone();
                    let watch = watch.clone();
                    Box::pin(async move {
                        let id = counter.fetch_add(1, Ordering::SeqCst);
                        let (resp_tx, resp_rx) = oneshot::channel();
                        let command = prompt.command.clone();
                        let req = ConfirmRequest {
                            id,
                            command: prompt.command,
                            reason: prompt.reason,
                            active: prompt.active,
                            deny_at: prompt.deny_at,
                            respond: resp_tx,
                        };
                        let _open = watch.open(&command);
                        if tx.send(req).is_err() {
                            return false;
                        }
                        matches!(resp_rx.await, Ok(ConfirmResponse::Allow))
                    })
                        as std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + Sync>>
                }) as lofi_code::ConfirmFn
            });
            let approve_dir: Option<lofi_code::ApproveDirFn> = self.confirm_tx.as_ref().map(|tx| {
                let tx = tx.clone();
                let counter = self.confirm_counter.clone();
                let watch = self.confirm_watch.clone();
                let root = self.root.clone();
                let config_grants = self.grants.clone();
                // The instant this call's clock closes the prompt: the same
                // one the tool refuses an answer after, so the countdown the
                // user reads ends where the ask does.
                let deny_at = clock.instant();
                Arc::new(move |approval: lofi_code::DirApproval| {
                    let tx = tx.clone();
                    let counter = counter.clone();
                    let watch = watch.clone();
                    let root = root.clone();
                    let config_grants = config_grants.clone();
                    Box::pin(async move {
                        // The prompt stays open for the whole approval: the
                        // stored-grant reads it does first are instantaneous
                        // next to the wait for an answer.
                        let _open = watch.open(&approval.dir.display().to_string());
                        approve_outside_dir(&root, approval, &config_grants, &tx, &counter, deny_at)
                            .await
                    })
                        as std::pin::Pin<
                            Box<
                                dyn std::future::Future<Output = Option<lofi_code::GrantedDir>>
                                    + Send
                                    + Sync,
                            >,
                        >
                }) as lofi_code::ApproveDirFn
            });
            // Children this call starts report their spend in the call's own
            // result, so a fan-out reads its cost from the value it already
            // gets back. The hook stays the session's; this wrapper records
            // what the call started, and asks that same hook to fold the runs
            // it spawned.
            let child_spend = Arc::new(Mutex::new(ChildSpend::default()));
            // The wall clock this call is under, handed to the tools it
            // starts: a command is clamped to what the call has left instead
            // of to a constant that has to stay below it by hand. A call with
            // no `timeoutMs` arms nothing, and its tools keep their own timeouts.
            // This is not the guest's CPU budget below: awaited time costs no
            // CPU, so the two clocks bound different things.
            let subagent: Option<lofi_code::SubagentFn> = self.subagent.clone().map(|hook| {
                let spend = child_spend.clone();
                let wait_clock = clock;
                Arc::new(move |op: lofi_code::SubagentOp, args: serde_json::Value| {
                    let hook = hook.clone();
                    let spend = spend.clone();
                    Box::pin(async move {
                        let (args, cut) = clamp_child(op, args, wait_clock);
                        // A resume installs the new run's totals over the
                        // entry's, so what the background run this call
                        // started billed is read before the hook replaces it.
                        let replaced = match op {
                            lofi_code::SubagentOp::Run => {
                                let id = args.get("id").and_then(serde_json::Value::as_str);
                                let ours = id.filter(|id| {
                                    lock(&spend).spawned.iter().any(|kept| kept == id)
                                });
                                match ours {
                                    Some(id) => Some(
                                        hook(
                                            lofi_code::SubagentOp::Status,
                                            serde_json::json!({ "id": id }),
                                        )
                                        .await,
                                    ),
                                    None => None,
                                }
                            }
                            _ => None,
                        };
                        let mut out = hook(op, args).await;
                        // The allowance this child asked for, when the call
                        // cut it: a parent can tell a child that ran out from
                        // one that was cut short.
                        if let Some(asked) = cut {
                            if let Some(map) = out.as_object_mut() {
                                map.insert(
                                    "requestedTimeoutMs".to_string(),
                                    serde_json::json!(asked),
                                );
                            }
                        }
                        let mut spend = lock(&spend);
                        match op {
                            // Only an admitted spawn returns an id: a
                            // rejection or a busy resume echoes the id it
                            // was asked for, and started nothing to fold.
                            lofi_code::SubagentOp::Spawn
                                if out.get("ok").and_then(serde_json::Value::as_bool)
                                    == Some(true) =>
                            {
                                if let Some(id) = out.get("id").and_then(serde_json::Value::as_str)
                                {
                                    spend.spawned.push(id.to_string());
                                    spend.claimed.push(id.to_string());
                                }
                            }
                            // A run body names its model and a rejection body
                            // does not. Every run body that reports usage was
                            // billed, completed or not.
                            lofi_code::SubagentOp::Run if out.get("model").is_some() => {
                                // The resume left the entry's outcome naming
                                // only this run, so the run it replaced is
                                // priced from the snapshot and stops being a
                                // child to fold.
                                if let Some(replaced) = &replaced {
                                    spend.add_run(replaced);
                                }
                                if let Some(id) = out.get("id").and_then(serde_json::Value::as_str)
                                {
                                    spend.forget(id);
                                }
                                spend.add_run(&out);
                            }
                            _ => {}
                        }
                        out
                    })
                        as std::pin::Pin<
                            Box<dyn std::future::Future<Output = serde_json::Value> + Send>,
                        >
                }) as lofi_code::SubagentFn
            });
            let exec_ctx = ExecCtx {
                session: self.tool_session(),
                strings,
                recall: recall.clone(),
                result: result.clone(),
                on_tool_event: Some(on_tool_event),
                confirm,
                approve_dir,
                // The stated description travels with this call's context,
                // so the gate judges this call's commands with this call's
                // intent however the call's future is dropped or queued.
                intent: Some(description.clone()),
                // A prompt must not outlive the run that raised it, so the
                // wait is cut to what the call has left. A spent clock leaves
                // the shortest wait there is rather than a disabled
                // auto-deny.
                approval_timeout: self
                    .auto_mode_config
                    .as_ref()
                    .and_then(|config| config.shell_policy.auto_mode.as_ref())
                    .and_then(lofi_types::AutoModeConfig::approval_timeout)
                    .map(|wait| {
                        clock
                            .remaining(std::time::Instant::now())
                            .map_or(wait, |left| {
                                wait.min(left).max(std::time::Duration::from_millis(1))
                            })
                    }),
                on_job_acquired: on_job_acquired.clone(),
                subagent: subagent.clone(),
            };
            let outcome = self
                .sandbox_worker()
                .exec(
                    &code,
                    exec_ctx,
                    ExecOptions {
                        // The guest's own allowance stays as it was: it counts
                        // CPU, so raising it to a long run cap would let a
                        // spin loop run for hours. The wall budget is the
                        // separate bound the caller named.
                        timeout: lofi_code::DEFAULT_GUEST_TIMEOUT,
                        deadline: exec_deadline,
                        max_value_bytes: MAX_EXEC_RESULT_BYTES,
                        cancel: cancel.cloned(),
                    },
                )
                .await;
            // Fold the runs this call spawned through the session hook; a
            // background child that is still running reports what it spent so
            // far. The runs it awaited carried their own spend already.
            let spend = lock(&child_spend).clone();
            let subagent_usage = match &self.subagent {
                Some(hook) if !spend.is_empty() => {
                    let folded = if spend.spawned.is_empty() {
                        None
                    } else {
                        Some(
                            hook(
                                lofi_code::SubagentOp::Usage,
                                serde_json::json!({ "ids": spend.spawned.clone() }),
                            )
                            .await,
                        )
                    };
                    Some(spend.record(folded.as_ref()))
                }
                _ => None,
            };
            // The registry drops the records this call priced, so a long
            // session does not keep spend nobody folds again. It runs even
            // when the fold above had nothing to add: the claim outlives it.
            if let Some(hook) = &self.subagent {
                if !spend.claimed.is_empty() {
                    hook(
                        lofi_code::SubagentOp::Release,
                        serde_json::json!({ "ids": spend.claimed.clone() }),
                    )
                    .await;
                }
            }
            // A tool may return a tagged image (`read` on an image file).
            // Upgrade it to a `ToolResultImage` carried on this result block,
            // so the model sees the image in the same round as the result on
            // every provider. The heavyweight base64 is stripped out of the
            // text payload, leaving a compact `bytes` count in the transcript.
            let (content, is_error, result_images) = match outcome {
                Ok(r) => {
                    let has_value = r.has_returned_value();
                    let mut value = r.value;
                    match upgrade_exec_image(&mut value, r.image, &self.image) {
                        Ok(img) => {
                            let is_error = !r.errors.is_empty();
                            let mut output = serde_json::Map::new();
                            output.insert("ok".into(), serde_json::Value::Bool(!is_error));
                            // One vocabulary with a run, a job and a bash call.
                            output.insert(
                                "status".into(),
                                serde_json::Value::String(
                                    if is_error { "failed" } else { "completed" }.into(),
                                ),
                            );
                            if is_error {
                                output.insert("errors".into(), serde_json::Value::Array(r.errors));
                            }
                            if has_value {
                                output.insert("value".into(), value);
                            }
                            if let Some(usage) = &subagent_usage {
                                output.insert("subagents".into(), usage.clone());
                            }
                            if !r.logs.is_empty() {
                                output.insert("logs".into(), serde_json::Value::String(r.logs));
                            }
                            if r.value_truncated {
                                output.insert("truncated".into(), serde_json::Value::Bool(true));
                            }
                            (cap_exec_envelope(output), is_error, img)
                        }
                        Err(error) => (
                            cap_exec_result(&failed_result(
                                &error.to_string(),
                                error.status(),
                                error.logs(),
                                subagent_usage.as_ref(),
                            )),
                            true,
                            None,
                        ),
                    }
                }
                Err(e) => (
                    cap_exec_result(&failed_result(
                        &e.to_string(),
                        e.status(),
                        e.logs(),
                        subagent_usage.as_ref(),
                    )),
                    true,
                    None,
                ),
            };
            if is_error {
                // A rejected guest promise can short-circuit concurrent native
                // calls before their futures emit `ToolEvent::End`. Close every
                // still-pending row explicitly so the live UI cannot leave a
                // spinner behind after the parent exec has already failed.
                // These synthetic cancellations are also persisted, making
                // resume reproduce the settled state.
                let pending = {
                    let mut pending = lock(&native_pending);
                    pending.drain().collect::<Vec<_>>()
                };
                for (call_id, (name, args)) in pending {
                    let result = "cancelled because parent exec failed".to_string();
                    if native_cursor.is_some() {
                        lock(&native_records).push(NativeToolRecord {
                            parent: parent.clone(),
                            call_id,
                            name,
                            args,
                            result: result.clone(),
                            is_error: true,
                        });
                    }
                    let _ = relay_tx.send(AgentEvent::NativeToolEnd {
                        parent: parent.clone(),
                        id: call_id,
                        result,
                        is_error: true,
                    });
                }
            }
            // A tool that outlives its turn may not extend a transcript no turn
            // will settle.
            let live = call
                .alive
                .as_ref()
                .is_none_or(|alive| alive.load(std::sync::atomic::Ordering::Acquire));
            if let (Some(cursor), true) = (native_cursor.as_ref(), live) {
                let records = std::mem::take(&mut *lock(&native_records));
                if let Err(error) =
                    cursor.record(crate::session::recorder::SessionRecord::NativeTools { records })
                {
                    let _ = relay_tx.send(AgentEvent::Notice(format!(
                        "transcript write failed for native tool result: {error}"
                    )));
                }
            }
            let elapsed_ms = dispatch.elapsed().as_millis() as u64;
            if !emit(
                call.events.as_ref(),
                AgentEvent::ToolEnd {
                    id: id.to_string(),
                    result: content.clone(),
                    is_error,
                    elapsed_ms,
                },
            )
            .await
            {
                return Err(Error::Cancelled);
            }
            results.push(ContentBlock::ToolResult {
                tool_use_id: id.to_string(),
                content,
                is_error,
                images: result_images.into_iter().collect(),
            });
        }
        Ok(results)
    }
}

/// Publish one tool event through the turn's sink.
///
/// The sink folds the event into the accounting the transcript summary is
/// built from, so a tool's elapsed time survives a resume. False when the
/// consumer is gone.
pub(super) async fn emit(
    events: Option<&crate::agent::turn_session::EventSink>,
    event: AgentEvent,
) -> bool {
    match events {
        Some(events) => events.emit(event).await,
        None => true,
    }
}

/// Normalize an image returned out of band by the exec sandbox. `QuickJS` carries
/// only a small token, so source image bytes never become a base64 guest string
/// or a second host JSON allocation. The durable result keeps only a byte count.
pub(in crate::agent) fn upgrade_exec_image(
    value: &mut serde_json::Value,
    image: Option<lofi_code::ExecImage>,
    config: &lofi_types::ImageConfig,
) -> Result<Option<lofi_types::ToolResultImage>> {
    use std::io::Seek as _;

    let Some(mut image) = image else {
        return Ok(None);
    };
    image.file.rewind()?;
    let reader = std::io::BufReader::new(image.file);
    let (bytes, media_type) = crate::image::normalize_reader(reader, config)?;
    if let Some(obj) = value.as_object_mut() {
        obj.insert("bytes".to_string(), serde_json::json!(bytes.len()));
        obj.insert("media_type".to_string(), serde_json::json!(&media_type));
    }
    Ok(Some(lofi_types::ToolResultImage {
        bytes: bytes.into(),
        media_type,
    }))
}

/// The failed body of an exec result, in the shape a successful call uses: a
/// program that threw after its children spent money still reports that
/// spend, and `status` names what stopped it in the vocabulary runs, jobs and
/// bash share.
pub(super) fn failed_result(
    message: &str,
    status: &str,
    logs: Option<&str>,
    subagents: Option<&serde_json::Value>,
) -> String {
    // The thrown message and the logs are the unbounded fields, and escaping
    // can outgrow the limit on its own, so the body is measured as it is
    // built: a cut body would lose the keys that sort after them, `status`
    // and `subagents` among them.
    let mut keep_message = message.len();
    let mut keep_logs = logs.map_or(0, str::len);
    loop {
        let message_end = message.floor_char_boundary(keep_message);
        let mut body = serde_json::json!({
            "ok": false,
            "status": status,
            "errors": [&message[..message_end]],
        });
        if let Some(usage) = subagents {
            body["subagents"] = usage.clone();
        }
        if let Some(logs) = logs {
            let logs_end = logs.floor_char_boundary(keep_logs);
            body["logs"] = serde_json::Value::String(logs[..logs_end].to_string());
        }
        // A cut field says so, the way a cut envelope does: the part that was
        // dropped is the most recent progress, and a reader cannot tell a
        // short log from a trimmed one.
        if message_end < message.len() || logs.is_some_and(|logs| keep_logs < logs.len()) {
            body["truncated"] = serde_json::Value::Bool(true);
        }
        let text = body.to_string();
        if text.len() <= MAX_EXEC_RESULT_BYTES || (message_end == 0 && keep_logs == 0) {
            return text;
        }
        // Halve whichever field is taking the room, so a large message does
        // not cost the logs and the other way around.
        if keep_logs > 0 && keep_logs >= keep_message {
            keep_logs /= 2;
        } else {
            keep_message /= 2;
        }
    }
}
