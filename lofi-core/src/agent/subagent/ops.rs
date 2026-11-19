use std::sync::Arc;
use std::time::Duration;

use lofi_code::{SubagentFn, SubagentOp};
use lofi_error::{Error, Result};
use serde_json::{json, Value};

use super::registry::{entry_session_id, Entry, Rollup, SubagentRegistry};
use super::runner::{start, CancelOnDrop, RunRejection};
use super::{Status, SubagentSupport, MIN_NOTIFY_INTERVAL_MS};
use crate::agent::lock;
use crate::agent::mailbox::{MailboxScope, PARENT_ADDRESS};
use crate::backend::BackendRef;

/// Hook for a child agent: mailbox only, scoped to the child's own address.
pub(super) fn mailbox_hook(scope: MailboxScope) -> SubagentFn {
    Arc::new(move |op: SubagentOp, args: Value| {
        let scope = scope.clone();
        Box::pin(async move {
            match op {
                SubagentOp::MailboxSend => scope.mailbox.send(&scope.me, false, &args),
                SubagentOp::MailboxReceive => scope.mailbox.receive(&scope.me, &args).await,
                SubagentOp::MailboxList => scope.mailbox.list(&scope.me, false, &args),
                _ => json!({
                    "ok": false,
                    "status": "unavailable",
                    "error": "subagent is not available in this context",
                }),
            }
        }) as std::pin::Pin<Box<dyn std::future::Future<Output = Value> + Send>>
    })
}

/// Absent, null and empty string all mean "not given"; any other non-string
/// value is a caller error, not a silent default.
pub(super) fn str_arg<'a>(args: &'a Value, key: &str) -> Result<Option<&'a str>> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_str() {
            Some(s) if !s.is_empty() => Ok(Some(s)),
            Some(_) => Ok(None),
            None => Err(Error::Tool(format!("{key} must be a string"))),
        },
    }
}

pub(super) fn u64_arg(args: &Value, key: &str) -> Result<Option<u64>> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_u64() {
            Some(n) if n > 0 => Ok(Some(n)),
            _ => Err(Error::Tool(format!("{key} must be a positive integer"))),
        },
    }
}

// The hook returns errors as a JSON body (never a tool error) so the parent
// model can react to `status` and resume by `id` after a failure.
pub(super) async fn dispatch(
    template: BackendRef,
    support: SubagentSupport,
    registry: SubagentRegistry,
    op: SubagentOp,
    args: Value,
) -> Value {
    match op {
        SubagentOp::Run => run(template, support, registry, args).await,
        SubagentOp::Spawn => spawn(&template, &support, &registry, &args).await,
        SubagentOp::Status => status_op(&registry, &args),
        SubagentOp::List => list_op(&registry, &args),
        SubagentOp::Usage => usage_op(&registry, &args),
        SubagentOp::Release => release_op(&registry, &args),
        SubagentOp::Wait => wait_op(&registry, args).await,
        SubagentOp::Read => read_op(&registry, &args),
        SubagentOp::Kill => kill_op(&registry, &args),
        SubagentOp::Notify => notify_op(&registry, &args),
        // The parent addresses runs by id; a child reaches the same store
        // through the scoped hook built below.
        SubagentOp::MailboxSend => registry.mailbox.send(PARENT_ADDRESS, true, &args),
        SubagentOp::MailboxReceive => registry.mailbox.receive(PARENT_ADDRESS, &args).await,
        SubagentOp::MailboxList => registry.mailbox.list(PARENT_ADDRESS, true, &args),
    }
}

/// Rejections carry no tool error so the model sees a structured body.
fn rejection_json(rejection: RunRejection, args: &Value) -> Value {
    let mut out = match rejection {
        RunRejection::Busy(state) => {
            // A busy id did not fail: the run is still going, and the parent
            // can resume it once it ends.
            json!({
                "ok": false,
                "status": "running",
                "error": format!("subagent is {state}; resume is only possible after a run ends"),
            })
        }
        RunRejection::Error(e) => {
            json!({ "ok": false, "status": "failed", "error": e.to_string() })
        }
    };
    if let Ok(Some(id)) = str_arg(args, "id") {
        out["id"] = json!(id);
    }
    out
}

fn arg_error_json(e: &Error) -> Value {
    json!({ "ok": false, "status": "failed", "error": e.to_string() })
}

/// Every id-addressed op shares these two bodies.
fn required_id(args: &Value) -> std::result::Result<String, Value> {
    match str_arg(args, "id") {
        Ok(Some(id)) => Ok(id.to_string()),
        Ok(None) => Err(json!({
            "ok": false,
            "status": "failed",
            "error": "id is required and must be a non-empty string",
        })),
        Err(e) => Err(arg_error_json(&e)),
    }
}

fn unknown_id(id: &str) -> Value {
    json!({ "ok": false, "id": id, "status": "failed", "error": "unknown subagent id" })
}

/// Foreground `lofi.subagent`: start the run and await it. Dropping this
/// future (exec timeout, Esc) arms `CancelOnDrop` and cancels the child.
async fn run(
    template: BackendRef,
    support: SubagentSupport,
    registry: SubagentRegistry,
    args: Value,
) -> Value {
    // Foreground runs return their outcome to the caller directly: a
    // terminal notice on top would queue a duplicate as a parent prompt.
    let started = match start(&template, &support, &registry, &args, false).await {
        Ok(started) => started,
        Err(rejection) => return rejection_json(rejection, &args),
    };
    let guard = CancelOnDrop {
        cancel: started.cancel.clone(),
        armed: true,
    };
    // A JoinError means the task panicked; its SettleOnDrop already stored
    // the failure outcome. Both paths read the entry's stored outcome.
    let _ = started.handle.await;
    guard.disarm();
    registry
        .outcome_of(&started.id)
        .unwrap_or_else(|| unknown_id(&started.id))
}

/// Background `lofi.subagentSpawn`: start the run and return its handle id
/// at once. No `CancelOnDrop`: the run survives the end of this exec.
async fn spawn(
    template: &BackendRef,
    support: &SubagentSupport,
    registry: &SubagentRegistry,
    args: &Value,
) -> Value {
    // `id` resumes that run in the background, so a run stopped at its
    // deadline continues without a foreground call that dies with its exec.
    if args.get("output_limit").is_some_and(|v| !v.is_null()) {
        return json!({
            "ok": false,
            "status": "failed",
            "error": "subagentSpawn does not accept output_limit",
        });
    }
    let started = match start(template, support, registry, args, true).await {
        Ok(started) => started,
        Err(rejection) => return rejection_json(rejection, args),
    };
    // Detaching a JoinHandle keeps the task alive; the entry tracks it.
    drop(started.handle);
    let mut out = json!({ "ok": true, "id": started.id, "status": "running" });
    if let Some(session_id) = started.session_id {
        out["session_id"] = json!(session_id);
    }
    out
}

fn status_op(registry: &SubagentRegistry, args: &Value) -> Value {
    let id = match required_id(args) {
        Ok(id) => id,
        Err(out) => return out,
    };
    let entries = lock(&registry.entries);
    let Some(entry) = entries.get(&id) else {
        return unknown_id(&id);
    };
    let mut out = json!({ "ok": true, "id": id, "status": entry.status.as_str() });
    if let Some(session_id) = entry_session_id(entry) {
        out["session_id"] = json!(session_id);
    }
    if let Some(note) = &entry.note {
        out["note"] = json!(note);
    }
    if let Some(outcome) = &entry.outcome {
        // The outcome flattens the run's progress into these keys, and the
        // status reports the same fields the settled result carries.
        for key in [
            "usage",
            "error",
            "rounds",
            "elapsed_ms",
            "last_tool",
            "stopped",
            "cap_ms",
            "requestedTimeoutMs",
        ] {
            if let Some(value) = outcome.get(key) {
                out[key] = value.clone();
            }
        }
    }
    out
}

fn list_op(registry: &SubagentRegistry, args: &Value) -> Value {
    let state = match str_arg(args, "state") {
        Ok(state) => state,
        Err(e) => return arg_error_json(&e),
    };
    let running_only = match state {
        None => None,
        Some("running") => Some(true),
        Some("terminal") => Some(false),
        Some(_) => {
            return json!({
                "ok": false,
                "status": "failed",
                "error": "state must be \"running\" or \"terminal\"",
            })
        }
    };
    let limit = match u64_arg(args, "limit") {
        Ok(limit) => limit.unwrap_or(20).min(100) as usize,
        Err(e) => return arg_error_json(&e),
    };
    let after = match str_arg(args, "cursor") {
        Ok(None) => 0,
        Ok(Some(cursor)) => match cursor.parse::<u64>() {
            Ok(born) => born,
            Err(_) => {
                return json!({
                    "ok": false,
                    "status": "failed",
                    "error": "cursor is not a valid page cursor",
                })
            }
        },
        Err(e) => return arg_error_json(&e),
    };
    let entries = lock(&registry.entries);
    let mut rows: Vec<(&String, &Entry)> = entries
        .iter()
        .filter(|(_, entry)| entry.born > after)
        .filter(|(_, entry)| running_only.is_none_or(|r| (entry.status == Status::Running) == r))
        .collect();
    rows.sort_by_key(|(_, entry)| entry.born);
    let more = rows.len() > limit;
    rows.truncate(limit);
    let cursor = more
        .then(|| rows.last().map(|(_, entry)| entry.born))
        .flatten();
    let list: Vec<Value> = rows
        .into_iter()
        .map(|(id, entry)| {
            let mut row = json!({ "id": id, "status": entry.status.as_str() });
            if let Some(session_id) = entry_session_id(entry) {
                row["session_id"] = json!(session_id);
            }
            if let Some(note) = &entry.note {
                row["note"] = json!(note);
            }
            row
        })
        .collect();
    let mut out = json!({ "ok": true, "entries": list });
    if let Some(cursor) = cursor {
        out["cursor"] = json!(cursor.to_string());
    }
    out
}

/// `subagentUsage`: fold the spend of this session's runs into one record.
/// `ids` narrows the fold to the named runs; without it every entry and
/// every evicted run counts.
/// An entry folds every run of its child session, so a resumed run reports
/// what that child billed across all of them. A restart prices a child only
/// when its transcript ends with a settled run: a resume that never settled
/// restores no spend for it, its earlier runs included.
/// A run the provider never billed contributes nothing. An evicted entry's
/// spend is folded from `retired`. An id in neither place is reported as
/// `missing` instead of failing the call: a rehydrated session restores only
/// some of its runs, and an id from another session is unknown here.
pub(super) fn usage_op(registry: &SubagentRegistry, args: &Value) -> Value {
    let ids = match id_list(args) {
        Ok(ids) => ids,
        Err(out) => return out,
    };
    let entries = lock(&registry.entries);
    let mut rollup = Rollup::default();
    let mut missing = Vec::new();
    match ids {
        None => {
            entries.values().for_each(|entry| rollup.add(entry));
            let total = lock(&registry.retired).total.json();
            rollup.merge(&total);
        }
        Some(ids) => {
            let retired = lock(&registry.retired);
            let mut seen = std::collections::HashSet::new();
            for id in ids {
                // A caller that builds the list by concatenation repeats ids.
                if !seen.insert(id.clone()) {
                    continue;
                }
                if let Some(entry) = entries.get(&id) {
                    rollup.add(entry);
                } else if let Some(spend) = retired.spend.get(&id) {
                    rollup.merge(spend);
                } else {
                    missing.push(id);
                }
            }
        }
    }
    let mut out = rollup.json();
    out["ok"] = json!(true);
    if !missing.is_empty() {
        out["missing"] = json!(missing);
    }
    out
}

/// `ids` as the caller gave it: absent or null means "every run".
fn id_list(args: &Value) -> std::result::Result<Option<Vec<String>>, Value> {
    match args.get("ids") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) => {
            let mut ids = Vec::with_capacity(items.len());
            for item in items {
                let Some(id) = item.as_str() else {
                    return Err(ids_error());
                };
                ids.push(id.to_string());
            }
            Ok(Some(ids))
        }
        Some(_) => Err(ids_error()),
    }
}

fn ids_error() -> Value {
    json!({
        "ok": false,
        "status": "failed",
        "error": "ids must be an array of strings",
    })
}

/// Internal counterpart of the fold: drop the spend records of the runs an
/// exec just priced. No guest binding produces this op, so a fold by the
/// model cannot take a record the spawning exec still needs.
pub(super) fn release_op(registry: &SubagentRegistry, args: &Value) -> Value {
    let ids = match id_list(args) {
        Ok(Some(ids)) => ids,
        Ok(None) => Vec::new(),
        Err(out) => return out,
    };
    lock(&registry.retired).release(&ids);
    json!({ "ok": true })
}

async fn wait_op(registry: &SubagentRegistry, args: Value) -> Value {
    let id = match required_id(&args) {
        Ok(id) => id,
        Err(out) => return out,
    };
    // Without `timeout_ms` there is no deadline of the wait's own: it ends
    // when the run settles, because a wait that expired at a constant would
    // report a run still working as if the wait had failed. A caller that
    // runs the wait inside a capped run is clamped to what that run has left.
    // Pass `timeout_ms` to poll instead.
    let timeout = match u64_arg(&args, "timeout_ms") {
        Ok(ms) => ms.map(Duration::from_millis),
        Err(e) => return arg_error_json(&e),
    };
    let mut rx = {
        let entries = lock(&registry.entries);
        match entries.get(&id) {
            Some(entry) => entry.state_tx.subscribe(),
            None => return unknown_id(&id),
        }
    };
    let mut deadline: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> =
        match timeout {
            Some(timeout) => Box::pin(tokio::time::sleep(timeout)),
            // Never resolves: the loop below then ends only on the run.
            None => Box::pin(std::future::pending()),
        };
    loop {
        if rx.borrow().is_terminal() {
            return registry.outcome_of(&id).unwrap_or_else(|| unknown_id(&id));
        }
        tokio::select! {
            biased;
            () = &mut deadline => {
                // A wait timeout never cancels: the run keeps going and can
                // be awaited again.
                return json!({ "ok": true, "id": id, "status": "running" });
            }
            changed = rx.changed() => {
                if changed.is_err() {
                    // The entry was evicted or cleared while waiting.
                    return registry
                        .outcome_of(&id)
                        .unwrap_or_else(|| unknown_id(&id));
                }
            }
        }
    }
}

fn read_op(registry: &SubagentRegistry, args: &Value) -> Value {
    let id = match required_id(args) {
        Ok(id) => id,
        Err(out) => return out,
    };
    let cursor = match u64_arg(args, "cursor") {
        Ok(cursor) => cursor.unwrap_or(0),
        Err(e) => return arg_error_json(&e),
    };
    let limit = match u64_arg(args, "limit") {
        Ok(limit) => limit.unwrap_or(50).min(200) as usize,
        Err(e) => return arg_error_json(&e),
    };
    let types: Option<Vec<String>> = match args.get("types") {
        None | Some(Value::Null) => None,
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item.as_str() {
                    Some(item) => out.push(item.to_string()),
                    None => {
                        return json!({
                            "ok": false,
                            "status": "failed",
                            "error": "types must be an array of strings",
                        })
                    }
                }
            }
            Some(out)
        }
        Some(_) => {
            return json!({
                "ok": false,
                "status": "failed",
                "error": "types must be an array of strings",
            })
        }
    };
    let entries = lock(&registry.entries);
    let Some(entry) = entries.get(&id) else {
        return unknown_id(&id);
    };
    let (events, next, dropped) = entry.events.page(cursor, limit, types.as_deref());
    let done = entry.status.is_terminal() && next >= entry.events.next_seq;
    json!({
        "ok": true,
        "id": id,
        "events": events,
        "cursor": next,
        "dropped": dropped,
        "done": done,
    })
}

fn kill_op(registry: &SubagentRegistry, args: &Value) -> Value {
    let id = match required_id(args) {
        Ok(id) => id,
        Err(out) => return out,
    };
    let reason = match str_arg(args, "reason") {
        Ok(reason) => reason,
        Err(e) => return arg_error_json(&e),
    };
    match registry.kill(&id, reason) {
        Some(status) => json!({ "ok": true, "id": id, "status": status }),
        None => unknown_id(&id),
    }
}

fn notify_op(registry: &SubagentRegistry, args: &Value) -> Value {
    let id = match required_id(args) {
        Ok(id) => id,
        Err(out) => return out,
    };
    let enabled = match args.get("enabled") {
        None | Some(Value::Null) => None,
        Some(v) => match v.as_bool() {
            Some(enabled) => Some(enabled),
            None => {
                return json!({
                    "ok": false,
                    "status": "failed",
                    "error": "enabled must be a boolean",
                })
            }
        },
    };
    let interval = match u64_arg(args, "interval_ms") {
        Ok(ms) => ms.map(|ms| Duration::from_millis(ms.max(MIN_NOTIFY_INTERVAL_MS))),
        Err(e) => return arg_error_json(&e),
    };
    let mut entries = lock(&registry.entries);
    let Some(entry) = entries.get_mut(&id) else {
        return unknown_id(&id);
    };
    if let Some(enabled) = enabled {
        entry.notify_terminal = enabled;
    }
    // Any reconfiguration retires the previous loop at once instead of
    // leaving it asleep until its old interval expires.
    entry.notify_tick += 1;
    if let Some(task) = entry.notify_task.take() {
        task.abort();
    }
    if let Some(interval) = interval {
        let tick = entry.notify_tick;
        if entry.status == Status::Running && entry.notify_terminal {
            let registry = registry.clone();
            let loop_id = id.clone();
            let task = registry.host_handle().spawn(async move {
                loop {
                    tokio::time::sleep(interval).await;
                    let active = {
                        let entries = lock(&registry.entries);
                        entries.get(&loop_id).is_some_and(|entry| {
                            entry.status == Status::Running
                                && entry.notify_terminal
                                && entry.notify_tick == tick
                        })
                    };
                    if !active {
                        break;
                    }
                    registry.publish_notice(&loop_id, format!("subagent {loop_id} still running"));
                }
            });
            entry.notify_task = Some(task.abort_handle());
        }
    }
    json!({ "ok": true, "id": id })
}
