use std::path::Path;

use lofi_types::{SessionEvent, SessionEventKind};
use serde_json::{json, Value};

use super::registry::{Rollup, SubagentRegistry};
use super::runner::cap_at_boundary;
use super::{
    Status, DEFAULT_OUTPUT_LIMIT, NOTICE_DETAIL_CAP, SETTLED_MARKER_LIMIT, SETTLED_TAIL_WINDOW,
};
use crate::agent::lock;
use crate::session::store::SessionCursor;

/// One-line terminal notice: status plus the first line of the outcome.
pub(in crate::agent) fn terminal_notice(id: &str, status: Status, outcome: &Value) -> String {
    let detail = outcome["result"]
        .as_str()
        .or_else(|| outcome["error"].as_str())
        .and_then(|text| text.lines().map(str::trim).find(|line| !line.is_empty()))
        // A detail that only repeats the status adds nothing.
        .filter(|line| *line != status.as_str());
    // Where the run got to, so a stopped run says more than that it stopped;
    // a run that finished its task has nothing to explain. A wrapped run
    // settles completed, so it names its deadline too: a summary is not the
    // task finishing.
    let deadline = outcome["stopped"].as_str();
    let where_to = (status != Status::Completed || deadline.is_some())
        .then(|| progress_note(outcome))
        .flatten();
    let note = match (deadline, &where_to) {
        (Some(deadline), Some(note)) => Some(format!("stopped: {deadline}; {note}")),
        (Some(deadline), None) => Some(format!("stopped: {deadline}")),
        (None, note) => note.clone(),
    };
    let head = match note {
        Some(note) => format!("subagent {id} {} ({note})", status.as_str()),
        None => format!("subagent {id} {}", status.as_str()),
    };
    match detail {
        Some(line) => format!("{head}: {}", cap_at_boundary(line, NOTICE_DETAIL_CAP)),
        None => head,
    }
}

/// "round 4 after 312 s, bash" for a settled run that reported progress.
fn progress_note(outcome: &Value) -> Option<String> {
    let rounds = outcome["rounds"].as_u64().filter(|rounds| *rounds > 0);
    let seconds = outcome["elapsed_ms"].as_u64().map(|ms| ms / 1000);
    let rounds = rounds.map(|rounds| format!("round {rounds}"));
    let seconds = seconds.map(|seconds| format!("{seconds} s"));
    let note = [rounds, seconds]
        .into_iter()
        .flatten()
        .reduce(|head, tail| format!("{head} after {tail}"))
        .unwrap_or_default();
    let note = match (note.is_empty(), outcome["last_tool"].as_str()) {
        (true, Some(tool)) => tool.to_string(),
        (false, Some(tool)) => format!("{note}, {tool}"),
        (_, None) => note,
    };
    (!note.is_empty()).then_some(note)
}

/// The JSON body stored at settle and returned to the foreground caller,
/// `subagentWait`, and (minus the result) `subagentStatus`.
pub(super) fn outcome_json(
    id: &str,
    session_id: Option<&str>,
    outcome: RunOutcome,
    model: &str,
) -> Value {
    let mut out = match (outcome.status, outcome.result, outcome.error) {
        (status @ Status::Completed, Some(result), _) => {
            json!({ "ok": true, "id": id, "status": status.as_str(), "result": result })
        }
        (status, _, Some(error)) => json!({
            "ok": false,
            "id": id,
            "status": status.as_str(),
            "error": error,
        }),
        (status, _, _) => json!({
            "ok": false,
            "id": id,
            "status": status.as_str(),
            "error": "subagent completed without a text result",
        }),
    };
    if let Some(full_bytes) = outcome.full_bytes {
        out["truncated"] = json!(true);
        out["full_bytes"] = json!(full_bytes);
    }
    if let Some(usage) = outcome.usage {
        out["usage"] = usage;
    }
    if let Some(stopped) = outcome.stopped {
        out["stopped"] = json!(stopped);
    }
    if let Some(progress) = outcome.progress {
        for (key, value) in progress.as_object().into_iter().flatten() {
            out[key] = value.clone();
        }
    }
    if let Some(session_id) = session_id {
        out["session_id"] = json!(session_id);
    }
    // The settled block names the model the run used; the result text and
    // the error never carry it.
    out["model"] = json!(model);
    out
}

/// The finalized state of one subagent run.
pub(super) struct RunOutcome {
    pub(super) status: Status,
    pub(super) result: Option<String>,
    pub(super) error: Option<String>,
    // Byte length of the full answer when `result` was cut to fit the output
    // limit.
    pub(super) full_bytes: Option<usize>,
    // Token and cost totals of the run, when the provider reported any.
    pub(super) usage: Option<Value>,
    // Where the run got to (rounds, elapsed, last tool), for the settle
    // report. A restored entry keeps no live progress.
    pub(super) progress: Option<Value>,
    // The deadline that stopped the run, when one did: a run that answered
    // its summary request settles completed but did not finish its task.
    pub(super) stopped: Option<String>,
}

/// Spend the entry's earlier runs already billed, for its settle marker.
/// Empty when the entry is gone: a session switch clears the registry while
/// a resumed child still runs, and that marker prices its own run alone.
pub(super) fn carried_spend(registry: &SubagentRegistry, id: &str) -> Rollup {
    lock(&registry.entries)
        .get(id)
        .map(|entry| entry.carried.clone())
        .unwrap_or_default()
}

/// Append the run's terminal record as the leaf of the transcript so a
/// rehydrate after a restart restores this exact status and outcome instead
/// of inferring them from the transcript tail. Appended last, so the marker
/// is the physical last line of the file: the rehydration tail
/// scan relies on that. The serialized line is bounded by
/// [`SETTLED_MARKER_LIMIT`] so the tail window always contains it whole.
/// Returns whether the marker was persisted; an idle entry keeps its history
/// in memory when the transcript cannot reproduce it.
pub(super) fn record_settled(
    cursor: Option<&SessionCursor>,
    status: Status,
    result: Option<&str>,
    error: Option<&str>,
    stopped: Option<&str>,
    totals: &Rollup,
    carried: &Rollup,
) -> bool {
    let Some(cursor) = cursor else { return false };
    let (usage, cost) = if totals.is_empty() {
        (None, None)
    } else {
        (Some(totals.usage()), Some(totals.cost))
    };
    // Callers already cap both texts at DEFAULT_OUTPUT_LIMIT; the measure
    // loop is the hard guarantee that escaping and multi-byte UTF-8 cannot
    // push the serialized line past the tail window.
    let mut result = result.map(|text| cap_at_boundary(text, DEFAULT_OUTPUT_LIMIT));
    let error = error.map(|text| cap_at_boundary(text, DEFAULT_OUTPUT_LIMIT));
    // The earlier runs of this child, so a restart still totals them.
    let prior = (!carried.is_empty()).then(|| lofi_types::RunSpend::from(carried));
    let build = |result: &Option<String>| SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::RunSettled {
            status: status.as_str().to_string(),
            result: result.clone(),
            error: error.clone(),
            usage,
            cost,
            prior,
            stopped: stopped.map(str::to_string),
        },
    };
    let mut event = build(&result);
    while let Some(text) = result.as_ref() {
        let Ok(line) = serde_json::to_string(&event) else {
            break;
        };
        if line.len() <= SETTLED_MARKER_LIMIT || text.is_empty() {
            break;
        }
        result = Some(cap_at_boundary(text, text.len() / 2));
        event = build(&result);
    }
    let mut events = vec![event];
    if let Err(e) = cursor.append_events(&mut events) {
        tracing::debug!(error = %e, "subagent settle marker not recorded");
        return false;
    }
    true
}

/// Terminal spend of the child session a transcript records, through its
/// last settled run. A dialog row restored from the store shows the columns
/// a live one has, and the process that held the cost is gone by then.
pub(crate) fn settled_cost(path: &Path) -> Option<f64> {
    let run = read_settled_marker(path)?;
    let spent = run
        .usage
        .as_ref()
        .and_then(|usage| usage.get("cost"))
        .and_then(Value::as_f64);
    // An earlier run is still spend, even when this one billed nothing.
    if spent.is_none() && run.prior.is_empty() {
        return None;
    }
    Some(run.prior.cost + spent.unwrap_or(0.0))
}

/// A subagent run's terminal record as persisted by the `RunSettled` marker.
struct SettledRun {
    status: Status,
    result: Option<String>,
    error: Option<String>,
    usage: Option<Value>,
    /// Spend of the child session's earlier runs, which a resume claims
    /// before it installs its own totals.
    prior: Rollup,
    /// The deadline that stopped the run, when one did.
    stopped: Option<String>,
}

/// Restore the terminal record from the transcript tail. The marker is the
/// physical last line of a settled run; anything else at the tail — turn
/// events of a newer run, a torn final line without a newline — means the
/// process exited before the last run settled.
fn read_settled_marker(path: &Path) -> Option<SettledRun> {
    use std::io::{Read as _, Seek as _, SeekFrom};
    let mut file = std::fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    file.seek(SeekFrom::Start(len.saturating_sub(SETTLED_TAIL_WINDOW)))
        .ok()?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).ok()?;
    // A final line without its newline is a torn write, not a marker.
    if !buf.ends_with(b"\n") {
        return None;
    }
    let text = String::from_utf8(buf).ok()?;
    let line = text.lines().rev().find(|line| !line.trim().is_empty())?;
    let event = crate::session::store::parse_event(line).ok()?;
    if let SessionEventKind::RunSettled {
        status,
        result,
        error,
        usage,
        cost,
        prior,
        stopped,
    } = event.kind
    {
        let usage = usage.map(|usage| {
            json!({
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "cost": cost.unwrap_or(0.0),
            })
        });
        let carried = prior.as_ref().map(Rollup::from).unwrap_or_default();
        Some(SettledRun {
            status: Status::from_marker(&status),
            result,
            error,
            usage,
            prior: carried,
            stopped,
        })
    } else {
        None
    }
}

/// Terminal state a transcript restores: the status and outcome its marker
/// records, and the spend of the runs that marker carries. A transcript
/// whose last line is not a marker restores neither.
pub(super) fn restore_settled(path: &Path) -> (Status, Rollup, RunOutcome) {
    match read_settled_marker(path) {
        Some(settled) => (
            settled.status,
            settled.prior,
            RunOutcome {
                status: settled.status,
                result: settled.result,
                error: settled.error,
                full_bytes: None,
                usage: settled.usage,
                progress: None,
                // A restored entry keeps only what the marker recorded.
                stopped: settled.stopped,
            },
        ),
        None => (
            Status::Cancelled,
            Rollup::default(),
            RunOutcome {
                status: Status::Cancelled,
                result: None,
                error: Some("lofi exited before the run settled".to_string()),
                full_bytes: None,
                usage: None,
                progress: None,
                stopped: None,
            },
        ),
    }
}
