use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use lofi_error::{Error, Result};
use lofi_types::{ContentBlock, Message, PromptKind, Role};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use super::child::prepare;
use super::ops::{str_arg, u64_arg};
use super::registry::SubagentRegistry;
use super::settled::{carried_spend, outcome_json, record_settled, terminal_notice, RunOutcome};
use super::{
    Status, SubagentSupport, DEFAULT_IDLE_MS, DEFAULT_OUTPUT_LIMIT, EVENT_TEXT_CAP, KILL_GRACE,
};
use crate::agent::tool_executor::{ActivityWatch, ConfirmWatch};
use crate::agent::{close_unanswered_tool_uses, lock, AgentEvent, RunControls, TurnRequest};
use crate::backend::BackendRef;
use crate::session::recorder::TurnOutcome;
use crate::session::store::SessionCursor;

/// Signal the child turn to stop when the hook future is dropped mid-run
/// (parent turn aborted, exec interrupted). The spawned task observes the
/// flag, then finalizes the entry itself, so an abandoned run still ends
/// with a consistent history and a terminal status.
pub(super) struct CancelOnDrop {
    pub(super) cancel: Arc<AtomicBool>,
    pub(super) armed: bool,
}

impl CancelOnDrop {
    pub(super) fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            crate::agent::signal_cancel(&self.cancel);
        }
    }
}

/// Finalizes an entry when its task ends unwatched: a panic, or an abort
/// from `subagentKill`'s backstop or `clear()`. Normal completion disarms
/// it; on the kill path the guard is what settles the entry, keeps the
/// history the run reached, and reports its spend. All work is
/// synchronous: Drop cannot await.
struct SettleOnDrop {
    registry: SubagentRegistry,
    id: String,
    cursor: Option<SessionCursor>,
    // The run's history. The turn borrows this vector, so an abort that
    // drops the turn future leaves it here instead of destroying it.
    messages: Vec<Message>,
    cancel: Arc<AtomicBool>,
    // Spend collected by the event drain. An aborted run never reaches the
    // report at the end of `run_turn_with_timeout`, so the guard folds in
    // what the drain saw instead of losing it.
    totals: Arc<std::sync::Mutex<ChildUsage>>,
    generation: u64,
    // The guard carries the model because a panic or a kill settles without
    // the normal path, and the settled block still has to name it.
    model: String,
    armed: bool,
}

impl SettleOnDrop {
    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for SettleOnDrop {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        {
            let entries = lock(&self.registry.entries);
            let Some(entry) = entries.get(&self.id) else {
                return;
            };
            if entry.status.is_terminal() {
                return;
            }
        }
        let status = if self.cancel.load(Ordering::SeqCst) {
            Status::Cancelled
        } else {
            Status::Failed
        };
        // Read the entry's earlier runs before taking the run's own lock:
        // readers take the entries lock first and then this one.
        let carried = carried_spend(&self.registry, &self.id);
        // The marker and the spend report do blocking store I/O, so they
        // stay outside the entries lock.
        let persisted = record_settled(
            self.cursor.as_ref(),
            status,
            None,
            Some(if status == Status::Cancelled {
                "cancelled"
            } else {
                "subagent task ended unexpectedly"
            }),
            // The guard does not know a deadline: it settles an abandoned run.
            None,
            &lock(&self.totals),
            &carried,
        );
        // The ended run never reached the report at the end of
        // run_turn_with_timeout; fold in what the drain saw. The generation
        // check happens under the lock, as in the normal reporting path, so
        // a concurrent clear() cannot split them.
        {
            let mut unreported = lock(&self.registry.unreported);
            if self.registry.generation() == self.generation {
                lock(&self.totals).report(&mut unreported);
            }
        }
        // Usage the drain collected before the abort. Buffered events may
        // still flush after this read — the drain task cannot be awaited
        // from Drop — so the figure can be short by one round.
        let usage = lock(&self.totals).json();
        let mut entries = lock(&self.registry.entries);
        // clear() may have run during the unlocked section, and the next
        // session's rehydrate reinserts this id from the parent's transcripts.
        // That entry belongs to the new session, so the generation decides.
        if self.registry.generation() != self.generation {
            return;
        }
        let Some(entry) = entries.get_mut(&self.id) else {
            return;
        };
        let mut outcome = json!({
            "ok": false,
            "id": self.id,
            "status": status.as_str(),
            "model": self.model,
            "error": if status == Status::Cancelled {
                "cancelled"
            } else {
                "subagent task ended unexpectedly"
            },
        });
        if let Some(usage) = usage {
            outcome["usage"] = usage;
        }
        entry.status = status;
        // Restamp like the normal settle does: an aborted run that keeps
        // its start stamp becomes the oldest eviction victim, and a caller
        // reading the kill result right after could lose it.
        entry.used = self.registry.stamp();
        entry.backend = entry
            .backend
            .with_tool_executor(entry.backend.tool_executor().reset_exec_worker());
        entry.outcome = Some(outcome);
        if persisted {
            // The transcript holds what the run reached, and the resume
            // replays it; the replay closes the call the abort left open.
            entry.messages = Vec::new();
            entry.history_loaded = false;
        } else {
            close_unanswered_tool_uses(&mut self.messages);
            entry.messages = std::mem::take(&mut self.messages);
            entry.history_loaded = true;
        }
        entry
            .events
            .push("state", json!({ "status": status.as_str() }));
        // send, not send_replace, would drop the value when no waiter is
        // subscribed: a wait started after the settle would hang.
        entry.state_tx.send_replace(status);
        // The close belongs to this critical section: a waiter that sees the
        // terminal status may claim a resume at once, and a close issued
        // after the lock was released would shut the fresh run's address.
        self.registry.mailbox.close(&self.id, status.as_str());
        let notify = entry.notify_terminal;
        drop(entries);
        if notify && self.registry.generation() == self.generation {
            self.registry.publish_notice(
                &self.id,
                format!("subagent {} {}", self.id, status.as_str()),
            );
        }
    }
}

/// Why a subagent call did not start. A busy id is not a failure: its run
/// keeps going and can be resumed once it ends.
pub(super) enum RunRejection {
    Busy(String),
    Error(Error),
}

impl From<Error> for RunRejection {
    fn from(e: Error) -> Self {
        Self::Error(e)
    }
}

/// A started run: prepared, registered, and spawned on the host runtime.
pub(super) struct Started {
    pub(super) id: String,
    pub(super) session_id: Option<String>,
    pub(super) cancel: Arc<AtomicBool>,
    pub(super) handle: tokio::task::JoinHandle<()>,
}

/// Shared start path for foreground runs, resumes, and background spawns:
/// parse and prepare, then spawn the supervised task on the host runtime.
/// Spawning on the sandbox worker runtime would stop the run unsettled
/// when the worker idles out; `set_host_handle` records the host runtime
/// at the start of every parent turn.
pub(super) async fn start(
    template: &BackendRef,
    support: &SubagentSupport,
    registry: &SubagentRegistry,
    args: &Value,
    notify: bool,
) -> std::result::Result<Started, RunRejection> {
    let prompt = str_arg(args, "prompt")?
        .filter(|p| !p.trim().is_empty())
        .ok_or_else(|| Error::Tool("prompt is required and must be non-empty".into()))?
        .to_string();
    // `timeout_ms` stays the hard cap; `idle_ms` is the silence that stops a
    // stalled run long before the cap.
    let limits = Deadlines::from_args(args)?;
    let output_limit = u64_arg(args, "output_limit")?.map_or(DEFAULT_OUTPUT_LIMIT, |v| v as usize);
    // The allowance the caller named before the parent's clock cut it, kept so
    // the settled run can report both numbers.
    let requested_ms = u64_arg(args, "requested_timeout_ms")?;
    let token_budget = u64_arg(args, "token_budget")?;
    if token_budget == Some(0) {
        return Err(Error::Tool("token_budget must be positive when set".into()).into());
    }

    let id = str_arg(args, "id")?.map(str::to_string);
    if args.get("id").is_some_and(|v| !v.is_null()) && id.is_none() {
        return Err(Error::Tool("id must be a non-empty string".into()).into());
    }
    let (backend, messages, cursor, cancel, id) =
        prepare(template, support, registry, id, args, notify).await?;
    // The transcript id lets the caller locate the persisted session later;
    // the registry id alone is meaningless after a restart or an eviction.
    let session_id = cursor.as_ref().map(SessionCursor::id);
    // History length before this run: the result text considers only what
    // this run appended.
    let base_len = messages.len();
    // Capture the generation at admission, not in the task: a clear()
    // between the two would let the task publish into the new session.
    let generation = registry.generation();
    {
        let mut entries = lock(&registry.entries);
        if let Some(entry) = entries.get_mut(&id) {
            entry
                .events
                .push("prompt", json!({ "text": cap_event_text(&prompt) }));
        }
    }
    let handle = {
        let registry = registry.clone();
        let id = id.clone();
        let cancel = cancel.clone();
        let task_session_id = session_id.clone();
        registry.host_handle().spawn(async move {
            run_task(
                backend,
                messages,
                cursor,
                cancel,
                registry,
                id,
                base_len,
                limits,
                prompt,
                output_limit,
                token_budget,
                requested_ms,
                generation,
                task_session_id,
            )
            .await;
        })
    };
    if let Some(entry) = lock(&registry.entries).get_mut(&id) {
        entry.abort = Some(handle.abort_handle());
    }
    Ok(Started {
        id,
        session_id,
        cancel,
        handle,
    })
}

/// The supervised body of one run: the turn, then the terminal record. The
/// `SettleOnDrop` guard covers the paths this body never reaches (panic,
/// abort), so every run settles exactly once.
#[allow(clippy::too_many_arguments)]
async fn run_task(
    backend: BackendRef,
    messages: Vec<Message>,
    cursor: Option<SessionCursor>,
    cancel: Arc<AtomicBool>,
    registry: SubagentRegistry,
    id: String,
    base_len: usize,
    limits: Deadlines,
    prompt: String,
    output_limit: usize,
    token_budget: Option<u64>,
    requested_ms: Option<u64>,
    generation: u64,
    session_id: Option<String>,
) {
    let totals = Arc::new(std::sync::Mutex::new(ChildUsage::default()));
    let model = backend.run_model().label();
    if let Some(entry) = lock(&registry.entries).get_mut(&id) {
        // The status surfaces read this accumulator through the entry, so a
        // live cost needs no second copy of the run's spend.
        entry.totals = Some(totals.clone());
        entry.model = Some(model.clone());
    }
    let mut guard = SettleOnDrop {
        registry: registry.clone(),
        id: id.clone(),
        cursor: cursor.clone(),
        messages,
        cancel: cancel.clone(),
        totals: totals.clone(),
        generation,
        model: model.clone(),
        armed: true,
    };
    let outcome = run_turn_with_timeout(
        backend,
        &mut guard.messages,
        cursor.clone(),
        cancel,
        registry.clone(),
        id.clone(),
        base_len,
        limits,
        prompt,
        output_limit,
        token_budget,
        totals.clone(),
        generation,
    )
    .await;
    let status = outcome.status;
    // Retention cap: the kept body bounds result and error so an idle entry
    // holds KBs regardless of the caller's output_limit. The full answer
    // stays in the transcript.
    let mut retained = outcome;
    // A capped run reports what it asked for beside what it got, so a
    // background run read back later still says the parent cut it short.
    if let Some(asked) = requested_ms {
        if let Some(progress) = retained.progress.as_mut() {
            progress["requestedTimeoutMs"] = json!(asked);
        }
    }
    if let Some(result) = retained.result.as_mut() {
        if result.len() > DEFAULT_OUTPUT_LIMIT {
            retained.full_bytes = Some(retained.full_bytes.unwrap_or(result.len()));
            *result = cap_at_boundary(result, DEFAULT_OUTPUT_LIMIT);
        }
    }
    if let Some(error) = retained.error.as_mut() {
        *error = cap_at_boundary(error, DEFAULT_OUTPUT_LIMIT);
    }
    let out = outcome_json(&id, session_id.as_deref(), retained, &model);
    // Read the entry's earlier runs before taking the run's own lock:
    // readers take the entries lock first and then this one.
    let carried = carried_spend(&registry, &id);
    // Durable terminal record: after a restart the rehydration tail scan
    // restores exactly this status and outcome.
    let persisted = record_settled(
        cursor.as_ref(),
        status,
        out["result"].as_str(),
        out["error"].as_str(),
        out["stopped"].as_str(),
        &lock(&totals),
        &carried,
    );
    let notice_text = terminal_notice(&id, status, &out);
    // Terminal write in one critical section: a resume observes either
    // the running entry or the fully settled one, never half of either.
    let notify = {
        let mut entries = lock(&registry.entries);
        // Close inside this critical section: a send racing the settle sees
        // either a live recipient or a rejected one, a child blocked in
        // receive wakes with the closed result, and a resume claimed the
        // moment the terminal status is visible finds the inbox already shut.
        // A clear() during the marker write removed this entry and let the
        // next session's rehydrate reinsert the id: that entry is not ours.
        let same_session = registry.generation() == generation;
        if same_session {
            registry.mailbox.close(&id, status.as_str());
        }
        match entries.get_mut(&id) {
            Some(entry) if same_session => {
                entry.status = status;
                entry.outcome = Some(out);
                entry
                    .events
                    .push("state", json!({ "status": status.as_str() }));
                entry.state_tx.send_replace(status);
                if persisted {
                    // The transcript holds the full history; an idle entry
                    // retains only the outcome. The next resume replays
                    // from the transcript.
                    entry.messages = Vec::new();
                    entry.history_loaded = false;
                } else {
                    entry.messages = std::mem::take(&mut guard.messages);
                    entry.history_loaded = true;
                }
                entry.notify_terminal
            }
            // clear() removed the entry, or the next session reinserted the
            // id: nothing to record, nothing to say.
            _ => false,
        }
    };
    if notify && registry.generation() == generation {
        registry.publish_notice(&id, notice_text);
    }
    guard.disarm();
}

/// Drain the child event stream into usage totals and the entry's public event
/// log. Sender clones escape the turn through tool contexts, so the channel may
/// never close: the caller stops the drain explicitly and the task flushes what
/// is buffered before exiting.
fn spawn_event_drain(
    mut rx: mpsc::Receiver<AgentEvent>,
    totals: Arc<std::sync::Mutex<ChildUsage>>,
    cancel: Arc<AtomicBool>,
    token_budget: Option<u64>,
    registry: SubagentRegistry,
    id: String,
    progress: Arc<RunProgress>,
) -> (
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel::<()>();
    let drain = tokio::spawn(async move {
        let flush_batch = |batch: &[AgentEvent]| {
            if batch.is_empty() {
                return;
            }
            let mut public: Vec<(&'static str, Value)> = Vec::new();
            let mut text_accum = String::new();
            let flush_text = |text_accum: &mut String, public: &mut Vec<(&'static str, Value)>| {
                if !text_accum.is_empty() {
                    let chunk = cap_event_text(text_accum);
                    text_accum.clear();
                    public.push(("text", json!({ "text": chunk })));
                }
            };
            for event in batch {
                // Every event is progress: the deadline and the settle report
                // both read this.
                progress.note(event);
                if let AgentEvent::RoundUsage { cost, usage } = event {
                    let spent = {
                        let mut totals = lock(&totals);
                        totals.add(*cost, usage);
                        totals.total()
                    };
                    // Cancel as soon as the budget is crossed; the settle path
                    // reports the breach as the failure reason.
                    if token_budget.is_some_and(|budget| spent > budget) {
                        crate::agent::signal_cancel(&cancel);
                    }
                }
                match event {
                    AgentEvent::Text(text) => {
                        if !text_accum.is_empty()
                            && text_accum.len().saturating_add(text.len()) > EVENT_TEXT_CAP
                        {
                            flush_text(&mut text_accum, &mut public);
                        }
                        text_accum.push_str(text);
                    }
                    AgentEvent::ToolStart { name, .. } => {
                        flush_text(&mut text_accum, &mut public);
                        public.push(("tool_start", json!({ "name": name })));
                    }
                    AgentEvent::ToolEnd {
                        result, is_error, ..
                    } => {
                        flush_text(&mut text_accum, &mut public);
                        public.push((
                            "tool_end",
                            json!({ "result": cap_event_text(result), "is_error": is_error }),
                        ));
                    }
                    AgentEvent::RoundUsage { cost, usage } => {
                        flush_text(&mut text_accum, &mut public);
                        public.push(("usage", json!({ "cost": cost, "usage": usage })));
                    }
                    _ => {}
                }
            }
            flush_text(&mut text_accum, &mut public);
            if let Some(entry) = lock(&registry.entries).get_mut(&id) {
                for event in batch {
                    entry.activity.note(event);
                }
                for (kind, data) in public {
                    entry.events.push(kind, data);
                }
            }
        };
        let mut batch = Vec::new();
        loop {
            batch.clear();
            tokio::select! {
                _ = &mut stop_rx => {
                    while let Ok(event) = rx.try_recv() {
                        batch.push(event);
                    }
                    flush_batch(&batch);
                    break;
                }
                event = rx.recv() => match event {
                    Some(first) => {
                        batch.push(first);
                        while let Ok(next) = rx.try_recv() {
                            batch.push(next);
                        }
                        flush_batch(&batch);
                    }
                    None => break,
                },
            }
        }
    });
    (stop_tx, drain)
}

/// Truncate at `cap` bytes, backing up to a UTF-8 boundary:
/// `String::truncate` panics off one.
pub(super) fn cap_at_boundary(text: &str, cap: usize) -> String {
    let mut text = text.to_string();
    if text.len() > cap {
        let mut end = cap;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
    text
}

fn cap_event_text(text: &str) -> String {
    cap_at_boundary(text, EVENT_TEXT_CAP)
}

/// Restamp the entry, drop its sandbox thread, and fold the run's spend into
/// the session's unreported total, which the parent turn loop reports after
/// the current tool round. Skipped when `clear()` removed the entry or the
/// generation moved on: that spend belongs to a session this registry no
/// longer owns.
fn settle_entry(
    registry: &SubagentRegistry,
    id: &str,
    totals: &Arc<std::sync::Mutex<ChildUsage>>,
    generation: u64,
) {
    let present = if let Some(entry) = lock(&registry.entries).get_mut(id) {
        // Restamp at settle: the stamp orders eviction, and a long run that
        // finishes last must not become the oldest victim.
        entry.used = registry.stamp();
        // A settled run is no owner any more, whatever ended it: the
        // reminders it armed would otherwise keep waking this session for a
        // run that is not running, the harm a kill already prevents.
        let tools = entry.backend.tool_executor();
        let _ = tools.timers().cancel_owner(Some(id));
        // Idle entries must not hold a sandbox thread: it restarts lazily on
        // the first exec of the next resume.
        entry.backend = entry.backend.with_tool_executor(tools.reset_exec_worker());
        true
    } else {
        false
    };
    if present {
        let mut unreported = lock(&registry.unreported);
        if registry.generation() == generation {
            lock(totals).report(&mut unreported);
        }
    }
}

/// Run one turn until it finishes or a deadline stops it. Returns the turn's
/// result (a cancelled turn when a deadline fired) with the deadline that
/// fired and the prompt the child was waiting on when it did.
#[allow(clippy::too_many_arguments)]
async fn run_under_watchdog(
    backend: &BackendRef,
    messages: &mut Vec<Message>,
    tx: mpsc::Sender<AgentEvent>,
    cursor: Option<&SessionCursor>,
    cancel: &Arc<AtomicBool>,
    prompt: String,
    progress: &Arc<RunProgress>,
    limits: &Deadlines,
) -> (Result<TurnOutcome>, Option<(Expiry, Option<String>)>) {
    // The turn stops on its own flag: the run's flag stays the caller's, so
    // a deadline that stops this turn does not read as a kill to the summary
    // turn that follows it.
    let stopping = crate::agent::new_cancel_flag();
    let cancel_notify = crate::agent::turn_session::lookup_cancel_notify(cancel);
    // A run whose task is aborted (a kill, a /new, a panic) still stops the
    // work it started: the sandbox watches this flag, not the task.
    let _stop_on_drop = CancelOnDrop {
        cancel: stopping.clone(),
        armed: true,
    };
    // The bounds are not announced as a notice: a notice is recorded as a turn
    // prompt, so it would sit in the transcript and replay stale on a resume.
    // The watchdog enforces them, and the settle report names the one that
    // fired.
    let request = TurnRequest::prompt(prompt, PromptKind::Task);
    let turn = backend.run_turn(
        messages,
        // The task is parent-written, so it must not read as user
        // authorization in the child's auto-mode context.
        request,
        tx,
        cursor,
        RunControls {
            cancel: Some(stopping.clone()),
            preempt: None,
        },
    );
    tokio::pin!(turn);
    let mut expired = None;
    let outcome = loop {
        tokio::select! {
            biased;
            res = &mut turn => break res,
            () = crate::agent::turn_session::wait_cancel_flag(cancel, cancel_notify.as_ref()) => {
                crate::agent::signal_cancel(&stopping);
                break tokio::time::timeout(KILL_GRACE, &mut turn)
                    .await
                    .unwrap_or(Ok(TurnOutcome::Cancelled));
            }
            () = tokio::time::sleep(WATCHDOG_TICK) => {
                // A kill, an Esc, or a dropped caller stops the turn it
                // started, and the run then settles cancelled. The turn still
                // finishes on its own so the round it reached is recorded and
                // the transcript sealed; only a turn that ignores the flag is
                // dropped, after the grace a kill already allows.
                if cancel.load(Ordering::SeqCst) {
                    crate::agent::signal_cancel(&stopping);
                    break tokio::time::timeout(KILL_GRACE, &mut turn)
                        .await
                        .unwrap_or(Ok(TurnOutcome::Cancelled));
                }
                if let Some(reason) = progress.expired(limits) {
                    // An open prompt settles as `blocked` only while it is
                    // still pending when the settle reads it, so give the ask
                    // a tick to be answered or to time out first: the same
                    // stop then reads the same way whichever timer is polled
                    // first.
                    if matches!(reason, Expiry::Cap(_)) && progress.confirms.pending() > 0 {
                        tokio::time::sleep(WATCHDOG_TICK).await;
                    }
                    // Read the open prompt now: dropping the turn closes it,
                    // and a blocked settle names what it waited on.
                    let waiting = (progress.confirms.pending() > 0)
                        .then(|| progress.confirms.command())
                        .flatten();
                    // The turn keeps the round it is in: a dropped future
                    // takes its partial answer with it. A turn that ignores
                    // the flag is dropped after the grace a kill allows; one
                    // blocked on a prompt has no partial to keep and closes
                    // that prompt now, so an answer in the window cannot
                    // start the command under a set stop flag.
                    let unwinding = waiting.is_none();
                    expired = Some((reason, waiting));
                    crate::agent::signal_cancel(&stopping);
                    if unwinding {
                        let _ = tokio::time::timeout(KILL_GRACE, &mut turn).await;
                    }
                    break Ok(TurnOutcome::Cancelled);
                }
            }
        }
    };
    (outcome, expired)
}

/// Run one turn to completion and finalize the registry entry: the open
/// tool call a timeout closes, the restamp, and the sandbox thread of an
/// idle entry. The caller owns `messages`, so the history outlives an
/// abort of this future.
#[allow(clippy::too_many_arguments)]
async fn run_turn_with_timeout(
    backend: BackendRef,
    messages: &mut Vec<Message>,
    cursor: Option<SessionCursor>,
    cancel: Arc<AtomicBool>,
    registry: SubagentRegistry,
    id: String,
    base_len: usize,
    limits: Deadlines,
    prompt: String,
    output_limit: usize,
    token_budget: Option<u64>,
    // Created by the caller so the settle guard can report an aborted run's
    // partial spend; the event drain fills it as RoundUsage events arrive.
    totals: Arc<std::sync::Mutex<ChildUsage>>,
    // Captured at admission: a /new or /resume bumps it, and a detached
    // settle must not fold spend into the new session.
    generation: u64,
) -> RunOutcome {
    let progress = Arc::new(RunProgress::new(
        run_started(),
        backend.tool_executor().confirm_watch.clone(),
        backend.tool_executor().activity.clone(),
    ));
    let (tx, rx) = mpsc::channel::<AgentEvent>(256);
    // The child events have no consumer: the parent transcript records the
    // tool call, not the child stream. Collect the usage so the spend is
    // still visible in the tool result; drain the rest.
    let (stop_tx, drain) = spawn_event_drain(
        rx,
        totals.clone(),
        cancel.clone(),
        token_budget,
        registry.clone(),
        id.clone(),
        progress.clone(),
    );
    // A watchdog, not one timeout: a child that keeps producing events is
    // working, and a child waiting on the user is not idle at all.
    let (outcome, expired) = run_under_watchdog(
        &backend,
        messages,
        tx,
        cursor.as_ref(),
        &cancel,
        prompt.clone(),
        &progress,
        &limits,
    )
    .await;
    // Sender clones may outlive the turn, so the channel may never close:
    // stop the drain explicitly; it flushes the buffer before exiting.
    let _ = stop_tx.send(());
    let _ = drain.await;
    // A stopped run is wrapped up: one short turn asks what it reached, so
    // the parent reads a summary. A run stopped while the user owed it an
    // answer never reaches its model.
    let blocked = match &expired {
        Some((Expiry::Cap(_), Some(command))) => Some(command.clone()),
        _ => None,
    };
    // The progress report describes the run, not the summary that follows it,
    // and states the budget it ran under: the elapsed time in it is what the
    // cap was spent on.
    let mut reported = progress.json();
    if let Some(cap) = limits.cap {
        reported["cap_ms"] = json!(u64::try_from(cap.as_millis()).unwrap_or(u64::MAX));
    }
    // A turn dropped mid-call (a deadline, or a kill that outran its grace)
    // leaves a tool call no provider will accept, so the history is repaired
    // before it settles. A stopped run keeps what it reached either way, so a
    // resume continues instead of paying for the work again.
    close_unanswered_tool_uses(messages);
    let mut wrapped = false;
    // A caller that already stopped the run, or a budget the drain's last
    // flush crossed, gets no summary request: it would only be dropped, and
    // its spend would never reach the drain.
    if expired.is_some() && blocked.is_none() && !cancel.load(Ordering::SeqCst) {
        wrapped = wrap_up(
            &backend,
            messages,
            cursor.as_ref(),
            &totals,
            &registry,
            &id,
            &progress,
            token_budget,
            &limits,
            &cancel,
        )
        .await;
        // A summary cut mid-call leaves the same unpaired call behind.
        close_unanswered_tool_uses(messages);
    }
    // The terminal text only promises a resume when the history holds a reply
    // or a tool exchange: the turn's own work, an earlier run's, and whatever
    // round a summary cut mid-exchange still completed.
    let kept = holds_work(messages);
    // The wrap-up spent tokens too, so a breached budget is only visible
    // now; the drain flushed, so it also catches a round the provider
    // finished before noticing the cancel flag.
    let budget_over = breach(&totals, token_budget);
    // A kill, an Esc or a dropped caller that landed while the summary ran
    // is a cancellation, not the deadline that had already passed.
    let stopped_by_caller = cancel.load(Ordering::SeqCst) && budget_over.is_none();
    let (mut status, mut error) = match (&expired, budget_over) {
        (None, _) => classify_outcome(outcome, budget_over, &id, kept),
        // The summary's own spend can cross the budget: a stopped run keeps
        // the failure contract of an ordinary turn.
        (Some(_), Some((spent, budget))) => budget_failure(spent, budget, &id, kept),
        (Some(_), None) if stopped_by_caller => (Status::Cancelled, Some("cancelled".to_string())),
        (Some(_), None) if wrapped => (Status::Completed, None),
        (Some((reason, _)), None) => deadline_status(*reason, blocked.as_deref(), &id, kept),
    };
    let (result, full_bytes) = if status == Status::Completed {
        last_text(&messages[base_len..], output_limit)
            .map_or((None, None), |(text, full)| (Some(text), full))
    } else {
        (None, None)
    };
    // A completed run without a final text is a failure: reporting
    // an empty string as a successful result would hide it. A summary that
    // came back empty is no summary, so a stopped run keeps the deadline
    // status and its resume hint instead.
    if status == Status::Completed && result.is_none() {
        match &expired {
            Some((reason, _)) => {
                (status, error) = deadline_status(*reason, blocked.as_deref(), &id, kept);
            }
            None => {
                (status, error) = (
                    Status::Failed,
                    Some("subagent completed without a text result".to_string()),
                );
            }
        }
    }
    // The terminal status is NOT set here: `run_task` writes status,
    // outcome, event, and the watch state in one critical section. Setting
    // it early would let a resume claim the entry in between and this
    // run's stale outcome would then land on the new run's waiters.
    settle_entry(&registry, &id, &totals, generation);
    let usage = lock(&totals).json();
    RunOutcome {
        status,
        result,
        error,
        full_bytes,
        usage,
        progress: Some(reported),
        stopped: expired.map(|(reason, _)| reason.as_str().to_string()),
    }
}

/// The budget a run crossed, if it did: the provider reports it per round,
/// and the wrap-up's own spend is only visible once it flushed.
fn breach(
    totals: &Arc<std::sync::Mutex<ChildUsage>>,
    token_budget: Option<u64>,
) -> Option<(u64, u64)> {
    let spent = lock(totals).total();
    token_budget
        .filter(|&budget| spent > budget)
        .map(|budget| (spent, budget))
}

/// One short turn asking a stopped run what it reached, so the parent reads
/// a summary instead of a bare timeout. Returns whether the child answered;
/// the caller settles `completed` on the answer and keeps the deadline
/// status when it has none.
#[allow(clippy::too_many_arguments)]
async fn wrap_up(
    backend: &BackendRef,
    messages: &mut Vec<Message>,
    cursor: Option<&SessionCursor>,
    totals: &Arc<std::sync::Mutex<ChildUsage>>,
    registry: &SubagentRegistry,
    id: &str,
    progress: &Arc<RunProgress>,
    token_budget: Option<u64>,
    limits: &Deadlines,
    // The run's own stop flag: a kill or a dropped caller stops the summary
    // like it stops any other turn.
    run_cancel: &Arc<AtomicBool>,
) -> bool {
    let (tx, rx) = mpsc::channel::<AgentEvent>(256);
    // The wrap-up is a request like any other, so its spend counts too; the
    // turn boundary keeps a per-turn cumulative cost from replacing the
    // run's total.
    lock(totals).start_turn();
    let cancel = crate::agent::new_cancel_flag();
    let cancel_notify = crate::agent::turn_session::lookup_cancel_notify(&cancel);
    let run_cancel_notify = crate::agent::turn_session::lookup_cancel_notify(run_cancel);
    // Dropping this future (an aborted caller, a panic) still stops the
    // sandbox work a summary call may have started.
    let _stop = CancelOnDrop {
        cancel: cancel.clone(),
        armed: true,
    };
    let (stop_tx, drain) = spawn_event_drain(
        rx,
        totals.clone(),
        cancel.clone(),
        token_budget,
        registry.clone(),
        id.to_string(),
        progress.clone(),
    );
    let turn = backend.run_turn(
        messages,
        TurnRequest::prompt(WRAP_UP_PROMPT.to_string(), PromptKind::Task),
        tx,
        cursor,
        RunControls {
            cancel: Some(cancel.clone()),
            preempt: None,
        },
    );
    tokio::pin!(turn);
    // The summary is deliberately outside the run's own bound, so it runs
    // under a budget of its own: a cap the caller set still bounds it, so a
    // tight caller never waits much past the deadline it named.
    let budget = limits
        .cap
        .map_or(WRAP_UP_TIMEOUT, |cap| WRAP_UP_TIMEOUT.min(cap));
    let deadline = tokio::time::sleep(budget);
    tokio::pin!(deadline);
    // A cap shorter than the wrap-up budget bounds it too, so a test or a
    // tight caller never waits much past the deadline it set.
    let finished = tokio::select! {
        biased;
        res = &mut turn => Some(res),
        () = &mut deadline => {
            // The summary keeps the round it is in for the same reason
            // the turn above does: dropping the future takes the reply
            // it was streaming, and its message with it. The deadline
            // stays authoritative, so this is not a wrap-up that
            // finished.
            crate::agent::signal_cancel(&cancel);
            let _ = tokio::time::timeout(KILL_GRACE, &mut turn).await;
            None
        }
        () = crate::agent::turn_session::wait_cancel_flag(
            run_cancel,
            run_cancel_notify.as_ref(),
        ) => {
            crate::agent::signal_cancel(&cancel);
            Some(
                tokio::time::timeout(KILL_GRACE, &mut turn)
                    .await
                    .unwrap_or(Ok(TurnOutcome::Cancelled)),
            )
        }
        () = crate::agent::turn_session::wait_cancel_flag(
            &cancel,
            cancel_notify.as_ref(),
        ) => {
            crate::agent::signal_cancel(&cancel);
            Some(
                tokio::time::timeout(KILL_GRACE, &mut turn)
                    .await
                    .unwrap_or(Ok(TurnOutcome::Cancelled)),
            )
        }
    };
    crate::agent::signal_cancel(&cancel);
    let _ = stop_tx.send(());
    let _ = drain.await;
    matches!(
        finished,
        Some(Ok(
            TurnOutcome::Finished | TurnOutcome::ContextPressure { resume: false }
        ))
    )
}

/// Whether the history holds work a resume would continue from: a reply with
/// text or a tool call, a tool result, or an image. An empty message, or one
/// whose only thinking was cut before its signature, carries nothing a
/// request replays, so a resume would have no history to continue from and
/// the hint must not promise one.
pub(super) fn holds_work(messages: &[Message]) -> bool {
    messages
        .iter()
        .filter(|message| matches!(message.role, Role::Assistant | Role::Tool))
        .any(|message| {
            message.blocks.iter().any(|block| match block {
                ContentBlock::Text { text } => !text.trim().is_empty(),
                ContentBlock::ToolUse { .. }
                | ContentBlock::ToolResult { .. }
                | ContentBlock::Image { .. } => true,
                ContentBlock::Thinking {
                    text, signature, ..
                } => !text.trim().is_empty() && signature.as_deref().is_some_and(|s| !s.is_empty()),
                ContentBlock::PartSignature { .. } => false,
            })
        })
}

/// Map the turn result to the terminal status. `budget_over` is `(spent,
/// budget)` when the run crossed its token budget; the drain flushed before
/// this call, so the breach is visible even when the provider finished the
/// round before noticing the cancel flag.
fn classify_outcome(
    outcome: Result<TurnOutcome>,
    budget_over: Option<(u64, u64)>,
    id: &str,
    kept: bool,
) -> (Status, Option<String>) {
    match outcome {
        Ok(outcome) => {
            if let Some((spent, budget)) = budget_over {
                budget_failure(spent, budget, id, kept)
            } else {
                match outcome {
                    TurnOutcome::Cancelled | TurnOutcome::Detached => {
                        (Status::Cancelled, Some("cancelled".to_string()))
                    }
                    // A pressure turn stops with the window full: a resume does
                    // not compact the child history, so it would stop again.
                    TurnOutcome::ContextPressure { resume: true } => {
                        (Status::Failed, Some("context window is full".to_string()))
                    }
                    // A child that stopped on its own keeps the answer it
                    // wrote, and the parent reads it as the child's result.
                    TurnOutcome::Finished | TurnOutcome::ContextPressure { resume: false } => {
                        (Status::Completed, None)
                    }
                    // A failed turn settles as `Err` from the turn session, so
                    // this arm is a guard on that contract, not a live path.
                    TurnOutcome::Failed(error) => (Status::Failed, Some(error)),
                }
            }
        }
        Err(e) => (Status::Failed, Some(e.to_string())),
    }
}

/// The status and text of a run that crossed its token budget: the parent
/// reads it as a failure, with the resume path when work was kept.
fn budget_failure(spent: u64, budget: u64, id: &str, kept: bool) -> (Status, Option<String>) {
    (
        Status::Failed,
        Some(resume_hint(
            &format!("token budget exceeded ({spent} > {budget})"),
            id,
            kept,
        )),
    )
}

/// The terminal status of a run its deadlines stopped. A run that kept its
/// work names the resume path, like a timeout always did.
fn deadline_status(
    expiry: Expiry,
    blocked: Option<&str>,
    id: &str,
    kept: bool,
) -> (Status, Option<String>) {
    match (expiry, blocked) {
        (_, Some(command)) => (
            Status::Blocked,
            Some(resume_hint(
                &format!("blocked on a confirmation prompt for {command}"),
                id,
                kept,
            )),
        ),
        (Expiry::Idle(idle), None) => (
            Status::TimedOut,
            Some(resume_hint(
                &format!("idle for {} ms", idle.as_millis()),
                id,
                kept,
            )),
        ),
        (Expiry::Cap(cap), None) => (
            Status::TimedOut,
            Some(resume_hint(
                &format!("timed out after {} ms", cap.as_millis()),
                id,
                kept,
            )),
        ),
    }
}

/// Append the resume path to a terminal error. The parent reads the first
/// line of the error inside `NOTICE_DETAIL_CAP` bytes, so the hint shares
/// that line with the reason instead of following it. A resume of a run that
/// kept nothing replays only its task prompt, which the parent would write
/// again anyway, so say that instead of promising work.
fn resume_hint(reason: &str, id: &str, kept: bool) -> String {
    if kept {
        format!("{reason}; resume with id {id} to continue from the work so far")
    } else {
        format!("{reason}; the run kept no work, so a resume would start over")
    }
}

/// Wall-clock bounds on one run: an idle deadline every event resets, and an
/// optional cap that no event postpones. There is no default cap: a fixed
/// clock cuts exactly the long-horizon work a subagent exists for, and a run
/// that keeps producing events is working, not stuck.
#[derive(Clone, Copy)]
pub(super) struct Deadlines {
    pub(super) idle: Duration,
    pub(super) cap: Option<Duration>,
}

impl Deadlines {
    pub(super) fn from_args(args: &Value) -> Result<Self> {
        let idle = Duration::from_millis(u64_arg(args, "idle_ms")?.unwrap_or(DEFAULT_IDLE_MS));
        // A zero idle deadline would stop the run on its first tick, before
        // any child work: reject it instead of silently killing the run.
        if idle.is_zero() {
            return Err(Error::Tool("idle_ms must be greater than zero".to_string()));
        }
        Ok(Self {
            idle,
            cap: u64_arg(args, "timeout_ms")?.map(Duration::from_millis),
        })
    }
}

/// Which bound of a run passed, carrying the value that fired so a settle
/// report can name it without reaching back into the configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Expiry {
    /// The cap.
    Cap(Duration),
    /// The idle allowance.
    Idle(Duration),
}

impl Expiry {
    /// The word a settle report uses for it.
    fn as_str(self) -> &'static str {
        match self {
            Self::Cap(_) => "cap",
            Self::Idle(_) => "idle",
        }
    }
}

/// The instant a run began, so the watchdog cannot grant a task that is
/// scheduled late a fresh allowance.
fn run_started() -> std::time::Instant {
    std::time::Instant::now()
}

/// How long the watchdog sleeps between checks. Small enough that a short
/// idle deadline means what a test says, cheap enough to ignore: it takes no
/// lock a child's turn needs.
const WATCHDOG_TICK: Duration = Duration::from_millis(25);

/// One short turn after a deadline stops a run, so the parent learns what the
/// child reached instead of only that it stopped. Small next to any deadline.
const WRAP_UP_TIMEOUT: Duration = Duration::from_mins(1);

const WRAP_UP_PROMPT: &str = "Your run was stopped at its deadline. In this reply only, state what you reached and what remains, in at most a few lines, and nothing else. Do not call tools in this reply.";

/// What a live run has done: the deadline reads it, and the settle report
/// names it, so a stopped run says where it stopped.
pub(super) struct RunProgress {
    pub(super) started: std::time::Instant,
    /// Milliseconds since `started` of the last event the child produced.
    pub(super) last_event: AtomicU64,
    rounds: AtomicU64,
    /// Milliseconds since `started` of the answer wait already spent at
    /// `last_event`; only the wait since that event is the child's own.
    waited_at_event: AtomicU64,
    last_tool: std::sync::Mutex<Option<String>>,
    /// Prompts this child is waiting on: the idle clock pauses while any is
    /// open, because the child cannot make progress until the user answers.
    confirms: Arc<ConfirmWatch>,
    /// Calls this child is running: the idle clock pauses while any runs,
    /// because a call reports nothing until it ends.
    activity: Arc<ActivityWatch>,
}

impl RunProgress {
    pub(super) fn new(
        started: std::time::Instant,
        confirms: Arc<ConfirmWatch>,
        activity: Arc<ActivityWatch>,
    ) -> Self {
        Self {
            started,
            last_event: AtomicU64::new(0),
            rounds: AtomicU64::new(0),
            waited_at_event: AtomicU64::new(confirms.waited().as_millis() as u64),
            last_tool: std::sync::Mutex::new(None),
            confirms,
            activity,
        }
    }

    /// Note one drained event. Every event counts as progress, not only tool
    /// rounds: a long stream of deltas is a working child.
    fn note(&self, event: &AgentEvent) {
        self.last_event
            .store(self.started.elapsed().as_millis() as u64, Ordering::SeqCst);
        self.waited_at_event
            .store(self.confirms.waited().as_millis() as u64, Ordering::SeqCst);
        match event {
            AgentEvent::RoundUsage { .. } => {
                self.rounds.fetch_add(1, Ordering::SeqCst);
            }
            // A nested call of the child's own agent loop reports the same
            // way, so both name what the child last reached for.
            AgentEvent::ToolStart { name, .. } | AgentEvent::NativeToolStart { name, .. } => {
                *lock(&self.last_tool) = Some(name.clone());
            }
            _ => {}
        }
    }

    /// The hard cap, once it has passed: the one deadline a run cannot work
    /// through.
    fn cap_expiry(&self, limits: &Deadlines) -> Option<Expiry> {
        let cap = limits.cap?;
        (self.started.elapsed() >= cap).then_some(Expiry::Cap(cap))
    }

    /// The deadline that has passed, if any. A run waiting on the user, or
    /// busy inside a tool call that reports nothing until it ends, is
    /// working: only the hard cap stops it.
    pub(super) fn expired(&self, limits: &Deadlines) -> Option<Expiry> {
        if let Some(cap) = self.cap_expiry(limits) {
            return Some(cap);
        }
        let elapsed = self.started.elapsed();
        let (busy, free) = self.activity.idle_mark();
        if self.confirms.pending() > 0 || busy > 0 {
            return None;
        }
        let last_event = Duration::from_millis(self.last_event.load(Ordering::SeqCst));
        // Answer waits are the user's time, not the child's: only the wait
        // since the last event is subtracted, so one long approval does not
        // stretch every later idle window.
        let waited = self.confirms.waited().saturating_sub(Duration::from_millis(
            self.waited_at_event.load(Ordering::SeqCst),
        ));
        let idle = elapsed.saturating_sub(last_event).saturating_sub(waited);
        // A call that has just ended is activity the drain has not noted yet:
        // without this, a tick racing the end of a long call stops a run that
        // was working all along.
        let idle = match free {
            Some(free) => idle.min(free),
            None => idle,
        };
        idle.ge(&limits.idle).then_some(Expiry::Idle(limits.idle))
    }

    /// Where the run got to, as the settle report and the notice name it.
    fn json(&self) -> Value {
        let mut out = json!({
            "rounds": self.rounds.load(Ordering::SeqCst),
            "elapsed_ms": self.started.elapsed().as_millis() as u64,
        });
        if let Some(tool) = lock(&self.last_tool).clone() {
            out["last_tool"] = json!(tool);
        }
        out
    }
}

/// Usage totals of one subagent run, collected from the drained events.
#[derive(Default)]
pub(super) struct ChildUsage {
    pub(super) input: u64,
    pub(super) output: u64,
    pub(super) cache_read: u64,
    pub(super) cache_write: u64,
    pub(super) cost: f64,
    /// The cost the current turn has reported so far.
    turn_cost: f64,
    pub(super) seen: bool,
}

impl ChildUsage {
    fn add(&mut self, cost: f64, usage: &lofi_types::Usage) {
        self.input += usage.input_tokens;
        self.output += usage.output_tokens;
        self.cache_read += usage.cache_read_tokens;
        self.cache_write += usage.cache_write_tokens;
        // RoundUsage reports the turn's cumulative cost: add what this turn
        // billed on top of the run's other turns.
        self.cost += cost - self.turn_cost;
        self.turn_cost = cost;
        self.seen = true;
    }

    /// Start another turn of the same run: its cost counts from zero again.
    fn start_turn(&mut self) {
        self.turn_cost = 0.0;
    }

    /// Total tokens across every category the provider reports.
    fn total(&self) -> u64 {
        self.input + self.output + self.cache_read + self.cache_write
    }

    pub(super) fn json(&self) -> Option<Value> {
        self.seen.then(|| {
            json!({
                "input_tokens": self.input,
                "output_tokens": self.output,
                "cache_read_tokens": self.cache_read,
                "cache_write_tokens": self.cache_write,
                "cost": self.cost,
            })
        })
    }

    /// Fold the run totals into the session's unreported subagent spend.
    fn report(&self, unreported: &mut (f64, lofi_types::Usage)) {
        if self.seen {
            unreported.0 += self.cost;
            unreported.1.input_tokens += self.input;
            unreported.1.output_tokens += self.output;
            unreported.1.cache_read_tokens += self.cache_read;
            unreported.1.cache_write_tokens += self.cache_write;
        }
    }
}

/// Answer of the run: the text of its final assistant message, truncated to
/// `limit` bytes at a UTF-8 boundary. `messages` must cover only the run in
/// progress, so a resumed subagent never returns an earlier turn's answer.
/// A run whose final message has no text has no answer: returning an earlier
/// message would report an intermediate note as the final result. Returns
/// the text and, when it was cut, the full byte length.
fn last_text(messages: &[Message], limit: usize) -> Option<(String, Option<usize>)> {
    let message = messages.last()?;
    if message.role != Role::Assistant {
        return None;
    }
    // Text blocks join with a newline: a provider may split one answer into
    // several blocks (thinking blocks in between), and bare concatenation
    // would glue words together.
    let mut text: String = message
        .blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() {
        return None;
    }
    let full = (text.len() > limit).then_some(text.len());
    if full.is_some() {
        let mut end = limit;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
    Some((text, full))
}
