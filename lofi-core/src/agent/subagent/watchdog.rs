use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use lofi_error::{Error, Result};
use lofi_types::{Message, PromptKind};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use super::ops::u64_arg;
use super::registry::{Rollup, SubagentRegistry};
use super::runner::{spawn_event_drain, CancelOnDrop};
use super::{DEFAULT_IDLE_MS, KILL_GRACE};
use crate::agent::tool_executor::{ActivityWatch, ConfirmWatch};
use crate::agent::{lock, AgentEvent, RunControls, TurnRequest};
use crate::backend::BackendRef;
use crate::session::recorder::TurnOutcome;
use crate::session::store::SessionCursor;

/// How long the watchdog sleeps between checks. Small enough that a short
/// idle deadline means what a test says, cheap enough to ignore: it takes no
/// lock a child's turn needs.
const WATCHDOG_TICK: Duration = Duration::from_millis(25);

/// One short turn after a deadline stops a run, so the parent learns what the
/// child reached instead of only that it stopped. Small next to any deadline.
const WRAP_UP_TIMEOUT: Duration = Duration::from_mins(1);

const WRAP_UP_PROMPT: &str = "Your run was stopped at its deadline. In this reply only, state what you reached and what remains, in at most a few lines, and nothing else. Do not call tools in this reply.";

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
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Cap(_) => "cap",
            Self::Idle(_) => "idle",
        }
    }
}

/// The instant a run began, so the watchdog cannot grant a task that is
/// scheduled late a fresh allowance.
pub(super) fn run_started() -> std::time::Instant {
    std::time::Instant::now()
}

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
    pub(super) fn note(&self, event: &AgentEvent) {
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
    pub(super) fn json(&self) -> Value {
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

/// Run one turn until it finishes or a deadline stops it. Returns the turn's
/// result (a cancelled turn when a deadline fired) with the deadline that
/// fired and the prompt the child was waiting on when it did.
#[allow(clippy::too_many_arguments)]
pub(super) async fn run_under_watchdog(
    backend: &BackendRef,
    messages: &mut Vec<Message>,
    tx: mpsc::Sender<AgentEvent>,
    cursor: Option<&SessionCursor>,
    cancel: &Arc<AtomicBool>,
    request: TurnRequest,
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

/// One short turn asking a stopped run what it reached, so the parent reads
/// a summary instead of a bare timeout. Returns whether the child answered;
/// the caller settles `completed` on the answer and keeps the deadline
/// status when it has none.
#[allow(clippy::too_many_arguments)]
pub(super) async fn wrap_up(
    backend: &BackendRef,
    messages: &mut Vec<Message>,
    cursor: Option<&SessionCursor>,
    totals: &Arc<std::sync::Mutex<Rollup>>,
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
