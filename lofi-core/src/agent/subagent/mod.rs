//! In-process subagents: a `lofi.subagent` hook that runs a child backend turn
//! on a forked message history and returns the child's final text to the
//! caller.
//!
//! The sandbox exposes the hook as `lofi.subagent`. Each spawn registers an
//! entry in the session's [`SubagentRegistry`]; passing `id` resumes that
//! entry with its stored history instead of starting fresh.
//!
//! A child is an ordinary backend from the same factory as its parent, so a
//! CLI parent spawns an HTTP child and the reverse. Nothing here branches on
//! the transport.

mod child;
mod ops;
mod registry;
mod runner;
mod settled;

#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::time::Duration;

use lofi_code::{SubagentFn, SubagentOp};
use serde_json::Value;

pub use child::SubagentSupport;
#[cfg(test)]
pub(in crate::agent) use child::{child_executor, inherited_request, isolate_auto_mode};
pub(in crate::agent) use registry::Rollup;
pub use registry::SubagentRegistry;
pub(crate) use settled::settled_cost;
#[cfg(test)]
pub(in crate::agent) use settled::terminal_notice;

use super::mailbox::PARENT_ADDRESS;
use super::Agent;
use crate::backend::BackendRef;
use crate::session::store::SessionStore;

/// A run that produces no event for this long is stopped: a stalled child
/// must not hold its entry, and nothing else ends a run by default.
pub(super) const DEFAULT_IDLE_MS: u64 = 300_000;
/// Cap on the `result` text returned to the parent when `output_limit` is
/// not set. Also the retention cap: the registry entry and the transcript
/// settle marker keep at most this much of the result and error text, so
/// one idle entry never holds more than a few KB.
pub(super) const DEFAULT_OUTPUT_LIMIT: usize = 32 * 1024;
/// Hard bound on the serialized settle-marker line. `record_settled` cuts
/// the result until the line fits, so the rehydration tail window always
/// contains the whole marker.
pub(super) const SETTLED_MARKER_LIMIT: usize = 128 * 1024;
/// Tail bytes scanned for the settle marker at rehydration: twice the
/// marker limit, so the marker line always starts inside the window.
pub(super) const SETTLED_TAIL_WINDOW: u64 = 2 * SETTLED_MARKER_LIMIT as u64;
/// Maximum retained subagent entries per session. Spawning past the cap
/// evicts a finished entry to make room; if every entry is still active the
/// spawn fails instead of growing the registry without bound.
pub(super) const MAX_ENTRIES: usize = 64;
/// Maximum concurrently running subagents per session. Bounds provider
/// load from fan-out; separate from the retention cap above.
pub(super) const MAX_RUNNING: usize = 8;
/// Retained events per run for `subagentRead`. Overflow drops the oldest
/// events; the read reports the gap via `dropped`.
pub(super) const EVENT_CAP: usize = 512;
/// Cap on any string carried by a public event.
pub(super) const EVENT_TEXT_CAP: usize = 8 * 1024;
/// Cap on the outcome line quoted in a terminal notice: the notice becomes
/// a parent prompt, so it stays short.
pub(super) const NOTICE_DETAIL_CAP: usize = 200;
/// Grace period between a kill's cooperative cancel and the task abort
/// backstop.
pub(super) const KILL_GRACE: Duration = Duration::from_secs(5);
/// Smallest periodic notice interval `subagentNotify` accepts.
pub(super) const MIN_NOTIFY_INTERVAL_MS: u64 = 5_000;

/// Lifecycle states of a subagent entry, reported in tool results.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::agent) enum Status {
    Running,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
    /// The hard cap passed while a confirmation prompt this child raised was
    /// still unanswered.
    Blocked,
}

impl Status {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::TimedOut => "timed_out",
            Self::Blocked => "blocked",
        }
    }

    pub(super) fn is_terminal(self) -> bool {
        !matches!(self, Self::Running)
    }

    /// Restore from a persisted settle marker. An unknown status (written by
    /// a newer build) degrades to failed rather than blocking the resume.
    pub(super) fn from_marker(status: &str) -> Self {
        match status {
            "completed" => Self::Completed,
            "cancelled" => Self::Cancelled,
            "timed_out" => Self::TimedOut,
            "blocked" => Self::Blocked,
            _ => Self::Failed,
        }
    }
}

/// One running subagent, surfaced on the parent's status line.
pub struct SubagentRunning {
    pub id: String,
    pub elapsed_ms: u64,
    pub task: String,
    /// What the run is doing now: the description of its exec block, or the
    /// start of the response it streams. Empty while it thinks.
    pub activity: String,
    /// The run's resolved model; `None` before the entry is claimed.
    pub model: Option<String>,
    /// The child session's spend so far; `None` until a run reports usage.
    pub cost: Option<f64>,
}

/// One registry entry as `/subagent` lists it: the durable fields a store
/// row cannot show for a run still in flight.
pub struct SubagentLive {
    pub id: String,
    pub session: Option<String>,
    pub running: bool,
    pub model: Option<String>,
    /// Elapsed milliseconds while the run is live; 0 once it settles.
    pub elapsed_ms: u64,
    /// The child session's spend; after settle it is the terminal cost.
    pub cost: Option<f64>,
    /// The one-line task the run was given. Empty once a resume restores the
    /// entry: the prompt lives in the child transcript, not in this process.
    pub task: String,
}

/// A state change of a background subagent, published to the host UI as a
/// notice prompt. Subagent ids are strings, so this does not reuse
/// `JobNotice`.
#[derive(Clone, Debug)]
pub struct SubagentNotice {
    pub id: String,
    pub text: String,
    pub kind: SubagentNoticeKind,
}

/// Why a notice fired. The host reacts in kind: a settled run gets a turn
/// that reports it, while a mailbox wake must not end a turn that is
/// already running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubagentNoticeKind {
    /// A run reached a terminal state, or is still running.
    Run,
    /// A message landed in the parent inbox.
    Mailbox,
}

/// The run-management hook of one backend. Every child is an ordinary
/// backend forked from this template, so the parent's transport does not
/// constrain the child's.
pub(crate) fn backend_hook(
    template: BackendRef,
    support: SubagentSupport,
    registry: SubagentRegistry,
) -> SubagentFn {
    Arc::new(move |op: SubagentOp, args: Value| {
        let template = template.clone();
        let support = support.clone();
        let registry = registry.clone();
        Box::pin(async move { ops::dispatch(template, support, registry, op, args).await })
            as std::pin::Pin<Box<dyn std::future::Future<Output = Value> + Send>>
    })
}

impl Agent {
    /// Build the `lofi.subagent*` hook for this agent, or `None` when the
    /// agent has no subagent support (nested subagent agents).
    #[must_use]
    pub fn subagent_hook(&self) -> Option<SubagentFn> {
        let Some(support) = self.executor.subagent_support.clone() else {
            // A child cannot manage runs, but it keeps the mailbox: the hook
            // answers the three mailbox ops and reports every other op as
            // unavailable.
            return self.executor.mailbox.clone().map(ops::mailbox_hook);
        };
        Some(backend_hook(
            Arc::new(self.clone()),
            support,
            self.executor.subagents.clone(),
        ))
    }

    /// Attach subagent support so the agent's sandbox exposes `lofi.subagent`.
    #[must_use]
    pub fn with_subagent_support(mut self, support: SubagentSupport) -> Self {
        self.executor = self.executor.with_subagent_support(support);
        self
    }

    /// Toggle transcript persistence for spawned subagents. No-op when the
    /// agent has no subagent support.
    #[must_use]
    pub fn with_subagent_persistence(mut self, persist_sessions: bool) -> Self {
        if let Some(support) = self.executor.subagent_support.take() {
            self.executor = self
                .executor
                .with_subagent_support(support.with_persistence(persist_sessions));
        }
        self
    }
}

/// Point the mailbox at this store's journal for the parent session, so a
/// resumed session replays only its own conversation.
pub(super) fn attach_mailbox_journal(registry: &SubagentRegistry, store: &SessionStore) {
    let name = registry
        .parent_session()
        .unwrap_or_else(|| PARENT_ADDRESS.to_string());
    registry
        .mailbox
        .attach_journal(&store.root().join(format!("mailbox-{name}.jsonl")));
}
