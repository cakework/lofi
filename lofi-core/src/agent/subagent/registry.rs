use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use lofi_error::Error;
use lofi_types::Message;
use serde_json::{json, Value};
use tokio::sync::{broadcast, watch};
use tokio::task::AbortHandle;

use super::child::restore_entry;
use super::runner::{ChildUsage, RunRejection};
use super::{
    attach_mailbox_journal, Status, SubagentLive, SubagentNotice, SubagentNoticeKind,
    SubagentRunning, SubagentSupport, EVENT_CAP, KILL_GRACE, MAX_ENTRIES,
};
use crate::agent::mailbox::Mailbox;
use crate::agent::{lock, AgentEvent};
use crate::backend::{Backend, BackendRef};
use crate::session::store::{SessionCursor, SessionStore};

pub(super) struct Entry {
    pub(super) status: Status,
    pub(super) backend: BackendRef,
    pub(super) messages: Vec<Message>,
    /// False only for entries rehydrated from a transcript after a restart:
    /// their history replays from the cursor on the first resume, so a
    /// session start does not pay for histories nobody resumes.
    pub(super) history_loaded: bool,
    pub(super) cancel: Arc<AtomicBool>,
    // Retained so the entry outlives a timed-out or abandoned run; reused on
    // the next resume.
    pub(super) cursor: Option<SessionCursor>,
    /// Transcript of a rehydrated entry whose cursor is not open yet; the
    /// first resume opens it and fills `cursor`. Opening at rehydrate would
    /// index every transcript at session start for entries nobody resumes.
    pub(super) transcript_path: Option<PathBuf>,
    // Transcript file name of this child; lets `/subagent` offer a drill-in
    // replay while the run is still live and the entry has not settled.
    pub(super) session: Option<String>,
    // Bumped on spawn and on every resume: at capacity the terminal entry
    // with the oldest stamp is evicted, never the one just finished.
    pub(super) used: u64,
    // Immutable creation order; the `subagentList` page cursor.
    pub(super) born: u64,
    // Terminal JSON body, stored at settle so `subagentWait` and
    // `subagentStatus` can return it to any later caller.
    pub(super) outcome: Option<Value>,
    // Status broadcast for `subagentWait`. Never blocks, so sending under
    // the entries lock is safe.
    pub(super) state_tx: watch::Sender<Status>,
    // Abort backstop for `subagentKill` and `clear()` when the cooperative
    // cancel flag is not observed.
    pub(super) abort: Option<AbortHandle>,
    // Bounded public event log paged by `subagentRead`.
    pub(super) events: EventLog,
    // Terminal notice switch; on by default for background runs, cleared by
    // `subagentNotify({enabled: false})` and by `clear()`.
    pub(super) notify_terminal: bool,
    // Generation of the periodic-notice loop: each `subagentNotify` with an
    // interval bumps it so a stale loop exits instead of doubling up.
    pub(super) notify_tick: u64,
    // Handle of the active periodic-notice loop; reconfiguration aborts it
    // instead of leaving it asleep until its old interval expires.
    pub(super) notify_task: Option<AbortHandle>,
    // Set when rehydration could not restore the transcript's model: the
    // entry keeps the session model instead. Surfaced by status and list.
    pub(super) note: Option<String>,
    // Reset at every claim (spawn and resume): the parent's status line
    // shows a running run's elapsed time and task.
    pub(super) started: std::time::Instant,
    pub(super) task: String,
    // Set at spawn, cleared at resume: the parent's status line shows only
    // background runs; a foreground run already blocks the parent's exec
    // block, so listing it there would duplicate it.
    pub(super) background: bool,
    // What the child is doing now, for the live line its status row shows.
    pub(super) activity: Activity,
    // Resolved model of the run, so a status line names what runs instead
    // of the session's model. A resume cannot change it.
    pub(super) model: Option<String>,
    // Spend of the current run, shared with the settle guard: the event
    // drain adds every `RoundUsage` here while the run is live.
    pub(super) totals: Option<Arc<std::sync::Mutex<ChildUsage>>>,
    // Spend of the runs this entry already paid for. A resume replaces the
    // run's own totals, so the fold of a child session adds these back.
    pub(super) carried: Rollup,
}

/// What a running child is doing, for the one-line row its parent shows.
/// The response is named by the start of its first line: the tail still
/// streams, and a row that followed it would scroll as the child writes.
/// A run that names nothing keeps the line it named last: a child that
/// thinks, or a tool whose input has not arrived, must not blank the row.
#[derive(Debug, Default)]
pub(super) struct Activity {
    /// What the run is doing now.
    now: Current,
    /// The last line the run named, shown while `now` names nothing.
    last: String,
}

/// What a run does at one moment. `Idle` names nothing.
#[derive(Debug, Default)]
enum Current {
    /// Nothing to name: the child is thinking, or its answer is the tail of
    /// the previous round.
    #[default]
    Idle,
    /// The description the child gave the exec block it is running, with
    /// whitespace collapsed onto one row.
    Exec(String),
    /// The first line of the response, kept whole once it is complete.
    Text { line: String, done: bool },
}

impl Activity {
    pub(super) fn note(&mut self, event: &AgentEvent) {
        self.now.note(event);
        let line = self.now.line();
        if !line.is_empty() {
            self.last = line;
        }
    }

    /// The row's text: what the run names now, or the last line it named.
    pub(super) fn line(&self) -> String {
        let line = self.now.line();
        if line.is_empty() {
            self.last.clone()
        } else {
            line
        }
    }
}

impl Current {
    /// Longest first line kept. The row clips anyway; the bound only keeps a
    /// single-paragraph answer from filling memory.
    const LINE_CAP: usize = 160;

    fn note(&mut self, event: &AgentEvent) {
        match event {
            // A tool names itself once its input arrives. Until then, and
            // while the child thinks, the row keeps the last line.
            AgentEvent::ToolStart { .. } => *self = Self::Idle,
            AgentEvent::ToolInput { label, .. } => {
                let label = label.as_deref().unwrap_or_default();
                *self = Self::Exec(label.split_whitespace().collect::<Vec<_>>().join(" "));
            }
            AgentEvent::Text(text) => self.push_text(text),
            _ => {}
        }
    }

    fn push_text(&mut self, text: &str) {
        if !matches!(self, Self::Text { .. }) {
            *self = Self::Text {
                line: String::new(),
                done: false,
            };
        }
        let Self::Text { line, done } = self else {
            return;
        };
        if *done {
            return;
        }
        for c in text.chars() {
            if c == '\n' {
                if !line.is_empty() {
                    *done = true;
                    return;
                }
                continue;
            }
            if c.is_whitespace() {
                if !line.is_empty() && !line.ends_with(' ') {
                    line.push(' ');
                }
                continue;
            }
            line.push(c);
            if line.chars().count() >= Self::LINE_CAP {
                *done = true;
                return;
            }
        }
    }

    /// The line this state names: an exec description, or the start of the
    /// response.
    fn line(&self) -> String {
        match self {
            Self::Idle => String::new(),
            Self::Exec(description) if description.is_empty() => "Exec".to_string(),
            Self::Exec(description) => format!("Exec: {description}"),
            Self::Text { line, .. } => line.trim_end().to_string(),
        }
    }
}

/// One-line task summary for status surfaces: the first line, bounded so a
/// pasted file body cannot flood a status line.
pub(super) fn task_preview(prompt: &str) -> String {
    const MAX: usize = 80;
    let first = prompt.lines().next().unwrap_or("").trim();
    let mut out: String = first.chars().take(MAX).collect();
    if first.chars().count() > MAX {
        out.push('\u{2026}');
    }
    out
}

/// Bounded per-run event buffer for `subagentRead`. Sequence numbers are
/// stable per run; overflow evicts from the front and the read reports the
/// gap as `dropped`.
#[derive(Default)]
pub(super) struct EventLog {
    events: VecDeque<Value>,
    pub(super) next_seq: u64,
}

impl Entry {
    /// The child session's spend so far: what its earlier runs billed plus
    /// what its record or live totals report for the current one. A
    /// rehydrated entry holds the last run in its record, not in totals.
    fn spend(&self) -> Option<f64> {
        let run = entry_usage(self).and_then(|usage| usage.get("cost").and_then(Value::as_f64));
        if run.is_none() && self.carried.is_empty() {
            return None;
        }
        Some(self.carried.cost + run.unwrap_or(0.0))
    }
}

impl EventLog {
    pub(super) fn push(&mut self, kind: &str, mut data: Value) {
        data["seq"] = json!(self.next_seq);
        data["type"] = json!(kind);
        self.next_seq += 1;
        self.events.push_back(data);
        if self.events.len() > EVENT_CAP {
            self.events.pop_front();
        }
    }

    /// Events from `cursor` on, filtered by `types`. Returns the page, the
    /// next cursor, and how many events before `cursor` were evicted.
    pub(super) fn page(
        &self,
        cursor: u64,
        limit: usize,
        types: Option<&[String]>,
    ) -> (Vec<Value>, u64, u64) {
        let first = self.next_seq - self.events.len() as u64;
        let dropped = first.saturating_sub(cursor);
        let mut out = Vec::new();
        let mut next = cursor.max(first);
        for event in &self.events {
            let seq = event["seq"].as_u64().unwrap_or(0);
            if seq < cursor {
                continue;
            }
            // The cursor advances over filtered-out events: a filtered page
            // must not stall the reader.
            next = seq + 1;
            if let Some(types) = types {
                if !types.iter().any(|t| event["type"].as_str() == Some(t)) {
                    continue;
                }
            }
            out.push(event.clone());
            if out.len() >= limit {
                break;
            }
        }
        (out, next, dropped)
    }
}

/// Per-session store of spawned subagents, shared across model switches.
#[derive(Clone)]
pub struct SubagentRegistry {
    pub(super) entries: Arc<Mutex<HashMap<String, Entry>>>,
    pub(super) next_used: Arc<AtomicU64>,
    /// Cost and tokens of settled runs not yet folded into the parent
    /// session record.
    pub(super) unreported: Arc<Mutex<(f64, lofi_types::Usage)>>,
    /// Session id of the owning session, stamped into child transcript
    /// headers so `/subagent` can list the children of one session.
    pub(super) parent_session: Arc<Mutex<Option<String>>>,
    /// Bumped by `clear()`. A detached run from before a session switch
    /// still calls `report` at settle; its spend must not land in the new
    /// session's totals.
    pub(super) generation: Arc<AtomicU64>,
    /// Creation order for `subagentList` cursors.
    pub(super) next_born: Arc<AtomicU64>,
    pub(super) ids: lofi_types::IdSpace,
    /// Runtime that owns background runs. Captured at construction: the
    /// agent is built on the app's long-lived runtime, while a turn may
    /// run on a per-turn worker runtime that dies when the turn ends.
    /// `set_host_handle` remains as the fallback for registries built
    /// outside any runtime.
    pub(super) host_handle: Arc<std::sync::OnceLock<tokio::runtime::Handle>>,
    /// Terminal and progress notices for the host UI.
    notices: Arc<NoticeHub>,
    /// Parent session id whose transcripts were already rehydrated into
    /// entries. Restart restore runs once per parent; `clear()` resets it.
    rehydrated_for: Arc<Mutex<Option<String>>>,
    /// Spend of entries evicted to make room, held until the fold of the
    /// exec that spawned them reads it.
    pub(super) retired: Arc<Mutex<Retired>>,
    /// Parent/child messaging. Shared by every child exec, which reaches it
    /// through a scoped capability rather than this registry.
    pub(crate) mailbox: Mailbox,
}

impl Default for SubagentRegistry {
    fn default() -> Self {
        let host_handle = Arc::new(std::sync::OnceLock::new());
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let _ = host_handle.set(handle);
        }
        let notices = Arc::new(NoticeHub::default());
        // A child holds only a mailbox scope, so it cannot publish a notice
        // itself: the mailbox wakes the host through this hub.
        let mailbox = Mailbox::default().with_sink({
            let notices = notices.clone();
            Arc::new(move |sender: &str| notices.publish_mailbox(sender))
        });
        Self {
            entries: Arc::default(),
            next_used: Arc::default(),
            unreported: Arc::default(),
            parent_session: Arc::default(),
            generation: Arc::default(),
            next_born: Arc::default(),
            ids: lofi_types::IdSpace::new(),
            host_handle,
            notices,
            rehydrated_for: Arc::default(),
            retired: Arc::default(),
            mailbox,
        }
    }
}

/// Bounded broadcast of subagent notices. Lagging subscribers skip ahead;
/// a missed terminal notice is recoverable through `subagentStatus`.
struct NoticeHub {
    tx: broadcast::Sender<SubagentNotice>,
}

impl Default for NoticeHub {
    fn default() -> Self {
        let (tx, _) = broadcast::channel(64);
        Self { tx }
    }
}

/// Pointer text for a mailbox wake. One wake covers every sender, so the
/// text names the oldest one without claiming it is the only one. It carries
/// no body: it becomes a prompt, and a child's words must reach the parent as
/// tool output the parent asked for, never as an instruction.
fn mailbox_notice_text(sender: &str) -> String {
    format!(
        "subagent {sender} sent a message; drain the parent inbox with lofi.subagentMailboxReceive"
    )
}

impl NoticeHub {
    fn publish_mailbox(&self, sender: &str) {
        let _ = self.tx.send(SubagentNotice {
            id: sender.to_string(),
            text: mailbox_notice_text(sender),
            kind: SubagentNoticeKind::Mailbox,
        });
    }
}

impl SubagentRegistry {
    /// Build on the handle space the session shares with its jobs and
    /// timers.
    #[must_use]
    pub fn with_ids(ids: lofi_types::IdSpace) -> Self {
        Self {
            ids,
            ..Self::default()
        }
    }

    /// Notice stream for the host UI. Re-subscribe after a session rebuild;
    /// a lagging receiver skips to the newest notices.
    #[must_use]
    pub fn subscribe_notices(&self) -> broadcast::Receiver<SubagentNotice> {
        self.notices.tx.subscribe()
    }

    /// Whether a wake for the parent inbox is still untaken. A notice that
    /// arrives after the wake was spent must not wake the host again.
    #[must_use]
    pub fn mailbox_wake_unclaimed(&self) -> bool {
        self.mailbox.parent_wake_unclaimed()
    }

    /// Take the parent wake for a turn the host is about to start. False for
    /// a spent wake, and for mail a running turn drained itself.
    #[must_use]
    pub fn claim_mailbox_wake(&self) -> bool {
        self.mailbox.claim_parent_wake()
    }

    /// The wake a broadcast lag may have dropped: one the host never took,
    /// for mail that is still queued. A taken wake is not rebuilt, because
    /// the turn it started may have been cancelled with the mail unread.
    #[must_use]
    pub fn lost_mailbox_wake(&self) -> Option<String> {
        if !self.mailbox.parent_wake_unclaimed() {
            return None;
        }
        self.mailbox
            .parent_inbox_front()
            .map(|sender| mailbox_notice_text(&sender))
    }

    pub(super) fn publish_notice(&self, id: &str, text: String) {
        let _ = self.notices.tx.send(SubagentNotice {
            id: id.to_string(),
            text,
            kind: SubagentNoticeKind::Run,
        });
    }

    /// Record the runtime that owns background runs when construction
    /// could not (built outside a runtime). First-wins: a turn may run on
    /// a per-turn worker runtime, so a later call must not replace the
    /// construction-time handle.
    pub(crate) fn set_host_handle(&self, handle: tokio::runtime::Handle) {
        let _ = self.host_handle.set(handle);
    }

    /// Whether both registries share one store (clones of one instance).
    /// A model switch that rebuilds the agent may keep the registry; the
    /// TUI must not cancel its live runs or re-subscribe then.
    #[must_use]
    pub fn same_instance(&self, other: &SubagentRegistry) -> bool {
        Arc::ptr_eq(&self.entries, &other.entries)
    }

    /// The runtime to spawn background runs on. Falls back to the current
    /// runtime (tests, print mode before the first turn); never panics
    /// because every caller runs under some runtime.
    pub(super) fn host_handle(&self) -> tokio::runtime::Handle {
        self.host_handle
            .get()
            .cloned()
            .unwrap_or_else(tokio::runtime::Handle::current)
    }

    /// Record the owning session's transcript id. The turn loop calls this
    /// at the start of every turn, so /new and /resume follow the live
    /// session without extra wiring.
    pub(crate) fn set_parent_session(&self, id: &str) {
        let mut p = lock(&self.parent_session);
        if p.as_deref() != Some(id) {
            *p = Some(id.to_string());
        }
    }

    pub(crate) fn parent_session(&self) -> Option<String> {
        lock(&self.parent_session).clone()
    }

    /// The terminal body stored at settle. `None` while the run is live or
    /// after the entry was evicted.
    pub(super) fn outcome_of(&self, id: &str) -> Option<Value> {
        lock(&self.entries)
            .get(id)
            .and_then(|entry| entry.outcome.clone())
    }

    /// Restore entries from this session's subagent transcripts so runs
    /// stay resumable across a process restart. Runs once per parent
    /// session id (the turn loop calls it at turn start); `clear()` resets
    /// the marker so a later /resume of the same parent re-runs it.
    ///
    /// Memory: reads only the header (via the listing) and a bounded tail
    /// window per file. Histories replay lazily on resume.
    pub fn rehydrate(&self, template: &dyn Backend, support: &SubagentSupport) {
        if !support.persist_sessions {
            return;
        }
        let Some(parent) = self.parent_session() else {
            return;
        };
        if lock(&self.rehydrated_for).as_deref() == Some(parent.as_str()) {
            return;
        }
        let generation = self.generation();
        let files = SessionStore::open().and_then(|store| {
            // Attach before restoring entries: a resumed session must replay
            // its own journal before the first spawn, or sends taken before
            // that point reuse sequence numbers the journal already holds.
            attach_mailbox_journal(self, &store);
            store.list_subagent_files_for_cwd(&support.cwd, &parent)
        });
        let files = match files {
            Ok(files) => files,
            Err(e) => {
                tracing::debug!(error = %e, "subagent rehydration skipped");
                return;
            }
        };
        let mut restored = Vec::new();
        // The listing is newest first: select within the cap, then build
        // oldest first so born order stays chronological.
        for file in files.into_iter().take(MAX_ENTRIES).rev() {
            let pair = restore_entry(template, support, self, &file);
            restored.push(pair);
        }
        let mut entries = lock(&self.entries);
        // A clear() during the scan switched sessions; these entries belong
        // to the old parent.
        if self.generation() != generation {
            return;
        }
        for (id, entry) in restored {
            if entries.len() >= MAX_ENTRIES {
                break;
            }
            // A spawn or a duplicate file with the same id wins over the
            // restored entry.
            let std::collections::hash_map::Entry::Vacant(slot) = entries.entry(id) else {
                continue;
            };
            // Every restored run is terminal: its address rejects sends, and a
            // message left pending by the replay stays visible as undelivered.
            // The resume claim reopens it. The close stays here, so it never
            // touches an address a live entry already owns.
            self.mailbox.close(slot.key(), entry.status.as_str());
            slot.insert(entry);
        }
        drop(entries);
        *lock(&self.rehydrated_for) = Some(parent);
    }

    pub(super) fn stamp(&self) -> u64 {
        self.next_used.fetch_add(1, Ordering::SeqCst)
    }

    /// Drop every entry and return the spend settled since the last take.
    /// Called when the session is replaced (/new, /resume), so ids from the
    /// old session cannot be resumed into the new one. The take and the
    /// generation bump happen while holding the unreported lock, so a
    /// settling run either folds into the returned value or is dropped by
    /// the generation check, never into the new session.
    ///
    /// Running entries are cancelled first: a background run can outlive
    /// the switch by minutes, and without the cancel it would keep spending
    /// against a session that no longer owns it. Their jobs die with them
    /// and the timers they armed are dropped; their terminal notices are
    /// silenced.
    #[must_use]
    pub fn clear(&self) -> (f64, lofi_types::Usage) {
        {
            let mut entries = lock(&self.entries);
            for (id, entry) in entries.iter_mut() {
                if entry.status.is_terminal() {
                    continue;
                }
                crate::agent::signal_cancel(&entry.cancel);
                entry.notify_terminal = false;
                let tools = entry.backend.tool_executor();
                tools.jobs().kill_owned(id);
                let _ = tools.timers().cancel_owner(Some(id.as_str()));
                if let Some(abort) = &entry.abort {
                    abort.abort();
                }
            }
            entries.clear();
        }
        lock(&self.retired).clear();
        self.mailbox.close_children("session cleared");
        let mut unreported = lock(&self.unreported);
        let usage = std::mem::take(&mut *unreported);
        *lock(&self.rehydrated_for) = None;
        self.generation.fetch_add(1, Ordering::SeqCst);
        usage
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    /// Spend of runs settled since the last take, if any. The parent turn
    /// loop takes it after each tool round, so runs that settle detached or
    /// after a cancel still count toward the session totals.
    #[must_use]
    pub fn take_unreported(&self) -> Option<(f64, lofi_types::Usage)> {
        let mut u = lock(&self.unreported);
        (u.0 > 0.0
            || u.1.input_tokens > 0
            || u.1.output_tokens > 0
            || u.1.cache_read_tokens > 0
            || u.1.cache_write_tokens > 0)
            .then(|| std::mem::take(&mut *u))
    }

    /// Live overlay rows for `/subagent`. The store listing is the durable
    /// half; this covers runs still in flight or not yet flushed.
    #[must_use]
    pub fn live(&self) -> Vec<SubagentLive> {
        lock(&self.entries)
            .iter()
            .map(|(id, e)| {
                let running = matches!(e.status, Status::Running);
                SubagentLive {
                    id: id.clone(),
                    session: e.session.clone(),
                    running,
                    model: e.model.clone(),
                    elapsed_ms: if running {
                        e.started.elapsed().as_millis() as u64
                    } else {
                        0
                    },
                    cost: e.spend(),
                    task: e.task.clone(),
                }
            })
            .collect()
    }

    /// Cancel one run, from `subagentKill` or the host's dialog. A
    /// terminal entry is left alone, so a kill stays idempotent and
    /// reports the state it found. `None` means no entry has that id.
    #[must_use]
    pub fn kill(&self, id: &str, reason: Option<&str>) -> Option<&'static str> {
        let (tools, abort, status) = {
            let mut entries = lock(&self.entries);
            let entry = entries.get_mut(id)?;
            if entry.status.is_terminal() {
                return Some(entry.status.as_str());
            }
            crate::agent::signal_cancel(&entry.cancel);
            let mut state = json!({ "status": "cancelling" });
            if let Some(reason) = reason {
                state["reason"] = json!(reason);
            }
            entry.events.push("state", state);
            (
                entry.backend.tool_executor(),
                entry.abort.clone(),
                entry.status.as_str(),
            )
        };
        // The child's process tree dies with it, and so do the reminders it
        // armed: their owner is gone, and the notices would otherwise keep
        // waking this session for a run that no longer exists.
        tools.jobs().kill_owned(id);
        let _ = tools.timers().cancel_owner(Some(id));
        // Backstop: a run that never observes the cancel flag is aborted
        // after a grace period; its SettleOnDrop then settles it as
        // cancelled.
        if let Some(abort) = abort {
            let registry = self.clone();
            let id = id.to_string();
            self.host_handle().spawn(async move {
                tokio::time::sleep(KILL_GRACE).await;
                let wedged = lock(&registry.entries)
                    .get(&id)
                    .is_some_and(|entry| !entry.status.is_terminal());
                if wedged {
                    abort.abort();
                }
            });
        }
        Some(status)
    }

    /// Currently running background runs, for the parent's status line.
    /// Foreground runs are excluded: the parent's exec block already shows
    /// them. Sorted by start time so the status line does not reshuffle
    /// between frames.
    #[must_use]
    pub fn running(&self) -> Vec<SubagentRunning> {
        let mut running: Vec<_> = lock(&self.entries)
            .iter()
            .filter(|(_, e)| matches!(e.status, Status::Running) && e.background)
            .map(|(id, e)| {
                (
                    e.started,
                    id.clone(),
                    e.task.clone(),
                    e.activity.line(),
                    e.spend(),
                    e.model.clone(),
                )
            })
            .collect();
        running.sort_by_key(|(started, ..)| *started);
        running
            .into_iter()
            .map(
                |(started, id, task, activity, cost, model)| SubagentRunning {
                    id,
                    elapsed_ms: started.elapsed().as_millis() as u64,
                    task,
                    activity,
                    model,
                    cost,
                },
            )
            .collect()
    }

    /// Whether a background run is still active. The tick loop polls this
    /// on every tick, so it must not copy the task strings the way
    /// [`Self::running`] does.
    #[must_use]
    pub fn has_running_background(&self) -> bool {
        lock(&self.entries)
            .values()
            .any(|e| matches!(e.status, Status::Running) && e.background)
    }

    /// Whether `id` names a run that is still active.
    #[must_use]
    pub fn is_running(&self, id: &str) -> bool {
        lock(&self.entries)
            .get(id)
            .is_some_and(|e| matches!(e.status, Status::Running))
    }

    #[cfg(test)]
    pub(crate) fn ids(&self) -> Vec<String> {
        lock(&self.entries).keys().cloned().collect()
    }

    #[cfg(test)]
    pub(crate) fn backend_of(&self, id: &str) -> Option<BackendRef> {
        lock(&self.entries).get(id).map(|e| e.backend.clone())
    }

    #[cfg(test)]
    pub(crate) fn status_of(&self, id: &str) -> Option<String> {
        lock(&self.entries)
            .get(id)
            .map(|e| e.status.as_str().to_string())
    }
}

/// Spend of one run: its terminal record once it settled, else the live
/// totals of a run still in flight. A rehydrated entry has only the record.
pub(super) fn entry_usage(entry: &Entry) -> Option<Value> {
    entry
        .outcome
        .as_ref()
        .and_then(|outcome| outcome.get("usage").cloned())
        .or_else(|| entry.totals.as_ref().and_then(|totals| lock(totals).json()))
}

/// Transcript id of an entry, whether the cursor is open (live or resumed
/// in this process) or not (rehydrated, first resume pending).
pub(super) fn entry_session_id(entry: &Entry) -> Option<String> {
    entry.cursor.as_ref().map(SessionCursor::id).or_else(|| {
        entry
            .session
            .as_deref()
            .map(|name| name.strip_suffix(".jsonl").unwrap_or(name).to_string())
    })
}

/// Sum of the runs `subagentUsage` folds, and of the child spend an exec
/// result reports.
#[derive(Clone, Default)]
pub(in crate::agent) struct Rollup {
    runs: u64,
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
    pub(super) cost: f64,
}

impl Rollup {
    pub(super) fn add(&mut self, entry: &Entry) {
        // The child session's spend: the runs it already paid for, then the
        // run its record or live totals report.
        self.add_rollup(&entry.carried);
        if let Some(usage) = entry_usage(entry) {
            self.add_usage(&usage);
        }
    }

    /// Add another fold onto this one.
    pub(super) fn add_rollup(&mut self, other: &Rollup) {
        self.runs += other.runs;
        self.input += other.input;
        self.output += other.output;
        self.cache_read += other.cache_read;
        self.cache_write += other.cache_write;
        self.cost += other.cost;
    }

    /// Add one run's usage record as one run.
    pub(in crate::agent) fn add_usage(&mut self, usage: &Value) {
        self.add_counts(usage, 1);
    }

    /// Add a record of this shape, keeping the run count it names.
    pub(in crate::agent) fn merge(&mut self, record: &Value) {
        let runs = record.get("runs").and_then(Value::as_u64).unwrap_or(0);
        self.add_counts(record, runs);
    }

    fn add_counts(&mut self, usage: &Value, runs: u64) {
        let count = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
        self.runs += runs;
        self.input += count("input_tokens");
        self.output += count("output_tokens");
        self.cache_read += count("cache_read_tokens");
        self.cache_write += count("cache_write_tokens");
        self.cost += usage.get("cost").and_then(Value::as_f64).unwrap_or(0.0);
    }

    pub(in crate::agent) fn is_empty(&self) -> bool {
        self.runs == 0
    }

    pub(in crate::agent) fn json(&self) -> Value {
        json!({
            "runs": self.runs,
            "input_tokens": self.input,
            "output_tokens": self.output,
            "cache_read_tokens": self.cache_read,
            "cache_write_tokens": self.cache_write,
            "cost": self.cost,
        })
    }
}

impl From<&Rollup> for lofi_types::RunSpend {
    fn from(rollup: &Rollup) -> Self {
        lofi_types::RunSpend {
            runs: rollup.runs,
            usage: lofi_types::Usage {
                input_tokens: rollup.input,
                output_tokens: rollup.output,
                cache_read_tokens: rollup.cache_read,
                cache_write_tokens: rollup.cache_write,
            },
            cost: rollup.cost,
        }
    }
}

impl From<&lofi_types::RunSpend> for Rollup {
    fn from(spent: &lofi_types::RunSpend) -> Self {
        Rollup {
            runs: spent.runs,
            input: spent.usage.input_tokens,
            output: spent.usage.output_tokens,
            cache_read: spent.usage.cache_read_tokens,
            cache_write: spent.usage.cache_write_tokens,
            cost: spent.cost,
        }
    }
}

/// Spend of evicted entries. `total` keeps every evicted run for a fold
/// without ids; that sum never shrinks, so it answers the same after any
/// fold. `spend` keeps a record only for a run in `claims`: a spawned run
/// whose exec has not folded it yet. Any other run is priced already, by its
/// own result or by its exec's fold, so only `total` counts it.
#[derive(Default)]
pub(super) struct Retired {
    claims: std::collections::HashSet<String>,
    pub(super) spend: HashMap<String, Value>,
    pub(super) total: Rollup,
}

impl Retired {
    /// Register a run an exec started and will fold at its end.
    pub(super) fn claim(&mut self, id: &str) {
        self.claims.insert(id.to_string());
    }

    pub(super) fn keep(&mut self, id: String, spend: Value) {
        self.total.merge(&spend);
        if self.claims.contains(&id) {
            self.spend.insert(id, spend);
        }
    }

    /// Drop the records of runs an exec has folded.
    pub(super) fn release(&mut self, ids: &[String]) {
        for id in ids {
            self.claims.remove(id);
            self.spend.remove(id);
        }
    }

    pub(super) fn clear(&mut self) {
        self.claims.clear();
        self.spend.clear();
        self.total = Rollup::default();
    }
}

/// Make room for one more entry by evicting the terminal entry with the
/// oldest stamp. Caller must hold the entries lock. The evicted run's spend
/// stays in `retired`, because a caller that spawned it can still name it.
pub(super) fn evict_for_spawn(
    registry: &SubagentRegistry,
    entries: &mut HashMap<String, Entry>,
) -> std::result::Result<(), RunRejection> {
    if entries.len() < MAX_ENTRIES {
        return Ok(());
    }
    let victim = entries
        .iter()
        .filter(|(_, entry)| entry.status.is_terminal())
        .min_by_key(|(_, entry)| entry.used)
        .map(|(key, _)| key.clone());
    match victim {
        Some(key) => {
            if let Some(entry) = entries.remove(&key) {
                let mut spend = Rollup::default();
                spend.add(&entry);
                lock(&registry.retired).keep(key, spend.json());
            }
            Ok(())
        }
        None => Err(Error::Tool(format!(
            "too many active subagents ({MAX_ENTRIES}); wait for one to finish"
        ))
        .into()),
    }
}
