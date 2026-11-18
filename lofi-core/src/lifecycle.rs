use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use lofi_types::{
    CompactionConfig, ContentBlock, Message, PromptKind, Role, SessionEvent, SessionEventKind,
    Usage,
};

use crate::compact::{compact, compacted_history, CompactOptions, Compaction};
use crate::recall::{
    recall, recall_cursor, CompactionTarget, RecallOutcome, RecallRequest, RecallScope,
};
use crate::session::recorder::SessionRecord;
use crate::session::replay::OutstandingJob;
use crate::session::store::{CompactionCounts, SessionCursor, SessionIndexEntry};
use crate::session::view::SessionView;
use crate::{CodeCompactionHook, Error, Result};

/// Core-owned mutable state and policy for an agent conversation.
/// Frontends render lifecycle outcomes, but do not plan compactions,
/// reconstruct event histories, execute recall, or decide policy eligibility.
#[derive(Clone)]
pub struct AgentLifecycle {
    history: Arc<Mutex<Vec<Message>>>,
    leased_stats: Arc<Mutex<Option<HistoryStats>>>,
    generation: Arc<AtomicU64>,
    compaction: CompactionConfig,
    context_window: u64,
    workspace_root: Option<PathBuf>,
}

/// Exclusive checkout of the live conversation history for one active turn.
///
/// Moves `Vec<Message>` out of [`AgentLifecycle`] instead of deep-cloning it
/// and restores it on drop unless a newer session/branch replaced history
/// while an unacknowledged turn was winding down.
pub struct HistoryLease {
    history: Arc<Mutex<Vec<Message>>>,
    leased_stats: Arc<Mutex<Option<HistoryStats>>>,
    generation: Arc<AtomicU64>,
    leased_generation: u64,
    messages: Vec<Message>,
}

impl std::ops::Deref for HistoryLease {
    type Target = Vec<Message>;

    fn deref(&self) -> &Self::Target {
        &self.messages
    }
}

impl std::ops::DerefMut for HistoryLease {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.messages
    }
}

impl Drop for HistoryLease {
    fn drop(&mut self) {
        if let Ok(mut stats) = self.leased_stats.lock() {
            *stats = None;
        }
        if self.generation.load(Ordering::Acquire) != self.leased_generation {
            return;
        }
        if let Ok(mut stored) = self.history.lock() {
            if self.generation.load(Ordering::Acquire) == self.leased_generation {
                *stored = std::mem::take(&mut self.messages);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum HardCompactOutcome {
    Compacted(Compaction),
    Cooldown,
    NotEnoughHistory,
}

/// Estimated retained allocations for the core-owned conversation history.
///
/// This accounts for collection and string capacities owned by messages. It
/// intentionally does not claim allocator precision (for example, JSON object
/// node overhead is implementation-specific).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HistoryStats {
    pub messages: usize,
    pub estimated_retained_bytes: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LineageJobReconciliation {
    pub killed: Vec<u64>,
    pub stale: Vec<OutstandingJob>,
}

impl AgentLifecycle {
    #[must_use]
    pub fn new(compaction: CompactionConfig, context_window: u64) -> Self {
        Self {
            history: Arc::new(Mutex::new(Vec::new())),
            leased_stats: Arc::new(Mutex::new(None)),
            generation: Arc::new(AtomicU64::new(0)),
            compaction,
            context_window,
            workspace_root: None,
        }
    }

    #[must_use]
    pub fn lease_history(&self) -> HistoryLease {
        let mut stored = self
            .history
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let leased_generation = self.generation.load(Ordering::Acquire);
        let stats = compute_history_stats(&stored);
        let messages = std::mem::take(&mut *stored);
        drop(stored);
        if let Ok(mut cached) = self.leased_stats.lock() {
            *cached = Some(stats);
        }
        HistoryLease {
            history: Arc::clone(&self.history),
            leased_stats: Arc::clone(&self.leased_stats),
            generation: Arc::clone(&self.generation),
            leased_generation,
            messages,
        }
    }

    #[must_use]
    pub fn shared_history(&self) -> Arc<Mutex<Vec<Message>>> {
        Arc::clone(&self.history)
    }

    /// Seed the live system prompt at the front of an empty history. Matches
    /// the durable transcript's lineage-birth pin: only a history with no
    /// messages yet is a fresh lineage, so this is a no-op after resume or any
    /// prior turn. Returns `true` when the system was inserted.
    ///
    /// # Errors
    /// Returns a state error when the shared history lock is poisoned.
    pub fn seed_system(&self, system_prompt: &str) -> Result<bool> {
        if system_prompt.is_empty() {
            return Ok(false);
        }
        let mut history = self
            .history
            .lock()
            .map_err(|_| Error::State("agent history lock poisoned".to_string()))?;
        if !history.is_empty() {
            return Ok(false);
        }
        history.push(Message {
            origin: None,
            role: Role::System,
            blocks: vec![ContentBlock::Text {
                text: system_prompt.to_string(),
            }],
            kind: PromptKind::default(),
        });
        Ok(true)
    }

    /// Reuses the existing buffer's capacity rather than dropping it and
    /// reallocating. Resume and `/tree` rollback both rebuild the full history
    /// each time; handing a fresh multi-MB `Vec` back to the allocator every
    /// cycle leaves glibc arenas bloated (freed-but-not-returned-to-OS), which
    /// surfaces as a one-time RSS step. Reusing the buffer avoids the churn.
    ///
    /// # Errors
    /// Returns a state error when the shared history lock is poisoned.
    pub fn replace_history(&self, messages: Vec<Message>) -> Result<()> {
        let mut history = self
            .history
            .lock()
            .map_err(|_| Error::State("agent history lock poisoned".to_string()))?;
        self.generation.fetch_add(1, Ordering::Release);
        if let Ok(mut cached) = self.leased_stats.lock() {
            *cached = None;
        }
        if history.capacity() >= messages.len() {
            history.clear();
            history.extend(messages);
        } else {
            *history = messages;
        }
        Ok(())
    }

    /// Rebuild the active agent context from an indexed durable lineage.
    /// # Errors
    /// Propagates transcript reads and poisoned history state.
    pub fn restore_history(
        &mut self,
        cursor: &SessionCursor,
        index: &[SessionIndexEntry],
        history_start: usize,
    ) -> Result<()> {
        let messages = crate::session::replay::history_from_index(cursor, index, history_start)?;
        self.replace_history(messages)?;
        Ok(())
    }

    /// Reconcile session-owned jobs after a branch switch has been loaded.
    /// Jobs acquired on another lineage are cancelled and removed. Started
    /// jobs that belong to this lineage, no longer have a live process, and
    /// did not release on any branch are reported as stale for the frontend
    /// to present.
    ///
    /// # Errors
    /// Propagates the lineage-marker read, which happens before a job is
    /// cancelled or removed, so its failure leaves the registry untouched.
    /// The finished-anywhere scan only narrows the stale rows: when it fails,
    /// the rows it would have cleared are reported, because an extra stale
    /// row costs less than a job left running on the branch that replaced it.
    pub fn reconcile_jobs_after_lineage_switch(
        &self,
        cursor: &SessionCursor,
        index: &[SessionIndexEntry],
        jobs: &lofi_code::tools::JobRegistry,
    ) -> Result<LineageJobReconciliation> {
        let lineage = crate::session::replay::job_lifecycle_ids_at(cursor, index)?;
        tracing::info!(
            target: "lofi::reconcile",
            spawn_ids = ?lineage.started,
            live = ?jobs.live_ids(),
            "reconcile"
        );
        // A job that completed on a pruned branch left its JobFinished
        // marker behind.  Rollback to a point between start and finish
        // cuts the marker off the lineage, but the process already
        // exited — reporting it as stale is a false positive.
        //
        // Only a job the registry does not count as live can be stale, so the
        // scan reads for those alone. The snapshot is safe because the sweep
        // below only unregisters jobs outside this lineage: it cannot turn a
        // live job into a stale row, and one that exits after the snapshot
        // records its own finish.
        let live = jobs.live_ids();
        let candidates: Vec<u64> = lineage
            .outstanding
            .iter()
            .map(|job| job.id)
            .filter(|id| !live.contains(id))
            .collect();
        let finished_elsewhere = if candidates.is_empty() {
            std::collections::HashSet::new()
        } else {
            cursor
                .finished_job_ids_anywhere(&candidates)
                .unwrap_or_else(|error| {
                    tracing::warn!(
                        target: "lofi::reconcile",
                        %error,
                        "finished scan failed; stale rows not narrowed"
                    );
                    std::collections::HashSet::new()
                })
        };
        let killed = jobs.kill_not_in(&lineage.started);
        let stale: Vec<OutstandingJob> = lineage
            .outstanding
            .into_iter()
            .filter(|job| !live.contains(&job.id) && !finished_elsewhere.contains(&job.id))
            .collect();
        Ok(LineageJobReconciliation { killed, stale })
    }

    /// # Errors
    /// Returns a state error when the shared history lock is poisoned.
    pub fn push_message(&self, message: Message) -> Result<()> {
        self.history
            .lock()
            .map_err(|_| Error::State("agent history lock poisoned".to_string()))?
            .push(message);
        Ok(())
    }

    /// # Errors
    /// Returns a state error when the shared history lock is poisoned.
    pub fn clear_history(&mut self) -> Result<()> {
        let mut history = self
            .history
            .lock()
            .map_err(|_| Error::State("agent history lock poisoned".to_string()))?;
        self.generation.fetch_add(1, Ordering::Release);
        if let Ok(mut cached) = self.leased_stats.lock() {
            *cached = None;
        }
        history.clear();
        Ok(())
    }

    #[must_use]
    pub fn history_stats(&self) -> HistoryStats {
        let Ok(messages) = self.history.lock() else {
            return HistoryStats::default();
        };
        if messages.is_empty() {
            if let Ok(cached) = self.leased_stats.lock() {
                if let Some(stats) = *cached {
                    return stats;
                }
            }
        }
        compute_history_stats(&messages)
    }

    pub fn set_context_window(&mut self, context_window: u64) {
        self.context_window = context_window;
    }

    pub fn set_workspace_root(&mut self, root: PathBuf) {
        self.workspace_root = Some(root);
    }

    #[must_use]
    pub fn compact_budget(&self) -> usize {
        self.compaction
            .soft_threshold(self.context_window)
            .or_else(|| self.compaction.hard_threshold(self.context_window))
            .map_or(0, |threshold| (threshold / 2) as usize)
    }

    /// Compact immediately, updating memory and the durable cursor as one
    /// logical operation. Ok(None) means there is not enough history.
    /// # Errors
    /// Propagates history-lock and transcript persistence failures.
    pub fn compact(
        &mut self,
        cursor: Option<&SessionCursor>,
        system_prompt: &str,
    ) -> Result<Option<Compaction>> {
        let Some(events) = self.compaction_events(cursor)? else {
            return Ok(None);
        };
        self.compact_loaded_events(&events, cursor, system_prompt)
    }

    fn compact_loaded_events(
        &mut self,
        events: &[SessionEvent],
        cursor: Option<&SessionCursor>,
        system_prompt: &str,
    ) -> Result<Option<Compaction>> {
        let options = CompactOptions {
            max_kept_tokens: self.compact_budget(),
            edit: self.compaction.edit.clone(),
            hooks: vec![Arc::new(CodeCompactionHook::new(
                self.workspace_root
                    .as_deref()
                    .and_then(|root| crate::state::WorkspaceMemory::open(root).ok())
                    .map(|memory| memory.path().to_path_buf())
                    .unwrap_or_default(),
            ))],
        };
        let Some(compaction) = compact(events, &options) else {
            return Ok(None);
        };
        let mut new_history = compacted_history(&compaction);
        if !system_prompt.is_empty() {
            new_history.insert(
                0,
                Message {
                    origin: None,
                    role: Role::System,
                    blocks: vec![ContentBlock::Text {
                        text: system_prompt.to_string(),
                    }],
                    kind: PromptKind::default(),
                },
            );
        }
        let mut history = self
            .history
            .lock()
            .map_err(|_| Error::State("agent history lock poisoned".to_string()))?;
        if let Some(cursor) = cursor {
            // System is re-pinned in the same record so the next restore
            // finds this latest System first.
            cursor.record(SessionRecord::Compaction {
                kept_messages: &compaction.kept_messages,
                summary: &compaction.summary,
                summarized_range: &compaction.summarized_range.clone().unwrap_or_default(),
                counts: CompactionCounts {
                    summarized: compaction.summarized_count,
                    represented: compaction.represented_count,
                    kept: compaction.kept_count,
                },
                system_prompt,
            })?;
        }
        self.generation.fetch_add(1, Ordering::Release);
        if let Ok(mut cached) = self.leased_stats.lock() {
            *cached = None;
        }
        *history = new_history;
        drop(history);
        // /compact can finish with no following agent run.
        crate::release_freed_memory();
        Ok(Some(compaction))
    }

    /// The cheap, pure gate [`Self::auto_compact`] applies before any
    /// history work. The UI checks it before starting a lifecycle operation,
    /// so a turn that stays under the threshold never occupies the lifecycle
    /// slot and drops a manual `/compact` submitted right after it.
    #[must_use]
    pub fn should_auto_compact(&self, usage: Usage) -> bool {
        self.compaction.auto.enable
            && self
                .compaction
                .soft_threshold(self.context_window)
                .is_some_and(|threshold| usage.context_tokens() > threshold)
    }

    /// Compact when the settled round's context fill exceeds the soft
    /// threshold.
    /// # Errors
    /// Propagates compaction failures.
    pub fn auto_compact(
        &mut self,
        usage: Usage,
        cursor: Option<&SessionCursor>,
        system_prompt: &str,
    ) -> Result<Option<Compaction>> {
        if !self.should_auto_compact(usage) {
            return Ok(None);
        }
        self.compact(cursor, system_prompt)
    }

    /// Apply hard-cap cooldown policy and compact when eligible.
    /// # Errors
    /// Propagates history and transcript failures.
    pub fn hard_compact(
        &mut self,
        cursor: Option<&SessionCursor>,
        system_prompt: &str,
    ) -> Result<HardCompactOutcome> {
        if let Some(cursor) = cursor {
            if cursor.assistant_messages_since_last_compact()?
                < self.compaction.min_messages_between_hard_compacts
            {
                return Ok(HardCompactOutcome::Cooldown);
            }
            return Ok(match self.compact(Some(cursor), system_prompt)? {
                Some(compaction) => HardCompactOutcome::Compacted(compaction),
                None => HardCompactOutcome::NotEnoughHistory,
            });
        }
        let Some(events) = self.compaction_events(None)? else {
            return Ok(HardCompactOutcome::NotEnoughHistory);
        };
        if assistant_messages_since_last_compact(&events)
            < self.compaction.min_messages_between_hard_compacts
        {
            return Ok(HardCompactOutcome::Cooldown);
        }
        Ok(
            match self.compact_loaded_events(&events, None, system_prompt)? {
                Some(compaction) => HardCompactOutcome::Compacted(compaction),
                None => HardCompactOutcome::NotEnoughHistory,
            },
        )
    }

    /// Execute recall against the durable session when present, otherwise the
    /// in-memory conversation. None means no history exists yet.
    /// # Errors
    /// Returns a state error when in-memory history cannot be read.
    pub fn recall_line(
        &self,
        cursor: Option<&SessionCursor>,
        line: &str,
    ) -> Result<Option<RecallOutcome>> {
        let request = parse_recall_line(line);
        if let Some(cursor) = cursor {
            return Ok(Some(recall_cursor(cursor, &request)));
        }
        let Some(events) = self.compaction_events(None)? else {
            return Ok(None);
        };
        Ok(Some(recall(&events, &request)))
    }

    /// Materialize the session lineage for compact/recall/context policy.
    ///
    /// This is the only place the lifecycle chooses between the transcript and
    /// the in-memory tail, and it does so by building a [`SessionView`]: the view
    /// owns the source choice, so this method is only a constructor plus a
    /// materialize call. Every consumer (compact, auto/hard compact, recall,
    /// messages-since-compact) reads through this one view.
    fn compaction_events(
        &self,
        cursor: Option<&SessionCursor>,
    ) -> Result<Option<Vec<SessionEvent>>> {
        let view = cursor.map_or_else(
            || SessionView::from_memory(&self.history),
            SessionView::from_transcript,
        );
        view.session_events()
    }
}

pub(crate) fn history_from_cursor(
    cursor: &SessionCursor,
    leaf_first_offsets: &[u64],
) -> Result<Vec<Message>> {
    let mut projection = crate::session::replay::AgentHistoryProjection::default();
    cursor.visit_events(leaf_first_offsets, |event| {
        projection.push(&event);
        Ok(())
    })?;
    Ok(projection.finish())
}

fn assistant_messages_since_last_compact(events: &[SessionEvent]) -> usize {
    let Some(last_compaction) = events
        .iter()
        .rposition(|event| matches!(event.kind, SessionEventKind::Compaction { .. }))
    else {
        return usize::MAX;
    };
    events[last_compaction + 1..]
        .iter()
        .filter(|event| {
            matches!(
                &event.kind,
                SessionEventKind::Message(message) if message.role == Role::Assistant
            )
        })
        .count()
}

#[allow(clippy::ptr_arg)]
fn compute_history_stats(messages: &Vec<Message>) -> HistoryStats {
    HistoryStats {
        messages: messages.len(),
        estimated_retained_bytes: messages.capacity() * std::mem::size_of::<Message>()
            + messages.iter().map(message_heap_bytes).sum::<usize>(),
    }
}

fn message_heap_bytes(message: &Message) -> usize {
    message.blocks.capacity() * std::mem::size_of::<ContentBlock>()
        + message.blocks.iter().map(content_heap_bytes).sum::<usize>()
        + message.origin.as_ref().map_or(0, |origin| {
            origin.provider.capacity() + origin.model.capacity()
        })
}

fn content_heap_bytes(block: &ContentBlock) -> usize {
    match block {
        ContentBlock::Text { text } => text.capacity(),
        ContentBlock::Thinking {
            text, signature, ..
        } => text.capacity() + signature.as_ref().map_or(0, String::capacity),
        ContentBlock::ToolUse { id, name, input } => {
            id.capacity() + name.capacity() + json_heap_bytes(input)
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            images,
            ..
        } => {
            tool_use_id.capacity()
                + content.capacity()
                + images
                    .iter()
                    .map(|img| img.bytes.len() + img.media_type.capacity())
                    .sum::<usize>()
        }
        ContentBlock::PartSignature {
            provider,
            model,
            format,
            signature,
        } => {
            provider.capacity()
                + model.capacity()
                + match format {
                    lofi_types::PartSignatureFormat::OpenAiExtraContent { namespace } => {
                        namespace.capacity()
                    }
                    lofi_types::PartSignatureFormat::Google
                    | lofi_types::PartSignatureFormat::OpenAiReasoningDetail => 0,
                }
                + signature.capacity()
        }
        ContentBlock::Image { bytes, media_type } => bytes.len() + media_type.capacity(),
    }
}

fn json_heap_bytes(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::String(value) => value.capacity(),
        serde_json::Value::Array(values) => {
            values.capacity() * std::mem::size_of::<serde_json::Value>()
                + values.iter().map(json_heap_bytes).sum::<usize>()
        }
        serde_json::Value::Object(values) => values
            .iter()
            .map(|(key, value)| key.capacity() + json_heap_bytes(value))
            .sum(),
        _ => 0,
    }
}

fn parse_recall_line(line: &str) -> RecallRequest {
    let raw = line.trim().strip_prefix("/recall").unwrap_or("").trim();
    let mut scope = RecallScope::default();
    let mut query = Vec::new();
    let mut page = 1usize;
    for token in raw.split_whitespace() {
        if let Some(value) = token.strip_prefix("scope:") {
            scope = match value {
                "all" => RecallScope::All,
                "lineage" => RecallScope::Lineage,
                "latest" => RecallScope::Compaction(CompactionTarget::Latest),
                value if value.starts_with("compaction:") => value
                    .strip_prefix("compaction:")
                    .and_then(|index| index.parse().ok())
                    .map_or(RecallScope::Lineage, |index| {
                        RecallScope::Compaction(CompactionTarget::Index(index))
                    }),
                _ => RecallScope::Lineage,
            };
        } else if let Some(value) = token.strip_prefix("page:") {
            page = value.parse::<usize>().unwrap_or(1).max(1);
        } else {
            query.push(token);
        }
    }
    let query = query.join(" ");
    RecallRequest {
        query: (!query.is_empty()).then_some(query),
        scope,
        page,
        expand: Vec::new(),
    }
}

/// How far the backend context is behind the session lineage.
#[derive(Debug)]
pub(crate) enum CatchUp {
    /// The backend saw the current lineage head.
    Current,
    /// The backend saw an ancestor; only these messages are missing.
    Delta(Vec<Message>),
    /// The backend saw another lineage, or the history was rewritten.
    Diverged,
}

/// Compare a backend checkpoint with the selected lineage up to the frontier a
/// turn admitted.
pub(crate) fn lineage_catch_up(
    cursor: &SessionCursor,
    frontier: Option<&str>,
    seen: Option<&str>,
) -> Result<CatchUp> {
    // The child lacks nothing between its last reply and the frame this turn
    // sends: the prompt is the frame, and its notices travel with it.
    if frontier == seen {
        return Ok(CatchUp::Current);
    }
    let Some(suffix) = cursor.lineage_after(seen)? else {
        return Ok(CatchUp::Diverged);
    };
    // Stop at the frontier: what this turn recorded after it is the frame, and
    // sending it as well would repeat the prompt or the child's own reply.
    // The events are leaf first, so the window the child missed runs from the
    // frontier down to the entry after `seen`.
    let suffix = match frontier {
        None => return Ok(CatchUp::Diverged),
        Some(frontier) => match suffix.iter().position(|entry| entry.id.matches(frontier)) {
            Some(position) => &suffix[position..],
            None => return Ok(CatchUp::Diverged),
        },
    };
    // Compaction and a fresh system prompt rewrite the context framing, so a
    // resume would keep stale framing.
    if suffix.iter().any(|entry| {
        matches!(
            entry.kind,
            crate::session::store::IndexKind::Compaction
                | crate::session::store::IndexKind::SystemMessage
        )
    }) {
        return Ok(CatchUp::Diverged);
    }
    // A failed turn stamps a boundary the projection reads, so a delta cannot
    // prove the child saw the head while the suffix holds one. Rebuild the
    // prompt from the lineage instead.
    if suffix
        .iter()
        .any(|entry| entry.kind == crate::session::store::IndexKind::TurnFailed)
    {
        return Ok(CatchUp::Diverged);
    }
    let offsets: Vec<u64> = suffix.iter().map(|entry| entry.offset).collect();
    let messages = history_from_cursor(cursor, &offsets)?;
    Ok(if messages.is_empty() {
        CatchUp::Current
    } else {
        CatchUp::Delta(messages)
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn recall_command_parsing_stays_in_core() {
        let request = parse_recall_line("/recall scope:compaction:2 page:3 needle");
        assert_eq!(request.page, 3);
        assert_eq!(request.query.as_deref(), Some("needle"));
        assert!(matches!(
            request.scope,
            RecallScope::Compaction(CompactionTarget::Index(2))
        ));
    }

    #[test]
    fn compact_budget_uses_soft_threshold() {
        let mut config = CompactionConfig::default();
        config.auto.context_ratio = Some(0.5);
        let lifecycle = AgentLifecycle::new(config, 100_000);
        assert_eq!(lifecycle.compact_budget(), 25_000);
    }

    #[test]
    fn compaction_summary_carries_the_workspace_memory_file() {
        let memory = "Next: wire the timer list into the run driver.";
        let mut config = CompactionConfig::default();
        config.auto.max_context_tokens = Some(50);
        // The memory file lives under the state dir, not the workspace.
        let _env = crate::state::STATE_ENV_LOCK.lock().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::env::set_var("LOFI_STATE_HOME", state.path());
        let dir = tempfile::tempdir().unwrap();
        // Resolving the path creates nothing, so the first write brings its
        // own directory, the way the write tool does.
        let target = crate::state::WorkspaceMemory::open(dir.path()).unwrap();
        let path = target.path().to_path_buf();
        crate::state::ensure_private_dir(path.parent().unwrap()).unwrap();
        std::fs::write(&path, memory).unwrap();
        let store = crate::session::store::SessionStore::new(dir.path().join("sessions"));
        let cursor = store.create_cursor(dir.path(), &"p/m".into()).unwrap();
        let mut events = Vec::new();
        for turn in 0..12 {
            for (role, text) in [
                (Role::User, format!("prompt {turn}")),
                (Role::Assistant, "x".repeat(200 * 1024)),
            ] {
                events.push(SessionEvent {
                    id: String::new(),
                    parent_id: None,
                    kind: SessionEventKind::Message(Message {
                        origin: None,
                        role,
                        blocks: vec![ContentBlock::Text { text }],
                        kind: PromptKind::default(),
                    }),
                });
            }
        }
        cursor.append_events(&mut events).unwrap();
        let mut lifecycle = AgentLifecycle::new(config, 100_000);
        lifecycle.set_workspace_root(dir.path().to_path_buf());

        let usage = Usage {
            input_tokens: 100,
            ..Usage::default()
        };
        let compaction = lifecycle
            .auto_compact(usage, Some(&cursor), "system prompt")
            .unwrap()
            .unwrap();

        assert!(
            compaction.summary.contains("[Workspace Memory]"),
            "{}",
            compaction.summary
        );
        assert!(compaction.summary.contains(memory));

        std::env::remove_var("LOFI_STATE_HOME");
    }

    #[test]
    fn auto_compact_counts_cache_writes_and_output_toward_the_threshold() {
        // Anthropic-style accounting on a cold cache: the prompt arrives as
        // cache writes, so input + cache read alone stays under the threshold
        // while the window is actually over it.
        let mut config = CompactionConfig::default();
        config.auto.max_context_tokens = Some(50);
        let dir = tempfile::tempdir().unwrap();
        let store = crate::session::store::SessionStore::new(dir.path().join("sessions"));
        let cursor = store.create_cursor(dir.path(), &"p/m".into()).unwrap();
        let make_msg = |role: Role, text: &str| SessionEvent {
            id: String::new(),
            parent_id: None,
            kind: SessionEventKind::Message(Message {
                origin: None,
                role,
                blocks: vec![ContentBlock::Text {
                    text: text.to_string(),
                }],
                kind: PromptKind::default(),
            }),
        };
        let mut events = Vec::new();
        for turn in 0..5 {
            events.push(make_msg(Role::User, &format!("u{turn}")));
            events.push(make_msg(Role::Assistant, &format!("a{turn}")));
        }
        cursor.append_events(&mut events).unwrap();
        let mut lifecycle = AgentLifecycle::new(config, 100_000);
        let snapshot = cursor.snapshot().unwrap();
        lifecycle
            .restore_history(&cursor, &snapshot.index, snapshot.history_start)
            .unwrap();

        let usage = Usage {
            input_tokens: 2,
            output_tokens: 3,
            cache_read_tokens: 0,
            cache_write_tokens: 60,
        };
        let compaction = lifecycle
            .auto_compact(usage, Some(&cursor), "system prompt")
            .unwrap();
        assert!(
            compaction.is_some(),
            "a cache-write-heavy turn past the threshold must auto-compact"
        );
    }

    #[test]
    fn auto_compact_preserves_stop_reasons_after_content_budget_is_exhausted() {
        let mut config = CompactionConfig::default();
        config.auto.max_context_tokens = Some(50);
        let dir = tempfile::tempdir().unwrap();
        let store = crate::session::store::SessionStore::new(dir.path().join("sessions"));
        let cursor = store.create_cursor(dir.path(), &"p/m".into()).unwrap();
        let mut events = Vec::new();
        let usage = Usage {
            input_tokens: 100,
            ..Usage::default()
        };
        for turn in 0..12 {
            for (role, text) in [
                (Role::User, format!("prompt {turn}")),
                (Role::Assistant, "x".repeat(200 * 1024)),
            ] {
                events.push(SessionEvent {
                    id: String::new(),
                    parent_id: None,
                    kind: SessionEventKind::Message(Message {
                        origin: None,
                        role,
                        blocks: vec![ContentBlock::Text { text }],
                        kind: PromptKind::default(),
                    }),
                });
            }
            events.push(SessionEvent {
                id: String::new(),
                parent_id: None,
                kind: SessionEventKind::TurnEnd {
                    model: "p/m".into(),
                    elapsed_ms: 1,
                    cost: 0.0,
                    usage,
                    stop_reason: Some(lofi_types::StopReason::EndTurn),
                },
            });
        }
        cursor.append_events(&mut events).unwrap();
        let mut lifecycle = AgentLifecycle::new(config, 100_000);

        let compaction = lifecycle
            .auto_compact(usage, Some(&cursor), "system prompt")
            .unwrap();

        assert!(compaction.is_some());
        let stored = cursor.load_events().unwrap();
        assert!(stored
            .iter()
            .any(|event| matches!(event.kind, SessionEventKind::Compaction { .. })));
        let stop_reasons: Vec<_> = stored
            .iter()
            .filter_map(|event| match event.kind {
                SessionEventKind::TurnEnd { stop_reason, .. } => stop_reason,
                _ => None,
            })
            .collect();
        assert_eq!(stop_reasons, vec![lofi_types::StopReason::EndTurn; 12]);
    }

    #[test]
    fn compaction_preserves_user_and_tool_images_across_resume() {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::session::store::SessionStore::new(dir.path().join("sessions"));
        let cursor = store.create_cursor(dir.path(), &"p/m".into()).unwrap();
        let mut events = Vec::new();
        for turn in 0..3 {
            for role in [Role::User, Role::Assistant] {
                events.push(SessionEvent {
                    id: String::new(),
                    parent_id: None,
                    kind: SessionEventKind::Message(Message {
                        origin: None,
                        role,
                        blocks: vec![ContentBlock::Text {
                            text: format!("turn {turn}"),
                        }],
                        kind: PromptKind::default(),
                    }),
                });
            }
        }
        let kept = vec![
            Message {
                origin: None,
                role: Role::User,
                blocks: vec![
                    ContentBlock::Text {
                        text: "inspect this image".into(),
                    },
                    ContentBlock::Image {
                        bytes: vec![1; 128 * 1024].into(),
                        media_type: "image/png".into(),
                    },
                ],
                kind: PromptKind::default(),
            },
            Message {
                origin: None,
                role: Role::Assistant,
                blocks: vec![ContentBlock::ToolUse {
                    id: "read-image".into(),
                    name: "exec".into(),
                    input: serde_json::json!({"code": "return await lofi.read('screenshot.png')"}),
                }],
                kind: PromptKind::default(),
            },
            Message {
                origin: None,
                role: Role::Tool,
                blocks: vec![ContentBlock::ToolResult {
                    tool_use_id: "read-image".into(),
                    content: "screenshot".into(),
                    is_error: false,
                    images: vec![lofi_types::ToolResultImage {
                        bytes: vec![2; 128 * 1024].into(),
                        media_type: "image/png".into(),
                    }],
                }],
                kind: PromptKind::default(),
            },
        ];
        events.extend(kept.iter().cloned().map(|message| SessionEvent {
            id: String::new(),
            parent_id: None,
            kind: SessionEventKind::Message(message),
        }));
        cursor.append_events(&mut events).unwrap();
        let mut lifecycle = AgentLifecycle::new(CompactionConfig::default(), 100_000);

        let compaction = lifecycle.compact(Some(&cursor), "system").unwrap().unwrap();
        assert_eq!(compaction.kept_messages, kept);
        let history = lifecycle.shared_history();
        assert_eq!(&history.lock().unwrap()[2..], kept.as_slice());

        let reopened = SessionCursor::open(cursor.path().to_path_buf()).unwrap();
        let snapshot = reopened.snapshot().unwrap();
        lifecycle
            .restore_history(&reopened, &snapshot.index, snapshot.history_start)
            .unwrap();
        assert_eq!(&history.lock().unwrap()[2..], kept.as_slice());
        let restored_events = reopened.load_compaction_events().unwrap();
        let restored_messages: Vec<_> = restored_events
            .iter()
            .filter_map(|event| match &event.kind {
                SessionEventKind::Message(message) if message.role != Role::System => {
                    Some(message.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(restored_messages, kept);
    }

    #[test]
    fn seed_system_populates_empty_history_once() {
        let lifecycle = AgentLifecycle::new(CompactionConfig::default(), 100_000);
        assert!(lifecycle.seed_system("sys").unwrap());
        assert!(
            !lifecycle.seed_system("sys-again").unwrap(),
            "second seed must be a no-op"
        );
        let history = lifecycle.shared_history().lock().unwrap().clone();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::System);
        assert_eq!(history.iter().filter(|m| m.role == Role::System).count(), 1);
    }

    #[test]
    fn seed_system_skips_after_any_message_and_ignores_empty() {
        let lifecycle = AgentLifecycle::new(CompactionConfig::default(), 100_000);
        assert!(!lifecycle.seed_system("").unwrap());
        lifecycle
            .push_message(Message {
                origin: None,
                role: Role::User,
                blocks: vec![ContentBlock::Text { text: "hi".into() }],
                kind: PromptKind::default(),
            })
            .unwrap();
        assert!(
            !lifecycle.seed_system("sys").unwrap(),
            "a history carrying a resume/turn is not a birth point"
        );
        assert_eq!(lifecycle.history_stats().messages, 1);
    }

    #[test]
    fn job_release_attaches_after_later_appends() {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::session::store::SessionStore::new(dir.path().join("sessions"));
        let cursor = store.create_cursor(dir.path(), &"p/m".into()).unwrap();
        let owner = cursor.record_job_started(42, "cargo test").unwrap();
        // Only a branch move can take the start marker off the selected
        // lineage, but the head does move as the job runs.
        cursor
            .record(SessionRecord::System { prompt: "sys" })
            .unwrap();
        cursor
            .record_job_finished(&owner, 42, lofi_code::JobRelease::Exited)
            .unwrap();

        let snapshot = cursor.snapshot().unwrap();
        let lineage =
            crate::session::replay::job_lifecycle_ids_at(&cursor, &snapshot.index).unwrap();
        assert_eq!(lineage.started, vec![42]);
        assert!(lineage.outstanding.is_empty());
    }

    #[test]
    fn job_release_attaches_to_an_abandoned_start_without_moving_the_head() {
        // Both paths must land the marker: a killed job reports Abandoned,
        // trusting the caller that its start left the lineage, while a natural
        // exit that raced the switch reports Exited and finds it by scanning.
        for release in [
            lofi_code::JobRelease::Abandoned,
            lofi_code::JobRelease::Exited,
        ] {
            assert_abandoned_release_attaches(release);
        }
    }

    fn assert_abandoned_release_attaches(release: lofi_code::JobRelease) {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::session::store::SessionStore::new(dir.path().join("sessions"));
        let cursor = store.create_cursor(dir.path(), &"p/m".into()).unwrap();
        let root = cursor.leaf_id();
        let owner = cursor.record_job_started(42, "cargo test").unwrap();
        let started = cursor.leaf_id();

        // A rollback drops the start from the selected lineage and the job dies
        // there. Dropping its release would leave the start open, so a later
        // return to that branch reads a job that is already dead as one that
        // outlived the run.
        cursor.restore_branch(root.clone()).unwrap();
        cursor.record_job_finished(&owner, 42, release).unwrap();

        let snapshot = cursor.snapshot().unwrap();
        let lineage =
            crate::session::replay::job_lifecycle_ids_at(&cursor, &snapshot.index).unwrap();
        assert!(
            lineage.started.is_empty(),
            "the marker stays off the selected lineage"
        );
        assert!(lineage.outstanding.is_empty());
        assert_eq!(cursor.leaf_id(), root, "the release does not move the head");
        assert_eq!(
            cursor.finished_job_ids_anywhere(&[42]).unwrap(),
            std::collections::HashSet::from([42])
        );

        // A reopen takes its head from the trailing cursor record, not from the
        // side event's place in the file.
        let reopened =
            crate::session::store::SessionCursor::open(cursor.path().to_path_buf()).unwrap();
        assert_eq!(reopened.leaf_id(), root);
        assert_eq!(
            reopened.finished_job_ids_anywhere(&[42]).unwrap(),
            std::collections::HashSet::from([42])
        );

        // Back on the branch that holds the start, the job reads as released.
        reopened.restore_branch(started).unwrap();
        let snapshot = reopened.snapshot().unwrap();
        let lineage =
            crate::session::replay::job_lifecycle_ids_at(&reopened, &snapshot.index).unwrap();
        assert_eq!(lineage.started, vec![42]);
        // The lineage-only view still names the job: its release sits on a side
        // branch. The finished-anywhere filter is what answers the caller.
        assert_eq!(lineage.outstanding.len(), 1);
        assert!(
            crate::session::replay::outstanding_jobs_at(&reopened, &snapshot.index)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn lineage_job_reconciliation_reports_only_outstanding_missing_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::session::store::SessionStore::new(dir.path().join("sessions"));
        let cursor = store.create_cursor(dir.path(), &"p/m".into()).unwrap();
        let owner = cursor.record_job_started(42, "cargo test").unwrap();
        let lifecycle = AgentLifecycle::new(CompactionConfig::default(), 100_000);
        let jobs = lofi_code::tools::JobRegistry::new();

        let snapshot = cursor.snapshot().unwrap();
        let reconciliation = lifecycle
            .reconcile_jobs_after_lineage_switch(&cursor, &snapshot.index, &jobs)
            .unwrap();
        assert!(reconciliation.killed.is_empty());
        assert_eq!(
            reconciliation.stale,
            vec![OutstandingJob {
                id: 42,
                cmd: Some("cargo test".into())
            }]
        );

        cursor
            .record_job_finished(&owner, 42, lofi_code::JobRelease::Exited)
            .unwrap();
        let snapshot = cursor.snapshot().unwrap();
        let reconciliation = lifecycle
            .reconcile_jobs_after_lineage_switch(&cursor, &snapshot.index, &jobs)
            .unwrap();
        assert!(reconciliation.killed.is_empty());
        assert!(reconciliation.stale.is_empty());
    }

    #[test]
    fn reconciliation_ignores_jobs_finished_on_a_pruned_branch() {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::session::store::SessionStore::new(dir.path().join("sessions"));
        let cursor = store.create_cursor(dir.path(), &"p/m".into()).unwrap();
        let owner = cursor.record_job_started(42, "cargo test").unwrap();
        let started_id = cursor.leaf_id();
        cursor
            .record_job_finished(&owner, 42, lofi_code::JobRelease::Exited)
            .unwrap();
        // Roll back to the JobStarted event so the JobFinished marker lands on
        // the pruned branch while JobStarted stays on the selected lineage.
        cursor.restore_branch(started_id).unwrap();

        let lifecycle = AgentLifecycle::new(CompactionConfig::default(), 100_000);
        // An empty registry models the state after a prior branch switch
        // already dropped the job, or a resumed process that never re-registered
        // it. Without the finished-anywhere check this reports a false stale.
        let jobs = lofi_code::tools::JobRegistry::new();
        let snapshot = cursor.snapshot().unwrap();
        let reconciliation = lifecycle
            .reconcile_jobs_after_lineage_switch(&cursor, &snapshot.index, &jobs)
            .unwrap();
        assert!(reconciliation.killed.is_empty());
        assert!(
            reconciliation.stale.is_empty(),
            "a job with a JobFinished marker elsewhere must not be stale: {:?}",
            reconciliation.stale
        );
    }

    #[test]
    fn restore_history_keeps_compacted_tail_before_failed_turn() {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::session::store::SessionStore::new(dir.path().join("sessions"));
        let cursor = store.create_cursor(dir.path(), &"p/m".into()).unwrap();
        let kept = vec![
            Message {
                origin: None,
                role: Role::User,
                blocks: vec![ContentBlock::Text {
                    text: "kept prompt".into(),
                }],
                kind: PromptKind::default(),
            },
            Message {
                origin: None,
                role: Role::Assistant,
                blocks: vec![ContentBlock::Text {
                    text: "kept reply".into(),
                }],
                kind: PromptKind::default(),
            },
        ];
        cursor
            .append_compaction(
                &kept,
                "summary",
                &[String::new(), String::new()],
                CompactionCounts {
                    summarized: 2,
                    represented: 2,
                    kept: 2,
                },
            )
            .unwrap();
        cursor.append_system("instructions").unwrap();
        let mut failed = vec![
            SessionEvent {
                id: String::new(),
                parent_id: None,
                kind: SessionEventKind::Message(Message {
                    origin: None,
                    role: Role::User,
                    blocks: vec![ContentBlock::Text {
                        text: "failed prompt".into(),
                    }],
                    kind: PromptKind::default(),
                }),
            },
            SessionEvent {
                id: String::new(),
                parent_id: None,
                kind: SessionEventKind::Message(Message {
                    origin: None,
                    role: Role::Assistant,
                    blocks: vec![ContentBlock::Text {
                        text: "failed partial".into(),
                    }],
                    kind: PromptKind::default(),
                }),
            },
            SessionEvent {
                id: String::new(),
                parent_id: None,
                kind: SessionEventKind::TurnFailed {
                    model: "p/m".into(),
                    elapsed_ms: 5,
                    error: "boom".into(),
                    cost: 0.0,
                    usage: Usage::default(),
                    keep_context: false,
                },
            },
        ];
        cursor.append_events(&mut failed).unwrap();
        let snapshot = cursor.snapshot().unwrap();
        let mut lifecycle = AgentLifecycle::new(CompactionConfig::default(), 100_000);

        lifecycle
            .restore_history(&cursor, &snapshot.index, snapshot.history_start)
            .unwrap();

        let history = lifecycle.shared_history();
        let messages = history.lock().unwrap();
        let texts: Vec<String> = messages
            .iter()
            .map(|message| match &message.blocks[..] {
                [ContentBlock::Text { text }] => text.clone(),
                _ => String::new(),
            })
            .collect();
        assert_eq!(
            texts,
            vec!["instructions", "summary", "kept prompt", "kept reply"]
        );
    }

    #[test]
    fn restore_history_keeps_cancelled_turn_content() {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::session::store::SessionStore::new(dir.path().join("sessions"));
        let cursor = store.create_cursor(dir.path(), &"p/m".into()).unwrap();
        let mut events = vec![
            SessionEvent {
                id: String::new(),
                parent_id: None,
                kind: SessionEventKind::Message(Message {
                    origin: None,
                    role: Role::User,
                    blocks: vec![ContentBlock::Text { text: "go".into() }],
                    kind: PromptKind::default(),
                }),
            },
            SessionEvent {
                id: String::new(),
                parent_id: None,
                kind: SessionEventKind::Message(Message {
                    origin: None,
                    role: Role::Assistant,
                    blocks: vec![ContentBlock::Text {
                        text: "partial".into(),
                    }],
                    kind: PromptKind::default(),
                }),
            },
            SessionEvent {
                id: String::new(),
                parent_id: None,
                kind: SessionEventKind::TurnCancelled {
                    model: "p/m".into(),
                    elapsed_ms: 5,
                    cost: 0.0,
                    usage: Usage::default(),
                },
            },
        ];
        cursor.append_events(&mut events).unwrap();
        let snapshot = cursor.snapshot().unwrap();
        let mut lifecycle = AgentLifecycle::new(CompactionConfig::default(), 100_000);

        lifecycle
            .restore_history(&cursor, &snapshot.index, snapshot.history_start)
            .unwrap();

        let history = lifecycle.shared_history();
        let messages = history.lock().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::User);
        assert!(matches!(
            &messages[1].blocks[..],
            [ContentBlock::Text { text }] if text == "partial"
        ));
    }
    #[test]
    fn compact_emits_system_before_history_and_restore_recovers_it() {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::session::store::SessionStore::new(dir.path().join("sessions"));
        let cursor = store.create_cursor(dir.path(), &"p/m".into()).unwrap();
        cursor.append_system("instructions v1").unwrap();
        let make_msg = |role: Role, text: &str| SessionEvent {
            id: String::new(),
            parent_id: None,
            kind: SessionEventKind::Message(Message {
                origin: None,
                role,
                blocks: vec![ContentBlock::Text {
                    text: text.to_string(),
                }],
                kind: PromptKind::default(),
            }),
        };
        let mut events = vec![make_msg(Role::User, "u0"), make_msg(Role::Assistant, "a0")];
        for turn in 1..=4 {
            events.push(make_msg(Role::User, &format!("u{turn}")));
            events.push(make_msg(Role::Assistant, &format!("a{turn}")));
        }
        cursor.append_events(&mut events).unwrap();
        let mut lifecycle = AgentLifecycle::new(CompactionConfig::default(), 1_000);
        let snapshot = cursor.snapshot().unwrap();
        lifecycle
            .restore_history(&cursor, &snapshot.index, snapshot.history_start)
            .unwrap();
        let live_before = lifecycle.shared_history().lock().unwrap().clone();
        assert_eq!(live_before.len(), 11);
        assert_eq!(live_before[0].role, Role::System);

        let compaction = lifecycle
            .compact(Some(&cursor), "instructions v1")
            .unwrap()
            .expect("compact produced a result");
        assert!(!compaction.summary.is_empty());

        let live = lifecycle.shared_history().lock().unwrap().clone();
        assert!(!live.is_empty());
        assert_eq!(live[0].role, Role::System);
        assert_eq!(live[1].role, Role::User);

        let logged = cursor.load_events().unwrap();
        let last_pos = logged
            .iter()
            .rposition(|event| {
                matches!(
                    &event.kind,
                    SessionEventKind::Message(m) if m.role == Role::System
                )
            })
            .expect("a System event exists on the log");
        assert!(matches!(
            &logged[last_pos].kind,
            SessionEventKind::Message(m) if matches!(&m.blocks[..], [ContentBlock::Text { text }] if text == "instructions v1")
        ));

        let snapshot = cursor.snapshot().unwrap();
        lifecycle
            .restore_history(&cursor, &snapshot.index, snapshot.history_start)
            .unwrap();
        let restored = lifecycle.shared_history().lock().unwrap().clone();
        assert_eq!(restored[0].role, Role::System);
        assert_eq!(
            restored.iter().filter(|m| m.role == Role::System).count(),
            1,
            "exactly one System after restore: {restored:?}"
        );
    }

    #[test]
    fn history_stats_measure_core_owned_capacities() {
        let mut lifecycle = AgentLifecycle::new(CompactionConfig::default(), 100_000);
        let text = String::with_capacity(4_096);
        lifecycle
            .push_message(Message {
                origin: None,
                role: Role::Assistant,
                blocks: vec![ContentBlock::Text { text }],
                kind: PromptKind::default(),
            })
            .unwrap();

        let stats = lifecycle.history_stats();
        assert_eq!(stats.messages, 1);
        assert!(
            stats.estimated_retained_bytes >= 4_096,
            "text capacity must be represented: {stats:?}"
        );

        lifecycle.clear_history().unwrap();
        let cleared = lifecycle.history_stats();
        assert_eq!(cleared.messages, 0);
        assert!(
            cleared.estimated_retained_bytes < stats.estimated_retained_bytes,
            "clearing drops nested allocations while retaining at most the outer vector: {cleared:?}"
        );
    }
}
