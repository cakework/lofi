use lofi_types::{ContentBlock, Message, PromptKind, Role, SessionEvent, SessionEventKind};

use super::super::store::{self, SessionCursor, SessionIndexEntry};
use crate::shell::UserShellResult;

/// The agent-visible message a session event contributes, if any: a chat
/// `Message` passes through unchanged; a `UserShell` becomes its context-text
/// user message; excluded bashes and everything else contribute none. Shared
/// by message reconstruction (`messages_from_events`, `compact`) so the
/// bash-to-context transform and exclusion rule live in one place.
#[must_use]
#[allow(clippy::match_same_arms)]
pub fn agent_message_for_event(kind: &SessionEventKind) -> Option<Message> {
    match kind {
        // Lineage bookkeeping only; the model learns about background work
        // from tool results and notices. Kept explicit against the wildcard
        // so the lifecycle rule is visible without scanning the `_` arm.
        SessionEventKind::JobStarted { .. } | SessionEventKind::JobFinished { .. } => None,
        // Accounting only; the model never sees subagent spend.
        SessionEventKind::SubagentUsage { .. } | SessionEventKind::TurnUsage { .. } => None,
        // Host bookkeeping: the timer set lives outside the conversation.
        SessionEventKind::TimerState { .. } => None,
        SessionEventKind::Message(m) => Some(m.clone()),
        SessionEventKind::UserShell {
            command,
            output,
            exit_code,
            signal,
            duration_ms,
            truncated,
            cancelled,
            exclude_from_context: false,
        } => {
            let result = UserShellResult::from_session(
                command.clone(),
                output.clone(),
                *exit_code,
                *signal,
                *duration_ms,
                *truncated,
                *cancelled,
            );
            Some(Message {
                origin: None,
                role: Role::User,
                blocks: vec![ContentBlock::Text {
                    text: result.context_text(),
                }],
                kind: PromptKind::default(),
            })
        }
        _ => None,
    }
}

#[derive(Default)]
pub(crate) struct LeafFirstContextFilter {
    skipping_failed_turn: bool,
    discarding_assistant: bool,
}

impl LeafFirstContextFilter {
    pub(crate) fn push(&mut self, kind: &SessionEventKind) -> Option<Message> {
        match kind {
            SessionEventKind::Compaction { .. }
            | SessionEventKind::TurnEnd { .. }
            | SessionEventKind::TurnCancelled { .. } => {
                self.skipping_failed_turn = false;
                self.discarding_assistant = false;
                None
            }
            SessionEventKind::TurnFailed { keep_context, .. } => {
                // A failure that recorded where it stopped keeps its prompt and
                // its completed rounds: the context filter only drops the round
                // its discard boundary names. A log from before the flag exists
                // has no such boundary, so its whole failed turn stays out.
                self.skipping_failed_turn = !keep_context;
                self.discarding_assistant = false;
                None
            }
            SessionEventKind::RoundDiscarded { .. } => {
                // A failed turn's discard marker bounds the skip to the
                // failed round only: the turn's prompt and completed rounds
                // stay in context. Logs from before this marker existed have
                // no discard and skip the whole failed turn instead.
                self.skipping_failed_turn = false;
                self.discarding_assistant = true;
                None
            }
            SessionEventKind::Message(message) if message.role == Role::System => {
                Some(message.clone())
            }
            SessionEventKind::Message(_) | SessionEventKind::UserShell { .. }
                if !self.skipping_failed_turn =>
            {
                let message = agent_message_for_event(kind)?;
                if self.discarding_assistant {
                    if message.role == Role::Assistant {
                        return None;
                    }
                    self.discarding_assistant = false;
                }
                Some(message)
            }
            _ => None,
        }
    }
}

/// Leaf-first durable-event projection into model context. Both complete
/// in-memory replay and the bounded file-backed restore path use this state
/// machine so compaction and system-message boundaries cannot diverge.
#[derive(Default)]
pub(crate) struct AgentHistoryProjection {
    messages: Vec<Message>,
    latest_system: Option<Message>,
    summary: Option<Message>,
    boundary: Option<String>,
    filter: LeafFirstContextFilter,
    done: bool,
}

impl AgentHistoryProjection {
    pub(crate) fn push(&mut self, event: &SessionEvent) {
        if self.done {
            return;
        }
        if let SessionEventKind::Compaction {
            summary,
            first_kept_entry_id,
            ..
        } = &event.kind
        {
            // Checkpoint copies omit TurnEnd, so compaction must end a failed-turn skip.
            self.filter.push(&event.kind);
            if !summary.is_empty() && self.summary.is_none() {
                self.summary = Some(Message {
                    origin: None,
                    role: Role::User,
                    blocks: vec![ContentBlock::Text {
                        text: summary.clone(),
                    }],
                    kind: PromptKind::default(),
                });
            }
            if first_kept_entry_id.is_empty() {
                self.done = true;
            } else {
                self.boundary = Some(first_kept_entry_id.clone());
            }
            return;
        }

        let Some(message) = self.filter.push(&event.kind) else {
            return;
        };
        if message.role == Role::System {
            if self.latest_system.is_none() {
                self.latest_system = Some(message);
            }
            return;
        }
        let boundary_hit = self
            .boundary
            .as_ref()
            .is_some_and(|boundary| boundary == &event.id);
        self.messages.push(message);
        if boundary_hit {
            self.done = true;
        }
    }

    #[must_use]
    pub(crate) fn finish(mut self) -> Vec<Message> {
        self.messages.reverse();
        let mut messages = Vec::with_capacity(
            self.messages.len()
                + usize::from(self.latest_system.is_some())
                + usize::from(self.summary.is_some()),
        );
        messages.extend(self.latest_system);
        messages.extend(self.summary);
        messages.extend(self.messages);
        messages
    }
}

/// Rebuild the agent-visible message history from the durable log along the
/// selected leaf. Mirrors `compacted_history`: current system prompt first,
/// then the compaction summary and retained tail. Failed and discarded rounds
/// remain visible in the transcript but are excluded from this model context.
#[must_use]
pub fn messages_from_events(events: &[SessionEvent]) -> Vec<Message> {
    let path = store::active_path_from_leaf(events);
    let mut projection = AgentHistoryProjection::default();
    for &index in path.iter().rev() {
        projection.push(&events[index]);
    }
    projection.finish()
}

/// Rebuild the model history of a session from its index: every event from
/// `history_start` on, projected the way a live turn projects it. Print mode
/// resumes a session with this, so a prompt the killed run never finished is
/// in the request the new run sends.
///
/// # Errors
/// Propagates transcript reads.
pub fn history_from_index(
    cursor: &SessionCursor,
    index: &[SessionIndexEntry],
    history_start: usize,
) -> lofi_error::Result<Vec<Message>> {
    let offsets: Vec<u64> = index[history_start.min(index.len())..]
        .iter()
        .rev()
        .map(|entry| entry.offset)
        .collect();
    let mut history = crate::lifecycle::history_from_cursor(cursor, &offsets)?;
    // A run killed while a tool was busy left its tool call without a result,
    // and a provider rejects that pairing. Close it before the resumed run
    // sends the history. The repair stays in memory: a restore reads, and the
    // next one repairs the same way.
    crate::agent::close_orphaned_tool_uses(&mut history);
    Ok(history)
}
