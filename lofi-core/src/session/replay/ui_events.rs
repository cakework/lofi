use std::collections::HashMap as Map;

use lofi_types::{
    ContentBlock, NativeToolRecord, PromptKind, Role, SessionEvent, SessionEventKind, Usage,
};

use super::visible_event_indices;
use crate::agent::AgentEvent;
use crate::exec_input_code_and_label;

pub fn replay_session_events(events: &[SessionEvent], emit: impl FnMut(AgentEvent)) {
    let visible: Vec<&SessionEvent> = visible_event_indices(events)
        .into_iter()
        .map(|i| &events[i])
        .collect();
    replay_visible_events(&visible, emit);
}

pub fn replay_selected_session_events(events: &[SessionEvent], emit: impl FnMut(AgentEvent)) {
    let visible: Vec<&SessionEvent> = events.iter().collect();
    replay_visible_events(&visible, emit);
}

/// Whether a checkpoint reporting `next_elapsed_ms` belongs to the turn whose
/// previous checkpoint reported `held_elapsed_ms`.
///
/// The turn clock starts at zero with every turn, so a checkpoint that reports
/// less elapsed time than the one before it opens a new turn. A turn that ends
/// without a marker — killed, or stopped at the context cap — has no prompt to
/// announce the next one, and this is all that separates them.
#[must_use]
pub fn continues_turn(held_elapsed_ms: u64, next_elapsed_ms: u64) -> bool {
    next_elapsed_ms >= held_elapsed_ms
}

#[allow(clippy::too_many_lines)]
fn replay_visible_events(visible: &[&SessionEvent], mut emit: impl FnMut(AgentEvent)) {
    // A checkpoint's spend stands for a turn no marker ended. The next turn
    // on the lineage is not the same turn, so it flushes this spend instead
    // of overwriting it: a resumed run starts by checkpointing zero. The
    // elapsed time is what tells the two apart when no prompt separates them.
    let mut unterminated: Option<(u64, f64, Usage)> = None;
    let mut tool_elapsed: Map<&str, u64> = Map::new();
    let mut native_by_parent: Map<&str, Vec<&NativeToolRecord>> = Map::new();
    let mut thinking_timing: Vec<u64> = Vec::new();
    for &ev in visible {
        match &ev.kind {
            SessionEventKind::ToolTiming {
                tool_call_id,
                elapsed_ms,
            } => {
                tool_elapsed.insert(tool_call_id.as_str(), *elapsed_ms);
            }
            SessionEventKind::ThinkingTiming { elapsed_ms } => {
                thinking_timing.push(*elapsed_ms);
            }
            SessionEventKind::NativeTool(rec) => {
                native_by_parent
                    .entry(rec.parent.as_str())
                    .or_default()
                    .push(rec);
            }
            _ => {}
        }
    }
    let mut thinking_idx = 0usize;
    for &ev in visible {
        match &ev.kind {
            SessionEventKind::Message(msg) => match msg.role {
                Role::User => {
                    if msg
                        .blocks
                        .iter()
                        .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
                    {
                        for b in &msg.blocks {
                            if let ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                is_error,
                                ..
                            } = b
                            {
                                emit(AgentEvent::ToolEnd {
                                    id: tool_use_id.clone(),
                                    result: if *is_error {
                                        format!("error: {content}")
                                    } else {
                                        content.clone()
                                    },
                                    is_error: *is_error,
                                    elapsed_ms: tool_elapsed
                                        .get(tool_use_id.as_str())
                                        .copied()
                                        .unwrap_or(0),
                                });
                            }
                        }
                        continue;
                    }
                    // A prompt opens a turn; a repair or auto-continue
                    // notice keeps it. A turn that died without a marker is
                    // settled here, before the new turn's own snapshots can
                    // replace what it spent.
                    if msg.kind != PromptKind::Notice {
                        if let Some((_, cost, usage)) = unterminated.take() {
                            emit(AgentEvent::TurnUsage { cost, usage });
                        }
                    }
                    if let Some(event) = AgentEvent::from_prompt(msg) {
                        emit(event);
                    }
                }
                Role::Assistant => {
                    for b in &msg.blocks {
                        match b {
                            ContentBlock::Text { text } => {
                                emit(AgentEvent::Text(text.clone()));
                            }
                            ContentBlock::Thinking { text, .. } => {
                                emit(AgentEvent::Thinking(text.clone()));
                                let elapsed =
                                    thinking_timing.get(thinking_idx).copied().unwrap_or(0);
                                thinking_idx += 1;
                                emit(AgentEvent::ThinkingEnd {
                                    elapsed_ms: elapsed,
                                });
                            }
                            ContentBlock::ToolUse { id, name, input } => {
                                emit(AgentEvent::ToolStart {
                                    id: id.clone(),
                                    name: name.clone(),
                                });
                                let (code, label) = if name == "exec" {
                                    exec_input_code_and_label(input)
                                } else {
                                    (input.to_string(), None)
                                };
                                emit(AgentEvent::ToolInput {
                                    id: id.clone(),
                                    code,
                                    label,
                                });
                                if let Some(natives) = native_by_parent.get(id.as_str()) {
                                    for rec in natives {
                                        emit(AgentEvent::NativeToolStart {
                                            parent: id.clone(),
                                            id: rec.call_id,
                                            name: rec.name.clone(),
                                            args: rec.args.clone(),
                                        });
                                        emit(AgentEvent::NativeToolEnd {
                                            parent: id.clone(),
                                            id: rec.call_id,
                                            result: rec.result.clone(),
                                            is_error: rec.is_error,
                                        });
                                    }
                                }
                            }
                            ContentBlock::ToolResult { .. }
                            | ContentBlock::PartSignature { .. }
                            | ContentBlock::Image { .. } => {}
                        }
                    }
                }
                Role::Tool => {
                    for b in &msg.blocks {
                        if let ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                            ..
                        } = b
                        {
                            emit(AgentEvent::ToolEnd {
                                id: tool_use_id.clone(),
                                result: if *is_error {
                                    format!("error: {content}")
                                } else {
                                    content.clone()
                                },
                                is_error: *is_error,
                                elapsed_ms: tool_elapsed
                                    .get(tool_use_id.as_str())
                                    .copied()
                                    .unwrap_or(0),
                            });
                        }
                    }
                }
                Role::System => {}
            },
            SessionEventKind::UserShell {
                command,
                output,
                exit_code,
                signal,
                duration_ms,
                truncated,
                cancelled,
                exclude_from_context,
            } => {
                emit(AgentEvent::UserShell {
                    command: command.clone(),
                    output: output.clone(),
                    exit_code: *exit_code,
                    signal: *signal,
                    duration_ms: *duration_ms,
                    truncated: *truncated,
                    cancelled: *cancelled,
                    exclude_from_context: *exclude_from_context,
                });
            }
            SessionEventKind::ContextReported { context } => {
                emit(AgentEvent::ContextReported { context: *context });
            }
            SessionEventKind::SubagentUsage { usage, cost } => {
                emit(AgentEvent::SubagentUsage {
                    cost: *cost,
                    usage: *usage,
                });
            }
            // A turn start settles the spend of a turn that ended without a
            // marker: the new turn may have no prompt of its own.
            SessionEventKind::TurnStarted => {
                if let Some((_, cost, usage)) = unterminated.take() {
                    emit(AgentEvent::TurnUsage { cost, usage });
                }
            }
            SessionEventKind::RoundDiscarded { .. }
            | SessionEventKind::NativeTool(_)
            | SessionEventKind::ToolTiming { .. }
            | SessionEventKind::ThinkingTiming { .. }
            | SessionEventKind::JobStarted { .. }
            | SessionEventKind::JobFinished { .. }
            | SessionEventKind::Cursor { .. }
            | SessionEventKind::RunSettled { .. }
            | SessionEventKind::TimerState { .. }
            | SessionEventKind::Unknown => {}
            SessionEventKind::Compaction {
                summarized,
                kept,
                summary,
                ..
            } => {
                emit(AgentEvent::Compaction {
                    summarized: *summarized,
                    kept: *kept,
                    summary: summary.clone(),
                });
            }
            SessionEventKind::TurnUsage {
                elapsed_ms,
                cost,
                usage,
            } => {
                // A turn that starts with no prompt — a continuation after the
                // context cap, a notice that opens its own turn — restarts the
                // turn clock. Settle the turn before it, or its spend is lost.
                if let Some((held_elapsed, held_cost, held_usage)) = unterminated {
                    if !continues_turn(held_elapsed, *elapsed_ms) {
                        emit(AgentEvent::TurnUsage {
                            cost: held_cost,
                            usage: held_usage,
                        });
                    }
                }
                unterminated = Some((*elapsed_ms, *cost, *usage));
            }
            SessionEventKind::TurnEnd {
                model,
                elapsed_ms,
                cost,
                usage,
                stop_reason,
                ..
            } => {
                unterminated = None;
                emit(AgentEvent::TurnEnd {
                    model: model.clone(),
                    elapsed_ms: *elapsed_ms,
                    cost: *cost,
                    usage: *usage,
                    stop_reason: *stop_reason,
                });
            }
            SessionEventKind::TurnFailed {
                model,
                elapsed_ms,
                error,
                cost,
                usage,
                ..
            } => {
                unterminated = None;
                emit(AgentEvent::TurnFailed {
                    model: model.clone(),
                    elapsed_ms: *elapsed_ms,
                    error: error.clone(),
                    cost: *cost,
                    usage: *usage,
                });
            }
            SessionEventKind::TurnCancelled {
                model,
                elapsed_ms,
                cost,
                usage,
            } => {
                unterminated = None;
                emit(AgentEvent::TurnCancelled {
                    model: model.clone(),
                    elapsed_ms: *elapsed_ms,
                    cost: *cost,
                    usage: *usage,
                });
            }
        }
    }
    // The last turn left no marker: report the spend its checkpoints kept.
    if let Some((_, cost, usage)) = unterminated {
        emit(AgentEvent::TurnUsage { cost, usage });
    }
}
