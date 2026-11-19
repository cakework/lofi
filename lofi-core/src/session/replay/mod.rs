//! `SessionEvent` → `AgentEvent` replay mapping. Owns the durable-log-to-live-
//! stream translation so the TUI only applies the resulting `AgentEvent`s to
//! its widgets. Tool results in the durable log travel as a separate `Message`
//! with `Role::Tool` (or, for legacy entries, `Role::User` carrying
//! `ToolResult` blocks) *after* the assistant's `ToolUse`. The live stream
//! delivers the result inline via `AgentEvent::ToolEnd`, so the replay adapter
//! defers each `ToolEnd` until the matching tool-result message arrives.

mod history;
mod lineage;
mod ui_events;

#[cfg(test)]
mod tests;

use std::collections::HashMap as Map;

use lofi_types::{ContentBlock, Role, SessionEvent, SessionEventKind};

use super::store::{self, IndexId, SessionIndexEntry};

pub use history::{agent_message_for_event, history_from_index, messages_from_events};
pub(crate) use history::{AgentHistoryProjection, LeafFirstContextFilter};
pub(crate) use lineage::job_lifecycle_ids_at;
pub use lineage::{
    compaction_status_from_index, last_run_model_from_index, outstanding_jobs_at, timer_records_at,
    OutstandingJob,
};
#[cfg(test)]
pub use lineage::{outstanding_job_ids, timer_records};
pub use ui_events::{continues_turn, replay_selected_session_events, replay_session_events};

/// Marks pre-compaction tail events hidden by checkpointed compaction markers
/// along `path`. `compaction_at` returns the marker's
/// `(checkpointed_tail, first_kept_entry_id)` for the event at the given path
/// position, or `None` when it is not a compaction; `id_matches` reports
/// whether the event at a path position has the given id. Parameterising these
/// two lookups lets one hiding rule serve both the in-memory event slice (kind
/// and id read inline) and the resume index (kind via `event_at`, id via the
/// index's normalized `IndexId`).
///
/// The target id is normalized into an `IndexId` once per marker: a compacted
/// transcript otherwise re-parses the 32-char hex string on every one of the
/// O(path) comparisons per marker, which dominates session-resume time on
/// long, heavily compacted sessions. The first position of each id comes from
/// a lazily built map — only materialized when a checkpointed marker exists —
/// so locating the hide start is O(1) per marker instead of another scan.
/// First-occurrence mapping reproduces the previous `(0..marker_pos).find`
/// walk exactly: both resolve to the smallest position holding the id, and
/// only positions before the marker count.
fn hidden_compaction_range(
    path: &[usize],
    mut compaction_at: impl FnMut(usize) -> Option<(bool, String)>,
    mut id_position: impl FnMut(&IndexId) -> Option<usize>,
) -> Vec<bool> {
    let mut hidden = vec![false; path.len()];
    for marker_pos in 0..path.len() {
        let Some((true, first_kept_entry_id)) = compaction_at(marker_pos) else {
            continue;
        };
        if first_kept_entry_id.is_empty() {
            continue;
        }
        let want = IndexId::borrow(&first_kept_entry_id);
        if let Some(start_pos) = id_position(&want).filter(|&p| p < marker_pos) {
            hidden[start_pos..marker_pos].fill(true);
        }
    }
    hidden
}

#[must_use]
pub fn visible_event_indices(events: &[SessionEvent]) -> Vec<usize> {
    let path = store::active_path_from_leaf(events);
    let mut positions: Option<Map<IndexId, usize>> = None;
    let hidden = hidden_compaction_range(
        &path,
        |p| match &events[path[p]].kind {
            SessionEventKind::Compaction {
                checkpointed_tail,
                first_kept_entry_id,
                ..
            } => Some((*checkpointed_tail, first_kept_entry_id.clone())),
            _ => None,
        },
        |id| {
            positions
                .get_or_insert_with(|| {
                    let mut map = Map::new();
                    for (pos, &i) in path.iter().enumerate() {
                        // First occurrence holds: `insert` overwrites and a
                        // later duplicate must not win the hide start.
                        map.entry(IndexId::borrow(&events[i].id)).or_insert(pos);
                    }
                    map
                })
                .get(id)
                .copied()
        },
    );
    path.into_iter()
        .zip(hidden)
        .filter_map(|(i, hide)| (!hide).then_some(i))
        .collect()
}

#[must_use]
pub fn visible_index_path(index: &[SessionIndexEntry]) -> Vec<usize> {
    index
        .iter()
        .enumerate()
        .filter_map(|(position, entry)| entry.visible.then_some(position))
        .collect()
}

/// Byte ranges per turn, derived from the visible event path. Test helper
/// shared with the TUI's resume projection.
#[must_use]
pub fn turn_byte_ranges_from_events(
    events: &[SessionEvent],
    offsets: &[u64],
    file_size: u64,
) -> Vec<Option<(u64, u64)>> {
    let mut ranges: Vec<Option<(u64, u64)>> = Vec::new();
    let mut cur_start: Option<u64> = None;
    for i in visible_event_indices(events) {
        let ev = &events[i];
        let is_turn_start = matches!(
            &ev.kind,
            SessionEventKind::Message(m)
                if m.role == Role::User
                    && !m.blocks.iter().any(|b| matches!(b, ContentBlock::ToolResult { .. }))
        );
        if is_turn_start {
            if let Some(start) = cur_start {
                if let Some(last) = ranges.last_mut() {
                    *last = Some((start, offsets[i]));
                }
            }
            cur_start = Some(offsets[i]);
            ranges.push(None);
        }
    }
    if let Some(start) = cur_start {
        if let Some(last) = ranges.last_mut() {
            *last = Some((start, file_size));
        }
    }
    ranges
}
