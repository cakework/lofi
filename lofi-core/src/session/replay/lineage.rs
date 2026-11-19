use std::collections::{HashMap as Map, HashSet};

#[cfg(test)]
use lofi_types::SessionEvent;
use lofi_types::{SessionEventKind, Usage};

use super::super::store::{self, IndexKind, SessionCursor, SessionIndexEntry};
#[cfg(test)]
use super::visible_event_indices;

#[must_use]
pub fn last_run_model_from_index(
    cursor: &SessionCursor,
    index: &[SessionIndexEntry],
) -> Option<lofi_types::RunModel> {
    for i in (0..index.len()).rev() {
        if !matches!(
            index[i].kind,
            IndexKind::TurnEnd | IndexKind::TurnFailed | IndexKind::TurnCancelled
        ) {
            continue;
        }
        let ev = cursor.event_at(index[i].offset).ok()?;
        if let Some(model) = store::turn_outcome_model(&ev.kind) {
            return Some(model.clone());
        }
    }
    None
}

/// Whether the session was compacted with no usage recorded after the marker
/// (so the status line shows the compacted banner) plus the latest usage.
#[must_use]
pub fn compaction_status_from_index(
    cursor: &SessionCursor,
    index: &[SessionIndexEntry],
) -> (bool, Option<Usage>) {
    // Read only the latest turn outcome: the prior shape fetched every
    // TurnEnd event on the transcript, and each fetch opens and seeks the
    // session file — hundreds of opens on a long resume when only the final
    // usage matters for the status line.
    let last_compaction_pos = index
        .iter()
        .rposition(|event| event.kind == IndexKind::Compaction);
    let last_usage = index
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, event)| {
            matches!(
                event.kind,
                IndexKind::TurnEnd
                    | IndexKind::TurnFailed
                    | IndexKind::TurnCancelled
                    | IndexKind::TurnUsage
            )
        })
        .find_map(|(pos, event)| {
            let ev = cursor.event_at(event.offset).ok()?;
            match ev.kind {
                SessionEventKind::TurnEnd { usage, .. }
                | SessionEventKind::TurnFailed { usage, .. }
                | SessionEventKind::TurnCancelled { usage, .. }
                | SessionEventKind::TurnUsage { usage, .. } => Some((pos, usage)),
                _ => None,
            }
        });
    let usage_after_compaction = last_usage
        .filter(|(pos, _)| last_compaction_pos.is_none_or(|compact_pos| *pos > compact_pos))
        .map(|(_, usage)| usage);
    (
        last_compaction_pos.is_some() && usage_after_compaction.is_none(),
        usage_after_compaction,
    )
}

/// Job ids with a `JobStarted` marker but no matching `JobFinished` marker
/// on the visible lineage. Test-only: it is the lineage-only answer the
/// finished-anywhere scan exists to replace, so handing it to a caller would
/// bring back the stale job it drops.
#[cfg(test)]
#[must_use]
pub fn outstanding_job_ids(events: &[SessionEvent]) -> Vec<u64> {
    let mut lifecycle = Lifecycle::default();
    for i in visible_event_indices(events) {
        lifecycle.track(&events[i].kind);
    }
    let (_, outstanding) = lifecycle.into_parts();
    outstanding.into_iter().map(|job| job.id).collect()
}

/// A job a resumed transcript opened but never closed. The process that owned
/// the command, the log, and the exit status is gone, so this is all a row can
/// say.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutstandingJob {
    pub id: u64,
    pub cmd: Option<String>,
}

/// The jobs a resumed transcript opened and never closed on the selected
/// lineage. An id whose `JobFinished` sits on any branch is dropped too.
/// Reads lifecycle markers directly from a cursor over an indexed lineage;
/// `index` must be lineage-scoped (see `SessionCursor::snapshot`). Only
/// `IndexKind::JobLifecycle` entries are touched.
///
/// A job that released before a rollback to a point between its start and its
/// release keeps `JobStarted` on the selected lineage while `JobFinished`
/// sits on the pruned branch, so marker absence alone would report a process
/// that is long gone as stale. The finished-anywhere scan re-indexes the whole
/// transcript and parses each lifecycle event, so it only runs when the lineage
/// names a start with no release.
/// # Errors
/// Propagates cursor I/O failures.
pub fn outstanding_jobs_at(
    cursor: &SessionCursor,
    index: &[SessionIndexEntry],
) -> lofi_error::Result<Vec<OutstandingJob>> {
    let outstanding = job_lifecycle_ids_at(cursor, index)?.outstanding;
    if outstanding.is_empty() {
        return Ok(outstanding);
    }
    let ids = outstanding.iter().map(|job| job.id).collect::<Vec<_>>();
    let finished = cursor.finished_job_ids_anywhere(&ids)?;
    Ok(outstanding
        .into_iter()
        .filter(|job| !finished.contains(&job.id))
        .collect())
}

/// The pending wake-up set a resume must re-arm: the last `TimerState` on
/// the visible lineage, or empty when the session recorded none. Test-only;
/// `timer_records_at` is the path a resume takes.
#[cfg(test)]
#[must_use]
pub fn timer_records(events: &[SessionEvent]) -> Vec<lofi_types::TimerRecord> {
    let mut latest: Vec<lofi_types::TimerRecord> = Vec::new();
    for i in visible_event_indices(events) {
        if let SessionEventKind::TimerState { timers } = &events[i].kind {
            latest.clone_from(timers);
        }
    }
    latest
}

/// The pending wake-up set a resume must re-arm: the last `TimerState`
/// on the selected lineage, or empty when the session recorded none. Reads
/// only the `TimerState` entries of an indexed lineage; `index` must be
/// lineage-scoped (see `SessionCursor::snapshot`).
/// # Errors
/// Propagates cursor I/O failures.
pub fn timer_records_at(
    cursor: &SessionCursor,
    index: &[SessionIndexEntry],
) -> lofi_error::Result<Vec<lofi_types::TimerRecord>> {
    // Only the last snapshot on the lineage matters, so read that one event.
    let Some(entry) = index
        .iter()
        .rev()
        .find(|entry| entry.kind == IndexKind::TimerState)
    else {
        return Ok(Vec::new());
    };
    match cursor.event_at(entry.offset)?.kind {
        SessionEventKind::TimerState { timers } => Ok(timers),
        _ => Ok(Vec::new()),
    }
}

#[derive(Default)]
struct Lifecycle {
    started: Vec<u64>,
    finished: HashSet<u64>,
    /// What each started job ran, for the rows a resume shows. Absent for
    /// ids recorded before the command was kept.
    commands: Map<u64, String>,
}

impl Lifecycle {
    fn track(&mut self, kind: &SessionEventKind) {
        match kind {
            SessionEventKind::JobStarted { job_id, cmd } => {
                self.started.push(*job_id);
                if let Some(cmd) = cmd {
                    self.commands.insert(*job_id, cmd.clone());
                }
            }
            SessionEventKind::JobFinished { job_id } => {
                self.finished.insert(*job_id);
            }
            _ => {}
        }
    }

    fn into_parts(mut self) -> (Vec<u64>, Vec<OutstandingJob>) {
        let mut outstanding = Vec::new();
        for id in &self.started {
            if !self.finished.contains(id) {
                outstanding.push(OutstandingJob {
                    id: *id,
                    cmd: self.commands.remove(id),
                });
            }
        }
        (self.started, outstanding)
    }
}

pub(crate) struct JobLifecycleIds {
    pub started: Vec<u64>,
    pub outstanding: Vec<OutstandingJob>,
}

pub(crate) fn job_lifecycle_ids_at(
    cursor: &SessionCursor,
    index: &[SessionIndexEntry],
) -> lofi_error::Result<JobLifecycleIds> {
    let mut lifecycle = Lifecycle::default();
    for entry in index.iter().filter(|e| e.kind == IndexKind::JobLifecycle) {
        lifecycle.track(&cursor.event_at(entry.offset)?.kind);
    }
    let (started, outstanding) = lifecycle.into_parts();
    Ok(JobLifecycleIds {
        started,
        outstanding,
    })
}
