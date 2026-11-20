use super::*;
use lofi_core::session::tree::{
    build_tree_row_skeletons as build_tree_entry_skeletons, build_tree_rows as build_tree_entries,
    hydrate_tree_rows as hydrate_tree_entry_rows, load_prompt_text,
};

fn run_tree_snapshot(
    generation: u64,
    generation_clock: &AtomicU64,
    cursor: &store::SessionCursor,
    tx: &tokio::sync::mpsc::UnboundedSender<PickerLoad>,
    state: &mut lofi_core::session::io::WorkerState,
) {
    match cursor.tree_index() {
        Ok(snapshot) => {
            // Pin the snapshot before the first cancellation check. Every
            // return path can then release it through the same worker-owned
            // drop-and-trim operation.
            let snapshot_id = state.insert_tree_snapshot(snapshot);
            if generation_clock.load(Ordering::Relaxed) != generation {
                state.remove_tree_snapshot(snapshot_id);
                return;
            }
            let Some(snapshot) = state.tree_snapshot(snapshot_id) else {
                return;
            };
            let skeletons = build_tree_entry_skeletons(&snapshot.index, snapshot.leaf_pos, cursor);
            if generation_clock.load(Ordering::Relaxed) != generation {
                drop(skeletons);
                state.remove_tree_snapshot(snapshot_id);
                return;
            }
            // The UI gets only an id. All subsequent hydration looks the
            // snapshot up by id, and picker close drops it here so its whole
            // lifecycle stays on the allocating worker.
            let start = skeletons.len().saturating_sub(20);
            let requested: Vec<_> = skeletons[start..]
                .iter()
                .cloned()
                .enumerate()
                .map(|(offset, row)| (start + offset, row))
                .collect();
            if tx
                .send(PickerLoad::TreeReady {
                    generation,
                    entries: skeletons,
                    snapshot_id,
                })
                .is_err()
            {
                state.remove_tree_snapshot(snapshot_id);
                return;
            }
            // Look the snapshot back up rather than holding a borrow across
            // the channel send above.
            let Some(snapshot) = state.tree_snapshot(snapshot_id) else {
                return;
            };
            hydrate_tree_entry_rows(
                &snapshot.index,
                cursor,
                &requested,
                |rows| tx.send(PickerLoad::TreeRows { generation, rows }).is_ok(),
                || generation_clock.load(Ordering::Relaxed) != generation,
            );
        }
        Err(error) => {
            let _ = tx.send(PickerLoad::TreeFailed {
                generation,
                error: error.to_string(),
            });
        }
    }
}

impl App {
    /// Return tree allocations to the IO worker that created them. The
    /// snapshot index and picker rows can each use multiple MiB. Dropping both
    /// on that worker before trimming lets glibc release its arena pages.
    pub(super) fn release_tree_rows(&self, rows: Vec<(usize, TreeEntry)>) {
        if let Some(sink) = self.session.sink.as_ref() {
            sink.submit_io(Box::new(move |_state| {
                drop(rows);
                lofi_types::memory::release_freed_memory();
            }));
        } else {
            drop(rows);
            lofi_types::memory::release_freed_memory();
        }
    }

    pub(super) fn release_tree_resources(
        &mut self,
        entries: Vec<TreeEntry>,
        snapshot_id: Option<u64>,
    ) {
        if entries.is_empty() && snapshot_id.is_none() {
            return;
        }
        if let Some(sink) = self.session.sink.as_ref() {
            sink.submit_io(Box::new(move |state| {
                drop(entries);
                if let Some(id) = snapshot_id {
                    state.remove_tree_snapshot(id);
                } else {
                    lofi_types::memory::release_freed_memory();
                }
            }));
        } else {
            drop(entries);
            lofi_types::memory::release_freed_memory();
        }
    }

    /// '/tree': open the branch-picker overlay over the active session's
    /// event log. Lists every user-prompt event (the natural branch points)
    /// with its preview. Confirmed entry feeds the prompt text back into the
    /// input (for editing) and moves the shared cursor so the next run starts
    /// as a sibling of that prompt rather than appending to the active leaf.
    pub(in crate::tui) fn open_tree_picker(&mut self) {
        let Some(cursor) = self.session.cursor.clone() else {
            self.notify(
                NotifyKind::Error,
                "no session file (ephemeral or --no-session)",
            );
            return;
        };
        // Release any open modal (and any prior tree snapshot on the IO thread)
        // before bumping generation for the new tree load.
        self.close_modal();
        let generation = self
            .picker_generation
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(1);
        if let (Some(tx), Some(sink)) = (self.picker_load_tx.clone(), self.session.sink.as_ref()) {
            self.modal = Some(ActiveModal::Tree {
                picker: TreePickerState {
                    list: PickerList::empty(),
                    generation,
                    loading: true,
                },
                snapshot: None,
                pending: std::collections::HashSet::new(),
            });
            let generation_clock = Arc::clone(&self.picker_generation);
            let cursor = cursor.clone();
            sink.submit_io(Box::new(move |state| {
                run_tree_snapshot(generation, &generation_clock, &cursor, &tx, state);
            }));
        } else {
            match cursor.tree_index() {
                Ok(snapshot) => {
                    let entries = build_tree_entries(&snapshot.index, snapshot.leaf_pos, &cursor);
                    if entries.is_empty() {
                        self.notify(NotifyKind::Info, "no branch points in this session yet");
                    } else {
                        let selected = entries.len().saturating_sub(1);
                        self.modal = Some(ActiveModal::Tree {
                            picker: TreePickerState {
                                list: PickerList::new(entries, selected),
                                generation,
                                loading: false,
                            },
                            snapshot: None,
                            pending: std::collections::HashSet::new(),
                        });
                    }
                }
                Err(e) => {
                    self.notify(NotifyKind::Error, format!("load session for /tree: {e}"));
                }
            }
        }
    }

    pub(super) fn request_tree_viewport(&mut self) {
        const VIEW_ROWS: usize = 20;
        let Some(tx) = self.picker_load_tx.clone() else {
            return;
        };
        let Some(cursor) = self.session.cursor.clone() else {
            return;
        };
        let (generation, snapshot_id, requested) = {
            let Some(ActiveModal::Tree {
                picker,
                snapshot: Some(snapshot_id),
                pending,
            }) = &mut self.modal
            else {
                return;
            };
            if !pending.is_empty() {
                return;
            }
            let generation = picker.generation;
            let len = picker.list.len();
            let start = picker
                .list
                .selected
                .saturating_sub(VIEW_ROWS.saturating_sub(1))
                .min(len.saturating_sub(VIEW_ROWS));
            let end = (start + VIEW_ROWS).min(len);
            let prefetch_start = start.saturating_sub(VIEW_ROWS);
            let requested: Vec<_> = (prefetch_start..end)
                .filter(|row| !picker.list.items[*row].hydrated && !pending.contains(row))
                .map(|row| (row, picker.list.items[row].clone()))
                .collect();
            if requested.is_empty() {
                return;
            }
            pending.extend(requested.iter().map(|(row, _)| *row));
            (generation, *snapshot_id, requested)
        };
        let Some(sink) = self.session.sink.as_ref() else {
            return;
        };
        let generation_clock = Arc::clone(&self.picker_generation);
        sink.submit_io(Box::new(move |state| {
            let Some(snapshot) = state.tree_snapshot(snapshot_id) else {
                return;
            };
            hydrate_tree_entry_rows(
                &snapshot.index,
                &cursor,
                &requested,
                |rows| tx.send(PickerLoad::TreeRows { generation, rows }).is_ok(),
                || generation_clock.load(Ordering::Relaxed) != generation,
            );
        }));
    }

    /// Jobs and wake-ups both belong to the lineage that was showing, so a
    /// successful rollback replaces them with the selected one's.
    pub(in crate::tui) fn adopt_selected_lineage_work(
        &mut self,
        cursor: &store::SessionCursor,
        index: &[store::SessionIndexEntry],
    ) {
        if let Some(jobs) = self.jobs() {
            match self
                .lifecycle
                .reconcile_jobs_after_lineage_switch(cursor, index, &jobs)
            {
                Ok(reconciliation) => self.present_lineage_job_reconciliation(&reconciliation),
                Err(error) => {
                    // The rollback already happened, so the rows and the
                    // notice left here describe a lineage the app is no
                    // longer on. Fail closed: showing none is true, showing
                    // the abandoned branch's is not.
                    self.stale_jobs.clear();
                    self.startup_notices.clear();
                    self.notify(NotifyKind::Error, format!("reconcile branch jobs: {error}"));
                }
            }
        }
        match lofi_core::session::replay::timer_records_at(cursor, index) {
            Ok(timers) => {
                // Restored as recorded, the same way a resume restores them:
                // a wake-up is owed whether or not its moment passed while
                // another branch was showing, and a timer dropped here would
                // be one the selected transcript still lists.
                if let Some(registry) = self.timers() {
                    registry.restore(&timers);
                }
                self.timers_recorded.clone_from(&timers);
            }
            Err(error) => self.notify(NotifyKind::Error, format!("restore branch timers: {error}")),
        }
    }

    pub(in crate::tui) fn present_lineage_job_reconciliation(
        &mut self,
        reconciliation: &lofi_core::LineageJobReconciliation,
    ) {
        // The rows and the pending handover notice describe the lineage that
        // was showing until now, so both are replaced by this one. The notice
        // rides the next prompt instead of starting a run of its own: the
        // user is still editing the restored prompt here.
        self.stale_jobs.clone_from(&reconciliation.stale);
        self.startup_notices.clear();
        if !reconciliation.killed.is_empty() {
            let ids = reconciliation
                .killed
                .iter()
                .map(|id| lofi_types::short_id(*id))
                .collect::<Vec<_>>()
                .join(", ");
            self.notify(
                NotifyKind::Info,
                format!(
                    "killed {} off-lineage job(s): {ids}",
                    reconciliation.killed.len()
                ),
            );
        }
        if !reconciliation.stale.is_empty() {
            let ids = reconciliation
                .stale
                .iter()
                .map(|job| lofi_types::short_id(job.id))
                .collect::<Vec<_>>()
                .join(", ");
            self.startup_notices.push(format!(
                "branch switch: jobs [{ids}] on this branch are no longer running; their ids are stale. Use jobSpawn for new background work."
            ));
        }
    }

    pub(in crate::tui) fn tree_picker_confirm_inner(
        &mut self,
        picker: TreePickerState,
        snapshot: Option<u64>,
    ) {
        let entry = picker.list.current().cloned();
        self.picker_generation.fetch_add(1, Ordering::Relaxed);
        self.release_tree_resources(picker.list.items, snapshot);
        if self.busy() {
            self.notify(
                NotifyKind::Warn,
                "wait for the running turn to finish before rolling back",
            );
            return;
        }
        let Some(entry) = entry else {
            return;
        };
        if !entry.hydrated {
            self.notify(NotifyKind::Info, "selected tree row is still loading");
            return;
        }
        // The branch-head move is a core-owned session write. The sink moves
        // the head and returns the new lineage snapshot in one call; the UI
        // then mirrors the cursor for reads and rolls the transcript back.
        let Some(cursor) = self.session.cursor.clone() else {
            return;
        };
        let old_leaf = cursor.leaf_id();
        // Move the head (core-owned) and grab the snapshot in a short borrow
        // scope so the mutable UI work below doesn't overlap the sink borrow.
        let snapshot = {
            let Some(sink) = self.session.sink.as_ref() else {
                return;
            };
            match sink.switch_branch(entry.branch_point.to_string()) {
                Ok(snapshot) => snapshot,
                Err(e) => {
                    self.notify(NotifyKind::Error, format!("persist session cursor: {e}"));
                    return;
                }
            }
        };
        self.session.refresh_cursor();
        let loaded = self.rollback_indexed(
            &cursor,
            &snapshot.index,
            snapshot.file_size,
            snapshot.history_start,
            snapshot.contiguous,
        );
        if let Err(e) = loaded {
            // Undo the head move (core-owned), then re-roll to the old lineage.
            let restored_snapshot = {
                self.session
                    .sink
                    .as_ref()
                    .and_then(|sink| sink.restore_branch(old_leaf.clone()).ok())
            };
            self.session.refresh_cursor();
            let restored = restored_snapshot.map(|snapshot| {
                self.rollback_indexed(
                    &cursor,
                    &snapshot.index,
                    snapshot.file_size,
                    snapshot.history_start,
                    snapshot.contiguous,
                )
                .ok()
            });
            let suffix = if restored.is_some() {
                String::new()
            } else {
                "; restore failed".to_string()
            };
            self.notify(
                NotifyKind::Error,
                format!("load session for /tree: {e}{suffix}"),
            );
            return;
        }
        self.adopt_selected_lineage_work(&cursor, &snapshot.index);
        let prefill = if !entry.prefill.is_empty() {
            entry.prefill.to_string()
        } else if entry.source_kind == store::IndexKind::UserPrompt {
            load_prompt_text(&cursor, entry.source_offset)
        } else {
            String::new()
        };
        if !prefill.is_empty() {
            self.prompt.text = prefill;
            self.prompt.cursor = self.prompt.text.len();
        }
    }

    #[cfg(test)]
    pub(in crate::tui) fn tree_picker_confirm(&mut self) {
        if matches!(self.modal, Some(ActiveModal::Tree { .. })) {
            let Some(ActiveModal::Tree {
                picker, snapshot, ..
            }) = self.modal.take()
            else {
                unreachable!()
            };
            self.tree_picker_confirm_inner(picker, snapshot);
        }
    }

    fn rollback_indexed(
        &mut self,
        cursor: &store::SessionCursor,
        index: &[store::SessionIndexEntry],
        file_size: u64,
        history_start: usize,
        contiguous: bool,
    ) -> Result<()> {
        self.restore_indexed_session(cursor, index, file_size, history_start, contiguous)?;
        // Rollback leaves no live run. Unlike resume, even the selected final
        // turn is immutable and file-backed, so retaining its potentially huge
        // blocks would recreate the RSS spike this path is meant to prevent.
        for turn in &mut self.transcript.turns {
            turn.blocks.clear();
        }
        self.bump_render_epoch();
        self.viewport.pinned = true;
        self.viewport.top_line = 0;
        Ok(())
    }
}
