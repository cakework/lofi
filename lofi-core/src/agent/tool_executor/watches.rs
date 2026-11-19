use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::agent::lock;

/// Confirmations one executor is waiting on, and how long their answers
/// took: a child's idle deadline pauses while one is open, and its hard cap
/// names the prompt it was waiting on.
#[derive(Default)]
pub(in crate::agent) struct ConfirmWatch {
    state: Mutex<WatchState>,
}

#[derive(Default)]
struct WatchState {
    next_id: u64,
    /// Reset count: a guard from an earlier run must not touch this run's
    /// prompts, because the tool work that opened it can outlive the reset.
    generation: u64,
    /// Still-open prompts, oldest first, with the command each asks about.
    open: Vec<(u64, String)>,
    /// When the first still-open prompt opened.
    since: Option<std::time::Instant>,
    /// Total time any prompt was open, including the interval in progress.
    waited: std::time::Duration,
}

impl ConfirmWatch {
    /// Prompts still unanswered.
    pub(in crate::agent) fn pending(&self) -> usize {
        lock(&self.state).open.len()
    }

    /// The oldest prompt still unanswered: the one a blocked run waited on,
    /// so answering a newer prompt first does not misname it.
    pub(in crate::agent) fn command(&self) -> Option<String> {
        lock(&self.state)
            .open
            .first()
            .map(|(_, command)| command.clone())
    }

    /// How long this child has spent waiting on answers.
    pub(in crate::agent) fn waited(&self) -> std::time::Duration {
        let state = lock(&self.state);
        state.waited.saturating_add(
            state
                .since
                .map_or(std::time::Duration::ZERO, |since| since.elapsed()),
        )
    }

    /// Count `command` as open until the guard drops.
    pub(in crate::agent) fn open(&self, command: &str) -> OpenConfirm<'_> {
        let mut state = lock(&self.state);
        let id = state.next_id;
        let generation = state.generation;
        state.next_id += 1;
        if state.since.is_none() {
            state.since = Some(std::time::Instant::now());
        }
        state.open.push((id, command.to_string()));
        drop(state);
        OpenConfirm {
            watch: self,
            id,
            generation,
        }
    }

    /// Forget prompts an earlier run left open: one must not pause or name
    /// the next run. Resetting in place reaches the executor clones a
    /// backend's worker captured.
    pub(in crate::agent) fn reset(&self) {
        let mut state = lock(&self.state);
        // Ids keep counting up: an earlier run's prompt that is still open
        // drops its own guard later and must not take a new one with it.
        state.generation += 1;
        state.open.clear();
        state.since = None;
        state.waited = std::time::Duration::ZERO;
    }
}

/// Tool calls one executor is running right now. A call reports nothing
/// until it ends, so a child inside a long one is working, not idle.
#[derive(Default)]
pub(crate) struct ActivityWatch {
    state: Mutex<ActivityState>,
}

#[derive(Default)]
struct ActivityState {
    /// Calls in flight, in the current generation.
    busy: usize,
    /// When the last call of the current generation ended. A call reports
    /// nothing until it ends, so this is the last sign of work the drain may
    /// not have seen yet.
    free_since: Option<std::time::Instant>,
    /// Reset count: a guard an earlier run left running belongs to that run,
    /// and an increment and a reset must not interleave.
    generation: u64,
}

impl ActivityWatch {
    /// Calls in flight, and how long an idle run has looked idle. The pair is
    /// read under one lock: a call starting between two reads would hide the
    /// call that just ended.
    pub(crate) fn idle_mark(&self) -> (usize, Option<std::time::Duration>) {
        let state = lock(&self.state);
        let free = (state.busy == 0)
            .then_some(state.free_since)
            .flatten()
            .map(|at| at.elapsed());
        (state.busy, free)
    }

    /// Forget calls an earlier run left running: they belong to that run,
    /// and a resumed one must not wait for them.
    pub(in crate::agent) fn reset(&self) {
        let mut state = lock(&self.state);
        state.generation += 1;
        state.busy = 0;
        state.free_since = None;
    }

    /// Count the call as running until the guard drops, so a failed call
    /// still clears it.
    pub(in crate::agent) fn enter(&self) -> RunningCall<'_> {
        let mut state = lock(&self.state);
        let generation = state.generation;
        state.busy += 1;
        state.free_since = None;
        drop(state);
        RunningCall {
            watch: self,
            generation,
        }
    }
}

/// One running call; see [`ActivityWatch::enter`].
pub(in crate::agent) struct RunningCall<'a> {
    watch: &'a ActivityWatch,
    generation: u64,
}

impl Drop for RunningCall<'_> {
    fn drop(&mut self) {
        let mut state = lock(&self.watch.state);
        // A call a previous run started is not this run's activity.
        if state.generation != self.generation {
            return;
        }
        state.busy = state.busy.saturating_sub(1);
        if state.busy == 0 {
            state.free_since = Some(std::time::Instant::now());
        }
    }
}

/// One open confirmation; see [`ConfirmWatch::open`]. It removes its own
/// prompt, so overlapping prompts leave the right one open.
pub(in crate::agent) struct OpenConfirm<'a> {
    watch: &'a ConfirmWatch,
    id: u64,
    generation: u64,
}

impl Drop for OpenConfirm<'_> {
    fn drop(&mut self) {
        let mut state = lock(&self.watch.state);
        // A guard an earlier run left open has no say over this one.
        if state.generation != self.generation {
            return;
        }
        state.open.retain(|(id, _)| *id != self.id);
        if state.open.is_empty() {
            if let Some(since) = state.since.take() {
                state.waited += since.elapsed();
            }
        }
    }
}

/// Clears a directory prompt's `active` flag when the ask ends. The UI drops
/// a prompt only while that flag stays true, so an exec call its deadline drops
/// mid-ask must still close its modal.
pub(super) struct ActivePrompt(pub(super) Arc<AtomicBool>);

impl Drop for ActivePrompt {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Relaxed);
    }
}
