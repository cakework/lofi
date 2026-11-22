#![allow(clippy::wildcard_imports)]

use super::*;

impl App {
    #[allow(clippy::too_many_lines)]
    pub(super) fn new(
        model_label: String,
        thinking: ThinkingLevel,
        service_tier: ServiceTier,
        ctx_limit: u64,
        compaction: lofi_types::CompactionConfig,
        system_prompt: String,
    ) -> Self {
        let model = ModelSelectionState::new(model_label, thinking, service_tier, ctx_limit);
        let ctx_window = model.ctx_limit;
        Self {
            transcript: Transcript::new(),
            layout: TranscriptLayout::new(),
            prompt: PromptEditor::default(),
            viewport: ViewportState::default(),
            modal: None,
            confirms: ConfirmQueue::default(),
            lifecycle: AgentLifecycle::new(compaction, ctx_window),
            system_prompt,
            model,
            usage: UsageState::default(),
            context_pressure: None,
            prompt_queue: Vec::new(),
            mailbox_wake: None,
            startup_notices: Vec::new(),
            notices_in_flight: false,
            active_run: None,
            retry: None,
            pending_prompt_start: false,
            lifecycle_tx: None,
            lifecycle_busy: false,
            continue_after_hard_compact: false,
            expanded_details: HashMap::new(),
            detail_focus: None,
            debug_after_draw: None,
            debug: None,
            should_quit: false,
            session: SessionState {
                sink: None,
                cursor: None,
                cwd: PathBuf::new(),
            },
            picker_load_tx: None,
            picker_generation: Arc::new(AtomicU64::new(0)),
            executor: None,
            theme: Theme::default(),
            theme_manager: theme::ThemeManager::default(),
            ctrl_c_at: None,
            yank_notify: None,
            notify: None,
            render_profile: Box::default(),
            timers_recorded: Vec::new(),
            resumed_timers: Vec::new(),
            stale_jobs: Vec::new(),
        }
    }

    pub(super) fn session_model(&self) -> String {
        self.model.session_label()
    }

    pub(crate) fn begin_active_run(&mut self) -> &mut ActiveRunState {
        self.active_run.insert(ActiveRunState::start(&self.model))
    }

    /// Apply one event while rebuilding a file-backed transcript. At each
    /// turn boundary the completed turn is immediately reduced to its prompt;
    /// its blocks can be reconstructed later from `turn_byte_ranges`. This
    /// keeps resume peak memory bounded to one visible turn instead of the
    /// entire transcript.
    pub(super) fn apply_file_backed_replay_event(&mut self, ev: AgentEvent) {
        if matches!(ev, AgentEvent::Prompt { .. }) {
            if let Some(turn) = self.transcript.turns.last_mut() {
                turn.blocks.clear();
            }
        }
        self.apply_event(ev);
    }

    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
    pub(super) fn apply_event(&mut self, ev: AgentEvent) {
        if let AgentEvent::RunStart = ev {
            self.usage.reset_turn();
            return;
        }
        if let AgentEvent::Prompt { prompt, kind } = ev {
            // A live run dates its prompt publication; replay of a resumed
            // file has no run and must not sample once per turn.
            if self.run_active() {
                self.debug_sample("core_prompt");
            }
            // spawn_agent_run pre-pushes the prompt at submit time so it
            // shows before the working indicator's first frame. The
            // matching Prompt is then a confirmation, not a new turn.
            let awaited = self.pending_prompt_start
                && self.transcript.turns.last().is_some_and(|turn| {
                    turn.blocks.is_empty() && turn.prompt == prompt && turn.kind == kind
                });
            self.pending_prompt_start = false;
            if !awaited {
                self.transcript.freeze_previous_file_backed_turn();
                self.transcript.push_turn(Turn {
                    prompt,
                    kind,
                    blocks: Vec::new(),
                });
            }
            return;
        }
        if let AgentEvent::TurnContinue = ev {
            self.usage.reset_turn();
            return;
        }
        match ev {
            AgentEvent::Notice(msg) => {
                self.notify(NotifyKind::Warn, msg);
                return;
            }
            AgentEvent::RetryStart {
                attempt,
                max_attempts,
                ..
            } => {
                self.retry = Some(RetryState {
                    attempt,
                    max_attempts,
                });
                return;
            }
            AgentEvent::RetryEnd { .. } => {
                self.retry = None;
                return;
            }
            AgentEvent::RoundCommitted {
                byte_start,
                byte_end,
            } => {
                // A round reaches the transcript before the prompt event that
                // opens the next turn: core commits the previous round, then
                // records the prompt. While the prompt event is outstanding,
                // those bytes belong to the turn before the pushed shell, and
                // attaching them to the shell would lose its prompt row.
                let pending = usize::from(self.pending_prompt_start);
                let idx = self.transcript.turns.len().saturating_sub(1 + pending);
                self.transcript.merge_turn_range(idx, byte_start, byte_end);
                if let Some(turn) = self.transcript.turns.last_mut() {
                    for block in &mut turn.blocks {
                        if let Block::Tool(tool) = block {
                            if tool.name == "exec" && tool.done && !tool.is_error {
                                tool.result_committed = true;
                                if tool
                                    .result
                                    .as_ref()
                                    .is_some_and(|result| !result.is_empty())
                                {
                                    tool.result_availability = ResultAvailability::Available;
                                }
                                tool.result = None;
                            }
                        }
                    }
                }
                return;
            }
            AgentEvent::TurnCommitted {
                byte_start,
                byte_end,
            } => {
                // The just-finished turn is now durably in the transcript
                // file over this byte range. Record it so the turn becomes
                // file-backed when the next prompt freezes it.
                let idx = self.transcript.turns.len().saturating_sub(1);
                self.transcript.merge_turn_range(idx, byte_start, byte_end);
                return;
            }
            AgentEvent::ModelReported { id, display } => {
                let run_label = self
                    .active_run
                    .as_ref()
                    .and_then(|r| r.model_label.as_deref())
                    .unwrap_or(&self.model.label);
                if let Some((provider, _)) = run_label.split_once('/') {
                    // The report names the model only. Keep the effort and the
                    // tier the turn runs with, so the working line keeps them.
                    let suffix = self.model.suffix();
                    let response_label = format!("{provider}/{id}{suffix}");
                    let display_label = format!("{provider}/{display}");
                    let same_selection = self
                        .active_run
                        .as_ref()
                        .and_then(|r| r.selected_model_label.as_deref())
                        .is_none_or(|selected| {
                            Some(selected) == self.model.selected_label.as_deref()
                        });
                    if same_selection {
                        self.model.label = display_label;
                    }
                    if let Some(active) = self.active_run.as_mut() {
                        active.model_label = Some(response_label);
                    }
                }
                return;
            }
            AgentEvent::ContextReported { context } => {
                // Spend arrives on the terminal turn events only. The context
                // window is the compaction denominator, so a report from any
                // engine updates it.
                if let Some(size) = context.and_then(|context| context.size) {
                    self.lifecycle.set_context_window(size);
                }
                self.usage.reported_context = context;
                return;
            }
            AgentEvent::SubagentUsage { cost, usage } => {
                // Subagent spend folds into the session totals only: child
                // tokens never occupy the parent context window, so the
                // gauge (`status_usage`) and the turn fold stay untouched.
                self.usage.fold_subagent(cost, usage);
                return;
            }
            AgentEvent::RoundUsage { cost, usage } => {
                // Per-round refresh of the context gauge and cost counter.
                // `cost` is the turn's cumulative cost so far; track it in
                // `turn_cost` (folded into `cost` at `TurnEnd`) so the
                // footer can show a live running total. Tokens accumulate
                // directly into the session totals; `TurnEnd` skips
                // re-adding them when `turn_has_round_usage` is set.
                self.usage.turn_cost = cost;
                self.usage.turn_has_round_usage = true;
                self.usage.total_in += usage.input_tokens;
                self.usage.total_out += usage.output_tokens;
                self.usage.status_usage = Some(usage);
                self.usage.settled_usage_fresh = true;
                self.usage.compacted = false;
                return;
            }
            AgentEvent::TurnEnd { cost, usage, .. } => {
                if self.usage.turn_has_round_usage {
                    self.usage.cost += self.usage.turn_cost;
                } else {
                    self.usage.cost += cost;
                    self.usage.total_in += usage.input_tokens;
                    self.usage.total_out += usage.output_tokens;
                    self.usage.status_usage = Some(usage);
                }
                self.usage.turn_cost = 0.0;
                self.usage.turn_has_round_usage = false;
                self.usage.compacted = false;
            }
            AgentEvent::TurnFailed { cost, usage, .. }
            | AgentEvent::TurnCancelled { cost, usage, .. } => {
                // A failed turn's consumed tokens count honestly. Same
                // fold logic as `TurnEnd`: the live path already applied
                // per-round tokens via `RoundUsage` and `turn_cost` holds
                // the cumulative cost; the resume path applies the bundled
                // totals. `status_usage` updates either way so the gauge
                // reflects the failed turn's last round.
                if self.usage.turn_has_round_usage {
                    self.usage.cost += self.usage.turn_cost;
                } else {
                    self.usage.cost += cost;
                    self.usage.total_in += usage.input_tokens;
                    self.usage.total_out += usage.output_tokens;
                }
                self.usage.status_usage = Some(usage);
                self.usage.turn_cost = 0.0;
                self.usage.turn_has_round_usage = false;
            }
            AgentEvent::ContextPressure {
                resume,
                cost,
                usage,
                ..
            } => {
                // The run stopped at the hard cap. Fold the partial turn's
                // cost/tokens like `TurnEnd`, record the over-cap usage, and
                // arm the flag so the run loop force-compacts on channel
                // close, and continues the turn when the stop allows it. No
                // block is rendered — the compaction marker (from
                // `compact_now`) is the visible signal, and the continue
                // appends to this turn.
                if self.usage.turn_has_round_usage {
                    self.usage.cost += self.usage.turn_cost;
                } else {
                    self.usage.cost += cost;
                    self.usage.total_in += usage.input_tokens;
                    self.usage.total_out += usage.output_tokens;
                }
                self.usage.status_usage = Some(usage);
                self.usage.turn_cost = 0.0;
                self.usage.turn_has_round_usage = false;
                self.context_pressure = Some(resume);
                return;
            }
            AgentEvent::TurnUsage { cost, usage } => {
                // The transcript's last turn never settled: the process died
                // before its terminal marker. Its checkpoints kept the spend,
                // so the resumed totals match what the provider billed.
                self.usage.cost += cost;
                self.usage.total_in += usage.input_tokens;
                self.usage.total_out += usage.output_tokens;
                self.usage.status_usage = Some(usage);
                return;
            }
            AgentEvent::Compaction { .. } => self.usage.mark_compacted(),
            _ => {}
        }
        // `UserShellStart` (live) opens the standalone turn; a finished
        // `UserShell` does too unless it finalizes an already-streamed one.
        let starts_standalone_turn = match &ev {
            AgentEvent::UserShellStart { .. } => true,
            AgentEvent::UserShell { .. } => !user_shell_running(&self.transcript.turns),
            _ => false,
        };
        if starts_standalone_turn {
            self.transcript.freeze_previous_file_backed_turn();
        }
        let previous_turns = self.transcript.turns.len();
        let native_end = match &ev {
            AgentEvent::NativeToolEnd { parent, id, .. } => Some((parent.clone(), *id)),
            _ => None,
        };
        apply_event_to_turns(&mut self.transcript.turns, ev);
        if let Some((parent, id)) = native_end {
            if let Some(native) = self
                .transcript
                .turns
                .last_mut()
                .and_then(|turn| tool_mut(&mut turn.blocks, &parent))
                .and_then(|tool| tool.native.iter_mut().find(|native| native.id == id))
            {
                view::blocks::cache_native_preview(native);
            }
        }
        if starts_standalone_turn {
            debug_assert_eq!(self.transcript.turns.len(), previous_turns + 1);
            self.transcript.turn_byte_ranges.push(None);
            self.transcript.turn_event_offsets.push(None);
            self.transcript.turn_compactions.push(Vec::new());
            // The output of a direct `!cmd` is the point of running it:
            // show it expanded instead of hiding it behind a detail toggle.
            // Covers the live event and the file-backed replay paths, both
            // of which flow through here.
            self.expanded_details.insert(
                DetailKey::UserShell(detail_block_id(previous_turns, 0)),
                DetailState {
                    turn: previous_turns,
                    scroll: None,
                },
            );
        }
    }

    pub(super) fn run_finished(&mut self) {
        if let Some(turn) = self.transcript.turns.last_mut() {
            finalize_open_thinking(turn);
        }
        // Keep the just-finished turn live in memory. It was built from the
        // lossless event stream and is the authoritative UI representation for
        // the frame in which the run settles. Clearing it here made visibility
        // depend on an immediate, fallible disk re-read; a range/read/replay
        // failure was intentionally mapped to an empty shell by
        // materialize_turn(), making an entire agent response disappear.
        // Submit/spawn_prompt freezes this turn when the next prompt starts, so
        // memory remains bounded to the viewport working set plus one latest
        // turn rather than growing with the session.
        // The turn-end marker (model, elapsed, cost, usage) arrives as an
        // `AgentEvent::TurnEnd` emitted by the engine, which also writes it
        // to the transcript — so there is nothing to stamp or persist here.
        self.active_run = None;
        self.retry = None;
        self.pending_prompt_start = false;
    }

    /// Push a submitted prompt immediately instead of waiting for the
    /// engine's `Prompt` event, so it renders before the working indicator's
    /// first frame.
    pub(super) fn begin_prompt_turn(&mut self, prompt: String, kind: lofi_types::PromptKind) {
        self.transcript.freeze_previous_file_backed_turn();
        self.transcript.push_turn(Turn {
            prompt,
            kind,
            blocks: Vec::new(),
        });
        self.pending_prompt_start = true;
    }

    pub(super) fn run_active(&self) -> bool {
        self.active_run.is_some()
    }

    /// Whether the user asked the current run to stop and it has not settled.
    pub(super) fn cancel_pending(&self) -> bool {
        self.active_run
            .as_ref()
            .is_some_and(|r| r.cancel_at.is_some())
    }

    /// Whether a cancel request met no answer: the turn was told to stop and
    /// has reported nothing since. The turn polls the flag itself, so silence
    /// for this long means it is blocked inside a call that never observes it.
    pub(super) fn cancel_unacknowledged(&self) -> bool {
        self.active_run
            .as_ref()
            .is_some_and(ActiveRunState::cancel_unacknowledged)
    }

    pub(super) fn busy(&self) -> bool {
        self.run_active() || self.lifecycle_busy
    }

    /// The mailbox turn, when the host is at rest and the inbox still holds
    /// mail. A queued prompt goes first, and a wake never cuts a run short.
    pub(super) fn ready_mailbox_turn(&mut self) -> Option<String> {
        if self.busy() || !self.prompt_queue.is_empty() {
            return None;
        }
        self.take_mailbox_wake()
    }

    /// Take the stored wake, and only when it is still unclaimed and the
    /// inbox still holds mail: the turn that just ended may have read it.
    fn take_mailbox_wake(&mut self) -> Option<String> {
        let text = self.mailbox_wake.take()?;
        let registry = self.subagents()?;
        registry.claim_mailbox_wake().then_some(text)
    }

    /// Whether a mailbox notice should be stored at all. A notice that was
    /// buffered across a session switch names a sender the new inbox does
    /// not hold.
    pub(super) fn has_unclaimed_mailbox_wake(&self) -> bool {
        self.subagents()
            .as_ref()
            .is_some_and(lofi_core::SubagentRegistry::mailbox_wake_unclaimed)
    }

    /// The wake a broadcast lag may have dropped. Only an untaken wake is
    /// rebuilt: one the host already spent may name mail a cancelled turn
    /// left unread.
    pub(super) fn lost_mailbox_wake(&self) -> Option<String> {
        self.subagents()
            .as_ref()
            .and_then(lofi_core::SubagentRegistry::lost_mailbox_wake)
    }

    pub(super) fn spinner_frame(&self) -> usize {
        self.active_run.as_ref().map_or(0, |r| r.spinner)
    }

    pub(super) fn run_label(&self) -> String {
        self.active_run
            .as_ref()
            .and_then(|r| r.model_label.clone())
            .unwrap_or_else(|| self.session_model())
    }

    pub(super) fn run_model(&self) -> RunModel {
        self.model.run_model()
    }

    pub(super) fn run_elapsed(&self) -> Duration {
        self.active_run
            .as_ref()
            .map_or(Duration::ZERO, |r| r.started_at.elapsed())
    }
}

impl App {
    fn render_compaction(&mut self, compaction: &lofi_core::Compaction) {
        if self.transcript.turns.is_empty() {
            self.transcript.push_turn(Turn {
                prompt: String::new(),
                kind: lofi_types::PromptKind::User,
                blocks: Vec::new(),
            });
        }
        // The marker belongs to the turn the compaction ended: the last
        // file-backed one. A queued prompt may already have popped and pushed an
        // empty live shell in front of it; a marker appended there describes a
        // turn the compaction never summarized, and the shell's freeze drops it.
        let target = self
            .transcript
            .turn_byte_ranges
            .iter()
            .rposition(Option::is_some)
            .unwrap_or(self.transcript.turns.len().saturating_sub(1));
        // Store the marker alongside the turn's byte range so frozen turns stay
        // unhydrated on disk and live turns can still release their blocks on
        // the next freeze without losing the marker.
        let file_end = self
            .transcript
            .turn_byte_ranges
            .get(target)
            .copied()
            .flatten()
            .map(|(_, end)| end);
        if let Some(end) = file_end {
            if let Some(markers) = self.transcript.turn_compactions.get_mut(target) {
                markers.push(transcript::CompactionMarker {
                    after_byte: end,
                    summarized: compaction.summarized_count,
                    kept: compaction.kept_count,
                    summary: compaction.summary.clone(),
                });
            }
            self.transcript.collapsed_turns.get_mut().remove(target);
        }
        if let Some(turn) = self.transcript.turns.get_mut(target) {
            if file_end.is_none() || !turn.blocks.is_empty() {
                turn.blocks.push(Block::Compaction {
                    summarized: compaction.summarized_count,
                    kept: compaction.kept_count,
                    summary: compaction.summary.clone(),
                });
            }
        }
        self.usage.mark_compacted();
        self.bump_render_epoch();
        self.debug_sample("compaction");
    }

    fn start_lifecycle_operation(&mut self, operation: LifecycleOperation) -> bool {
        let Some(tx) = self.lifecycle_tx.clone() else {
            return false;
        };
        if self.run_active() {
            self.notify(NotifyKind::Warn, "wait for the running turn to finish");
            return true;
        }
        if self.lifecycle_busy {
            self.notify(NotifyKind::Warn, "wait for the memory operation to finish");
            return true;
        }
        self.lifecycle_busy = true;
        let mut lifecycle = self.lifecycle.clone();
        let cursor = self.session.cursor.clone();
        let system_prompt = self.system_prompt.clone();
        let task = tokio::task::spawn_blocking(move || match operation {
            LifecycleOperation::Compact => {
                LifecycleResult::Compact(lifecycle.compact(cursor.as_ref(), &system_prompt))
            }
            LifecycleOperation::AutoCompact(usage) => LifecycleResult::AutoCompact(
                lifecycle.auto_compact(usage, cursor.as_ref(), &system_prompt),
            ),
            LifecycleOperation::HardCompact => LifecycleResult::HardCompact(
                lifecycle.hard_compact(cursor.as_ref(), &system_prompt),
            ),
            LifecycleOperation::Recall(line) => LifecycleResult::Recall {
                result: lifecycle.recall_line(cursor.as_ref(), &line),
                line,
            },
        });
        tokio::task::spawn_local(async move {
            let result = task.await.unwrap_or_else(|error| {
                LifecycleResult::Failed(format!("memory operation failed: {error}"))
            });
            let _ = tx.send(result);
        });
        true
    }

    fn present_compact_result(&mut self, result: Result<Option<lofi_core::Compaction>>) -> bool {
        match result {
            Ok(Some(compaction)) => {
                self.render_compaction(&compaction);
                true
            }
            Ok(None) => {
                self.notify(NotifyKind::Warn, "not enough history to compact yet");
                false
            }
            Err(error) => {
                self.notify(NotifyKind::Error, format!("could not compact: {error}"));
                false
            }
        }
    }

    /// Request an immediate core-owned compaction and render its outcome.
    pub(super) fn compact_now(&mut self) -> bool {
        if self.start_lifecycle_operation(LifecycleOperation::Compact) {
            return true;
        }
        let cursor = self.session.cursor.clone();
        let system_prompt = self.system_prompt.clone();
        let result = self.lifecycle.compact(cursor.as_ref(), &system_prompt);
        self.present_compact_result(result)
    }

    fn present_recall_result(
        &mut self,
        line: &str,
        result: Result<Option<lofi_core::recall::RecallOutcome>>,
    ) {
        match result {
            Ok(Some(outcome)) => {
                self.transcript.push_turn(Turn {
                    prompt: line.trim().to_string(),
                    kind: lofi_types::PromptKind::User,
                    blocks: vec![Block::Text(outcome.text)],
                });
                self.notify(NotifyKind::Info, outcome.status);
                self.bump_render_epoch();
            }
            Ok(None) => self.notify(NotifyKind::Warn, "no session history yet"),
            Err(error) => self.notify(NotifyKind::Error, format!("recall failed: {error}")),
        }
    }

    /// Request core-owned recall and render the read-only result.
    pub(super) fn recall_now(&mut self, line: &str) {
        if self.start_lifecycle_operation(LifecycleOperation::Recall(line.to_string())) {
            return;
        }
        let cursor = self.session.cursor.as_ref();
        let result = self.lifecycle.recall_line(cursor, line);
        self.present_recall_result(line, result);
    }

    /// Let core evaluate soft-compaction policy; the UI only renders outcomes.
    pub(super) fn maybe_auto_compact(&mut self) {
        if !std::mem::take(&mut self.usage.settled_usage_fresh) {
            return;
        }
        let Some(usage) = self.usage.status_usage else {
            return;
        };
        // The gate is cheap and pure, so an under-threshold turn never
        // starts a lifecycle operation: the slot it would occupy for a
        // scheduler-dependent moment drops a manual /compact instead.
        if !self.lifecycle.should_auto_compact(usage) {
            return;
        }
        if self.start_lifecycle_operation(LifecycleOperation::AutoCompact(usage)) {
            return;
        }
        let cursor = self.session.cursor.clone();
        let system_prompt = self.system_prompt.clone();
        let result = self
            .lifecycle
            .auto_compact(usage, cursor.as_ref(), &system_prompt);
        if let Ok(Some(compaction)) = result {
            self.render_compaction(&compaction);
        } else if let Err(error) = result {
            self.notify(NotifyKind::Error, format!("could not compact: {error}"));
        }
    }

    pub(super) fn hard_compact(&mut self) -> Option<HardCompactOutcome> {
        if self.start_lifecycle_operation(LifecycleOperation::HardCompact) {
            return None;
        }
        let cursor = self.session.cursor.clone();
        let system_prompt = self.system_prompt.clone();
        let result = self.lifecycle.hard_compact(cursor.as_ref(), &system_prompt);
        Some(self.present_hard_compact_result(result))
    }

    fn present_hard_compact_result(
        &mut self,
        result: Result<HardCompactOutcome>,
    ) -> HardCompactOutcome {
        match result {
            Ok(HardCompactOutcome::Compacted(compaction)) => {
                self.render_compaction(&compaction);
                HardCompactOutcome::Compacted(compaction)
            }
            Ok(outcome) => outcome,
            Err(error) => {
                self.notify(NotifyKind::Error, format!("could not compact: {error}"));
                HardCompactOutcome::NotEnoughHistory
            }
        }
    }

    pub(super) fn apply_lifecycle_result(&mut self, result: LifecycleResult) -> bool {
        self.lifecycle_busy = false;
        match result {
            LifecycleResult::Compact(result) => {
                self.present_compact_result(result);
                false
            }
            LifecycleResult::AutoCompact(result) => {
                match result {
                    Ok(Some(compaction)) => self.render_compaction(&compaction),
                    Ok(None) => {}
                    Err(error) => {
                        self.notify(NotifyKind::Error, format!("could not compact: {error}"));
                    }
                }
                false
            }
            LifecycleResult::Recall { line, result } => {
                self.present_recall_result(&line, result);
                false
            }
            LifecycleResult::HardCompact(result) => {
                match self.present_hard_compact_result(result) {
                    HardCompactOutcome::Compacted(_) => true,
                    other => {
                        if let Some(reason) = other.blocked_reason() {
                            self.notify(NotifyKind::Warn, reason);
                        }
                        false
                    }
                }
            }
            LifecycleResult::Failed(error) => {
                self.notify(NotifyKind::Error, error);
                false
            }
        }
    }
}
