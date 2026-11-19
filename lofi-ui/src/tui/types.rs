#[allow(clippy::wildcard_imports)]
use super::*;

// Tree rows are produced by the core session projection; the TUI keeps the
// historical TreeEntry name for its picker state.
pub(crate) use lofi_core::session::tree::TreeRow as TreeEntry;

pub(crate) fn detail_block_id(turn: usize, block: usize) -> u64 {
    ((turn as u64) << 32) | block as u64
}

pub(crate) fn assign_detail_ids(turn: &mut Turn, turn_index: usize) {
    for (block_index, block) in turn.blocks.iter_mut().enumerate() {
        let id = detail_block_id(turn_index, block_index);
        match block {
            Block::Tool(tool) => tool.detail_id = id,
            Block::UserShell { id: block_id, .. } => *block_id = id,
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub(crate) enum DetailKey {
    Exec(u64),
    ExecResult(u64),
    NativeTool { parent: u64, id: u64 },
    UserShell(u64),
}

impl DetailKey {
    pub(crate) fn retains_expansion(&self) -> bool {
        matches!(self, Self::UserShell(_))
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DetailState {
    pub(crate) turn: usize,
    pub(crate) scroll: Option<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct DetailFocus {
    pub(crate) key: DetailKey,
    pub(crate) cursor: usize,
    pub(crate) col: usize,
}

impl DetailFocus {
    pub(crate) fn on(target: &view::DetailTarget, col: usize) -> Self {
        Self {
            key: target.key.clone(),
            // A released result leaves the collapsed header's total stale, so
            // a tail detail cannot know its last row yet. usize::MAX defers
            // the clamp to the next render, which sees the expanded layout.
            cursor: target
                .row
                .unwrap_or(if target.tail { usize::MAX } else { 0 }),
            col,
        }
    }
}

/// A native tool call (`lofi.bash`/`lofi.read`/…) observed inside an `exec`
/// block, surfaced so the UI can render each one under its parent exec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct NativePreview {
    pub(crate) header_suffix: Option<String>,
    pub(crate) lines: Vec<String>,
    pub(crate) total_lines: usize,
    pub(crate) preview_start: usize,
    pub(crate) numbered: bool,
    pub(crate) start_line: usize,
    pub(crate) is_diff: bool,
    pub(crate) notice: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct NativeTool {
    pub(crate) id: u64,
    pub(crate) name: String,
    pub(crate) args: String,
    pub(crate) result: Option<String>,
    pub(crate) preview: Option<Box<NativePreview>>,
    pub(crate) is_error: bool,
    pub(crate) done: bool,
}

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum ResultAvailability {
    #[default]
    None,
    Available,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ToolCall {
    #[serde(skip)]
    pub(crate) detail_id: u64,
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) input: String,
    pub(crate) label: Option<String>,
    pub(crate) native: Vec<NativeTool>,
    pub(crate) result: Option<String>,
    #[serde(default)]
    pub(crate) result_availability: ResultAvailability,
    pub(crate) result_committed: bool,
    pub(crate) is_error: bool,
    pub(crate) done: bool,
    pub(crate) elapsed: Option<Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ThinkingBlock {
    pub(crate) text: String,
    #[serde(skip, default = "Instant::now")]
    pub(crate) start: Instant,
    pub(crate) elapsed: Option<Duration>,
}

#[derive(Debug, Clone)]
pub(crate) struct RetryState {
    pub(crate) attempt: u32,
    pub(crate) max_attempts: u32,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct UsageState {
    pub(crate) status_usage: Option<Usage>,
    /// The context state the engine reported after its last step.
    pub(crate) reported_context: Option<lofi_types::ContextUsage>,
    pub(crate) total_in: u64,
    pub(crate) total_out: u64,
    pub(crate) cost: f64,
    pub(crate) turn_cost: f64,
    pub(crate) turn_has_round_usage: bool,
    /// True when a settled round supplied usage that core policy has not
    /// evaluated yet.
    pub(crate) settled_usage_fresh: bool,
    pub(crate) compacted: bool,
}

impl UsageState {
    pub(crate) fn reset_session(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn reset_turn(&mut self) {
        self.turn_cost = 0.0;
        self.turn_has_round_usage = false;
        self.settled_usage_fresh = false;
    }

    pub(crate) fn reset_compaction_gauges(&mut self) {
        self.status_usage = None;
        // A rewritten history invalidates the fill the engine last reported;
        // the next round or turn reports the new one.
        self.reported_context = None;
        self.settled_usage_fresh = false;
        self.compacted = false;
    }

    pub(crate) fn mark_compacted(&mut self) {
        self.reset_compaction_gauges();
        self.compacted = true;
    }

    pub(crate) fn fold_subagent(&mut self, cost: f64, usage: Usage) {
        self.cost += cost;
        self.total_in += usage.input_tokens;
        self.total_out += usage.output_tokens;
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ModelSelectionState {
    pub(crate) label: String,
    /// The `provider/model` the user picked, kept while a turn reports the id
    /// it actually ran so the header can show both.
    pub(crate) selected_label: Option<String>,
    pub(crate) thinking_label: Option<String>,
    pub(crate) thinking: ThinkingLevel,
    pub(crate) service_label: Option<String>,
    pub(crate) service_tier: ServiceTier,
    pub(crate) ctx_limit: u64,
    pub(crate) choices: Vec<lofi_types::ModelChoice>,
    pub(crate) pending_switch: Option<String>,
    pub(crate) no_models_hint: Option<String>,
}

impl ModelSelectionState {
    pub(crate) fn new(
        label: String,
        thinking: ThinkingLevel,
        service_tier: ServiceTier,
        ctx_limit: u64,
    ) -> Self {
        let thinking_label =
            (thinking != ThinkingLevel::Off).then(|| format!(":{}", thinking.as_str()));
        let service_label =
            (service_tier != ServiceTier::Auto).then(|| format!("@{}", service_tier.as_str()));
        Self {
            label,
            selected_label: None,
            thinking_label,
            thinking,
            service_label,
            service_tier,
            ctx_limit: if ctx_limit > 0 {
                ctx_limit
            } else {
                DEFAULT_CTX_LIMIT
            },
            choices: Vec::new(),
            pending_switch: None,
            no_models_hint: None,
        }
    }

    /// The label a model switch or capability lookup must carry: the selected
    /// entry when a backend reported a resolved display name over it.
    pub(crate) fn switch_label(&self) -> &str {
        self.selected_label.as_deref().unwrap_or(&self.label)
    }

    pub(crate) fn suffix(&self) -> String {
        format!(
            "{}{}",
            self.thinking_label.as_deref().unwrap_or(""),
            self.service_label.as_deref().unwrap_or("")
        )
    }

    pub(crate) fn session_label(&self) -> String {
        format!("{}{}", self.label, self.suffix())
    }

    /// Build a `provider/model:effort[@tier]` switch query. Explicit `@auto` is
    /// emitted only when leaving a non-`auto` tier so switching back overrides
    /// any configured provider/model default without polluting ordinary `auto`
    /// queries.
    pub(crate) fn format_switch_query(
        &self,
        base_label: &str,
        thinking: &ThinkingLevel,
        tier: &ServiceTier,
    ) -> String {
        let tier_suffix = if *tier != ServiceTier::Auto {
            format!("@{}", tier.as_str())
        } else if self.service_tier != ServiceTier::Auto {
            "@auto".to_string()
        } else {
            String::new()
        };
        format!("{}:{}{}", base_label, thinking.as_str(), tier_suffix)
    }

    /// Switch query needed to align the live selection with a resumed turn's
    /// recorded [`RunModel`], or `None` when model, effort, and tier already
    /// match.
    pub(crate) fn restore_switch_query(&self, model: &RunModel) -> Option<String> {
        let target_base = format!("{}/{}", model.provider, model.selected_or_id());
        if target_base == self.switch_label()
            && model.thinking == self.thinking
            && model.service_tier == self.service_tier
        {
            return None;
        }
        Some(self.format_switch_query(&target_base, &model.thinking, &model.service_tier))
    }

    pub(crate) fn run_model(&self) -> RunModel {
        let (provider, id) = match self.label.split_once('/') {
            Some((p, r)) => (p, r),
            None => ("", self.label.as_str()),
        };
        let selected = self
            .selected_label
            .as_deref()
            .and_then(|label| label.split_once('/'))
            .filter(|(label_provider, _)| *label_provider == provider)
            .map_or(id, |(_, selected)| selected);
        RunModel {
            provider: provider.to_string(),
            id: id.to_string(),
            selected: selected.to_string(),
            display: (selected != id).then(|| id.to_string()),
            thinking: self.thinking.clone(),
            service_tier: self.service_tier.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ActiveRunState {
    pub(crate) spinner: usize,
    pub(crate) started_at: Instant,
    /// When the current run last delivered an event.
    pub(crate) event_at: Instant,
    /// When the user first asked the current run to stop.
    pub(crate) cancel_at: Option<Instant>,
    /// Model label captured when the run started. A mid-run `/model` switch
    /// updates the footer; the "Working for ..." line keeps showing the model
    /// the in-flight run is actually using.
    pub(crate) model_label: Option<String>,
    pub(crate) selected_model_label: Option<String>,
}

impl ActiveRunState {
    pub(crate) fn start(model: &ModelSelectionState) -> Self {
        let now = Instant::now();
        Self {
            spinner: 0,
            started_at: now,
            event_at: now,
            cancel_at: None,
            model_label: Some(model.session_label()),
            selected_model_label: Some(model.switch_label().to_string()),
        }
    }

    pub(crate) fn cancel_unacknowledged(&self) -> bool {
        let Some(requested) = self.cancel_at else {
            return false;
        };
        self.event_at.max(requested).elapsed() >= CANCEL_STALL
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum Block {
    Text(String),
    Thinking(ThinkingBlock),
    Tool(ToolCall),
    UserShell {
        #[serde(skip)]
        id: u64,
        command: String,
        output: String,
        exit_code: Option<i32>,
        signal: Option<i32>,
        duration: Duration,
        truncated: bool,
        cancelled: bool,
        /// Live-only: the command is still running and `output` is the
        /// stream accumulated so far. Default `false` (finished) matches any
        /// value serialized before the field existed.
        #[serde(default)]
        running: bool,
        exclude_from_context: bool,
    },
    Error(String),
    TurnEnd {
        label: String,
        elapsed: Duration,
    },
    TurnFailed {
        label: String,
        elapsed: Duration,
        error: String,
    },
    TurnCancelled {
        label: String,
        elapsed: Duration,
    },
    Compaction {
        summarized: usize,
        kept: usize,
        summary: String,
    },
}

/// A prompt waiting to be run: typed input queued while a run is live, or
/// an app-injected notice waking the agent. The kind rides along so the
/// resulting turn renders with the right marker.
#[derive(Debug, Clone)]
pub(crate) struct QueuedPrompt {
    pub text: String,
    pub kind: lofi_types::PromptKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Turn {
    pub(crate) prompt: String,
    /// Where the prompt came from. The kind is not persisted to the session
    /// log, so restored turns always report `User`.
    #[serde(default)]
    pub(crate) kind: lofi_types::PromptKind,
    pub(crate) blocks: Vec<Block>,
}

#[derive(Debug)]
pub(crate) struct SessionState {
    /// Core-owned session write endpoint. `None` in ephemeral mode
    /// (`--no-session`). The UI never assembles or appends session events;
    /// every mutation flows through this sink.
    pub(crate) sink: Option<lofi_core::session::sink::SessionSink>,
    pub(crate) cursor: Option<store::SessionCursor>,
    pub(crate) cwd: PathBuf,
}

impl SessionState {
    pub(crate) fn path(&self) -> Option<&Path> {
        self.cursor.as_ref().map(store::SessionCursor::path)
    }

    /// Attach a cursor to the session owner and refresh the read mirror.
    /// Keeping this as one operation prevents resume from rendering one lineage
    /// while subsequent writes and tree operations still use another.
    pub(crate) fn attach_cursor(&mut self, cursor: store::SessionCursor) {
        if let Some(sink) = self.sink.as_mut() {
            sink.set_cursor(cursor.clone());
        }
        self.cursor = Some(cursor);
    }

    /// Sync `cursor` from the sink after any sink-side mutation. The sink owns
    /// cursor creation and branch moves; this mirror exists only for reads.
    pub(crate) fn refresh_cursor(&mut self) {
        self.cursor = self.sink.as_ref().and_then(|s| s.cursor().cloned());
    }

    /// Borrow the sink for a core-side session mutation. Returns `None` in
    /// ephemeral (`--no-session`) mode, so callers skip session recording.
    pub(crate) fn sink_mut(&mut self) -> Option<&mut lofi_core::session::sink::SessionSink> {
        self.sink.as_mut()
    }
}

pub(crate) struct SessionConfig {
    /// Core-owned session write endpoint; `None` for ephemeral sessions.
    pub(crate) sink: Option<lofi_core::session::sink::SessionSink>,
    pub(crate) cursor: Option<store::SessionCursor>,
    pub(crate) index: Vec<store::SessionIndexEntry>,
    pub(crate) file_size: u64,
    pub(crate) history_start: usize,
    pub(crate) contiguous: bool,
    pub(crate) cwd: PathBuf,
}

impl SessionConfig {
    pub(crate) fn ephemeral(cwd: PathBuf) -> Self {
        Self {
            sink: None,
            cursor: None,
            index: Vec::new(),
            file_size: 0,
            history_start: 0,
            contiguous: true,
            cwd,
        }
    }

    /// Fresh session: no file on disk yet. The sink creates it lazily on the
    /// first run or user bash. `sink` is `None` only for `--no-session`.
    pub(crate) fn fresh(sink: lofi_core::session::sink::SessionSink, cwd: PathBuf) -> Self {
        Self {
            sink: Some(sink),
            cursor: None,
            index: Vec::new(),
            file_size: 0,
            history_start: 0,
            contiguous: true,
            cwd,
        }
    }

    /// Resume an existing session at its selected cursor.
    pub(crate) fn resumed(
        sink: lofi_core::session::sink::SessionSink,
        cursor: store::SessionCursor,
        index: Vec<store::SessionIndexEntry>,
        file_size: u64,
        history_start: usize,
        contiguous: bool,
        cwd: PathBuf,
    ) -> Self {
        Self {
            sink: Some(sink),
            cursor: Some(cursor),
            index,
            file_size,
            history_start,
            contiguous,
            cwd,
        }
    }

    #[must_use]
    pub(crate) fn last_run_model(&self) -> Option<RunModel> {
        self.cursor
            .as_ref()
            .and_then(|cursor| last_run_model_from_index(cursor, &self.index))
    }
}

pub(crate) struct Selection {
    pub(crate) start: (usize, usize),
    pub(crate) end: (usize, usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Input,
    Navigate,
    Select,
}

pub(crate) enum LifecycleOperation {
    Compact,
    AutoCompact(Usage),
    HardCompact,
    Recall(String),
}

pub(crate) enum LifecycleResult {
    Compact(Result<Option<lofi_core::Compaction>>),
    AutoCompact(Result<Option<lofi_core::Compaction>>),
    HardCompact(Result<HardCompactOutcome>),
    Recall {
        line: String,
        result: Result<Option<lofi_core::recall::RecallOutcome>>,
    },
    Failed(String),
}

pub(crate) enum PickerLoad {
    ResumePreviews {
        generation: u64,
        rows: Vec<(usize, String)>,
    },
    ResumeRows {
        generation: u64,
        rows: Vec<(usize, SessionEntry)>,
    },
    TreeReady {
        generation: u64,
        entries: Vec<TreeEntry>,
        /// Handle for the IO-thread-owned snapshot. The full
        /// `Vec<EventIndex>` stays on the worker that built it so its free
        /// happens on the same thread that allocated — sending the Vec itself
        /// would push ~9 MB onto the main thread and leave the freed slack in
        /// the worker's arena on close.
        snapshot_id: u64,
    },
    TreeRows {
        generation: u64,
        rows: Vec<(usize, TreeEntry)>,
    },
    TreeFailed {
        generation: u64,
        error: String,
    },
}
