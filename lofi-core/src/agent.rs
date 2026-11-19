use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use lofi_types::{ContentBlock, Message, Model, Role, RunModel, ServiceTier, ThinkingLevel, Usage};
use tokio::sync::mpsc::Sender;

use crate::backend;
use crate::models::ModelRegistry;
use crate::state;
use lofi_error::{Error, Result};
use lofi_providers::ToolSchema;
use lofi_providers::{open, Provider};
use lofi_types::BashConfig;
#[cfg(test)]
use lofi_types::StreamingEvent;
#[cfg(test)]
use std::time::Duration;

mod agent_run;
#[cfg(test)]
use agent_run::assistant_announces_tool_intent;
pub(crate) use agent_run::messages::{
    close_orphaned_tool_uses, close_unanswered_tool_uses, MAX_REQUEST_IMAGE_BYTES,
};
mod auto_mode;
mod event;
mod exec;
mod mailbox;
mod model;
pub(crate) mod native;
mod subagent;
mod tool_executor;
pub(crate) use tool_executor::ActivityWatch;
pub(crate) mod turn;
pub(crate) mod turn_session;
pub use lofi_code::SubagentOp;
pub(crate) use subagent::{backend_hook, settled_cost};
pub use subagent::{
    SubagentLive, SubagentNotice, SubagentNoticeKind, SubagentRegistry, SubagentRunning,
    SubagentSupport,
};
pub use tool_executor::{ConfirmRequest, ConfirmResponse, ToolCallContext, ToolExecutor};
pub(crate) use turn_session::ToolRecorder;
pub use turn_session::{new_cancel_flag, signal_cancel, RunControls, TurnInput, TurnRequest};

#[cfg(test)]
mod tests;

pub(crate) use auto_mode::collect_text;
pub use event::AgentEvent;
#[cfg(test)]
pub(crate) use exec::extract_code_prefix;
pub(crate) use exec::{
    cap_description, exec_timeout_ms, exec_tool_schema_bytes, exec_tool_schema_ref,
    normalize_exec_input, normalize_exec_input_in_place,
};
pub(crate) use exec::{cap_exec_envelope, cap_exec_result, cap_tool_result, CodePrefixDecoder};
pub use exec::{
    exec_input_code_and_label, exec_label, exec_result_display, exec_tool_schema, parse_exec_input,
};
pub(crate) use model::select_model_for_subagent;
pub(crate) use model::{parse_model_query, NO_MODELS_HINT};
pub use model::{rebuild_agent, select_model, select_model_with_level};

pub const SYSTEM_PROMPT: &str = include_str!("prompts/system.md");

/// Resolves `[[permissions]]` entries into granted directories: `~` is
/// expanded, then the deepest existing ancestor is canonicalized and the
/// missing tail re-joined, so a grant for a not-yet-created path matches
/// what a later resolution computes.
pub(crate) fn granted_dirs(
    permissions: &[lofi_types::PermissionEntry],
) -> Vec<lofi_code::GrantedDir> {
    permissions
        .iter()
        .map(|entry| {
            let dir = crate::bash_env::expand_tilde(&entry.path);
            lofi_code::GrantedDir {
                dir: lofi_code::canonicalize_up(&dir).unwrap_or(dir),
                access: entry.access,
            }
        })
        .collect()
}

/// Maximum streamed bytes of one reply the model re-reads: text, thinking, and
/// tool input, plus a fixed cost per event, so a stream of many tiny deltas is
/// bounded too. A round over it keeps what it wrote, and the turn reports the
/// cut.
const MAX_ROUND_BYTES: usize = 1024 * 1024;
/// Maximum cumulative bytes of provider signatures and redacted reasoning
/// kept in one round. The reply budget ignores them because the model never
/// re-reads them, but they are retained, so they keep their own bound.
const MAX_OPAQUE_ROUND_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_TOOL_RESULT_BYTES: usize = 50 * 1024;
/// Outer cap on the whole exec result sent back to the provider. Native
/// sub-tool results are already individually capped to `MAX_TOOL_RESULT_BYTES`;
/// this guards the aggregate (many sub-tool results + `logs`) so a long
/// `Promise.all` burst or a verbose `print` loop can't balloon the context.
pub(crate) const MAX_EXEC_RESULT_BYTES: usize = 200 * 1024;

const PER_EVENT_OVERHEAD: usize = 64;
/// User-role notice appended once per turn when the provider reports a
/// token-limit stop, nudging the model to pick up where it was cut off.
pub(crate) const TRUNCATION_CONTINUATION_PROMPT: &str =
    "Your previous response was cut off at the token limit. Continue where you left off.";
pub(crate) const THINKING_ONLY_CONTINUATION_PROMPT: &str = "Your previous response contained reasoning but no final answer. Continue the task and give the final answer.";
pub(crate) const LOST_TOOL_CONTINUATION_PROMPT: &str = "Your tool call was not received. Continue the task by issuing the required tool call again. If no tool call is required, give the final answer.";
pub(crate) const INTENT_CONTINUATION_PROMPT: &str = "Continue the task now. If the task is already complete, give the final answer. Do not only describe the next action.";
pub(crate) const BYTE_BUDGET_CONTINUATION_PROMPT: &str = "Your previous response was cut off at the reply size limit. Answer briefly: give the result and omit long reasoning.";

/// Spend of one complete usage report, or `None` when the model carries no
/// price. The same arithmetic bills a finished round and a one-shot call.
#[must_use]
pub fn usage_cost(model: &Model, usage: Usage) -> Option<f64> {
    let (Some(input), Some(output)) = (model.input_price, model.output_price) else {
        return model.per_request_price;
    };
    let cache_read = model.cache_read_price.unwrap_or(input);
    let cache_write = model.cache_write_price.unwrap_or(input);
    let mut cost = usage.input_tokens as f64 / 1_000_000.0 * input
        + usage.cache_read_tokens as f64 / 1_000_000.0 * cache_read
        + usage.cache_write_tokens as f64 / 1_000_000.0 * cache_write
        + usage.output_tokens as f64 / 1_000_000.0 * output;
    if let Some(per_request) = model.per_request_price {
        cost += per_request;
    }
    Some(cost)
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Cheaply cloneable because the provider and session handles are reference-counted.
#[derive(Clone)]
pub struct Agent {
    pub(crate) provider: Arc<dyn Provider>,
    pub(crate) model: Model,
    pub(crate) executor: ToolExecutor,
    retry: crate::retry::RetryPolicy,
    pub(crate) system_prompt: String,
    max_output_tokens: Option<u64>,
    compaction: lofi_types::CompactionConfig,
    auto_continue: lofi_types::AutoContinuePolicy,
    /// The config and registry this backend was opened from. Model selection
    /// rebuilds the transport from here, so no caller keeps a second copy.
    pub(crate) source: Option<Arc<backend::NativeSource>>,
}

impl Agent {
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider: Box<dyn Provider>,
        model: Model,
        root: PathBuf,
        tmp_dir: PathBuf,
        system_prompt: String,
        max_output_tokens: Option<u64>,
        compaction: lofi_types::CompactionConfig,
        bash: &BashConfig,
        env: &[(String, String)],
        truncate: lofi_types::TruncateConfig,
        image: lofi_types::ImageConfig,
        shell_policy_config: &lofi_types::ShellPolicyConfig,
    ) -> Self {
        let bash_env = crate::bash_env::resolve_bash_env(bash, env);
        let truncate = lofi_code::TruncatedCap {
            max_lines: truncate.max_lines,
            max_bytes: truncate.max_bytes,
        };
        let shell_policy = lofi_code::policy::defaults::resolve(shell_policy_config);
        // One handle space for everything this session owns, so a job, a
        // subagent, and a timer can never share an id.
        let ids = lofi_types::IdSpace::new();
        let memory = state::WorkspaceMemory::open(&root).ok().map(Into::into);
        let executor = ToolExecutor {
            root,
            tmp_dir,
            tmp_lease: None,
            bash_env,
            shell_policy,
            truncate,
            image,
            confirm_tx: None,
            confirm_counter: Arc::new(AtomicU64::new(0)),
            confirm_watch: Arc::new(tool_executor::ConfirmWatch::default()),
            activity: Arc::new(tool_executor::ActivityWatch::default()),
            auto_mode: None,
            auto_mode_config: None,
            auto_mode_hints: auto_mode::hint_handle(),
            policy_override: lofi_code::policy::PolicyOverride::default(),
            skills_dir: None,
            grants: Vec::new(),
            memory,
            jobs: lofi_code::tools::JobRegistry::with_ids(ids.clone()),
            timers: lofi_code::tools::TimerRegistry::with_ids(ids.clone()),
            job_owner: None,
            exec_worker: std::sync::Arc::new(std::sync::OnceLock::new()),
            subagent: None,
            mailbox: None,
            subagents: SubagentRegistry::with_ids(ids),
            subagent_support: None,
            models: None,
            session_cache: Arc::new(std::sync::OnceLock::new()),
        };
        Self::from_executor(provider, model, executor, system_prompt, compaction)
            .with_max_output_tokens(max_output_tokens)
    }

    #[must_use]
    pub(crate) fn from_executor(
        provider: Box<dyn Provider>,
        model: Model,
        mut executor: ToolExecutor,
        system_prompt: String,
        compaction: lofi_types::CompactionConfig,
    ) -> Self {
        executor.models = executor
            .subagent_support
            .as_ref()
            .map(SubagentSupport::models_fn);
        executor.invalidate_session();
        Self {
            provider: Arc::from(provider),
            model,
            executor,
            retry: crate::retry::RetryPolicy::default(),
            system_prompt,
            max_output_tokens: None,
            compaction,
            auto_continue: lofi_types::AutoContinuePolicy::default(),
            source: None,
        }
    }

    fn with_max_output_tokens(mut self, max_output_tokens: Option<u64>) -> Self {
        self.max_output_tokens = max_output_tokens;
        self
    }

    /// Attaches config-sourced granted directories outside the workspace
    /// root. A model switch keeps them because every rebuild clones the
    /// executor.
    #[must_use]
    pub fn with_grants(mut self, grants: Vec<lofi_code::GrantedDir>) -> Self {
        self.executor.grants = grants;
        self.executor.invalidate_session();
        self
    }

    /// The config-sourced grants, for the grant hook and the paths manager.
    #[must_use]
    pub fn grants(&self) -> &[lofi_code::GrantedDir] {
        &self.executor.grants
    }

    fn with_tmp_lease(mut self, lease: state::SessionTempDir) -> Self {
        let lease = Arc::new(lease);
        self.executor.tmp_dir = lease.path().to_path_buf();
        self.executor.tmp_lease = Some(lease);
        self.executor.invalidate_session();
        self
    }

    #[must_use]
    pub fn with_model(&self, provider: Box<dyn Provider>, model: Model) -> Self {
        Self {
            provider: Arc::from(provider),
            model,
            ..self.clone()
        }
    }

    #[must_use]
    pub fn with_system_prompt(&self, system_prompt: String) -> Self {
        Self {
            system_prompt,
            ..self.clone()
        }
    }

    #[must_use]
    pub fn with_skills_dir(&self, skills_dir: Option<PathBuf>) -> Self {
        let mut next = self.clone();
        next.executor = next.executor.with_skills_dir(skills_dir);
        next
    }

    /// Replace every session-scoped field from an executor.
    #[must_use]
    pub fn with_executor(&self, executor: &ToolExecutor) -> Self {
        let mut next = self.clone();
        next.executor = executor.clone();
        if next.executor.models.is_none() && next.executor.subagent_support.is_some() {
            next.executor.models = next
                .executor
                .subagent_support
                .as_ref()
                .map(SubagentSupport::models_fn);
            next.executor.invalidate_session();
        }
        next
    }

    /// A new conversation that shares the backend configuration and nothing
    /// else: its own transcript slot, its own nested-subagent scope.
    #[must_use]
    pub fn forked(&self, executor: &ToolExecutor) -> Self {
        let mut child = self.with_executor(executor);
        child.executor.mailbox = None;
        child.executor.subagent = None;
        child.executor.models = None;
        child.executor.subagent_support = None;
        child.executor.subagents = SubagentRegistry::default();
        child.executor.invalidate_session();
        child
    }

    #[must_use]
    pub fn with_source(&self, source: Arc<backend::NativeSource>) -> Self {
        Self {
            source: Some(source),
            ..self.clone()
        }
    }

    #[must_use]
    pub fn source(&self) -> Option<&Arc<backend::NativeSource>> {
        self.source.as_ref()
    }

    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    #[must_use]
    pub fn provider(&self) -> &Arc<dyn Provider> {
        &self.provider
    }

    #[must_use]
    pub fn with_confirm_tx(
        &self,
        confirm_tx: tokio::sync::mpsc::UnboundedSender<ConfirmRequest>,
    ) -> Self {
        let mut next = self.clone();
        next.executor.confirm_tx = Some(confirm_tx);
        next
    }

    /// Share the session's jobs, policy and sandbox with a backend.
    #[must_use]
    pub fn tool_executor(&self) -> ToolExecutor {
        let mut executor = self.executor.clone();
        executor.subagent = self.subagent_hook();
        if executor.models.is_none() && executor.subagent_support.is_some() {
            executor.models = executor
                .subagent_support
                .as_ref()
                .map(SubagentSupport::models_fn);
            executor.invalidate_session();
        }
        executor
    }

    /// Keep execution state when switching from another backend.
    #[must_use]
    pub fn with_tool_executor(mut self, mut executor: ToolExecutor) -> Self {
        if let Some(support) = self.executor.subagent_support {
            executor.models = Some(support.models_fn());
            executor.subagent_support = Some(support);
            executor.invalidate_session();
        } else if executor.models.is_none() && executor.subagent_support.is_some() {
            executor.models = executor
                .subagent_support
                .as_ref()
                .map(SubagentSupport::models_fn);
            executor.invalidate_session();
        }
        self.executor = executor;
        self
    }

    #[must_use]
    pub fn with_auto_mode(&self, auto_mode: lofi_code::AutoModeFn) -> Self {
        let mut next = self.clone();
        next.executor = next.executor.with_auto_mode(Some(auto_mode));
        next
    }

    /// The turn-context handle the evaluator reads. The turn loop replaces
    /// its contents at the start of every turn.
    #[must_use]
    pub fn auto_mode_hints(&self) -> &lofi_code::AutoModeHintHandle {
        &self.executor.auto_mode_hints
    }

    #[must_use]
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    #[must_use]
    pub fn root(&self) -> &PathBuf {
        &self.executor.root
    }

    #[must_use]
    pub fn tmp_dir(&self) -> &PathBuf {
        &self.executor.tmp_dir
    }

    /// The session-scoped `/policy` approval override handle; the TUI
    /// writes to it, every exec reads from it.
    #[must_use]
    pub fn policy_override(&self) -> &lofi_code::policy::PolicyOverride {
        &self.executor.policy_override
    }

    /// Whether auto-mode approval evaluation is configured for this agent.
    #[must_use]
    pub fn has_auto_mode(&self) -> bool {
        self.executor.auto_mode.is_some()
    }

    /// The session's background-job registry. Cheap to clone (shares the
    /// same map); the UI clones it once to render the job list and badge
    /// and to kill jobs from the `/job` modal.
    #[must_use]
    pub fn jobs(&self) -> lofi_code::tools::JobRegistry {
        self.executor.jobs.clone()
    }

    /// Session-scoped wake-up timers, for a host to poll the notices fired
    /// since its last pull.
    #[must_use]
    pub fn timers(&self) -> lofi_code::tools::TimerRegistry {
        self.executor.timers.clone()
    }

    /// The session's subagent registry (shared by every clone). The UI keeps
    /// a handle so a new session can drop the entries of the old one.
    #[must_use]
    pub fn subagents(&self) -> SubagentRegistry {
        self.executor.subagents.clone()
    }

    /// The raw model identity stamped onto [`AgentEvent::TurnEnd`] (and the
    /// transcript's `TurnEnd` marker, via the recorder). Built from the
    /// resolved [`Model`] so it is authoritative across a model switch on
    /// resume — the persisted turn keeps its original model, and the UI
    /// renders it to `provider/id:level` at display time rather than storing
    /// a formatted string.
    #[must_use]
    pub fn run_model(&self) -> RunModel {
        RunModel {
            provider: self.model.provider.clone(),
            id: self.model.id.clone(),
            selected: self.model.id.clone(),
            display: None,
            thinking: self.model.thinking.clone(),
            service_tier: self.model.service_tier.clone(),
        }
    }

    #[must_use]
    pub fn with_retry(mut self, retry: crate::retry::RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    #[must_use]
    pub fn with_auto_continue(mut self, policy: lofi_types::AutoContinuePolicy) -> Self {
        self.auto_continue = policy;
        self
    }

    #[must_use]
    pub fn max_output_tokens(&self) -> Option<u64> {
        self.max_output_tokens
    }

    /// Effective `max_tokens` for the next request, clipped so the request
    /// cannot demand more output than the remaining context window. Some
    /// providers reject `max_tokens >= context_window - input_tokens`; the
    /// prior round's [`Usage::context_tokens`] approximates the next
    /// request's input. Returns `None` (omit the field) when the
    /// window or model cap is unknown.
    #[must_use]
    fn clipped_max_tokens(&self, prev_input_tokens: Option<u64>) -> Option<u64> {
        let cap = self.max_output_tokens.or(self.model.max_tokens)?;
        let window = self.model.context_window?;
        let input = prev_input_tokens.unwrap_or(0);
        let headroom = window.saturating_sub(input);
        (headroom > 0).then(|| cap.min(headroom))
    }
}
