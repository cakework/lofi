mod approvals;
mod dispatch;
mod spend;
mod watches;

#[cfg(test)]
mod tests;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use lofi_code::policy::ResolvedPolicy;
use lofi_code::{BashEnv, RecallFn, ResultFn};
use lofi_error::{Error, Result};
use lofi_types::ContentBlock;

pub use approvals::{ConfirmRequest, ConfirmResponse};
use dispatch::emit;
#[cfg(test)]
pub(in crate::agent) use dispatch::upgrade_exec_image;
pub(crate) use watches::ActivityWatch;
pub(in crate::agent) use watches::ConfirmWatch;

use super::{AgentEvent, SubagentRegistry, SubagentSupport};
use crate::state;

/// Session-scoped execution state shared by native and external backends.
#[derive(Clone)]
pub struct ToolExecutor {
    pub(in crate::agent) root: PathBuf,
    pub(in crate::agent) tmp_dir: PathBuf,
    pub(in crate::agent) tmp_lease: Option<Arc<state::SessionTempDir>>,
    pub(in crate::agent) bash_env: BashEnv,
    pub(in crate::agent) shell_policy: ResolvedPolicy,
    pub(in crate::agent) truncate: lofi_types::TruncateConfig,
    pub(in crate::agent) image: lofi_types::ImageConfig,
    pub(in crate::agent) confirm_tx: Option<tokio::sync::mpsc::UnboundedSender<ConfirmRequest>>,
    pub(in crate::agent) confirm_counter: Arc<AtomicU64>,
    /// Confirmations this executor has open. A child's idle deadline pauses
    /// while one is unanswered, and its hard cap names what it waited on.
    pub(in crate::agent) confirm_watch: Arc<ConfirmWatch>,
    /// Calls this executor is running. A child's idle deadline and a CLI
    /// child's frame clock pause while one runs, because it reports nothing
    /// until it ends.
    pub(crate) activity: Arc<ActivityWatch>,
    pub(in crate::agent) auto_mode: Option<lofi_code::AutoModeFn>,
    /// Config the evaluator was opened from, with its provider resolved. A
    /// child re-attaches from it, so its requests carry the same credentials.
    pub(in crate::agent) auto_mode_config: Option<Arc<lofi_types::Config>>,
    /// Turn context the evaluator reads; see [`super::Agent::auto_mode_hints`].
    pub(in crate::agent) auto_mode_hints: lofi_code::AutoModeHintHandle,
    pub(in crate::agent) policy_override: lofi_code::policy::PolicyOverride,
    pub(in crate::agent) skills_dir: Option<PathBuf>,
    /// Config-sourced granted directories outside the workspace root, with
    /// their access levels.
    pub(in crate::agent) grants: Vec<lofi_code::GrantedDir>,
    /// Workspace memory file `lofi.memory` reports, resolved once from the
    /// workspace root, with the modes its first write applies.
    pub(in crate::agent) memory: Option<lofi_types::HostFile>,
    pub(in crate::agent) jobs: lofi_code::tools::JobRegistry,
    pub(in crate::agent) timers: lofi_code::tools::TimerRegistry,
    pub(in crate::agent) job_owner: Option<String>,
    pub(in crate::agent) exec_worker: Arc<std::sync::OnceLock<lofi_code::ExecWorker>>,
    /// `lofi.subagent` hook, `None` where subagents are unsupported (nested
    /// subagent sandboxes).
    pub(in crate::agent) subagent: Option<lofi_code::SubagentFn>,
    /// Mailbox capability of a spawned child: its own inbox plus the right to
    /// write to the parent. `None` on a session executor, which addresses
    /// runs through the registry instead.
    pub(in crate::agent) mailbox: Option<super::mailbox::MailboxScope>,
    /// Spawned subagents of this session, resumable by id. The registry lives
    /// here so a CLI backend exposes the same runs an HTTP one does.
    pub(in crate::agent) subagents: SubagentRegistry,
    /// Config and registry the subagent hook resolves model overrides with.
    /// `None` on a child, which cannot spawn its own subagents.
    pub(in crate::agent) subagent_support: Option<SubagentSupport>,
    /// `lofi.models` hook, `None` where the host has no model registry.
    pub(in crate::agent) models: Option<lofi_code::ModelsFn>,
    pub(in crate::agent) session_cache: Arc<std::sync::OnceLock<Arc<lofi_code::ToolSession>>>,
}

/// Hooks for one tool call. All callbacks belong to the owning session.
#[derive(Clone, Default)]
pub(crate) struct ToolCallContext {
    pub events: Option<crate::agent::turn_session::EventSink>,
    pub session: Option<crate::session::store::SessionCursor>,
    pub recall: Option<RecallFn>,
    pub result: Option<ResultFn>,
    pub cancel: Option<Arc<AtomicBool>>,
    pub on_job_acquired: Option<lofi_code::JobAcquireFn>,
    pub record_tool: Option<Arc<dyn Fn(ContentBlock) -> Result<()> + Send + Sync>>,
    /// The lifetime of the turn that owns this context. A tool that settles
    /// after the turn dropped must not extend the transcript. None means the
    /// caller is not a turn, so nothing is sealed.
    pub alive: Option<Arc<AtomicBool>>,
}

impl ToolExecutor {
    /// Create execution state without constructing a model provider.
    /// # Errors
    /// Returns an error when the state directory cannot be resolved.
    pub fn new(
        root: &std::path::Path,
        config: &lofi_types::Config,
        env: &[(String, String)],
    ) -> Result<Self> {
        let lease = state::create_session_tmp_dir(root)?;
        let mut exec = Self::in_memory(root);
        exec.tmp_dir = lease.path().to_path_buf();
        exec.tmp_lease = Some(Arc::new(lease));
        exec.bash_env = crate::bash_env::resolve_bash_env(&config.bash, env);
        exec.shell_policy = lofi_code::policy::defaults::resolve(&config.shell_policy);
        exec.truncate = config.truncate;
        exec.image = config.image;
        exec.grants = super::granted_dirs(&config.permissions);
        exec.memory = state::WorkspaceMemory::open(root).ok().map(Into::into);
        Ok(exec)
    }

    /// Construct execution state in memory without creating a temporary
    /// directory or opening workspace state files on disk.
    #[must_use]
    pub fn in_memory(root: impl Into<PathBuf>) -> Self {
        let config = lofi_types::Config::default();
        let ids = lofi_types::IdSpace::new();
        Self {
            root: root.into(),
            tmp_dir: PathBuf::new(),
            tmp_lease: None,
            bash_env: crate::bash_env::resolve_bash_env(&config.bash, &[]),
            shell_policy: lofi_code::policy::defaults::resolve(&config.shell_policy),
            truncate: config.truncate,
            image: config.image,
            confirm_tx: None,
            confirm_counter: Arc::new(AtomicU64::new(0)),
            confirm_watch: Arc::new(ConfirmWatch::default()),
            activity: Arc::new(ActivityWatch::default()),
            auto_mode: None,
            auto_mode_config: None,
            auto_mode_hints: super::auto_mode::hint_handle(),
            policy_override: lofi_code::policy::PolicyOverride::default(),
            skills_dir: None,
            grants: Vec::new(),
            memory: None,
            jobs: lofi_code::tools::JobRegistry::with_ids(ids.clone()),
            timers: lofi_code::tools::TimerRegistry::with_ids(ids.clone()),
            job_owner: None,
            exec_worker: Arc::new(std::sync::OnceLock::new()),
            subagent: None,
            mailbox: None,
            subagents: SubagentRegistry::with_ids(ids),
            subagent_support: None,
            models: None,
            session_cache: Arc::new(std::sync::OnceLock::new()),
        }
    }

    pub(crate) fn invalidate_session(&mut self) {
        self.session_cache = Arc::new(std::sync::OnceLock::new());
    }

    pub(super) fn tool_session(&self) -> Arc<lofi_code::ToolSession> {
        self.session_cache
            .get_or_init(|| {
                let mut session = lofi_code::ToolSession::new(
                    self.root.clone(),
                    self.tmp_dir.clone(),
                    self.bash_env.clone(),
                    self.shell_policy.clone(),
                )
                .with_skills_dir(self.skills_dir.clone())
                .with_grants(self.grants.clone())
                .with_memory(self.memory.clone());
                session.auto_mode.clone_from(&self.auto_mode);
                session.policy_override = self.policy_override.clone();
                session.truncate = self.truncate;
                session.jobs = self.jobs.clone();
                session.timers = self.timers.clone();
                session.job_owner.clone_from(&self.job_owner);
                session.models.clone_from(&self.models);
                Arc::new(session)
            })
            .clone()
    }

    /// Configure shell approval without loading unrelated model providers.
    /// # Errors
    /// Returns evaluator configuration, credential or model registry errors.
    pub async fn with_auto_mode_config(
        self,
        config: &lofi_types::Config,
        env: &[(String, String)],
    ) -> Result<Self> {
        super::auto_mode::configure(self, config, env).await
    }

    #[must_use]
    pub fn subagent_hook(&self) -> Option<lofi_code::SubagentFn> {
        self.subagent.clone()
    }

    /// Install the `lofi.subagent` hook the sandbox reports from.
    #[must_use]
    pub fn with_subagent_hook(mut self, hook: Option<lofi_code::SubagentFn>) -> Self {
        self.subagent = hook;
        self
    }

    /// Spawned subagents of this session, shared by every clone.
    #[must_use]
    pub fn subagents(&self) -> SubagentRegistry {
        self.subagents.clone()
    }

    /// The allowed-paths view the /paths dialog reads and edits.
    #[must_use]
    pub fn paths(&self) -> crate::paths::PathsHandle {
        crate::paths::PathsHandle::new(
            self.root.clone(),
            (!self.tmp_dir.as_os_str().is_empty()).then(|| self.tmp_dir.clone()),
            self.skills_dir.clone(),
            self.memory.as_ref().map(|file| file.path.clone()),
            self.grants.clone(),
        )
    }

    #[must_use]
    pub fn with_grants(mut self, grants: Vec<lofi_code::GrantedDir>) -> Self {
        self.grants = grants;
        self.invalidate_session();
        self
    }

    #[must_use]
    pub fn with_jobs(mut self, jobs: lofi_code::tools::JobRegistry) -> Self {
        self.jobs = jobs;
        self.invalidate_session();
        self
    }

    #[must_use]
    pub fn with_timers(mut self, timers: lofi_code::tools::TimerRegistry) -> Self {
        self.timers = timers;
        self.invalidate_session();
        self
    }

    #[must_use]
    pub fn with_subagents(mut self, subagents: SubagentRegistry) -> Self {
        self.subagents = subagents;
        self
    }

    #[must_use]
    pub fn with_policy_override(
        mut self,
        policy_override: lofi_code::policy::PolicyOverride,
    ) -> Self {
        self.policy_override = policy_override;
        self.invalidate_session();
        self
    }

    /// Config and registry a subagent hook resolves model overrides with.
    #[must_use]
    pub fn subagent_support(&self) -> Option<SubagentSupport> {
        self.subagent_support.clone()
    }

    /// Attach subagent support: the session's registry, its model listing and
    /// the config a child's model override resolves against.
    #[must_use]
    pub fn with_subagent_support(mut self, support: SubagentSupport) -> Self {
        self.models = Some(support.models_fn());
        self.subagent_support = Some(support);
        self.invalidate_session();
        self
    }

    #[must_use]
    pub fn has_auto_mode(&self) -> bool {
        self.auto_mode.is_some()
    }

    /// Drop the subagent hook. Used when the executor moves to a backend
    /// whose registry the UI cannot reach for cleanup. The models listing
    /// goes with it: `lofi.models` exists so a guest can pick a model for
    /// `lofi.subagent`, and leaving it behind would advertise overrides the
    /// session cannot use.
    #[must_use]
    pub fn without_subagent(mut self) -> Self {
        self.subagent = None;
        self.models = None;
        self.subagent_support = None;
        self.invalidate_session();
        self
    }

    #[must_use]
    pub fn with_confirm_tx(
        mut self,
        tx: tokio::sync::mpsc::UnboundedSender<ConfirmRequest>,
    ) -> Self {
        self.confirm_tx = Some(tx);
        self
    }

    #[must_use]
    pub fn with_skills_dir(mut self, path: Option<PathBuf>) -> Self {
        self.skills_dir = path;
        self.invalidate_session();
        self
    }

    /// Execution state for a spawned child: the session's jobs, policy and
    /// sandbox configuration stay shared, while the identity, the hook and
    /// the sandbox thread are the child's own.
    #[must_use]
    pub(crate) fn with_child_identity(
        mut self,
        owner: &str,
        mailbox: Option<super::mailbox::MailboxScope>,
        subagent: Option<lofi_code::SubagentFn>,
        models: Option<lofi_code::ModelsFn>,
    ) -> Self {
        self.job_owner = Some(owner.to_string());
        self.mailbox = mailbox;
        self.subagent = subagent;
        self.models = models;
        // A child owns its own runs and cannot spawn: a shared registry would
        // also form an Arc cycle with the entries that hold the children.
        self.subagents = SubagentRegistry::default();
        self.subagent_support = None;
        self.invalidate_session();
        self.reset_exec_worker()
    }

    /// A fresh sandbox thread, so an idle entry does not hold one: it
    /// restarts lazily on the first exec of the next resume.
    #[must_use]
    pub fn reset_exec_worker(mut self) -> Self {
        // Sharing the parent's worker deadlocks: the parent worker is busy
        // awaiting the subagent hook while the child's exec jobs queue
        // behind it.
        self.exec_worker = Arc::new(std::sync::OnceLock::new());
        self
    }

    /// Owning subagent run id, when this executor belongs to a child.
    #[must_use]
    pub fn job_owner(&self) -> Option<&str> {
        self.job_owner.as_deref()
    }

    /// Give this executor a fresh auto-mode turn context, so a child does not
    /// read the session hints its parent's evaluator captured.
    #[must_use]
    pub fn with_fresh_auto_mode_hints(mut self) -> Self {
        self.auto_mode_hints = super::auto_mode::hint_handle();
        self
    }

    /// Replace the auto-mode evaluator, or drop it.
    #[must_use]
    pub fn with_auto_mode(mut self, auto_mode: Option<lofi_code::AutoModeFn>) -> Self {
        self.auto_mode = auto_mode;
        self.invalidate_session();
        self
    }

    /// The prompt this executor's run reads. Every backend that builds a
    /// prompt asks here rather than assembling one of its own.
    #[must_use]
    pub(crate) fn system_prompt(&self) -> String {
        super::model::assemble_system_prompt(
            self.skills_dir.as_deref().and_then(std::path::Path::parent),
            &self.root,
        )
    }

    #[must_use]
    pub fn jobs(&self) -> lofi_code::tools::JobRegistry {
        self.jobs.clone()
    }

    /// Session-scoped wake-up timers shared with the owning agent.
    #[must_use]
    pub fn timers(&self) -> lofi_code::tools::TimerRegistry {
        self.timers.clone()
    }

    #[must_use]
    pub fn policy_override(&self) -> lofi_code::policy::PolicyOverride {
        self.policy_override.clone()
    }

    /// Execute through the same path as the native agent.
    /// # Errors
    /// Returns an error if the consumer disconnects or execution cannot start.
    pub(crate) async fn execute(
        &self,
        id: &str,
        input: &serde_json::Value,
        context: ToolCallContext,
    ) -> Result<ContentBlock> {
        if context
            .cancel
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return Err(Error::Cancelled);
        }
        if !emit(
            context.events.as_ref(),
            AgentEvent::ToolStart {
                id: id.into(),
                name: "exec".into(),
            },
        )
        .await
        {
            return Err(Error::Cancelled);
        }
        if let Some(record) = &context.record_tool {
            record(ContentBlock::ToolUse {
                id: id.into(),
                name: "exec".into(),
                input: crate::agent::normalize_exec_input(input),
            })?;
        }
        let result = self
            .execute_tools(&[(id, "exec", input)], &context)
            .await?
            .pop()
            .ok_or_else(|| Error::State("exec returned no result".into()))?;
        if let Some(record) = &context.record_tool {
            record(result.clone())?;
        }
        Ok(result)
    }

    pub(super) fn sandbox_worker(&self) -> &lofi_code::ExecWorker {
        self.exec_worker.get_or_init(lofi_code::ExecWorker::spawn)
    }
}
