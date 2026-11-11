#![allow(clippy::wildcard_imports)]

use super::*;

/// Confirmations one executor is waiting on, and how long their answers
/// took: a child's idle deadline pauses while one is open, and its hard cap
/// names the prompt it was waiting on.
#[derive(Default)]
pub(super) struct ConfirmWatch {
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
    pub(super) fn pending(&self) -> usize {
        lock(&self.state).open.len()
    }

    /// The oldest prompt still unanswered: the one a blocked run waited on,
    /// so answering a newer prompt first does not misname it.
    pub(super) fn command(&self) -> Option<String> {
        lock(&self.state)
            .open
            .first()
            .map(|(_, command)| command.clone())
    }

    /// How long this child has spent waiting on answers.
    pub(super) fn waited(&self) -> std::time::Duration {
        let state = lock(&self.state);
        state.waited.saturating_add(
            state
                .since
                .map_or(std::time::Duration::ZERO, |since| since.elapsed()),
        )
    }

    /// Count `command` as open until the guard drops.
    pub(super) fn open(&self, command: &str) -> OpenConfirm<'_> {
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
}

impl ConfirmWatch {
    /// Forget prompts an earlier run left open: one must not pause or name
    /// the next run. Resetting in place reaches the executor clones a
    /// backend's worker captured.
    pub(super) fn reset(&self) {
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
    pub(super) fn reset(&self) {
        let mut state = lock(&self.state);
        state.generation += 1;
        state.busy = 0;
        state.free_since = None;
    }

    /// Count the call as running until the guard drops, so a failed call
    /// still clears it.
    pub(super) fn enter(&self) -> RunningCall<'_> {
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
pub(super) struct RunningCall<'a> {
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
pub(super) struct OpenConfirm<'a> {
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

/// Session-scoped execution state shared by native and external backends.
#[derive(Clone)]
pub struct ToolExecutor {
    pub(super) root: PathBuf,
    pub(super) tmp_dir: PathBuf,
    pub(super) tmp_lease: Option<Arc<state::SessionTempDir>>,
    pub(super) bash_env: BashEnv,
    pub(super) shell_policy: ResolvedPolicy,
    pub(super) truncate: lofi_code::TruncatedCap,
    pub(super) image: lofi_types::ImageConfig,
    pub(super) confirm_tx: Option<tokio::sync::mpsc::UnboundedSender<ConfirmRequest>>,
    pub(super) confirm_counter: Arc<AtomicU64>,
    /// Confirmations this executor has open. A child's idle deadline pauses
    /// while one is unanswered, and its hard cap names what it waited on.
    pub(super) confirm_watch: Arc<ConfirmWatch>,
    /// Calls this executor is running. A child's idle deadline and a CLI
    /// child's frame clock pause while one runs, because it reports nothing
    /// until it ends.
    pub(crate) activity: Arc<ActivityWatch>,
    pub(super) auto_mode: Option<lofi_code::AutoModeFn>,
    /// Config the evaluator was opened from, with its provider resolved. A
    /// child re-attaches from it, so its requests carry the same credentials.
    pub(super) auto_mode_config: Option<Arc<lofi_types::Config>>,
    /// Turn context the evaluator reads; see [`Agent::auto_mode_hints`].
    pub(super) auto_mode_hints: lofi_code::AutoModeHintHandle,
    pub(super) policy_override: lofi_code::policy::PolicyOverride,
    pub(super) skills_dir: Option<PathBuf>,
    /// Config-sourced granted directories outside the workspace root, with
    /// their access levels.
    pub(super) grants: Vec<lofi_code::GrantedDir>,
    /// Workspace memory file `lofi.memory` reports, resolved once from the
    /// workspace root, with the modes its first write applies.
    pub(super) memory: Option<lofi_types::HostFile>,
    pub(super) jobs: lofi_code::tools::JobRegistry,
    pub(super) timers: lofi_code::tools::TimerRegistry,
    pub(super) job_owner: Option<String>,
    pub(super) exec_worker: Arc<std::sync::OnceLock<lofi_code::ExecWorker>>,
    /// `lofi.subagent` hook, `None` where subagents are unsupported (nested
    /// subagent sandboxes).
    pub(super) subagent: Option<lofi_code::SubagentFn>,
    /// Mailbox capability of a spawned child: its own inbox plus the right to
    /// write to the parent. `None` on a session executor, which addresses
    /// runs through the registry instead.
    pub(super) mailbox: Option<super::mailbox::MailboxScope>,
    /// Spawned subagents of this session, resumable by id. The registry lives
    /// here so a CLI backend exposes the same runs an HTTP one does.
    pub(super) subagents: super::SubagentRegistry,
    /// Config and registry the subagent hook resolves model overrides with.
    /// `None` on a child, which cannot spawn its own subagents.
    pub(super) subagent_support: Option<super::SubagentSupport>,
    /// `lofi.models` hook, `None` where the host has no model registry.
    pub(super) models: Option<lofi_code::ModelsFn>,
}

/// Hooks for one tool call. All callbacks belong to the owning session.
#[derive(Clone, Default)]
pub struct ToolCallContext {
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

/// Report one rejected tool call: emit the error `ToolEnd`, then record the
/// `ToolResult`. Shared by every pre-execution rejection in `execute_tools`.
async fn reject(
    call: &ToolCallContext,
    results: &mut Vec<ContentBlock>,
    id: &str,
    content: impl Into<String>,
    dispatch: std::time::Instant,
) -> Result<()> {
    let content = content.into();
    let elapsed_ms = dispatch.elapsed().as_millis() as u64;
    if !emit(
        call.events.as_ref(),
        AgentEvent::ToolEnd {
            id: id.to_string(),
            result: content.clone(),
            is_error: true,
            elapsed_ms,
        },
    )
    .await
    {
        return Err(Error::Cancelled);
    }
    results.push(ContentBlock::ToolResult {
        tool_use_id: id.to_string(),
        content,
        is_error: true,
        images: Vec::new(),
    });
    Ok(())
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
        let ids = lofi_types::IdSpace::new();
        Ok(Self {
            root: root.to_path_buf(),
            tmp_dir: lease.path().to_path_buf(),
            tmp_lease: Some(Arc::new(lease)),
            bash_env: crate::bash_env::resolve_bash_env(&config.bash, env),
            shell_policy: lofi_code::policy::defaults::resolve(&config.shell_policy),
            truncate: lofi_code::TruncatedCap {
                max_lines: config.truncate.max_lines,
                max_bytes: config.truncate.max_bytes,
            },
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
            grants: super::granted_dirs(&config.permissions),
            memory: state::WorkspaceMemory::open(root).ok().map(Into::into),
            jobs: lofi_code::tools::JobRegistry::with_ids(ids.clone()),
            timers: lofi_code::tools::TimerRegistry::with_ids(ids.clone()),
            job_owner: None,
            exec_worker: Arc::new(std::sync::OnceLock::new()),
            subagent: None,
            mailbox: None,
            subagents: super::SubagentRegistry::with_ids(ids),
            subagent_support: None,
            models: None,
        })
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
    pub fn subagents(&self) -> super::SubagentRegistry {
        self.subagents.clone()
    }

    /// The allowed-paths view the /paths dialog reads and edits.
    #[must_use]
    pub fn paths(&self) -> crate::paths::PathsHandle {
        crate::paths::PathsHandle::new(
            self.root.clone(),
            Some(self.tmp_dir.clone()),
            self.skills_dir.clone(),
            self.memory.as_ref().map(|file| file.path.clone()),
            self.grants.clone(),
        )
    }

    /// Config and registry a subagent hook resolves model overrides with.
    #[must_use]
    pub fn subagent_support(&self) -> Option<super::SubagentSupport> {
        self.subagent_support.clone()
    }

    /// Attach subagent support: the session's registry, its model listing and
    /// the config a child's model override resolves against.
    #[must_use]
    pub fn with_subagent_support(mut self, support: super::SubagentSupport) -> Self {
        self.models = Some(support.models_fn());
        self.subagent_support = Some(support);
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
        self.subagents = super::SubagentRegistry::default();
        self.subagent_support = None;
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
    pub async fn execute(
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

    fn sandbox_worker(&self) -> &lofi_code::ExecWorker {
        self.exec_worker.get_or_init(lofi_code::ExecWorker::spawn)
    }

    #[allow(clippy::too_many_lines)]
    pub(super) async fn execute_tools(
        &self,
        tool_uses: &[(&str, &str, &serde_json::Value)],
        call: &ToolCallContext,
    ) -> Result<Vec<ContentBlock>> {
        if tool_uses.is_empty() {
            return Ok(Vec::new());
        }
        // The child's idle deadline pauses for the whole run of calls: a long
        // one emits no event until it returns.
        let _running = self.activity.enter();
        // The scratch directory arrives with the first tool run, so a
        // session that never runs one leaves no directory behind.
        if let Some(lease) = &self.tmp_lease {
            lease.ensure()?;
        }
        let native_cursor = call.session.clone();
        let recall = call.recall.clone();
        let result = call.result.clone();
        let cancel = call.cancel.as_ref();
        let on_job_acquired = call.on_job_acquired.clone();
        let mut results: Vec<ContentBlock> = Vec::with_capacity(tool_uses.len());
        // Native tool events are emitted from a *sync* `on_tool_event`
        // callback inside the sandbox, so they can't `await` on the bounded
        // UI channel. `try_send` would drop events when the channel fills
        // (e.g. a `Promise.all` burst of Start/End events), leaving native
        // tiles stuck "running" even after the exec succeeds. Relay them
        // through an unbounded channel whose forwarder drains onto the
        // bounded channel with `send().await` — lossless and order-preserving.
        let (relay_tx, mut relay_rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
        let relay_dst = call.events.clone();
        let _forwarder = tokio::spawn(async move {
            while let Some(ev) = relay_rx.recv().await {
                match &relay_dst {
                    Some(dst) => {
                        if !dst.emit(ev).await {
                            break;
                        }
                    }
                    None => {
                        drop(ev);
                    }
                }
            }
        });
        for (id, name, input) in tool_uses {
            let dispatch = std::time::Instant::now();
            if *name != "exec" {
                reject(
                    call,
                    &mut results,
                    id,
                    format!("unknown tool: {name}"),
                    dispatch,
                )
                .await?;
                continue;
            }
            let (code, strings, description) = parse_exec_input(input);
            if description.is_empty() {
                reject(
                    call,
                    &mut results,
                    id,
                    "exec: missing required 'description' argument",
                    dispatch,
                )
                .await?;
                continue;
            }
            if code.is_empty() {
                reject(
                    call,
                    &mut results,
                    id,
                    "exec: missing required 'code' argument",
                    dispatch,
                )
                .await?;
                continue;
            }
            // The call's own bound: what it asked, as an absolute instant, so
            // the prompt, the compile, and every tool it starts come out of it
            // instead of adding to it. Nothing above the call bounds it.
            if input
                .get("timeoutMs")
                .filter(|asked| !asked.is_null())
                .is_some_and(|asked| asked.as_u64().is_none_or(|ms| ms == 0))
            {
                reject(
                    call,
                    &mut results,
                    id,
                    failed_result(
                        "exec: timeoutMs must be a positive integer",
                        "failed",
                        None,
                        None,
                    ),
                    dispatch,
                )
                .await?;
                continue;
            }
            let exec_deadline = exec_timeout_ms(input).and_then(|ms| {
                std::time::Instant::now().checked_add(std::time::Duration::from_millis(ms))
            });
            // The bound this call's tools run under: the call's own `timeoutMs`.
            let clock = exec_deadline.map_or(lofi_types::Deadline::NONE, lofi_types::Deadline::at);
            // The ToolInput event must land before we invoke the sandbox;
            // if the consumer is gone, stop instead of running tools (which
            // may shell out or mutate files) for a dead receiver.
            if !emit(
                call.events.as_ref(),
                AgentEvent::ToolInput {
                    id: id.to_string(),
                    code: code.clone(),
                    label: Some(description.clone()),
                },
            )
            .await
            {
                return Err(Error::Cancelled);
            }
            let native_tx = relay_tx.clone();
            let parent = id.to_string();
            let native_pending: Arc<Mutex<HashMap<u64, (String, String)>>> =
                Arc::new(Mutex::new(HashMap::new()));
            let native_records: Arc<Mutex<Vec<NativeToolRecord>>> =
                Arc::new(Mutex::new(Vec::new()));
            let on_tool_event: Arc<dyn Fn(ToolEvent) + Send + Sync> = {
                let native_pending = native_pending.clone();
                let native_records = native_records.clone();
                let retain_records = native_cursor.is_some();
                let event_parent = parent.clone();
                Arc::new(move |ev: ToolEvent| match ev {
                    ToolEvent::Start { id, name, args } => {
                        lock(&native_pending).insert(id, (name.clone(), args.clone()));
                        let _ = native_tx.send(AgentEvent::NativeToolStart {
                            parent: event_parent.clone(),
                            id,
                            name,
                            args,
                        });
                    }
                    ToolEvent::End {
                        id,
                        result,
                        is_error,
                    } => {
                        let display_result = cap_tool_result(&result);
                        if let Some((name, args)) = lock(&native_pending).remove(&id) {
                            if retain_records {
                                lock(&native_records).push(NativeToolRecord {
                                    parent: event_parent.clone(),
                                    call_id: id,
                                    name,
                                    args,
                                    result,
                                    is_error,
                                });
                            }
                        }
                        let _ = native_tx.send(AgentEvent::NativeToolEnd {
                            parent: event_parent.clone(),
                            id,
                            result: display_result,
                            is_error,
                        });
                    }
                }) as Arc<dyn Fn(ToolEvent) + Send + Sync>
            };
            let confirm: Option<lofi_code::ConfirmFn> = self.confirm_tx.as_ref().map(|tx| {
                let tx = tx.clone();
                let counter = self.confirm_counter.clone();
                let watch = self.confirm_watch.clone();
                Arc::new(move |prompt: lofi_code::ConfirmPrompt| {
                    let tx = tx.clone();
                    let counter = counter.clone();
                    let watch = watch.clone();
                    Box::pin(async move {
                        let id = counter.fetch_add(1, Ordering::SeqCst);
                        let (resp_tx, resp_rx) = oneshot::channel();
                        let command = prompt.command.clone();
                        let req = ConfirmRequest {
                            id,
                            command: prompt.command,
                            reason: prompt.reason,
                            active: prompt.active,
                            deny_at: prompt.deny_at,
                            respond: resp_tx,
                        };
                        let _open = watch.open(&command);
                        if tx.send(req).is_err() {
                            return false;
                        }
                        matches!(resp_rx.await, Ok(ConfirmResponse::Allow))
                    })
                        as std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + Sync>>
                }) as lofi_code::ConfirmFn
            });
            let approve_dir: Option<lofi_code::ApproveDirFn> = self.confirm_tx.as_ref().map(|tx| {
                let tx = tx.clone();
                let counter = self.confirm_counter.clone();
                let watch = self.confirm_watch.clone();
                let root = self.root.clone();
                let config_grants = self.grants.clone();
                // The instant this call's clock closes the prompt: the same
                // one the tool refuses an answer after, so the countdown the
                // user reads ends where the ask does.
                let deny_at = clock.instant();
                Arc::new(move |approval: lofi_code::DirApproval| {
                    let tx = tx.clone();
                    let counter = counter.clone();
                    let watch = watch.clone();
                    let root = root.clone();
                    let config_grants = config_grants.clone();
                    Box::pin(async move {
                        // The prompt stays open for the whole approval: the
                        // stored-grant reads it does first are instantaneous
                        // next to the wait for an answer.
                        let _open = watch.open(&approval.dir.display().to_string());
                        approve_outside_dir(&root, approval, &config_grants, &tx, &counter, deny_at)
                            .await
                    })
                        as std::pin::Pin<
                            Box<
                                dyn std::future::Future<Output = Option<lofi_code::GrantedDir>>
                                    + Send
                                    + Sync,
                            >,
                        >
                }) as lofi_code::ApproveDirFn
            });
            // Children this call starts report their spend in the call's own
            // result, so a fan-out reads its cost from the value it already
            // gets back. The hook stays the session's; this wrapper records
            // what the call started, and asks that same hook to fold the runs
            // it spawned.
            let child_spend = Arc::new(Mutex::new(ChildSpend::default()));
            // The wall clock this call is under, handed to the tools it
            // starts: a command is clamped to what the call has left instead
            // of to a constant that has to stay below it by hand. A call with
            // no `timeoutMs` arms nothing, and its tools keep their own timeouts.
            // This is not the guest's CPU budget below: awaited time costs no
            // CPU, so the two clocks bound different things.
            let subagent: Option<lofi_code::SubagentFn> = self.subagent.clone().map(|hook| {
                let spend = child_spend.clone();
                let wait_clock = clock;
                Arc::new(move |op: lofi_code::SubagentOp, args: serde_json::Value| {
                    let hook = hook.clone();
                    let spend = spend.clone();
                    Box::pin(async move {
                        let (args, cut) = clamp_child(op, args, wait_clock);
                        // A resume installs the new run's totals over the
                        // entry's, so what the background run this call
                        // started billed is read before the hook replaces it.
                        let replaced = match op {
                            lofi_code::SubagentOp::Run => {
                                let id = args.get("id").and_then(serde_json::Value::as_str);
                                let ours = id.filter(|id| {
                                    lock(&spend).spawned.iter().any(|kept| kept == id)
                                });
                                match ours {
                                    Some(id) => Some(
                                        hook(
                                            lofi_code::SubagentOp::Status,
                                            serde_json::json!({ "id": id }),
                                        )
                                        .await,
                                    ),
                                    None => None,
                                }
                            }
                            _ => None,
                        };
                        let mut out = hook(op, args).await;
                        // The allowance this child asked for, when the call
                        // cut it: a parent can tell a child that ran out from
                        // one that was cut short.
                        if let Some(asked) = cut {
                            if let Some(map) = out.as_object_mut() {
                                map.insert(
                                    "requestedTimeoutMs".to_string(),
                                    serde_json::json!(asked),
                                );
                            }
                        }
                        let mut spend = lock(&spend);
                        match op {
                            // Only an admitted spawn returns an id: a
                            // rejection or a busy resume echoes the id it
                            // was asked for, and started nothing to fold.
                            lofi_code::SubagentOp::Spawn
                                if out.get("ok").and_then(serde_json::Value::as_bool)
                                    == Some(true) =>
                            {
                                if let Some(id) = out.get("id").and_then(serde_json::Value::as_str)
                                {
                                    spend.spawned.push(id.to_string());
                                    spend.claimed.push(id.to_string());
                                }
                            }
                            // A run body names its model and a rejection body
                            // does not. Every run body that reports usage was
                            // billed, completed or not.
                            lofi_code::SubagentOp::Run if out.get("model").is_some() => {
                                // The resume left the entry's outcome naming
                                // only this run, so the run it replaced is
                                // priced from the snapshot and stops being a
                                // child to fold.
                                if let Some(replaced) = &replaced {
                                    spend.add_run(replaced);
                                }
                                if let Some(id) = out.get("id").and_then(serde_json::Value::as_str)
                                {
                                    spend.forget(id);
                                }
                                spend.add_run(&out);
                            }
                            _ => {}
                        }
                        out
                    })
                        as std::pin::Pin<
                            Box<dyn std::future::Future<Output = serde_json::Value> + Send>,
                        >
                }) as lofi_code::SubagentFn
            });
            let exec_ctx = ExecCtx {
                root: self.root.clone(),
                tmp_dir: self.tmp_dir.clone(),
                strings,
                recall: recall.clone(),
                result: result.clone(),
                on_tool_event: Some(on_tool_event),
                bash_env: self.bash_env.clone(),
                shell_policy: self.shell_policy.clone(),
                confirm,
                approve_dir,
                grants: self.grants.clone(),
                auto_mode: self.auto_mode.clone(),
                // The stated description travels with this call's context,
                // so the gate judges this call's commands with this call's
                // intent however the call's future is dropped or queued.
                intent: Some(description.clone()),
                // A prompt must not outlive the run that raised it, so the
                // wait is cut to what the call has left. A spent clock leaves
                // the shortest wait there is rather than a disabled
                // auto-deny.
                approval_timeout: self
                    .auto_mode_config
                    .as_ref()
                    .and_then(|config| config.shell_policy.auto_mode.as_ref())
                    .and_then(lofi_types::AutoModeConfig::approval_timeout)
                    .map(|wait| {
                        clock
                            .remaining(std::time::Instant::now())
                            .map_or(wait, |left| {
                                wait.min(left).max(std::time::Duration::from_millis(1))
                            })
                    }),
                policy_override: self.policy_override.clone(),
                skills_dir: self.skills_dir.clone(),
                memory: self.memory.clone(),
                truncate: self.truncate,
                jobs: self.jobs.clone(),
                timers: self.timers.clone(),
                job_owner: self.job_owner.clone(),
                on_job_acquired: on_job_acquired.clone(),
                subagent: subagent.clone(),
                models: self.models.clone(),
            };
            let outcome = self
                .sandbox_worker()
                .exec(
                    &code,
                    exec_ctx,
                    ExecOptions {
                        // The guest's own allowance stays as it was: it counts
                        // CPU, so raising it to a long run cap would let a
                        // spin loop run for hours. The wall budget is the
                        // separate bound the caller named.
                        timeout: lofi_code::DEFAULT_GUEST_TIMEOUT,
                        deadline: exec_deadline,
                        max_value_bytes: MAX_EXEC_RESULT_BYTES,
                        cancel: cancel.cloned(),
                    },
                )
                .await;
            // Fold the runs this call spawned through the session hook; a
            // background child that is still running reports what it spent so
            // far. The runs it awaited carried their own spend already.
            let spend = lock(&child_spend).clone();
            let subagent_usage = match &self.subagent {
                Some(hook) if !spend.is_empty() => {
                    let folded = if spend.spawned.is_empty() {
                        None
                    } else {
                        Some(
                            hook(
                                lofi_code::SubagentOp::Usage,
                                serde_json::json!({ "ids": spend.spawned.clone() }),
                            )
                            .await,
                        )
                    };
                    Some(spend.record(folded.as_ref()))
                }
                _ => None,
            };
            // The registry drops the records this call priced, so a long
            // session does not keep spend nobody folds again. It runs even
            // when the fold above had nothing to add: the claim outlives it.
            if let Some(hook) = &self.subagent {
                if !spend.claimed.is_empty() {
                    hook(
                        lofi_code::SubagentOp::Release,
                        serde_json::json!({ "ids": spend.claimed.clone() }),
                    )
                    .await;
                }
            }
            // A tool may return a tagged image (`read` on an image file).
            // Upgrade it to a `ToolResultImage` carried on this result block,
            // so the model sees the image in the same round as the result on
            // every provider. The heavyweight base64 is stripped out of the
            // text payload, leaving a compact `bytes` count in the transcript.
            let (content, is_error, result_images) = match outcome {
                Ok(r) => {
                    let has_value = r.has_returned_value();
                    let mut value = r.value;
                    match upgrade_exec_image(&mut value, r.image, &self.image) {
                        Ok(img) => {
                            let is_error = !r.errors.is_empty();
                            let mut output = serde_json::Map::new();
                            output.insert("ok".into(), serde_json::Value::Bool(!is_error));
                            // One vocabulary with a run, a job and a bash call.
                            output.insert(
                                "status".into(),
                                serde_json::Value::String(
                                    if is_error { "failed" } else { "completed" }.into(),
                                ),
                            );
                            if is_error {
                                output.insert("errors".into(), serde_json::Value::Array(r.errors));
                            }
                            if has_value {
                                output.insert("value".into(), value);
                            }
                            if let Some(usage) = &subagent_usage {
                                output.insert("subagents".into(), usage.clone());
                            }
                            if !r.logs.is_empty() {
                                output.insert("logs".into(), serde_json::Value::String(r.logs));
                            }
                            if r.value_truncated {
                                output.insert("truncated".into(), serde_json::Value::Bool(true));
                            }
                            (cap_exec_envelope(output), is_error, img)
                        }
                        Err(error) => (
                            cap_exec_result(&failed_result(
                                &error.to_string(),
                                error.status(),
                                error.logs(),
                                subagent_usage.as_ref(),
                            )),
                            true,
                            None,
                        ),
                    }
                }
                Err(e) => (
                    cap_exec_result(&failed_result(
                        &e.to_string(),
                        e.status(),
                        e.logs(),
                        subagent_usage.as_ref(),
                    )),
                    true,
                    None,
                ),
            };
            if is_error {
                // A rejected guest promise can short-circuit concurrent native
                // calls before their futures emit `ToolEvent::End`. Close every
                // still-pending row explicitly so the live UI cannot leave a
                // spinner behind after the parent exec has already failed.
                // These synthetic cancellations are also persisted, making
                // resume reproduce the settled state.
                let pending = {
                    let mut pending = lock(&native_pending);
                    pending.drain().collect::<Vec<_>>()
                };
                for (call_id, (name, args)) in pending {
                    let result = "cancelled because parent exec failed".to_string();
                    if native_cursor.is_some() {
                        lock(&native_records).push(NativeToolRecord {
                            parent: parent.clone(),
                            call_id,
                            name,
                            args,
                            result: result.clone(),
                            is_error: true,
                        });
                    }
                    let _ = relay_tx.send(AgentEvent::NativeToolEnd {
                        parent: parent.clone(),
                        id: call_id,
                        result,
                        is_error: true,
                    });
                }
            }
            // A tool that outlives its turn may not extend a transcript no turn
            // will settle.
            let live = call
                .alive
                .as_ref()
                .is_none_or(|alive| alive.load(std::sync::atomic::Ordering::Acquire));
            if let (Some(cursor), true) = (native_cursor.as_ref(), live) {
                let records = std::mem::take(&mut *lock(&native_records));
                if let Err(error) =
                    cursor.record(crate::session::recorder::SessionRecord::NativeTools { records })
                {
                    let _ = relay_tx.send(AgentEvent::Notice(format!(
                        "transcript write failed for native tool result: {error}"
                    )));
                }
            }
            let elapsed_ms = dispatch.elapsed().as_millis() as u64;
            if !emit(
                call.events.as_ref(),
                AgentEvent::ToolEnd {
                    id: id.to_string(),
                    result: content.clone(),
                    is_error,
                    elapsed_ms,
                },
            )
            .await
            {
                return Err(Error::Cancelled);
            }
            results.push(ContentBlock::ToolResult {
                tool_use_id: id.to_string(),
                content,
                is_error,
                images: result_images.into_iter().collect(),
            });
        }
        Ok(results)
    }
}

/// Publish one tool event through the turn's sink.
///
/// The sink folds the event into the accounting the transcript summary is
/// built from, so a tool's elapsed time survives a resume. False when the
/// consumer is gone.
async fn emit(events: Option<&crate::agent::turn_session::EventSink>, event: AgentEvent) -> bool {
    match events {
        Some(events) => events.emit(event).await,
        None => true,
    }
}

/// Normalize an image returned out of band by the exec sandbox. `QuickJS` carries
/// only a small token, so source image bytes never become a base64 guest string
/// or a second host JSON allocation. The durable result keeps only a byte count.
pub(super) fn upgrade_exec_image(
    value: &mut serde_json::Value,
    image: Option<lofi_code::ExecImage>,
    config: &lofi_types::ImageConfig,
) -> Result<Option<lofi_types::ToolResultImage>> {
    use std::io::Seek as _;

    let Some(mut image) = image else {
        return Ok(None);
    };
    image.file.rewind()?;
    let reader = std::io::BufReader::new(image.file);
    let (bytes, media_type) = crate::image::normalize_reader(reader, config)?;
    if let Some(obj) = value.as_object_mut() {
        obj.insert("bytes".to_string(), serde_json::json!(bytes.len()));
        obj.insert("media_type".to_string(), serde_json::json!(&media_type));
    }
    Ok(Some(lofi_types::ToolResultImage { bytes, media_type }))
}

/// Clears a directory prompt's `active` flag when the ask ends. The UI drops
/// a prompt only while that flag stays true, so an exec call its deadline drops
/// mid-ask must still close its modal.
struct ActivePrompt(Arc<AtomicBool>);

impl Drop for ActivePrompt {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Relaxed);
    }
}

/// Asks the user to grant one directory tree for use outside the workspace
/// root, and remembers the grant in the per-workspace state file so later
/// sessions do not ask again. A remembered grant that already allows the
/// request answers at once. Returns the granted directory and its access;
/// a refused prompt, or a closed confirmation channel, keeps the path
/// rejected.
async fn approve_outside_dir(
    root: &std::path::Path,
    approval: lofi_code::DirApproval,
    config_grants: &[lofi_code::GrantedDir],
    tx: &tokio::sync::mpsc::UnboundedSender<ConfirmRequest>,
    counter: &Arc<AtomicU64>,
    deny_at: Option<std::time::Instant>,
) -> Option<lofi_code::GrantedDir> {
    // A grant that already covers the requested directory answers at once,
    // whatever its width: the prompt's unit is the project root, and a
    // remembered grant for the directory itself still allows the path.
    let requested_dir = approval
        .narrow
        .clone()
        .unwrap_or_else(|| approval.dir.clone());
    // A state file that cannot be opened only costs the prompt: the grant
    // is then not remembered for the next session.
    let state = state::WorkspaceState::open(root).ok();
    if let Some(grant) = state
        .as_ref()
        .and_then(|s| s.grant_allowing(&requested_dir, approval.requested))
    {
        return Some(lofi_code::GrantedDir {
            dir: grant.dir,
            access: grant.access,
        });
    }
    // A configured grant that already allows the request answers at once.
    if let Some(grant) = config_grants
        .iter()
        .filter(|g| requested_dir.starts_with(&g.dir))
        .max_by_key(|g| g.access)
        .filter(|g| g.access.allows(approval.requested))
    {
        return Some(grant.clone());
    }
    // The strongest covering grant, remembered or configured, names what a
    // widening prompt starts from. Its directory travels with the access, so
    // the prompt describes only the tree the grant actually holds.
    let current = state
        .as_ref()
        .and_then(|s| s.strongest_covering(&requested_dir))
        .map(|grant| lofi_code::GrantedDir {
            dir: grant.dir,
            access: grant.access,
        })
        .or_else(|| {
            config_grants
                .iter()
                .filter(|g| requested_dir.starts_with(&g.dir))
                .max_by_key(|g| g.access)
                .cloned()
        });
    let id = counter.fetch_add(1, Ordering::SeqCst);
    let (resp_tx, resp_rx) = oneshot::channel();
    let active = Arc::new(AtomicBool::new(true));
    let _closes = ActivePrompt(active.clone());
    let req = ConfirmRequest {
        id,
        command: approval.dir.display().to_string(),
        reason: Arc::new(Mutex::new(lofi_code::ConfirmReason::OutsideRoot {
            dir: approval.dir.display().to_string(),
            narrow: approval
                .narrow
                .as_ref()
                .map(|dir| dir.display().to_string()),
            marker: approval.marker,
            requested: approval.requested,
            current,
        })),
        active,
        deny_at: Arc::new(Mutex::new(deny_at)),
        respond: resp_tx,
    };
    if tx.send(req).is_err() {
        return None;
    }
    let ConfirmResponse::Grant { access, narrow } = resp_rx.await.ok()? else {
        return None;
    };
    // A grant that cannot satisfy the request is a refusal.
    if !access.allows(approval.requested) {
        return None;
    }
    let dir = if narrow {
        approval.narrow.unwrap_or(approval.dir)
    } else {
        approval.dir
    };
    // An answer that lands after the call's clock closed the prompt is
    // reported but not kept: the call refuses the work it was asked for, so
    // the access it asked for must not outlive it.
    let late = deny_at.is_some_and(|end| std::time::Instant::now() >= end);
    // The grant is reused only by this call's retry; a failed save means
    // the next request asks again.
    if let Some(state) = state.as_ref().filter(|_| !late) {
        let _ = state.remember_grant(&dir, access);
    }
    Some(lofi_code::GrantedDir { dir, access })
}

/// Child spend of one exec call: the runs it spawned, which the session
/// hook folds at the end, and the spend of the foreground runs it awaited,
/// which their own results carried.
#[derive(Clone, Default)]
struct ChildSpend {
    spawned: Vec<String>,
    /// Every id this call started, including the ones a resume priced from
    /// their own body: the call releases all of them when it ends.
    claimed: Vec<String>,
    awaited: super::subagent::Rollup,
}

impl ChildSpend {
    /// Take one run body's usage. A resume installs the new run's totals over
    /// the entry's, so a run's body is where its own spend is read from.
    fn add_run(&mut self, body: &serde_json::Value) {
        if let Some(usage) = body.get("usage") {
            self.awaited.add_usage(usage);
        }
    }

    /// Drop a spawned run from the fold: its spend is priced here instead.
    fn forget(&mut self, id: &str) {
        self.spawned.retain(|kept| kept != id);
    }

    fn is_empty(&self) -> bool {
        self.spawned.is_empty() && self.awaited.is_empty()
    }

    /// The result's `subagents` record: the session's fold of the spawned
    /// runs plus the runs this call awaited itself.
    fn record(&self, folded: Option<&serde_json::Value>) -> serde_json::Value {
        let mut total = self.awaited.clone();
        if let Some(folded) = folded {
            total.merge(folded);
        }
        total.json()
    }
}

/// A bound ending `left` from now, as a call's own `timeout_ms` states it.
#[cfg(test)]
fn clock(left: std::time::Duration) -> lofi_types::Deadline {
    lofi_types::Deadline::at(std::time::Instant::now() + left)
}

fn clamp_child(
    op: lofi_code::SubagentOp,
    mut args: serde_json::Value,
    clock: lofi_types::Deadline,
) -> (serde_json::Value, Option<u64>) {
    // A wait or a mailbox receive must not block past the call that awaits
    // them, and a foreground child must not outlive it either, so each takes
    // what is left. A spawned run outlives the call that started it, so its
    // own timeout stands as given and one that sets none stays unbounded: it
    // is left alone below. Each kind names its bound with its own key.
    let (key, child, fill) = match op {
        lofi_code::SubagentOp::Run => ("timeout_ms", true, true),
        lofi_code::SubagentOp::Wait => ("timeout_ms", false, true),
        // A wait that names nothing waits for the run; a receive that names
        // nothing returns at once, so its absent bound stays absent.
        lofi_code::SubagentOp::MailboxReceive => ("wait_ms", false, false),
        _ => return (args, None),
    };
    let Some(left) = clock.remaining(std::time::Instant::now()) else {
        return (args, None);
    };
    let asked = match args.get(key) {
        None | Some(serde_json::Value::Null) => None,
        Some(asked) => match asked.as_u64() {
            Some(asked) => Some(asked),
            // A value that is not a positive integer is the hook's to report.
            None => return (args, None),
        },
    };
    if asked.is_none() && !fill {
        return (args, None);
    }
    // A spent clock still asks the hook: a wait resolves at once on a run
    // that has ended, and a child starts and expires rather than hanging.
    // Zero would be read as an invalid argument.
    // Rounded up, like every bound taken from the clock: a wait that ends a
    // millisecond before the wall reads as its own outcome instead of the
    // budget that cut it.
    let left = u64::try_from(left.as_nanos().div_ceil(1_000_000))
        .unwrap_or(u64::MAX)
        .max(1);
    let used = asked.map_or(left, |asked| asked.min(left));
    // A wait returns the run's own shape, so the cap it was cut to is not the
    // wait's to report: the run it names already said what it was given.
    let cut = if child {
        asked.filter(|asked| *asked > left)
    } else {
        None
    };
    if let Some(map) = args.as_object_mut() {
        map.insert(key.to_string(), serde_json::json!(used));
        // What the caller asked for, so the run reports the allowance it
        // was cut from and not only what it was left with.
        if let Some(asked) = cut {
            map.insert("requested_timeout_ms".to_string(), serde_json::json!(asked));
        }
    }
    (args, cut)
}

/// The failed body of an exec result, in the shape a successful call uses: a
/// program that threw after its children spent money still reports that
/// spend, and `status` names what stopped it in the vocabulary runs, jobs and
/// bash share.
fn failed_result(
    message: &str,
    status: &str,
    logs: Option<&str>,
    subagents: Option<&serde_json::Value>,
) -> String {
    // The thrown message and the logs are the unbounded fields, and escaping
    // can outgrow the limit on its own, so the body is measured as it is
    // built: a cut body would lose the keys that sort after them, `status`
    // and `subagents` among them.
    let mut keep_message = message.len();
    let mut keep_logs = logs.map_or(0, str::len);
    loop {
        let message_end = message.floor_char_boundary(keep_message);
        let mut body = serde_json::json!({
            "ok": false,
            "status": status,
            "errors": [&message[..message_end]],
        });
        if let Some(usage) = subagents {
            body["subagents"] = usage.clone();
        }
        if let Some(logs) = logs {
            let logs_end = logs.floor_char_boundary(keep_logs);
            body["logs"] = serde_json::Value::String(logs[..logs_end].to_string());
        }
        // A cut field says so, the way a cut envelope does: the part that was
        // dropped is the most recent progress, and a reader cannot tell a
        // short log from a trimmed one.
        if message_end < message.len() || logs.is_some_and(|logs| keep_logs < logs.len()) {
            body["truncated"] = serde_json::Value::Bool(true);
        }
        let text = body.to_string();
        if text.len() <= MAX_EXEC_RESULT_BYTES || (message_end == 0 && keep_logs == 0) {
            return text;
        }
        // Halve whichever field is taking the room, so a large message does
        // not cost the logs and the other way around.
        if keep_logs > 0 && keep_logs >= keep_message {
            keep_logs /= 2;
        } else {
            keep_message /= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// The guest's own return value escapes too: a value that fits the raw
    /// budget can outgrow the limit once serialized, and it used to pass
    /// through uncut.
    #[test]
    fn an_oversized_exec_value_is_cut_and_says_so() {
        let mut output = serde_json::Map::new();
        output.insert("ok".into(), true.into());
        output.insert("status".into(), "completed".into());
        output.insert(
            "value".into(),
            "\0".repeat(MAX_EXEC_RESULT_BYTES / 2).into(),
        );

        let text = cap_exec_envelope(output);

        assert!(text.len() <= MAX_EXEC_RESULT_BYTES, "{} bytes", text.len());
        let value: serde_json::Value =
            serde_json::from_str(&text).expect("a trimmed envelope stays JSON");
        assert_eq!(value["status"], "completed");
        assert_eq!(value["truncated"], true, "a cut body has to say so");
    }

    /// An escaped object key is payload too: it gets cut rather than the
    /// status giving up its room for it.
    #[test]
    fn an_oversized_object_key_is_cut_before_the_status() {
        let mut payload = serde_json::Map::new();
        payload.insert("\0".repeat(MAX_EXEC_RESULT_BYTES / 2), 1.into());
        let mut output = serde_json::Map::new();
        output.insert("ok".into(), true.into());
        output.insert("status".into(), "completed".into());
        output.insert("value".into(), serde_json::Value::Object(payload));

        let text = cap_exec_envelope(output);

        assert!(text.len() <= MAX_EXEC_RESULT_BYTES, "{} bytes", text.len());
        let value: serde_json::Value =
            serde_json::from_str(&text).expect("a trimmed envelope stays JSON");
        assert_eq!(value["status"], "completed", "the status keeps its word");
        assert_eq!(value["truncated"], true, "a cut body has to say so");
    }

    /// Many short strings outgrow the limit together, each escaping to six
    /// times its length. Trimming them one per pass would serialize the whole
    /// envelope once per string, so the top half-length band is cut in one
    /// pass.
    #[test]
    fn many_short_escaped_strings_are_trimmed_together() {
        let mut output = serde_json::Map::new();
        output.insert("ok".into(), true.into());
        output.insert("status".into(), "completed".into());
        output.insert(
            "value".into(),
            serde_json::Value::Array(
                (0..9000)
                    .map(|_| serde_json::Value::String("\0".repeat(20)))
                    .collect(),
            ),
        );

        let started = std::time::Instant::now();
        let text = cap_exec_envelope(output);

        assert!(text.len() <= MAX_EXEC_RESULT_BYTES, "{} bytes", text.len());
        let value: serde_json::Value =
            serde_json::from_str(&text).expect("a trimmed envelope stays JSON");
        assert_eq!(value["status"], "completed");
        assert_eq!(value["truncated"], true, "a cut body has to say so");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "trimming took {:?}",
            started.elapsed()
        );
    }

    /// The logs escape when serialized, so text that fits the sandbox's raw
    /// budget can outgrow the envelope. Trimming must keep the JSON and the
    /// keys that follow the logs, `status` among them.
    #[test]
    fn an_oversized_exec_envelope_keeps_its_status() {
        let mut output = serde_json::Map::new();
        output.insert("ok".into(), false.into());
        output.insert("status".into(), "failed".into());
        output.insert("logs".into(), "\n".repeat(MAX_EXEC_RESULT_BYTES / 2).into());

        let text = cap_exec_envelope(output);

        assert!(text.len() <= MAX_EXEC_RESULT_BYTES, "{} bytes", text.len());
        let value: serde_json::Value =
            serde_json::from_str(&text).expect("a trimmed envelope stays JSON");
        assert_eq!(value["status"], "failed");
    }

    fn state_env() -> tempfile::TempDir {
        let state_home = tempfile::tempdir().unwrap();
        std::env::set_var("LOFI_STATE_HOME", state_home.path());
        state_home
    }

    #[tokio::test]
    async fn a_dropped_directory_ask_closes_its_prompt() {
        let ws = tempfile::tempdir().unwrap();
        let counter = Arc::new(AtomicU64::new(0));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
        let mut ask = Box::pin(approve_outside_dir(
            ws.path(),
            dir_approval("/data/dir", lofi_types::Access::Read),
            &[],
            &tx,
            &counter,
            None,
        ));
        let req = tokio::select! {
            req = rx.recv() => req.unwrap(),
            answered = &mut ask => panic!("the ask answered itself: {answered:?}"),
        };
        assert!(req.active.load(Ordering::Relaxed));
        drop(ask);
        assert!(
            !req.active.load(Ordering::Relaxed),
            "a dropped ask must close its modal"
        );
    }

    fn dir_approval(dir: &str, requested: lofi_types::Access) -> lofi_code::DirApproval {
        lofi_code::DirApproval {
            dir: PathBuf::from(dir),
            narrow: None,
            marker: None,
            requested,
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn grants_are_remembered_across_sessions() {
        let _lock = state::STATE_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _state_home = state_env();
        let ws = tempfile::tempdir().unwrap();
        let counter = Arc::new(AtomicU64::new(0));
        let write = lofi_types::Access::Write;

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
        let responder = tokio::spawn(async move {
            let req = rx.recv().await.unwrap();
            assert_eq!(req.command, "/data/dir");
            let reason = req
                .reason
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(
                matches!(*reason, lofi_code::ConfirmReason::OutsideRoot { requested, current: None, .. } if requested == write)
            );
            req.respond
                .send(ConfirmResponse::Grant {
                    access: write,
                    narrow: false,
                })
                .unwrap();
        });
        let granted = approve_outside_dir(
            ws.path(),
            dir_approval("/data/dir", write),
            &[],
            &tx,
            &counter,
            None,
        )
        .await;
        responder.await.unwrap();
        assert_eq!(
            granted,
            Some(lofi_code::GrantedDir {
                dir: PathBuf::from("/data/dir"),
                access: write
            })
        );

        // A later session reads the state file: the call returns at once and
        // no prompt opens on the channel.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
        let second = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            approve_outside_dir(
                ws.path(),
                dir_approval("/data/dir/sub", write),
                &[],
                &tx,
                &counter,
                None,
            ),
        )
        .await
        .expect("remembered grant must not wait for a prompt");
        assert_eq!(
            second,
            Some(lofi_code::GrantedDir {
                dir: PathBuf::from("/data/dir"),
                access: write
            })
        );
        assert!(rx.try_recv().is_err());

        std::env::remove_var("LOFI_STATE_HOME");
    }

    fn project_approval(root: &str, narrow: Option<&str>) -> lofi_code::DirApproval {
        lofi_code::DirApproval {
            dir: PathBuf::from(root),
            narrow: narrow.map(PathBuf::from),
            marker: Some(".jj"),
            requested: lofi_types::Access::Write,
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn a_project_grant_covers_a_sibling_without_asking() {
        // The prompt offers the project root; answering it once must cover
        // every sibling directory, not only the one that asked.
        let _lock = state::STATE_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _state_home = state_env();
        let ws = tempfile::tempdir().unwrap();
        let counter = Arc::new(AtomicU64::new(0));
        let write = lofi_types::Access::Write;

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
        let responder = tokio::spawn(async move {
            let req = rx.recv().await.unwrap();
            assert_eq!(req.command, "/data/repo");
            let reason = req
                .reason
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(
                matches!(&*reason, lofi_code::ConfirmReason::OutsideRoot { narrow: Some(narrow), marker: Some(".jj"), .. } if narrow == "/data/repo/src"),
                "the prompt names both units and the marker"
            );
            req.respond
                .send(ConfirmResponse::Grant {
                    access: write,
                    narrow: false,
                })
                .unwrap();
        });
        let granted = approve_outside_dir(
            ws.path(),
            project_approval("/data/repo", Some("/data/repo/src")),
            &[],
            &tx,
            &counter,
            None,
        )
        .await;
        responder.await.unwrap();
        assert_eq!(
            granted,
            Some(lofi_code::GrantedDir {
                dir: PathBuf::from("/data/repo"),
                access: write
            })
        );

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
        let sibling = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            approve_outside_dir(
                ws.path(),
                project_approval("/data/repo", Some("/data/repo/lib")),
                &[],
                &tx,
                &counter,
                None,
            ),
        )
        .await
        .expect("a project grant must not ask again for a sibling");
        assert_eq!(
            sibling,
            Some(lofi_code::GrantedDir {
                dir: PathBuf::from("/data/repo"),
                access: write
            })
        );
        assert!(rx.try_recv().is_err());

        std::env::remove_var("LOFI_STATE_HOME");
    }

    /// An answer that lands after the call's clock closed the prompt is
    /// reported so the tool can refuse the work, but the access it asked for
    /// is not kept past the call that asked.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn a_grant_answered_after_the_wall_is_not_remembered() {
        let _lock = state::STATE_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _state_home = state_env();
        let ws = tempfile::tempdir().unwrap();
        let counter = Arc::new(AtomicU64::new(0));
        let write = lofi_types::Access::Write;
        let past = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_millis(1))
            .unwrap_or_else(std::time::Instant::now);

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
        let responder = tokio::spawn(async move {
            let req = rx.recv().await.unwrap();
            req.respond
                .send(ConfirmResponse::Grant {
                    access: write,
                    narrow: false,
                })
                .unwrap();
        });
        let granted = approve_outside_dir(
            ws.path(),
            project_approval("/data/repo", None),
            &[],
            &tx,
            &counter,
            Some(past),
        )
        .await;
        responder.await.unwrap();
        assert!(
            granted.is_some(),
            "the answer reaches the tool, which refuses"
        );

        // Nothing was kept: the next request asks again.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
        let asked = tokio::time::timeout(
            std::time::Duration::from_millis(200),
            approve_outside_dir(
                ws.path(),
                project_approval("/data/repo", None),
                &[],
                &tx,
                &counter,
                None,
            ),
        )
        .await;
        assert!(asked.is_err(), "a late grant is not remembered");
        assert!(rx.try_recv().is_ok(), "the next request asks");

        std::env::remove_var("LOFI_STATE_HOME");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn a_narrow_grant_remembers_the_requested_directory() {
        let _lock = state::STATE_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _state_home = state_env();
        let ws = tempfile::tempdir().unwrap();
        let counter = Arc::new(AtomicU64::new(0));
        let write = lofi_types::Access::Write;

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
        let responder = tokio::spawn(async move {
            let req = rx.recv().await.unwrap();
            req.respond
                .send(ConfirmResponse::Grant {
                    access: write,
                    narrow: true,
                })
                .unwrap();
        });
        let granted = approve_outside_dir(
            ws.path(),
            project_approval("/data/repo", Some("/data/repo/src")),
            &[],
            &tx,
            &counter,
            None,
        )
        .await;
        responder.await.unwrap();
        assert_eq!(
            granted,
            Some(lofi_code::GrantedDir {
                dir: PathBuf::from("/data/repo/src"),
                access: write
            })
        );

        // The narrower grant answers inside itself and asks again beside it.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
        let inside = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            approve_outside_dir(
                ws.path(),
                project_approval("/data/repo", Some("/data/repo/src/deep")),
                &[],
                &tx,
                &counter,
                None,
            ),
        )
        .await
        .expect("a remembered grant covers its own subdirectory");
        assert_eq!(
            inside,
            Some(lofi_code::GrantedDir {
                dir: PathBuf::from("/data/repo/src"),
                access: write
            })
        );
        assert!(rx.try_recv().is_err());

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
        let responder = tokio::spawn(async move {
            let req = rx.recv().await.unwrap();
            req.respond
                .send(ConfirmResponse::Grant {
                    access: write,
                    narrow: false,
                })
                .unwrap();
        });
        let sibling = approve_outside_dir(
            ws.path(),
            project_approval("/data/repo", Some("/data/repo/lib")),
            &[],
            &tx,
            &counter,
            None,
        )
        .await;
        responder.await.unwrap();
        assert_eq!(
            sibling,
            Some(lofi_code::GrantedDir {
                dir: PathBuf::from("/data/repo"),
                access: write
            }),
            "a sibling is not covered by the narrower grant"
        );

        std::env::remove_var("LOFI_STATE_HOME");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn read_request_under_readonly_grant_never_prompts() {
        let _lock = state::STATE_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _state_home = state_env();
        let ws = tempfile::tempdir().unwrap();
        let counter = Arc::new(AtomicU64::new(0));

        // A config read grant answers at once even though the workspace
        // state remembers nothing.
        let config = vec![lofi_code::GrantedDir {
            dir: PathBuf::from("/data/dir"),
            access: lofi_types::Access::Read,
        }];
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
        let hit = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            approve_outside_dir(
                ws.path(),
                dir_approval("/data/dir/sub", lofi_types::Access::Read),
                &config,
                &tx,
                &counter,
                None,
            ),
        )
        .await
        .expect("config grant must not wait for a prompt");
        assert_eq!(
            hit,
            Some(lofi_code::GrantedDir {
                dir: PathBuf::from("/data/dir"),
                access: lofi_types::Access::Read
            })
        );
        assert!(rx.try_recv().is_err());

        // A write under the same read-only grant offers the widening prompt.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
        let responder = tokio::spawn(async move {
            let req = rx.recv().await.unwrap();
            let reason = req
                .reason
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(
                matches!(&*reason, lofi_code::ConfirmReason::OutsideRoot { requested, current: Some(c), .. } if *requested == lofi_types::Access::Write && c.access == lofi_types::Access::Read && c.dir == std::path::Path::new("/data/dir"))
            );
            req.respond
                .send(ConfirmResponse::Grant {
                    access: lofi_types::Access::Write,
                    narrow: false,
                })
                .unwrap();
        });
        let widened = approve_outside_dir(
            ws.path(),
            dir_approval("/data/dir", lofi_types::Access::Write),
            &config,
            &tx,
            &counter,
            None,
        )
        .await;
        responder.await.unwrap();
        assert_eq!(
            widened,
            Some(lofi_code::GrantedDir {
                dir: PathBuf::from("/data/dir"),
                access: lofi_types::Access::Write
            })
        );

        std::env::remove_var("LOFI_STATE_HOME");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn refusal_is_not_remembered() {
        let _lock = state::STATE_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _state_home = state_env();
        let ws = tempfile::tempdir().unwrap();
        let counter = Arc::new(AtomicU64::new(0));

        // A refusal is not cached: both calls ask again and both are denied.
        for _ in 0..2 {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
            let responder = tokio::spawn(async move {
                let req = rx.recv().await.unwrap();
                req.respond.send(ConfirmResponse::Deny).unwrap();
            });
            let granted = approve_outside_dir(
                ws.path(),
                dir_approval("/data/dir", lofi_types::Access::Write),
                &[],
                &tx,
                &counter,
                None,
            )
            .await;
            responder.await.unwrap();
            assert_eq!(granted, None);
        }
        assert_eq!(
            state::WorkspaceState::open(ws.path())
                .unwrap()
                .grant_allowing(std::path::Path::new("/data/dir"), lofi_types::Access::Read),
            None
        );

        std::env::remove_var("LOFI_STATE_HOME");
    }

    #[test]
    fn a_wait_takes_the_time_the_run_has_left() {
        let (args, cut) = clamp_child(
            lofi_code::SubagentOp::Wait,
            serde_json::json!({ "id": "a" }),
            clock(Duration::from_secs(30)),
        );

        let wait = args["timeout_ms"].as_u64().unwrap();
        assert!(
            (29_000..=30_000).contains(&wait),
            "the wait takes what the run has left: {args}"
        );
        assert_eq!(cut, None, "a wait the caller did not bound was not cut");
    }

    #[test]
    fn a_mailbox_receive_waits_no_longer_than_the_call() {
        let clock = clock(Duration::from_secs(3));
        let (args, cut) = clamp_child(
            lofi_code::SubagentOp::MailboxReceive,
            serde_json::json!({ "limit": 5, "wait_ms": 10_000 }),
            clock,
        );

        let wait = args["wait_ms"].as_u64().unwrap();
        assert!(
            (2_000..=3_000).contains(&wait),
            "the receive must end with the call, not block past it: {args}"
        );
        assert_eq!(cut, None, "a receive names no cap of its own");
        assert_eq!(args["limit"], serde_json::json!(5));

        // A receive that named no wait is a poll: filling the bound in would
        // make it block for the call instead.
        let (poll, _) = clamp_child(
            lofi_code::SubagentOp::MailboxReceive,
            serde_json::json!({ "limit": 5 }),
            clock,
        );
        assert_eq!(poll["wait_ms"], serde_json::Value::Null);
    }

    #[test]
    fn a_spent_clock_still_asks_for_a_short_wait() {
        let (args, _) = clamp_child(
            lofi_code::SubagentOp::Wait,
            serde_json::json!({ "id": "a" }),
            clock(Duration::ZERO),
        );

        assert_eq!(
            args["timeout_ms"],
            serde_json::json!(1),
            "zero would be read as an invalid argument: {args}"
        );
    }

    #[test]
    fn a_wait_that_is_not_a_number_is_left_to_the_hook() {
        let clock = clock(Duration::from_secs(30));

        for asked in [
            serde_json::json!("bad"),
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!(0),
        ] {
            let (args, _) = clamp_child(
                lofi_code::SubagentOp::Wait,
                serde_json::json!({ "id": "a", "timeout_ms": asked }),
                clock,
            );
            assert_eq!(args["timeout_ms"], asked, "{args}");
        }
    }

    #[test]
    fn a_child_reports_what_its_caller_asked_for_when_the_cap_cut_it() {
        let clock = clock(Duration::from_mins(1));

        let (args, cut) = clamp_child(
            lofi_code::SubagentOp::Run,
            serde_json::json!({ "prompt": "x", "timeout_ms": 600_000 }),
            clock,
        );
        let (unset, _) = clamp_child(
            lofi_code::SubagentOp::Run,
            serde_json::json!({ "prompt": "x" }),
            clock,
        );

        assert_eq!(cut, Some(600_000));
        assert_eq!(
            args["requested_timeout_ms"],
            serde_json::json!(600_000),
            "the run states the allowance it was cut from, not only what it was left: {args}"
        );
        assert!(
            unset.get("requested_timeout_ms").is_none(),
            "a caller that set no timeout has none to report: {unset}"
        );
    }

    #[test]
    fn an_uncapped_run_leaves_a_wait_alone() {
        let (args, _) = clamp_child(
            lofi_code::SubagentOp::Wait,
            serde_json::json!({ "id": "a" }),
            lofi_types::Deadline::NONE,
        );

        assert_eq!(args, serde_json::json!({ "id": "a" }), "{args}");
    }

    #[test]
    fn a_wait_is_clamped_to_the_run_and_not_stretched_by_it() {
        let clock = clock(Duration::from_secs(2));

        let (asked, cut) = clamp_child(
            lofi_code::SubagentOp::Wait,
            serde_json::json!({ "id": "a", "timeout_ms": 900_000 }),
            clock,
        );
        let (short, _) = clamp_child(
            lofi_code::SubagentOp::Wait,
            serde_json::json!({ "id": "a", "timeout_ms": 500 }),
            clock,
        );

        assert!(
            asked["timeout_ms"].as_u64().unwrap() <= 2_000,
            "a wait longer than the run has would block past it: {asked}"
        );
        assert_eq!(
            cut, None,
            "a wait returns the run's own shape, so its timeout is not the run's cap"
        );
        assert_eq!(short["timeout_ms"], serde_json::json!(500));
    }

    #[test]
    fn a_spawned_child_keeps_its_own_timeout() {
        let clock = clock(Duration::from_mins(1));

        let (asked, cut) = clamp_child(
            lofi_code::SubagentOp::Spawn,
            serde_json::json!({ "prompt": "x", "timeout_ms": 600_000 }),
            clock,
        );
        let (unset, uncut) = clamp_child(
            lofi_code::SubagentOp::Spawn,
            serde_json::json!({ "prompt": "x" }),
            clock,
        );

        assert_eq!(
            asked["timeout_ms"],
            serde_json::json!(600_000),
            "a spawned run outlives the call that started it: {asked}"
        );
        assert_eq!(cut, None, "the call's bound is not the child's");
        assert!(
            unset.get("timeout_ms").is_none(),
            "a spawn that set no timeout stays unbounded: {unset}"
        );
        assert_eq!(uncut, None, "a child that set no timeout was not cut");
    }

    #[test]
    fn a_foreground_child_takes_what_the_call_has_left() {
        let clock = clock(Duration::from_mins(1));

        let (run, cut) = clamp_child(
            lofi_code::SubagentOp::Run,
            serde_json::json!({ "prompt": "x" }),
            clock,
        );
        let (asked, asked_cut) = clamp_child(
            lofi_code::SubagentOp::Run,
            serde_json::json!({ "prompt": "x", "timeout_ms": 600_000 }),
            clock,
        );

        let run_ms = run["timeout_ms"].as_u64().unwrap();
        assert!(
            run_ms > 0 && run_ms <= 60_000,
            "a foreground run ends with the call that awaits it: {run}"
        );
        assert_eq!(cut, None, "the call's bound is not a timeout to report");
        assert_eq!(
            asked["timeout_ms"],
            serde_json::json!(60_000),
            "a timeout longer than the call is cut to it: {asked}"
        );
        assert_eq!(
            asked_cut,
            Some(600_000),
            "the allowance it asked for is reported"
        );
    }

    #[test]
    fn a_call_that_runs_nothing_is_left_alone() {
        let clock = clock(Duration::from_secs(2));

        for op in [
            lofi_code::SubagentOp::Kill,
            lofi_code::SubagentOp::Status,
            lofi_code::SubagentOp::List,
        ] {
            let (args, cut) = clamp_child(
                op,
                serde_json::json!({ "id": "a", "timeout_ms": 900_000 }),
                clock,
            );
            assert_eq!(
                args,
                serde_json::json!({ "id": "a", "timeout_ms": 900_000 })
            );
            assert_eq!(cut, None);
        }
    }
}
