use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use lofi_error::{Error, Result};
use lofi_types::{Config, ContentBlock, Message, Role, ThinkingLevel};
use serde_json::{json, Value};
use tokio::sync::watch;

use super::ops::{mailbox_hook, str_arg};
use super::registry::{
    entry_usage, evict_for_spawn, task_preview, Activity, Entry, EventLog, Rollup, SubagentRegistry,
};
use super::runner::RunRejection;
use super::settled::{outcome_json, restore_settled};
use super::{attach_mailbox_journal, Status, MAX_RUNNING};
use crate::agent::mailbox::MailboxScope;
use crate::agent::tool_executor::{ActivityWatch, ConfirmWatch};
use crate::agent::{auto_mode, close_unanswered_tool_uses, lock, ToolExecutor};
use crate::backend::{Backend, BackendFactory, BackendRef};
use crate::models::ModelRegistry;
use crate::session::store::{SessionCursor, SessionFile, SessionStore};

/// What the hook needs to resolve subagent model overrides.
#[derive(Clone)]
pub struct SubagentSupport {
    pub(super) registry: ModelRegistry,
    pub(super) config: Config,
    /// Workspace root the subagent transcript is filed under.
    pub(super) cwd: PathBuf,
    /// Whether subagent turns write transcripts. Disabled by the host when
    /// the parent session does not persist (`--no-session`, print mode).
    pub(super) persist_sessions: bool,
    /// Environment specs the session was opened with, so a child that
    /// overrides its model to another provider resolves that provider's
    /// credentials the same way.
    env: Vec<(String, String)>,
    /// Skill directory next to the session's config file.
    skills_dir: Option<PathBuf>,
}

impl SubagentSupport {
    #[must_use]
    pub fn new(registry: ModelRegistry, config: Config, cwd: PathBuf) -> Self {
        Self {
            registry,
            config,
            cwd,
            persist_sessions: true,
            env: Vec::new(),
            skills_dir: None,
        }
    }

    #[must_use]
    pub fn with_persistence(mut self, persist_sessions: bool) -> Self {
        self.persist_sessions = persist_sessions;
        self
    }

    #[must_use]
    pub fn with_env(mut self, env: Vec<(String, String)>) -> Self {
        self.env = env;
        self
    }

    #[must_use]
    pub fn with_skills_dir(mut self, dir: Option<PathBuf>) -> Self {
        self.skills_dir = dir;
        self
    }

    /// The factory a model override opens its child through. Built on demand:
    /// a spawn without an override keeps the template's own selection and
    /// never resolves another provider.
    ///
    /// The factory carries this catalog, so an override can name a model the
    /// provider discovered at startup.
    #[allow(clippy::unnecessary_wraps)]
    pub(in crate::agent) fn factory(&self) -> Result<BackendFactory> {
        Ok(BackendFactory::from_resolved(
            self.config.clone(),
            self.registry.clone(),
            &self.cwd,
            self.env.clone(),
        )
        .with_skills_dir(self.skills_dir.clone())
        .with_subagent_persistence(self.persist_sessions))
    }

    /// Build the `lofi.models` hook: the registry's available choices as JSON.
    #[must_use]
    pub fn models_fn(&self) -> lofi_code::ModelsFn {
        let choices: Vec<Value> = self
            .registry
            .choices()
            .iter()
            .map(|c| {
                json!({
                    "model": format!("{}/{}", c.provider, c.id),
                    "name": c.name,
                    "thinking_levels": c.thinking_levels.iter().map(lofi_types::ThinkingLevel::as_str).collect::<Vec<_>>(),
                    "service_tiers": c.service_tiers.iter().map(lofi_types::ServiceTier::as_str).collect::<Vec<_>>(),
                })
            })
            .collect();
        Arc::new(move || json!({ "models": choices }))
    }
}

/// Give a child its own turn context and its own evaluator. A clone of the
/// template shares the handle the parent evaluator captured, so without this
/// a child reads the session's context and its own turns overwrite it.
pub(in crate::agent) fn isolate_auto_mode(executor: &mut ToolExecutor, support: &SubagentSupport) {
    executor.auto_mode_hints = auto_mode::hint_handle();
    if support.config.shell_policy.auto_mode.is_none() {
        executor.auto_mode = None;
        executor.auto_mode_config = None;
        executor.invalidate_session();
        return;
    }
    // The session config is unresolved and its provider table is pruned to the
    // model's own provider, so the child takes the evaluator config its parent
    // opened: same catalog entry, same resolved credentials.
    let Some(config) = executor.auto_mode_config.clone() else {
        executor.auto_mode = None;
        executor.invalidate_session();
        return;
    };
    let mut child = executor.clone();
    match auto_mode::attach(&mut child, &config, &support.registry) {
        Ok(()) => {
            executor.auto_mode = child.auto_mode;
            executor.auto_mode_config = child.auto_mode_config;
        }
        Err(error) => {
            tracing::warn!(%error, "subagent: auto-mode disabled for the child");
            executor.auto_mode = None;
            executor.auto_mode_config = None;
        }
    }
    executor.invalidate_session();
}

/// The child's execution state: the session's jobs, policy, truncation and
/// sandbox configuration stay shared; the sandbox thread, the auto-mode
/// context and the run-management hook are the child's own.
/// The user request a child inherits: the request its parent inherited
/// (a nested child), else the parent session's current request. Read from
/// the parent's hints handle, which the turn session fills from user input,
/// so a spawn call can never supply it.
pub(in crate::agent) fn inherited_request(template: &dyn Backend) -> Option<String> {
    let executor = template.tool_executor();
    let hints = executor.auto_mode_hints.snapshot();
    hints.root_request.or(hints.prompt)
}

pub(in crate::agent) fn child_executor(
    template: &dyn Backend,
    support: &SubagentSupport,
    root_request: Option<String>,
) -> ToolExecutor {
    let mut executor = template
        .tool_executor()
        .reset_exec_worker()
        .with_fresh_auto_mode_hints();
    // A child inherits the session's confirm channel, so its own open
    // prompts and running calls need their own count: the idle deadline
    // pauses on them, and a sibling's work is not this child's progress.
    executor.confirm_watch = Arc::new(ConfirmWatch::default());
    executor.activity = Arc::new(ActivityWatch::default());
    isolate_auto_mode(&mut executor, support);
    // Seed the inherited request into the handle the turn loop preserves:
    // the child's own history holds only its task, so a later turn cannot
    // recover the request the user typed in the parent session. Seeded after
    // `isolate_auto_mode`, which hands the child a fresh handle.
    if let Some(request) = root_request {
        *executor
            .auto_mode_hints
            .hints
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = lofi_code::AutoModeHints {
            root_request: Some(request),
            ..lofi_code::AutoModeHints::default()
        };
    }
    // A child cannot manage runs; it gets its own inbox once it has an id.
    executor.with_child_identity("", None, None, None)
}

/// The same child at another thinking level. The target checks the level, so a
/// session model that declares no levels still runs at Off, and a level it
/// rejects fails with the same text an override gets. A level the child
/// already runs at returns it unchanged, which keeps a CLI child alive.
fn with_thinking(
    child: BackendRef,
    level: &ThinkingLevel,
) -> std::result::Result<BackendRef, RunRejection> {
    let selection = child.selection();
    if selection.thinking == *level {
        return Ok(child);
    }
    Ok(child.with_selection(&format!(
        "{}/{}:{}",
        child.name(),
        selection.model,
        level.as_str()
    ))?)
}

/// The child backend: the template's transport and selection, unless the call
/// names another model, in which case the factory opens whichever transport
/// serves it.
async fn child_backend(
    template: &BackendRef,
    support: &SubagentSupport,
    args: &Value,
) -> std::result::Result<BackendRef, RunRejection> {
    let model = str_arg(args, "model")?;
    // An unknown level becomes a custom one: the target decides whether it
    // accepts it, the same way a `:level` suffix does.
    let level = str_arg(args, "thinking")?.map(ThinkingLevel::parse_or_custom);
    // Captured once, before the fork, from the parent session's live hints.
    let root_request = inherited_request(template.as_ref());
    let executor = child_executor(template.as_ref(), support, root_request);
    // The subagent hook is installed when the worker starts, so its template
    // keeps the selection the worker was opened with: fork reads the live one,
    // which a model or thinking switch has since changed.
    let parent = template.fork(executor.clone());
    let mut child = if let Some(query) = model {
        // The factory resolves the name against whichever catalog holds it, and
        // an explicit level is resolved with the model so it is validated
        // against the supported levels of the target.
        let factory = support.factory()?;
        let query = factory
            .resolve_model_query(&parent.selection(), query, level)
            .map_err(|e| Error::Tool(format!("invalid model override: {e}")))?;
        // The factory opens a session backend; a child forks from it so it
        // runs without run management of its own.
        let opened = factory
            .open(Some(&query), executor.clone())
            .await
            .map_err(|e| Error::Tool(format!("invalid model override: {e}")))?;
        opened.fork(executor)
    } else {
        match level {
            Some(level) => with_thinking(parent, &level)?,
            None => parent,
        }
    };
    if let Some(system) = str_arg(args, "system")? {
        child = child.with_system_prompt(system.to_string());
    }
    Ok(child)
}

/// Build or resume the entry, and hand out the state the turn needs.
#[allow(clippy::type_complexity, clippy::too_many_lines)]
pub(super) async fn prepare(
    template: &BackendRef,
    support: &SubagentSupport,
    registry: &SubagentRegistry,
    id: Option<String>,
    args: &Value,
    notify: bool,
) -> std::result::Result<
    (
        BackendRef,
        Vec<Message>,
        Option<SessionCursor>,
        Arc<AtomicBool>,
        String,
    ),
    RunRejection,
> {
    if let Some(id) = id {
        return prepare_resume(registry, &id, args, notify);
    }

    // Built before the entry exists, so a rejected override leaves no state.
    let mut agent = child_backend(template, support, args).await?;
    // The job registry stays shared with the session: an isolated registry
    // would leak child processes at session end.
    let jobs = template.tool_executor().jobs();

    let cancel = Arc::new(AtomicBool::new(false));
    // Reserve capacity, the id, and a placeholder entry under one lock;
    // transcript creation does blocking store I/O and must not hold it.
    // A rejected spawn returns inside the block, so it leaves no persistent
    // state behind; the placeholder is filled in with the transcript below.
    let id = {
        let mut entries = lock(&registry.entries);
        let running = entries
            .values()
            .filter(|entry| !entry.status.is_terminal())
            .count();
        if running >= MAX_RUNNING {
            return Err(Error::Tool(format!(
                "concurrency cap reached ({MAX_RUNNING} running subagents)"
            ))
            .into());
        }
        evict_for_spawn(registry, &mut entries)?;
        let id = loop {
            let id = lofi_types::short_id(registry.ids.next());
            // A killed owner's job bar outlives its entry (its old worker
            // may still finish in-flight work): do not hand the id to a
            // new run, which would then be unable to spawn jobs.
            if !entries.contains_key(&id) && !jobs.owner_cancelled(&id) {
                break id;
            }
        };
        registry.mailbox.open(&id);
        // A spawned run is folded by its exec at the end, so its spend must
        // survive eviction until then. A foreground run's body carries the
        // spend to that exec already.
        if notify {
            lock(&registry.retired).claim(&id);
        }
        let (state_tx, _) = watch::channel(Status::Running);
        entries.insert(
            id.clone(),
            Entry {
                status: Status::Running,
                backend: agent.clone(),
                messages: Vec::new(),
                history_loaded: true,
                cancel: cancel.clone(),
                session: None,
                cursor: None,
                transcript_path: None,
                used: registry.stamp(),
                // Born 0 would collide with the absent page cursor.
                born: registry.next_born.fetch_add(1, Ordering::SeqCst) + 1,
                notify_tick: 0,
                notify_task: None,
                outcome: None,
                state_tx,
                abort: None,
                events: EventLog::default(),
                notify_terminal: notify,
                note: None,
                started: std::time::Instant::now(),
                activity: Activity::default(),
                task: task_preview(str_arg(args, "prompt")?.unwrap_or("")),
                // The notify argument marks the spawn path: only
                // `subagentSpawn` starts a run with notices on.
                background: notify,
                model: Some(agent.run_model().label()),
                totals: None,
                carried: Rollup::default(),
            },
        );
        id
    };
    // The child has an identity now: its jobs are tagged with the run id so a
    // kill or a session switch cancels exactly its process tree, and its
    // sandbox gets its own inbox.
    let scope = MailboxScope {
        mailbox: registry.mailbox.clone(),
        me: id.clone(),
    };
    agent = agent.with_tool_executor(agent.tool_executor().with_child_identity(
        &id,
        Some(scope.clone()),
        Some(mailbox_hook(scope)),
        None,
    ));
    if let Some(entry) = lock(&registry.entries).get_mut(&id) {
        entry.backend = agent.clone();
    }
    let cursor = if support.persist_sessions {
        let created = SessionStore::open().and_then(|store| {
            attach_mailbox_journal(registry, &store);
            let cursor = store.create_subagent_cursor(
                &support.cwd,
                &agent.run_model(),
                registry.parent_session().as_deref(),
                &id,
            )?;
            let system = agent.system_prompt();
            if !system.is_empty() {
                cursor
                    .record(crate::session::recorder::SessionRecord::System { prompt: system })?;
            }
            Ok(cursor)
        });
        match created {
            Ok(cursor) => Some(cursor),
            Err(e) => {
                tracing::debug!(error = %e, "subagent transcript disabled");
                None
            }
        }
    } else {
        None
    };
    // Fill the placeholder in with the transcript. The entry can be gone
    // when clear() ran during a session switch; the run then finishes
    // untracked, and its settle fold skips the missing entry.
    if let Some(cursor) = &cursor {
        if let Some(entry) = lock(&registry.entries).get_mut(&id) {
            entry.session = cursor
                .path()
                .file_name()
                .map(|n| n.to_string_lossy().into_owned());
            entry.cursor = Some(cursor.clone());
        }
    }
    Ok((agent, Vec::new(), cursor, cancel, id))
}

/// Build one registry entry from a transcript: identity, model, and the
/// persisted outcome only. The history replays on the first resume; the
/// system prompt comes from the replayed lineage (a header-level read could
/// pick a record from an abandoned branch).
pub(super) fn restore_entry(
    template: &dyn Backend,
    support: &SubagentSupport,
    registry: &SubagentRegistry,
    file: &SessionFile,
) -> (String, Entry) {
    let meta = file.meta();
    // Transcripts written before rehydration existed carry no run_id; the
    // file stem stands in so they stay resumable.
    let id = meta.run_id.clone().unwrap_or_else(|| file.id());
    let session_id = file.id();
    let (status, carried, outcome) = restore_settled(file.path());
    // Restore the model the transcript ran with. When it is gone (provider
    // or model removed from the config), keep the session model and say so
    // on the entry instead of failing the restore.
    let mut note = None;
    // A restored entry lost the capture across the restart, so it inherits
    // nothing: the current parent request did not start this run.
    let executor = child_executor(template, support, None);
    let mut child = template.fork(executor.clone());
    let selection = child.selection();
    let current = format!(
        "{}/{}:{}",
        child.name(),
        selection.model,
        selection.thinking.as_str()
    );
    let restored = format!(
        "{}/{}:{}",
        meta.model.provider,
        meta.model.selected_or_id(),
        meta.model.thinking.as_str()
    );
    if restored != current {
        match support
            .factory()
            .and_then(|factory| factory.open_resolved(Some(&restored), executor.clone()))
        {
            // The factory opens a session backend; a restored child runs as a
            // child, so it forks from it.
            Ok(opened) => child = opened.fork(executor.clone()),
            Err(e) => {
                note = Some(format!(
                    "transcript model {} is unavailable ({e}); resume uses the session model",
                    meta.model.label(),
                ));
            }
        }
    }
    // A restored child cannot manage runs: it messages its parent instead.
    let scope = MailboxScope {
        mailbox: registry.mailbox.clone(),
        me: id.clone(),
    };
    child = child.with_tool_executor(child.tool_executor().with_child_identity(
        &id,
        Some(scope.clone()),
        Some(mailbox_hook(scope)),
        None,
    ));
    let (state_tx, _) = watch::channel(status);
    // The entry's model is the one a resume would run. The settled outcome
    // names the model that produced it: the transcript's, even when the
    // resume fell back to the session model.
    let model = child.run_model().label();
    let outcome = outcome_json(&id, Some(&session_id), outcome, &meta.model.label());
    let entry = Entry {
        status,
        backend: child,
        messages: Vec::new(),
        history_loaded: false,
        cancel: Arc::new(AtomicBool::new(false)),
        session: file
            .path()
            .file_name()
            .map(|name| name.to_string_lossy().into_owned()),
        cursor: None,
        transcript_path: Some(file.path().to_path_buf()),
        used: registry.stamp(),
        // Born 0 would collide with the absent page cursor.
        born: registry.next_born.fetch_add(1, Ordering::SeqCst) + 1,
        outcome: Some(outcome),
        state_tx,
        abort: None,
        events: EventLog::default(),
        notify_terminal: false,
        notify_tick: 0,
        notify_task: None,
        note,
        started: std::time::Instant::now(),
        activity: Activity::default(),
        task: String::new(),
        // A rehydrated entry is settled; a resume re-claims it as
        // foreground and flips this flag itself.
        background: false,
        model: Some(model),
        totals: None,
        carried,
    };
    (id, entry)
}

/// Resume path of `prepare`: claim a settled entry for a fresh run. A
/// background resume is folded by its exec at the end, like a spawn, so its
/// spend must survive eviction until then; a rejected or unloadable resume
/// claims nothing, because no exec would ever release it.
/// A resumed run: the backend it continues on, the history it replays, the
/// transcript to record into, its cancel flag, and its id.
type ResumedRun = (
    BackendRef,
    Vec<Message>,
    Option<SessionCursor>,
    Arc<AtomicBool>,
    String,
);

fn prepare_resume(
    registry: &SubagentRegistry,
    id: &str,
    args: &Value,
    notify: bool,
) -> std::result::Result<ResumedRun, RunRejection> {
    let started = prepare_resume_run(registry, id.to_string(), args, notify);
    if notify && started.is_ok() {
        lock(&registry.retired).claim(id);
    }
    started
}

/// The body of `prepare_resume`, which claims the retirement slot itself.
#[allow(clippy::too_many_lines)]
fn prepare_resume_run(
    registry: &SubagentRegistry,
    id: String,
    args: &Value,
    notify: bool,
) -> std::result::Result<ResumedRun, RunRejection> {
    let (agent, history, cursor, transcript_path, cancel, loaded) = {
        let mut entries = lock(&registry.entries);
        match entries.get(&id) {
            None => return Err(Error::Tool(format!("unknown subagent id: {id}")).into()),
            Some(entry) if !entry.status.is_terminal() => {
                return Err(RunRejection::Busy(entry.status.as_str().to_string()))
            }
            Some(_) => {}
        }
        // A resume re-enters the running set, so it is subject to the same cap.
        let running = entries
            .values()
            .filter(|entry| !entry.status.is_terminal())
            .count();
        if running >= MAX_RUNNING {
            return Err(Error::Tool(format!(
                "concurrency cap reached ({MAX_RUNNING} running subagents)"
            ))
            .into());
        }
        let entry = entries
            .get_mut(&id)
            .ok_or_else(|| Error::Tool(format!("unknown subagent id: {id}")))?;
        // The previous run's prompts and calls are not this run's: a prompt it
        // left open would otherwise pause this run's idle deadline forever.
        let executor = entry.backend.tool_executor();
        executor.confirm_watch.reset();
        executor.activity.reset();
        if str_arg(args, "model")?.is_some()
            || str_arg(args, "thinking")?.is_some()
            || str_arg(args, "system")?.is_some()
        {
            return Err(Error::Tool(
                "model, thinking and system only apply when spawning a subagent".into(),
            )
            .into());
        }
        // Parse before the claim: a failure must not strand the entry in
        // the running state.
        let task = task_preview(str_arg(args, "prompt")?.unwrap_or(""));
        // Claim the entry under the lock so a concurrent resume of the same
        // id is rejected instead of racing this run.
        entry.status = Status::Running;
        entry.used = registry.stamp();
        entry.started = std::time::Instant::now();
        // The previous run's last words are not this run's live line.
        entry.activity = Activity::default();
        entry.task = task;
        // The run this resume replaces moves onto the entry's earlier runs,
        // so a fold of the child session still counts what it billed. This
        // run installs its own accumulator when it starts.
        if let Some(spent) = entry_usage(entry) {
            entry.carried.add_usage(&spent);
        }
        entry.totals = None;
        // A previous kill barred this owner from acquiring jobs; the new run
        // needs them again.
        entry.backend.tool_executor().jobs().uncancel_owner(&id);
        // Fresh cancel flag per run: the previous run's flag may still be
        // observed by its lingering sandbox work, and resetting it would
        // uncancel that work.
        entry.cancel = Arc::new(AtomicBool::new(false));
        // The previous run's outcome and abort handle belong to that run; the
        // new run stores its own. Waiters see the state move back to running.
        entry.outcome = None;
        entry.abort = None;
        // Notices and the periodic loop follow the caller: a background
        // resume keeps them, and a foreground one silences the terminal
        // notice and retires the previous run's loop, which would otherwise
        // wake up on the running status.
        entry.notify_terminal = notify;
        entry.background = notify;
        entry.notify_tick += 1;
        if let Some(task) = entry.notify_task.take() {
            task.abort();
        }
        entry.state_tx.send_replace(Status::Running);
        // The previous run closed this inbox at settle; a resume continues the
        // same conversation and must receive what that run never read.
        registry.mailbox.open(&id);
        let transcript_backed = entry.cursor.is_some() || entry.transcript_path.is_some();
        // The run owns the history while it runs: a transcript-backed entry
        // gets it back at settle, normal or guard, or reloads it from the
        // transcript when the settle marker persisted, so nothing retains two
        // copies. An entry without a transcript keeps its only copy here.
        let history = match (entry.history_loaded, transcript_backed) {
            (true, true) => std::mem::take(&mut entry.messages),
            (true, false) => entry.messages.clone(),
            (false, _) => Vec::new(),
        };
        (
            entry.backend.clone(),
            history,
            entry.cursor.clone(),
            entry.transcript_path.clone(),
            entry.cancel.clone(),
            entry.history_loaded,
        )
    };
    if loaded {
        return Ok((agent, history, cursor, cancel, id));
    }
    // First resume after a rehydrate or an unload: replay the transcript.
    // The store I/O stays outside the entries lock; a concurrent kill sets
    // the claimed cancel flag, which the run observes on its first round.
    match load_entry_history(cursor, transcript_path) {
        Ok((history, cursor, system)) => {
            let agent = match system {
                Some(prompt) if prompt != agent.system_prompt() => agent.with_system_prompt(prompt),
                _ => agent,
            };
            let mut entries = lock(&registry.entries);
            let Some(entry) = entries.get_mut(&id) else {
                return Err(Error::Tool(format!("unknown subagent id: {id}")).into());
            };
            entry.cursor = Some(cursor.clone());
            entry.history_loaded = true;
            Ok((agent, history, Some(cursor), entry.cancel.clone(), id))
        }
        Err(e) => {
            unclaim(registry, &id, &e);
            Err(Error::Tool(format!("subagent transcript could not be loaded: {e}")).into())
        }
    }
}

/// A resume whose transcript failed to load must not wedge the entry in the
/// running state: publish the failure and hand the entry back to waiters.
fn unclaim(registry: &SubagentRegistry, id: &str, error: &Error) {
    let mut entries = lock(&registry.entries);
    let Some(entry) = entries.get_mut(id) else {
        return;
    };
    entry.status = Status::Failed;
    entry.outcome = Some(json!({
        "ok": false,
        "id": id,
        "status": Status::Failed.as_str(),
        "error": format!("subagent transcript could not be loaded: {error}"),
    }));
    entry.state_tx.send_replace(Status::Failed);
    // The claim reopened the address before the load; a failed claim must
    // close it again, or a send would be accepted for a run that never starts.
    registry.mailbox.close(id, Status::Failed.as_str());
}

/// Replay a transcript into the resume history: a bounded projection of the
/// selected lineage (events stream one at a time; the full event vector is
/// never materialized), the latest system prompt split out for the agent,
/// and every tool call without a result closed with an interrupted result.
fn load_entry_history(
    cursor: Option<SessionCursor>,
    transcript_path: Option<PathBuf>,
) -> Result<(Vec<Message>, SessionCursor, Option<String>)> {
    let (cursor, snapshot) = if let Some(cursor) = cursor {
        let snapshot = cursor.snapshot()?;
        (cursor, snapshot)
    } else {
        let path = transcript_path
            .ok_or_else(|| Error::State("subagent entry has no transcript".to_string()))?;
        SessionCursor::open_snapshot(path)?
    };
    // Same leaf-first recipe as the parent session's restore: from
    // history_start so a compaction boundary is never crossed.
    let offsets: Vec<u64> = snapshot.index[snapshot.history_start.min(snapshot.index.len())..]
        .iter()
        .rev()
        .map(|entry| entry.offset)
        .collect();
    let mut projection = crate::session::replay::AgentHistoryProjection::default();
    cursor.visit_events(&offsets, |event| {
        projection.push(&event);
        Ok(())
    })?;
    let mut messages = projection.finish();
    drop(snapshot);
    lofi_types::memory::release_freed_memory();
    // The projection puts the latest system prompt first; the agent carries
    // it instead, matching entries that never reloaded.
    let system = if messages
        .first()
        .is_some_and(|message| message.role == Role::System)
    {
        let system = messages.remove(0);
        let text = system
            .blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        Some(text)
    } else {
        None
    };
    close_unanswered_tool_uses(&mut messages);
    Ok((messages, cursor, system))
}
