use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lofi_error::{Error, Result};
use lofi_types::ModelChoice;
use tokio::sync::mpsc;

use super::claude_engine::{spawn_worker, ClaudeEngine, Work, WorkerInit};
use super::contract::{Backend, BackendRef, Completion, CompletionRequest, ModelSelection};
use crate::agent::turn_session::{ExchangeDriver, TurnPolicy};
use crate::agent::ToolExecutor;

/// The thinking levels one CLI backend accepts. The picker and the check read
/// this one table, so they cannot disagree.
#[must_use]
fn levels() -> Vec<lofi_types::ThinkingLevel> {
    super::claude::effort_levels()
}

/// Whether a CLI backend runs at a level. Off always means "pass no effort
/// flag", so it needs no entry in the table.
#[must_use]
pub(super) fn accepts(level: &lofi_types::ThinkingLevel) -> bool {
    *level == lofi_types::ThinkingLevel::Off || levels().contains(level)
}

/// Model choices of one CLI backend, in picker order: the configured model
/// first, then the configured list, then "default" when nothing is named.
#[must_use]
pub fn configured_models(
    name: &str,
    config: &lofi_types::BackendConfig,
    selected: Option<&str>,
) -> Vec<ModelChoice> {
    let mut models = config.models().to_vec();
    if let Some(model) = selected {
        models.retain(|id| id != model);
        models.insert(0, model.into());
    }
    if models.is_empty() {
        models.push("default".into());
    }
    models
        .into_iter()
        .map(|id| ModelChoice {
            provider: name.into(),
            name: id.clone(),
            id,
            thinking_levels: levels(),
            service_tiers: Vec::new(),
            supports_image: true,
            // Learned from the child at turn end; the choice reads the cell.
            context_window: None,
        })
        .collect()
}

/// The model ids a backend accepts in a query, in picker order: the configured
/// model, then its `models`, then `default`, which opens the CLI's own
/// default model.
fn model_ids(config: &lofi_types::BackendConfig) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();
    if let Some(model) = config.model() {
        ids.push(model.to_string());
    }
    for id in config.models() {
        if !ids.iter().any(|known| known == id) {
            ids.push(id.clone());
        }
    }
    if !ids.iter().any(|known| known == "default") {
        ids.push("default".to_string());
    }
    ids
}

/// Check one selection against a backend's config: the model it names, the
/// level it runs at, and the tier it asks for.
///
/// Every caller that names a backend model goes through here — the factory
/// opening a session, a live model switch, and a subagent override — so the
/// rule and its error text exist once.
///
/// # Errors
/// Returns [`Error::Config`] for a model outside the config, a service tier
/// other than `auto`, or a level the backend does not accept.
pub(super) fn check_selection(
    config: &lofi_types::BackendConfig,
    backend: &str,
    model: &str,
    level: &lofi_types::ThinkingLevel,
    tier: &lofi_types::ServiceTier,
) -> Result<()> {
    let ids = model_ids(config);
    if !ids.iter().any(|id| id == model) {
        return Err(Error::Config(format!(
            "no model `{model}` for backend `{backend}`; configured: {}",
            ids.join(", ")
        )));
    }
    if *tier != lofi_types::ServiceTier::Auto {
        return Err(Error::Config(format!(
            "service tier `{}` is not supported by `{backend}`",
            tier.as_str()
        )));
    }
    if !accepts(level) {
        let allowed = levels();
        let names: Vec<&str> = allowed
            .iter()
            .map(lofi_types::ThinkingLevel::as_str)
            .collect();
        return Err(Error::Config(format!(
            "thinking level `{}` is not supported by `{backend}`; allowed: {}",
            level.as_str(),
            names.join(", ")
        )));
    }
    Ok(())
}

#[derive(Clone)]
pub(super) struct ExternalBackend {
    name: String,
    model: Option<String>,
    config: lofi_types::BackendConfig,
    root: PathBuf,
    system_prompt: String,
    executor: ToolExecutor,
    thinking: lofi_types::ThinkingLevel,
    worker: Arc<Mutex<Option<mpsc::Sender<Work>>>>,
    /// The context window the CLI reported for the current model. Shared by
    /// every clone, so the picker and the gauge learn it.
    context_window: Arc<Mutex<Option<u64>>>,
    /// The selection the live worker serves. The subagent hook is installed
    /// once, when the worker starts, so a child forks from this cell instead
    /// of the fields its template was copied with: a model switch reuses the
    /// worker and must reach the subagents it spawns.
    live: Arc<Mutex<(Option<String>, lofi_types::ThinkingLevel)>>,
    /// The compaction config, so this transport stops at the same fill the
    /// provider path does. Lofi owns the cap on this transport too, so the child
    /// is never asked to decide when the context is full.
    compaction: lofi_types::CompactionConfig,
    /// The same retry ladder the native path applies to a failed request.
    retry: crate::retry::RetryPolicy,
}

impl ExternalBackend {
    /// Configure a CLI backend without starting it or reading credentials.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn configure(
        name: String,
        config: lofi_types::BackendConfig,
        root: PathBuf,
        executor: ToolExecutor,
        model: Option<String>,
        thinking: lofi_types::ThinkingLevel,
        system_prompt: String,
        compaction: lofi_types::CompactionConfig,
    ) -> Self {
        Self {
            live: Arc::new(Mutex::new((model.clone(), thinking.clone()))),
            // None means the backend default: the caller resolved the query.
            model,
            name,
            config,
            root,
            system_prompt,
            executor,
            thinking,
            worker: Arc::new(Mutex::new(None)),
            context_window: Arc::new(Mutex::new(None)),
            compaction,
            retry: crate::retry::RetryPolicy::default(),
        }
    }

    /// Apply the session retry ladder. The factory holds the config that
    /// declares it; the backend only carries the resolved policy.
    #[must_use]
    pub(super) fn with_retry(mut self, retry: crate::retry::RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// A copy that opens its own child. The worker cell holds the sender of
    /// the thread it started, so sharing it would keep another thread's
    /// channel open, and a running child keeps the system prompt and the
    /// executor it was opened with. The selection cell is per thread too:
    /// each worker writes the model it serves, and a copy must not inherit
    /// another thread's.
    fn detached(&self) -> Self {
        Self {
            worker: Arc::new(Mutex::new(None)),
            live: Arc::new(Mutex::new((self.model.clone(), self.thinking.clone()))),
            ..self.clone()
        }
    }

    /// The template the worker's executor carries. It opens its own child and
    /// its own turn slot, but reads the selection cell of the worker it was
    /// installed in, which survives a model switch.
    fn worker_template(&self) -> Self {
        Self {
            worker: Arc::new(Mutex::new(None)),
            ..self.clone()
        }
    }

    /// The fill above which the turn stops and the UI compacts, on the rule
    /// the provider path applies. A turn decides it before it runs: the window
    /// the child reported, or the one the config declares before the first
    /// turn.
    fn hard_compact_threshold(&self) -> Option<u64> {
        self.context_window()
            .and_then(|window| self.compaction.hard_threshold(window))
    }

    /// The context window the child reported, if any turn has finished.
    fn learned_window(&self) -> Option<u64> {
        self.context_window.lock().ok().and_then(|cell| *cell)
    }

    /// The model and effort the live worker serves, falling back to the
    /// configured pair when the cell is unreadable.
    fn learned_selection(&self) -> (Option<String>, lofi_types::ThinkingLevel) {
        self.live.lock().map_or_else(
            |_| (self.model.clone(), self.thinking.clone()),
            |cell| cell.clone(),
        )
    }

    /// Start the worker thread, or hand back the one already running.
    fn open_worker(&self) -> Result<mpsc::Sender<Work>> {
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| Error::State("backend worker lock poisoned".into()))?;
        if let Some(tx) = worker.as_ref() {
            return Ok(tx.clone());
        }
        // The bridge reports lofi.subagent availability from the hook it
        // carries, so install the run-management hook here where the backend
        // exists.
        let executor = self
            .executor
            .clone()
            .with_subagent_hook(self.subagent_hook());
        let tx = spawn_worker(WorkerInit {
            name: self.name.clone(),
            config: self.config.clone(),
            root: self.root.clone(),
            executor,
            // The child's system prompt comes from the backend, not from the
            // executor: a fork carries the role prompt its subagent was given.
            system_prompt: self.system_prompt.clone(),
            live: self.live.clone(),
        })?;
        *worker = Some(tx.clone());
        Ok(tx)
    }
}

#[async_trait]
impl Backend for ExternalBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn selection(&self) -> ModelSelection {
        ModelSelection {
            model: self.model.clone().unwrap_or_else(|| "default".to_owned()),
            thinking: self.thinking.clone(),
            service_tier: lofi_types::ServiceTier::Auto,
        }
    }

    fn choice(&self) -> ModelChoice {
        let current = self.selection().model;
        let mut choice = configured_models(&self.name, &self.config, self.model.as_deref())
            .into_iter()
            .find(|choice| choice.id == current)
            .unwrap_or(ModelChoice {
                provider: self.name.clone(),
                id: current.clone(),
                name: current,
                thinking_levels: levels(),
                service_tiers: Vec::new(),
                supports_image: true,
                context_window: None,
            });
        // The child reports the window it runs; before the first turn the
        // config declares one, exactly as a provider model does.
        choice.context_window = self
            .learned_window()
            .or_else(|| self.config.context_window());
        choice
    }

    fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    fn tool_executor(&self) -> ToolExecutor {
        self.executor.clone()
    }

    fn with_selection(&self, query: &str) -> Result<BackendRef> {
        let parsed = super::resolve::parse_query(query)?;
        if parsed.provider != self.name {
            return Err(Error::Config(format!(
                "unknown backend \"{}\"; this session runs on \"{}\"",
                parsed.provider, self.name
            )));
        }
        // A switch that names no level keeps the one the session runs at.
        let level = parsed
            .level
            .clone()
            .unwrap_or_else(|| self.thinking.clone());
        check_selection(
            &self.config,
            &self.name,
            &parsed.model,
            &level,
            &parsed.tier,
        )?;
        let mut next = self.clone();
        let model = (parsed.model != "default").then_some(parsed.model);
        // The learned window belongs to the model that reported it.
        if model != self.model {
            next.context_window = Arc::new(Mutex::new(None));
        }
        next.model = model;
        // A query without a level keeps the session's effort, as it does on
        // the native path.
        next.thinking = level;
        // The worker writes the cell when it takes this turn: a switch made
        // while another turn runs must not change the model that turn uses.
        Ok(Arc::new(next))
    }

    fn with_tool_executor(&self, executor: ToolExecutor) -> BackendRef {
        // A model switch keeps the live child and changes its model in place,
        // so the worker survives a new executor.
        let mut next = self.clone();
        next.executor = executor;
        Arc::new(next)
    }

    fn with_system_prompt(&self, prompt: String) -> BackendRef {
        // A live child keeps the prompt it was opened with, so the next turn
        // must open its own instead of reusing the running worker.
        let mut next = self.detached();
        next.system_prompt = prompt;
        Arc::new(next)
    }

    fn subagents(&self) -> crate::SubagentRegistry {
        // The CLI backend keeps its exec bridge, which carries the same
        // registry an HTTP backend exposes.
        self.executor.subagents()
    }

    fn subagent_hook(&self) -> Option<lofi_code::SubagentFn> {
        match self.executor.subagent_support() {
            // A child gets its own inbox from the spawner; a session builds
            // the run-management hook, so its children are ordinary backends.
            None => self.executor.subagent_hook(),
            // The hook travels inside the worker thread's executor. A template
            // that shared this backend's worker cell would keep that thread's
            // own sender alive, so the thread would never see its channel
            // close and its child would never stop.
            Some(support) => Some(crate::agent::backend_hook(
                Arc::new(self.worker_template()),
                support,
                self.executor.subagents(),
            )),
        }
    }

    fn fork(&self, executor: ToolExecutor) -> BackendRef {
        // The subagent hook is installed once, when the worker starts, so its
        // template outlives a model switch: take the selection the worker
        // serves, not the one the template was copied with.
        let (model, thinking) = self.learned_selection();
        Arc::new(Self {
            name: self.name.clone(),
            model: model.clone(),
            config: self.config.clone(),
            root: self.root.clone(),
            system_prompt: self.system_prompt.clone(),
            executor,
            thinking: thinking.clone(),
            // A child owns its own child process and its own turn slot, and
            // the window it reports belongs to its model, not the parent's.
            worker: Arc::new(Mutex::new(None)),
            context_window: Arc::new(Mutex::new(None)),
            live: Arc::new(Mutex::new((model, thinking))),
            compaction: self.compaction.clone(),
            retry: self.retry,
        })
    }

    fn open_turn(&self) -> Result<Box<dyn ExchangeDriver>> {
        let worker = self.open_worker()?;
        Ok(Box::new(ClaudeEngine::open(
            worker,
            &self.name,
            self.model.clone(),
            self.thinking.clone(),
            self.context_window.clone(),
        )))
    }

    fn turn_policy(&self) -> TurnPolicy {
        TurnPolicy {
            auto_continue: lofi_types::AutoContinuePolicy::default(),
            retry: self.retry,
            hard_cap: self.hard_compact_threshold(),
        }
    }

    fn context_window(&self) -> Option<u64> {
        self.learned_window()
            .or_else(|| self.config.context_window())
    }

    async fn complete(&self, request: CompletionRequest) -> Result<Completion> {
        let lofi_types::BackendConfig::ClaudeCode { command, .. } = &self.config;
        super::oneshot::complete(
            command,
            &self.root,
            self.model.as_deref(),
            &self.thinking,
            &request,
        )
        .await
    }
}
