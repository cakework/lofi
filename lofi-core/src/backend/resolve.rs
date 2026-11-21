use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use lofi_error::{Error, Result};
use lofi_types::{Config, ModelChoice, ServiceTier, ThinkingLevel};

use super::contract::{BackendRef, ModelSelection};
use super::session::ExternalBackend;
use crate::agent::ToolExecutor;
use crate::models::ModelRegistry;

/// A model query split into the backend that runs it and the selection inside
/// that backend.
pub(crate) struct ParsedQuery {
    pub provider: String,
    pub model: String,
    pub level: Option<ThinkingLevel>,
    pub tier: ServiceTier,
}

pub(crate) fn parse_query(query: &str) -> Result<ParsedQuery> {
    let parsed = crate::agent::parse_model_query(query)?;
    Ok(ParsedQuery {
        provider: parsed.provider,
        model: parsed.model,
        level: parsed.level,
        tier: parsed.tier.unwrap_or(ServiceTier::Auto),
    })
}

/// Opens a model query as a backend, whichever transport it names.
///
/// The provider table and the backend table share one namespace: a query names
/// a provider or a backend and the factory picks the transport. Nothing above
/// this point knows which one it got.
///
/// Credentials stay lazy. Building the catalog reads no credential command, and
/// opening a provider resolves only that provider.
#[derive(Clone)]
pub struct BackendFactory {
    config: Arc<Config>,
    registry: Arc<RwLock<ModelRegistry>>,
    root: PathBuf,
    env: Arc<Vec<(String, String)>>,
    skills_dir: Option<PathBuf>,
    persist_sessions: bool,
}

impl BackendFactory {
    /// Build the catalog from an unresolved config: a provider's own
    /// environment variable fills in its key for listing only, so the picker
    /// shows the models it would be able to open.
    ///
    /// # Errors
    /// Propagates static model registry errors.
    pub fn new(config: Config, root: &Path, env: Vec<(String, String)>) -> Result<Self> {
        let mut choices_config = config.clone();
        for provider in choices_config.providers.values_mut() {
            if provider.api_key.is_none() {
                provider.api_key = provider
                    .env_name
                    .as_ref()
                    .and_then(|name| std::env::var(name).ok());
            }
        }
        let registry = ModelRegistry::load_with_cache(&choices_config)?;
        Ok(Self::from_resolved(config, registry, root, env))
    }

    /// Construct a factory from a config and an already-loaded model catalog,
    /// avoiding a redundant `discovery.json` read and static catalog build.
    #[must_use]
    pub fn from_resolved(
        config: Config,
        registry: ModelRegistry,
        root: &Path,
        env: Vec<(String, String)>,
    ) -> Self {
        Self::from_shared_registry(config, Arc::new(RwLock::new(registry)), root, env)
    }

    #[must_use]
    pub(crate) fn from_shared_registry(
        config: Config,
        registry: Arc<RwLock<ModelRegistry>>,
        root: &Path,
        env: Vec<(String, String)>,
    ) -> Self {
        Self {
            config: Arc::new(config),
            registry,
            root: root.to_path_buf(),
            env: Arc::new(env),
            skills_dir: None,
            persist_sessions: true,
        }
    }

    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    #[must_use]
    pub fn registry(&self) -> ModelRegistry {
        self.registry
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn env(&self) -> &[(String, String)] {
        &self.env
    }

    /// The skill directory the session reads, next to its config file.
    #[must_use]
    pub fn with_skills_dir(mut self, dir: Option<PathBuf>) -> Self {
        self.skills_dir = dir;
        self
    }

    /// Whether subagent turns write transcripts. A session without a
    /// transcript turns them off for every backend it opens.
    #[must_use]
    pub fn with_subagent_persistence(mut self, persist: bool) -> Self {
        self.persist_sessions = persist;
        self
    }

    /// Every model of every transport, as the picker lists them.
    #[must_use]
    pub fn choices(&self) -> Vec<ModelChoice> {
        self.registry
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .choices()
    }

    /// The session-scoped state a backend opened by this factory starts with.
    ///
    /// # Errors
    /// Propagates session temporary directory creation.
    pub fn executor(&self) -> Result<ToolExecutor> {
        Ok(ToolExecutor::new(&self.root, &self.config, &self.env)?
            .with_subagent_support(self.support()))
    }

    /// What a subagent hook resolves model overrides with.
    #[must_use]
    pub fn support(&self) -> crate::SubagentSupport {
        crate::SubagentSupport::with_shared_registry(
            Arc::clone(&self.registry),
            (*self.config).clone(),
            self.root.clone(),
        )
        .with_persistence(self.persist_sessions)
        .with_env(self.env.to_vec())
        .with_skills_dir(self.skills_dir.clone())
    }

    /// Open the backend that a query names, or the configured default.
    ///
    /// # Errors
    /// Returns a configuration error for an unknown name, a model the backend
    /// does not declare, or a missing credential.
    pub async fn open(&self, query: Option<&str>, executor: ToolExecutor) -> Result<BackendRef> {
        let query = self.resolve_query(query)?;
        let parsed = parse_query(&query)?;
        if self.config.backends.contains_key(&parsed.provider) {
            return self.open_resolved(Some(&query), executor);
        }
        if !self.config.providers.contains_key(&parsed.provider) {
            let mut known: Vec<&str> = self.config.providers.keys().map(String::as_str).collect();
            known.extend(self.config.backends.keys().map(String::as_str));
            return Err(Error::Config(format!(
                "unknown backend \"{}\"; configured: {}",
                parsed.provider,
                known.join(", ")
            )));
        }
        // Resolve one provider only: opening a model must not run another
        // provider's credential command.
        let _env_guard = crate::config_loader::apply_env_specs(&self.env);
        let mut config = (*self.config).clone();
        config.providers.retain(|name, _| name == &parsed.provider);
        crate::config_loader::resolve_config(&mut config).await?;
        let registry = match ModelRegistry::load_async(&config).await {
            Ok(registry) => registry,
            Err(_) => ModelRegistry::load(&config)?,
        };
        // Propagate freshly discovered models back into the shared catalog so
        // the active picker, `lofi.models()`, and subagent overrides see them.
        self.registry
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .merge_from(&registry);
        BackendFactory::from_resolved(config, registry, &self.root, self.env.to_vec())
            .with_skills_dir(self.skills_dir.clone())
            .with_subagent_persistence(self.persist_sessions)
            .open_resolved(Some(&query), executor)
    }

    /// Open a query against the config and registry this factory already
    /// holds. Credentials must be resolved.
    ///
    /// # Errors
    /// Returns a configuration error for an unknown name or a model the
    /// backend does not declare.
    pub fn open_resolved(&self, query: Option<&str>, executor: ToolExecutor) -> Result<BackendRef> {
        let query = self.resolve_query(query)?;
        let parsed = parse_query(&query)?;
        if self.config.backends.contains_key(&parsed.provider) {
            return self.open_external(&parsed, executor);
        }
        if !self.config.providers.contains_key(&parsed.provider) {
            let mut known: Vec<&str> = self.config.providers.keys().map(String::as_str).collect();
            known.extend(self.config.backends.keys().map(String::as_str));
            return Err(Error::Config(format!(
                "unknown backend \"{}\"; configured: {}",
                parsed.provider,
                known.join(", ")
            )));
        }
        let registry = self.registry();
        let (mut model, level) = crate::agent::select_model(&registry, &self.config, Some(&query))?;
        model.thinking = level;
        let provider_cfg = self
            .config
            .providers
            .get(&model.provider)
            .ok_or_else(|| Error::Config(format!("provider not found: {}", model.provider)))?;
        let provider = lofi_providers::open(model.api, provider_cfg)?;
        let auto_continue = registry.auto_continue_policy(&self.config.agent, &model);
        // The prompt is the executor's own: it states the tools and the exec
        // contract, and the backend only carries it.
        let executor = executor.with_skills_dir(self.skills_dir.clone());
        let system_prompt = executor.system_prompt();
        Ok(Arc::new(
            crate::agent::Agent::from_executor(
                provider,
                model,
                executor,
                system_prompt,
                self.config.compaction.clone(),
            )
            .with_retry(self.config.retry.into())
            .with_auto_continue(auto_continue)
            .with_source(crate::backend::NativeSource::new(
                registry,
                (*self.config).clone(),
            )),
        ))
    }

    /// Replace the static catalog with a discovered one. A caller that
    /// already resolved remote models passes them through instead of
    /// rebuilding a catalog that cannot name them.
    #[must_use]
    pub fn with_resolved_registry(self, registry: ModelRegistry) -> Self {
        *self
            .registry
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = registry;
        self
    }

    fn open_external(&self, parsed: &ParsedQuery, executor: ToolExecutor) -> Result<BackendRef> {
        let config = self
            .config
            .backends
            .get(&parsed.provider)
            .ok_or_else(|| Error::Config(format!("unknown backend \"{}\"", parsed.provider)))?;
        let thinking = parsed.level.clone().unwrap_or(ThinkingLevel::Off);
        super::session::check_selection(
            config,
            &parsed.provider,
            &parsed.model,
            &thinking,
            &parsed.tier,
        )?;
        let model = (parsed.model != "default").then_some(parsed.model.clone());
        let system_prompt = executor.system_prompt();
        Ok(Arc::new(
            ExternalBackend::configure(
                parsed.provider.clone(),
                config.clone(),
                self.root.clone(),
                executor,
                model,
                thinking,
                system_prompt,
                self.config.compaction.clone(),
            )
            .with_retry(self.config.retry.into()),
        ))
    }

    /// The same conversation on another backend, or the same selection with
    /// another model inside the same backend.
    ///
    /// # Errors
    /// Propagates resolution and provider construction errors.
    pub async fn switch(&self, current: &BackendRef, query: &str) -> Result<BackendRef> {
        let parsed = parse_query(query)?;
        if parsed.provider == current.name() {
            let next = current.with_selection(query)?;
            return Ok(next.with_tool_executor(self.prepare(next.tool_executor())));
        }
        let executor = self.prepare(current.tool_executor());
        self.open(Some(query), executor).await
    }

    /// Session state for a backend opened by this factory.
    fn prepare(&self, executor: ToolExecutor) -> ToolExecutor {
        executor.with_skills_dir(self.skills_dir.clone())
    }

    /// The query a model name selects, with the level and tier the target will
    /// run at.
    ///
    /// The name may belong to a provider or to a CLI backend. Neither the
    /// caller nor the backend it opens learns which one: this is the one place
    /// that reads both catalogs.
    ///
    /// `parent` supplies what the name leaves unsaid: a backend selection that
    /// names no level runs at the parent's, when that backend accepts it, and at
    /// `off` otherwise; the service tier is inherited when the target declares
    /// it.
    ///
    /// # Errors
    /// Returns a configuration error for an unqualified or unknown name, a
    /// model the target does not declare, or a level it does not accept.
    pub(crate) fn resolve_model_query(
        &self,
        parent: &ModelSelection,
        query: &str,
        level: Option<ThinkingLevel>,
    ) -> Result<String> {
        let parsed = crate::agent::parse_model_query(query)?;
        if let Some(config) = self.config.backends.get(&parsed.provider) {
            // A parent's level carries over only when this backend runs it: a
            // provider model may think at a level no CLI offers.
            let desired = match level.or(parsed.level) {
                Some(level) => level,
                None if super::session::accepts(&parent.thinking) => parent.thinking.clone(),
                None => ThinkingLevel::Off,
            };
            let tier = parsed.tier.unwrap_or(ServiceTier::Auto);
            super::session::check_selection(
                config,
                &parsed.provider,
                &parsed.model,
                &desired,
                &tier,
            )?;
            return Ok(format!(
                "{}/{}:{}@{}",
                parsed.provider,
                parsed.model,
                desired.as_str(),
                tier.as_str()
            ));
        }
        let registry = self.registry();
        let (model, thinking) = crate::agent::select_model_for_subagent(
            &registry,
            &self.config,
            query,
            level,
            &parent.service_tier,
        )?;
        Ok(format!(
            "{}/{}:{}@{}",
            model.provider,
            model.id,
            thinking.as_str(),
            model.service_tier.as_str()
        ))
    }

    /// Resolve the configured default when the caller names no model.
    fn resolve_query(&self, query: Option<&str>) -> Result<String> {
        if let Some(query) = query {
            return Ok(query.to_owned());
        }
        if let Some(default) = self.config.default_model.clone() {
            return Ok(default);
        }
        let registry = self.registry();
        if let Some(name) = self.config.default_provider.clone() {
            if let Some(config) = self.config.backends.get(&name) {
                let model = config
                    .model()
                    .or_else(|| config.models().first().map(String::as_str))
                    .unwrap_or("default");
                return Ok(format!("{name}/{model}"));
            }
            if let Some(choice) = registry.choices().into_iter().find(|c| c.provider == name) {
                return Ok(format!("{name}/{}", choice.id));
            }
        }
        // Config order, not picker order: the first model the user wrote is
        // the default, not the alphabetically first one.
        if let Some(model) = registry.available().first() {
            return Ok(format!("{}/{}", model.provider, model.id));
        }
        if let Some((name, config)) = self.config.backends.first() {
            let model = config
                .model()
                .or_else(|| config.models().first().map(String::as_str))
                .unwrap_or("default");
            return Ok(format!("{name}/{model}"));
        }
        Err(Error::NoModels(crate::agent::NO_MODELS_HINT.to_string()))
    }

    /// The selection a restored session names.
    #[must_use]
    pub fn selection_of(&self, model: &str, thinking: ThinkingLevel) -> ModelSelection {
        ModelSelection {
            model: model.to_owned(),
            thinking,
            service_tier: ServiceTier::Auto,
        }
    }
}

/// Prepared session configuration, model factory, and tool executor shared by
/// interactive startup, no-model fallback, and headless print runs.
pub struct SessionBootstrap {
    pub config: Config,
    pub factory: BackendFactory,
    pub executor: ToolExecutor,
}

impl SessionBootstrap {
    /// Load configuration, build the model catalog, and initialize session
    /// execution state (`SessionTempDir` and auto-mode evaluator) once.
    ///
    /// # Errors
    /// Propagates configuration, temporary-directory, or auto-mode errors.
    pub async fn prepare(
        config_path: Option<PathBuf>,
        root: &Path,
        env: Vec<(String, String)>,
        persist_subagents: bool,
    ) -> Result<Self> {
        let path = config_path.map_or_else(crate::config_loader::user_config_path, Ok)?;
        let config = crate::config_loader::load_backend_config(&path)?;
        let skills_dir = path.parent().map(|dir| dir.join("skills"));
        let factory = BackendFactory::new(config.clone(), root, env.clone())?
            .with_skills_dir(skills_dir.clone())
            .with_subagent_persistence(persist_subagents);
        let executor = factory
            .executor()?
            .with_skills_dir(skills_dir)
            .with_auto_mode_config(&config, &env)
            .await?;
        Ok(Self {
            config,
            factory,
            executor,
        })
    }

    /// Open a backend for `query` constrained by an optional `--backend` flag,
    /// cloning the prepared session executor so fallback attempts reuse it.
    ///
    /// # Errors
    /// Propagates backend constraint, model selection, or credential errors.
    pub async fn open(&self, backend: Option<&str>, query: Option<&str>) -> Result<BackendRef> {
        let constrained = constrain_backend_query(&self.config, backend, query)?;
        self.factory
            .open(constrained.as_deref(), self.executor.clone())
            .await
    }
}

/// Honour `--backend`: constrain `query` to `backend` instead of silently
/// switching providers, or select `backend`'s configured default model.
///
/// # Errors
/// Returns `Error::Config` when a qualified `query` names a different backend.
pub fn constrain_backend_query(
    config: &Config,
    backend: Option<&str>,
    query: Option<&str>,
) -> Result<Option<String>> {
    let Some(name) = backend else {
        return Ok(query.map(str::to_owned));
    };
    if let Some(query) = query {
        return match query.split_once('/') {
            Some((prefix, _)) if prefix == name => Ok(Some(query.to_owned())),
            _ => Err(Error::Config(format!(
                "model \"{query}\" conflicts with --backend \"{name}\""
            ))),
        };
    }
    let model = config
        .backends
        .get(name)
        .and_then(|entry| {
            entry
                .model()
                .or_else(|| entry.models().first().map(String::as_str))
        })
        .unwrap_or("default");
    Ok(Some(format!("{name}/{model}")))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use indexmap::IndexMap;
    use lofi_types::{Api, AutoModelsConfig, Config, FieldMappings, ProviderConfig};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    struct RestoreEnv(Option<std::ffi::OsString>);

    impl Drop for RestoreEnv {
        fn drop(&mut self) {
            match self.0.take() {
                Some(v) => std::env::set_var("XDG_STATE_HOME", v),
                None => std::env::remove_var("XDG_STATE_HOME"),
            }
        }
    }

    fn auto_models_test_config(addr: std::net::SocketAddr) -> Config {
        let mut api_types = IndexMap::new();
        api_types.insert(
            Api::OpenAiCompletions.id().to_string(),
            lofi_types::ApiTypeMapping {
                path: None,
                pricing_field_mappings: None,
            },
        );
        let mut providers = IndexMap::new();
        providers.insert(
            "local".to_string(),
            ProviderConfig {
                api_type: Some(Api::OpenAiCompletions),
                api_types,
                base_url: Some(format!("http://{addr}/v1")),
                pricing_convention: lofi_types::PricingConvention::PerToken,
                pricing_field_mappings: lofi_types::PricingFieldMappings::default(),
                env_name: None,
                api_key: Some("test-key".to_string()),
                headers: None,
                models: IndexMap::new(),
                auto_models: Some(AutoModelsConfig {
                    enabled: true,
                    auth: true,
                    models_url: Some(format!("http://{addr}/v1/models")),
                    path: "data".to_string(),
                    api_type_field: None,
                    api_type_mappings: HashMap::new(),
                    field_mappings: FieldMappings::default(),
                    thinking_levels: Vec::new(),
                    thinking_level: None,
                    service_tiers: Vec::new(),
                    service_tier: None,
                    ttl_seconds: Some(60),
                }),
                no_auth: false,
                response_start_timeout_ms: 90_000,
                body_idle_timeout_ms: 300_000,
                thinking_level: None,
                thinking_levels: Vec::new(),
                service_tier: None,
                service_tiers: Vec::new(),
                auto_continue: lofi_types::AutoContinueConfig::default(),
            },
        );
        Config {
            providers,
            ..Config::default()
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn open_propagates_discovered_auto_models_to_factory_and_models_fn() {
        let _env_lock = crate::state::STATE_ENV_LOCK.lock().unwrap();
        let state_dir = tempfile::tempdir().unwrap();
        let _restore = RestoreEnv(std::env::var_os("XDG_STATE_HOME"));
        std::env::set_var("XDG_STATE_HOME", state_dir.path());

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 2048];
            let _ = sock.read(&mut buf).await;
            let body = r#"{"data":[{"id":"fresh-one","name":"Fresh One"},{"id":"fresh-two","name":"Fresh Two"}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = sock.write_all(response.as_bytes()).await;
        });

        let root = tempfile::tempdir().unwrap();
        let factory =
            BackendFactory::new(auto_models_test_config(addr), root.path(), Vec::new()).unwrap();
        assert!(factory.choices().is_empty(), "cache starts empty");

        // Capture support + lofi.models() before open(), matching SessionBootstrap::prepare.
        let support = factory.support();
        let models_fn = support.models_fn();
        assert_eq!(models_fn()["models"].as_array().unwrap().len(), 0);

        let executor = factory.executor().unwrap();
        let _backend = tokio::time::timeout(
            Duration::from_secs(5),
            factory.open(Some("local/fresh-one"), executor),
        )
        .await
        .unwrap()
        .unwrap();
        let _ = server.await;

        let choice_ids: Vec<String> = factory
            .choices()
            .into_iter()
            .map(|c| format!("{}/{}", c.provider, c.id))
            .collect();
        assert!(
            choice_ids.contains(&"local/fresh-one".to_string())
                && choice_ids.contains(&"local/fresh-two".to_string()),
            "factory choices should contain freshly discovered models: {choice_ids:?}"
        );

        let hook_models: Vec<String> = models_fn()["models"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v["model"].as_str().map(str::to_owned))
            .collect();
        assert!(
            hook_models.contains(&"local/fresh-one".to_string())
                && hook_models.contains(&"local/fresh-two".to_string()),
            "lofi.models() should reflect freshly discovered models: {hook_models:?}"
        );
    }
}
