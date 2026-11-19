use std::path::PathBuf;

use lofi_error::{Error, Result};
use lofi_types::{ServiceTier, ThinkingLevel};

pub mod tui;

#[derive(Debug, Clone)]
pub struct InteractiveOptions {
    pub root: PathBuf,
    pub config_path: Option<PathBuf>,
    pub model: Option<String>,
    pub backend: Option<String>,
    pub env: Vec<String>,
    pub continue_last: bool,
    pub resume: Option<String>,
    pub no_session: bool,
}

impl InteractiveOptions {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            config_path: None,
            model: None,
            backend: None,
            env: Vec::new(),
            continue_last: false,
            resume: None,
            no_session: false,
        }
    }

    #[must_use]
    pub fn with_config_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config_path = Some(path.into());
        self
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    #[must_use]
    pub fn with_envs<'a>(mut self, envs: impl IntoIterator<Item = &'a str>) -> Self {
        self.env = envs.into_iter().map(str::to_string).collect();
        self
    }

    #[must_use]
    pub fn with_continue(mut self) -> Self {
        self.continue_last = true;
        self
    }

    #[must_use]
    pub fn with_resume(mut self, id: impl Into<String>) -> Self {
        self.resume = Some(id.into());
        self
    }

    #[must_use]
    pub fn with_no_session(mut self) -> Self {
        self.no_session = true;
        self
    }
}

/// A missing-models outcome is *not* fatal: the TUI still launches so the
/// user can read transcripts, browse `/resume`, and quit. A hint explaining
/// how to configure a model is shown in the log, and submitting a prompt
/// re-surfaces that hint instead of running.
/// # Errors
/// Propagates [`Error`] from config/session resolution or the terminal
/// session. Model-resolution failures that reduce to "no active model" are
/// swallowed into the no-model TUI mode described above.
#[allow(clippy::too_many_lines)]
pub async fn run_interactive(opts: InteractiveOptions) -> Result<()> {
    let env = lofi_core::config_loader::parse_env_specs(&opts.env)?;
    let session = resolve_session(&opts)?;
    let bootstrap = lofi_core::backend::SessionBootstrap::prepare(
        opts.config_path.clone(),
        &opts.root,
        env,
        !opts.no_session,
    )
    .await?;
    // Restore the model, effort and tier the user last ran with unless
    // --model was given: the resumed session's final turn marker carries
    // them, so a fresh session falls through to the configured default.
    let restored = opts
        .model
        .is_none()
        .then(|| session.last_run_model())
        .flatten();
    let query = opts
        .model
        .clone()
        .or_else(|| restored.as_ref().map(lofi_types::RunModel::query));
    let opened = match bootstrap
        .open(opts.backend.as_deref(), query.as_deref())
        .await
    {
        Ok(backend) => Ok(backend),
        // A restored model that no longer resolves must not stop a resume.
        Err(_) if restored.is_some() => bootstrap.open(opts.backend.as_deref(), None).await,
        Err(error) => Err(error),
    };
    let backend = match opened {
        Ok(backend) => backend,
        Err(error) => return no_model_tui(bootstrap, error, session).await,
    };
    let lofi_core::backend::SessionBootstrap {
        config,
        factory,
        executor: _,
    } = bootstrap;
    let selection = backend.selection();
    let choice = backend.choice();
    // A resumed session shows the label its last turn recorded, so a backend
    // that reports a longer name (a CLI reports the context window it runs)
    // keeps the header the user left.
    let label = match restored.filter(|model| model.provider == backend.name()) {
        Some(model) => format!("{}/{}", backend.name(), model.display_or_selected()),
        None => format!("{}/{}", backend.name(), selection.model),
    };
    let ctx_limit = choice.context_window.filter(|&l| l > 0).unwrap_or(0);
    let system_prompt = backend.system_prompt().to_string();
    let switcher = tui::ModelSwitcher::new(factory).with_subagent_persistence(!opts.no_session);
    Box::pin(tui::run(
        Some(backend),
        label,
        selection.thinking,
        selection.service_tier,
        config.ui.theme,
        session,
        None,
        ctx_limit,
        config.compaction,
        Some(switcher),
        system_prompt,
    ))
    .await
}

/// A missing-model launch is not fatal: the TUI still opens so the user can
/// read transcripts, browse /resume and quit.
async fn no_model_tui(
    bootstrap: lofi_core::backend::SessionBootstrap,
    error: Error,
    session: tui::SessionConfig,
) -> Result<()> {
    let Error::NoModels(hint) = error else {
        return Err(error);
    };
    let lofi_core::backend::SessionBootstrap {
        config,
        factory,
        executor,
    } = bootstrap;
    let switcher = tui::ModelSwitcher::new(factory).with_tool_executor(executor);
    Box::pin(tui::run(
        None,
        "(no model)".to_string(),
        ThinkingLevel::Off,
        ServiceTier::Auto,
        config.ui.theme,
        session,
        Some(hint),
        0,
        config.compaction,
        Some(switcher),
        String::new(),
    ))
    .await
}

/// Independent of model resolution so the no-model launch path still gets a
/// session (and thus `/resume`, transcript browsing, etc.).
/// # Errors
/// Returns `Error::State` when `--resume <id>` matches no session, or
/// propagates `Error::Io` from the store.
fn resolve_session(opts: &InteractiveOptions) -> Result<tui::SessionConfig> {
    if opts.no_session {
        return Ok(tui::SessionConfig::ephemeral(opts.root.clone()));
    }
    // The UI never opens a session store: `--resume`/`--continue` resolve
    // entirely inside core, returning a sink (which owns the store) plus the
    // selected cursor and snapshot.
    if let Some(id) = &opts.resume {
        let (sink, cursor, snapshot) =
            lofi_core::session::sink::SessionSink::resume_by_id(&opts.root, id)?;
        return Ok(tui::SessionConfig::resumed(
            sink,
            cursor,
            snapshot.index,
            snapshot.file_size,
            snapshot.history_start,
            snapshot.contiguous,
            opts.root.clone(),
        ));
    }
    if opts.continue_last {
        if let Some((sink, cursor, snapshot)) =
            lofi_core::session::sink::SessionSink::resume_last(&opts.root)?
        {
            return Ok(tui::SessionConfig::resumed(
                sink,
                cursor,
                snapshot.index,
                snapshot.file_size,
                snapshot.history_start,
                snapshot.contiguous,
                opts.root.clone(),
            ));
        }
    }
    Ok(tui::SessionConfig::fresh(
        lofi_core::session::sink::SessionSink::open(&opts.root)?,
        opts.root.clone(),
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn dual_provider_config(dir: &std::path::Path) -> PathBuf {
        let path = dir.join("lofi.toml");
        std::fs::write(
            &path,
            r#"
default_model = "alpha/a1"

[providers.alpha]
api_type = "openai-completions"
api_key = "alpha-key"

[providers.alpha.models]
a1 = {}

[providers.beta]
api_type = "openai-completions"
api_key = "beta-key"

[providers.beta.models]
b1 = {}
"#,
        )
        .unwrap();
        path
    }

    #[tokio::test]
    async fn startup_model_flag_honored() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = dual_provider_config(tmp.path());
        let opts = InteractiveOptions::new(tmp.path())
            .with_config_path(&cfg)
            .with_model("beta/b1");
        let env = lofi_core::config_loader::parse_env_specs(&opts.env).unwrap();
        let bootstrap = lofi_core::backend::SessionBootstrap::prepare(
            opts.config_path.clone(),
            &opts.root,
            env,
            !opts.no_session,
        )
        .await
        .unwrap();
        let backend = bootstrap
            .open(opts.backend.as_deref(), opts.model.as_deref())
            .await
            .unwrap();
        assert_eq!(backend.name(), "beta");
        assert_eq!(backend.selection().model, "b1");
    }

    #[tokio::test]
    async fn startup_default_model_when_no_flag() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = dual_provider_config(tmp.path());
        let opts = InteractiveOptions::new(tmp.path()).with_config_path(&cfg);
        let env = lofi_core::config_loader::parse_env_specs(&opts.env).unwrap();
        let bootstrap = lofi_core::backend::SessionBootstrap::prepare(
            opts.config_path.clone(),
            &opts.root,
            env,
            !opts.no_session,
        )
        .await
        .unwrap();
        let backend = bootstrap.open(opts.backend.as_deref(), None).await.unwrap();
        assert_eq!(backend.name(), "alpha");
        assert_eq!(backend.selection().model, "a1");
    }
}
