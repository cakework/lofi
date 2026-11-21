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
        let tx = spawn_worker(WorkerInit {
            name: self.name.clone(),
            config: self.config.clone(),
            root: self.root.clone(),
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
        // Build the turn executor per turn so updates via with_tool_executor
        // and with_confirm_tx reach an already-running worker and bridge.
        let executor = self
            .executor
            .clone()
            .with_subagent_hook(self.subagent_hook());
        Ok(Box::new(ClaudeEngine::open(
            worker,
            &self.name,
            self.model.clone(),
            self.thinking.clone(),
            executor,
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
        // One-shot evaluation runs with thinking off, matching Agent::complete.
        super::oneshot::complete(
            command,
            &self.root,
            self.model.as_deref(),
            &lofi_types::ThinkingLevel::Off,
            &request,
        )
        .await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    use serde_json::{json, Value};

    use super::*;
    use crate::agent::{ConfirmRequest, ConfirmResponse, RunControls, TurnInput, TurnRequest};

    fn write_fake_claude_script(dir: &std::path::Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let script = dir.join("fake-claude");
        let exe = std::env::current_exe().unwrap();
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf '%s\\0' \"$@\" > claude-args\nLOFI_CORE_FAKE_CLAUDE=1 exec '{}' --exact backend::session::tests::fake_cli_child --nocapture 3>&1 1>/dev/null\n",
                exe.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        script
    }

    fn read_json_line(reader: &mut impl BufRead) -> Value {
        let mut line = String::new();
        assert!(reader.read_line(&mut line).unwrap() > 0, "unexpected EOF");
        serde_json::from_str(&line).unwrap()
    }

    fn send_json_line(writer: &mut impl Write, value: &Value) {
        writeln!(writer, "{value}").unwrap();
        writer.flush().unwrap();
    }

    #[test]
    fn fake_cli_child() {
        if std::env::var_os("LOFI_CORE_FAKE_CLAUDE").is_none() {
            return;
        }
        let args = std::fs::read_to_string("claude-args").unwrap();
        let args: Vec<_> = args.trim_end_matches('\0').split('\0').collect();
        let flag_val = |flag| args[args.iter().position(|a| *a == flag).unwrap() + 1];
        let session_arg = args
            .iter()
            .find(|a| a.starts_with("--session-id=") || a.starts_with("--resume="))
            .unwrap();
        let (_, id) = session_arg.split_once('=').unwrap();
        std::fs::write("child-pid", std::process::id().to_string()).unwrap();

        let mcp_cfg: Value =
            serde_json::from_slice(&std::fs::read(flag_val("--mcp-config")).unwrap()).unwrap();
        let env = &mcp_cfg["mcpServers"]["lofi"]["env"];
        let socket = env["LOFI_EXEC_SOCKET"].as_str().unwrap();
        let token = env["LOFI_EXEC_TOKEN"].as_str().unwrap();

        let stream = UnixStream::connect(socket).unwrap();
        let mut bridge_in = stream.try_clone().unwrap();
        let mut bridge_out = BufReader::new(stream);
        send_json_line(
            &mut bridge_in,
            &json!({"jsonrpc":"2.0","method":"_lofi/connect","params":{"token":token}}),
        );
        send_json_line(
            &mut bridge_in,
            &json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        );
        let _ = read_json_line(&mut bridge_out);
        send_json_line(
            &mut bridge_in,
            &json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        );

        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .open("/dev/fd/3")
            .unwrap();
        let stdin = std::io::stdin();
        let mut input = stdin.lock();
        let mut next_id = 2u64;
        let mut initialized = false;
        loop {
            let mut line = String::new();
            if input.read_line(&mut line).unwrap() == 0 {
                break;
            }
            let prompt: Value = serde_json::from_str(&line).unwrap();
            if !initialized {
                send_json_line(
                    &mut output,
                    &json!({
                        "type": "system",
                        "subtype": "init",
                        "session_id": id,
                        "tools": ["mcp__lofi__exec"],
                        "mcp_servers": [{"name": "lofi", "status": "connected"}],
                        "model": "test-model",
                    }),
                );
                initialized = true;
            }
            let text = prompt["message"]["content"].as_str().unwrap_or("");
            if text == "run bash" {
                send_json_line(
                    &mut bridge_in,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": next_id,
                        "method": "tools/call",
                        "params": {
                            "name": "exec",
                            "arguments": {
                                "code": r#"return await lofi.bash({cmd:"echo ok",timeoutMs:2000})"#,
                                "description": "test exec"
                            }
                        }
                    }),
                );
                next_id += 1;
                let resp = read_json_line(&mut bridge_out);
                assert!(
                    resp.get("error").is_none(),
                    "unexpected bridge error: {resp}"
                );
            }
            send_json_line(
                &mut output,
                &json!({
                    "type": "result",
                    "subtype": "success",
                    "is_error": false,
                    "session_id": id,
                    "result": "done",
                }),
            );
        }
    }

    #[tokio::test]
    async fn with_confirm_tx_updates_a_live_worker_executor() {
        tokio::time::timeout(std::time::Duration::from_secs(15), async {
            let dir = tempfile::tempdir().unwrap();
            let script = write_fake_claude_script(dir.path());
            let store = crate::session::store::SessionStore::new(dir.path().join("sessions"));
            let cursor = store
                .create_cursor(dir.path(), &"claude/test-model".into())
                .unwrap();
            let (first_send, mut first_recv) =
                tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
            let executor = ToolExecutor::in_memory(dir.path()).with_confirm_tx(first_send);
            let backend: BackendRef = Arc::new(ExternalBackend::configure(
                "claude".into(),
                lofi_types::BackendConfig::ClaudeCode {
                    command: script.display().to_string(),
                    model: Some("test-model".into()),
                    models: vec!["test-model".into()],
                    context_window: Some(200_000),
                    frame_idle_timeout_ms: Some(10_000),
                },
                dir.path().to_path_buf(),
                executor,
                Some("test-model".into()),
                lofi_types::ThinkingLevel::Off,
                "You are lofi".into(),
                lofi_types::CompactionConfig::default(),
            ));

            // Turn 1 starts the live worker and child process.
            let mut messages = Vec::new();
            let (ev_tx, mut ev_rx) = tokio::sync::mpsc::channel(64);
            let drain = tokio::spawn(async move { while ev_rx.recv().await.is_some() {} });
            backend
                .run_turn(
                    &mut messages,
                    TurnRequest {
                        input: TurnInput::Prompt {
                            text: "warmup".into(),
                            kind: lofi_types::PromptKind::User,
                            attachments: Vec::new(),
                        },
                        notices: Vec::new(),
                    },
                    ev_tx,
                    Some(&cursor),
                    RunControls::default(),
                )
                .await
                .unwrap();
            let _ = drain.await;
            let pid_before = std::fs::read_to_string(dir.path().join("child-pid")).unwrap();

            // Update confirm_tx after the worker is already live.
            let (second_send, mut second_recv) =
                tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
            let updated = backend.with_confirm_tx(second_send);

            let saw_on_second = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let saw_flag = saw_on_second.clone();
            let approver = tokio::spawn(async move {
                tokio::select! {
                    Some(req) = second_recv.recv() => {
                        saw_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                        let _ = req.respond.send(ConfirmResponse::Allow);
                    }
                    Some(req) = first_recv.recv() => {
                        let _ = req.respond.send(ConfirmResponse::Allow);
                    }
                }
            });

            let (ev_tx, mut ev_rx) = tokio::sync::mpsc::channel(64);
            let drain = tokio::spawn(async move { while ev_rx.recv().await.is_some() {} });
            updated
                .run_turn(
                    &mut messages,
                    TurnRequest {
                        input: TurnInput::Prompt {
                            text: "run bash".into(),
                            kind: lofi_types::PromptKind::User,
                            attachments: Vec::new(),
                        },
                        notices: Vec::new(),
                    },
                    ev_tx,
                    Some(&cursor),
                    RunControls::default(),
                )
                .await
                .unwrap();
            let _ = drain.await;
            approver.abort();

            assert!(
                saw_on_second.load(std::sync::atomic::Ordering::SeqCst),
                "confirm request went to the stale ToolExecutor instead of the updated one"
            );
            let pid_after = std::fs::read_to_string(dir.path().join("child-pid")).unwrap();
            assert_eq!(pid_before, pid_after, "child should be reused in place");
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn external_backend_complete_forces_thinking_off() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("fake-eval");
        std::fs::write(
            &script,
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > eval-args\nread -r _\nprintf '%s\\n' '{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":false,\"result\":\"ok\"}'\n",
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();

        let backend = ExternalBackend::configure(
            "claude".into(),
            lofi_types::BackendConfig::ClaudeCode {
                command: script.display().to_string(),
                model: Some("test-model".into()),
                models: vec!["test-model".into()],
                context_window: Some(200_000),
                frame_idle_timeout_ms: Some(10_000),
            },
            dir.path().to_path_buf(),
            ToolExecutor::in_memory(dir.path()),
            Some("test-model".into()),
            lofi_types::ThinkingLevel::High,
            "You are lofi".into(),
            lofi_types::CompactionConfig::default(),
        );

        let completion = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            backend.complete(CompletionRequest {
                system: "evaluator prompt".into(),
                prompt: "echo hi".into(),
                max_tokens: Some(64),
            }),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(completion.text, "ok");

        let args = std::fs::read_to_string(dir.path().join("eval-args")).unwrap();
        assert!(
            !args.contains("--effort="),
            "one-shot evaluator must not inherit session thinking effort: {args}"
        );
    }
}
