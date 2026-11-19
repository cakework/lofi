use std::io::Write;
use std::path::PathBuf;

use lofi_core::AgentEvent;
use lofi_error::{Error, Result};

use crate::print_summary::PrintSummary;

/// Where a print-mode run keeps its transcript.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) enum PrintSession {
    /// Write no transcript; the default.
    #[default]
    None,
    /// Start a new session for this run.
    Fresh,
    /// Continue the most recent session of this workspace.
    Continue,
    /// Resume the session whose id starts with this prefix.
    Resume(String),
}

#[derive(Debug, Clone)]
pub(crate) struct PrintOptions {
    pub(crate) prompt: String,
    pub(crate) root: PathBuf,
    pub(crate) model: Option<String>,
    pub(crate) backend: Option<String>,
    pub(crate) env: Vec<String>,
    /// Where to write one JSON record of the run; `None` writes nothing.
    pub(crate) summary: Option<PathBuf>,
    /// Where the run keeps its transcript, so a killed run can be resumed.
    pub(crate) session: PrintSession,
}

impl PrintOptions {
    #[must_use]
    pub(crate) fn new(prompt: impl Into<String>, root: impl Into<PathBuf>) -> Self {
        Self {
            prompt: prompt.into(),
            root: root.into(),
            model: None,
            backend: None,
            env: Vec::new(),
            summary: None,
            session: PrintSession::None,
        }
    }

    #[must_use]
    pub(crate) fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    #[must_use]
    pub(crate) fn with_envs<'a>(mut self, envs: impl IntoIterator<Item = &'a str>) -> Self {
        self.env = envs.into_iter().map(str::to_string).collect();
        self
    }

    /// Write one JSON record of the run to `path` when it ends.
    #[must_use]
    pub(crate) fn with_summary(mut self, path: impl Into<PathBuf>) -> Self {
        self.summary = Some(path.into());
        self
    }

    /// Start a new session for this run.
    #[must_use]
    pub(crate) fn with_session(mut self) -> Self {
        self.session = PrintSession::Fresh;
        self
    }

    /// Continue the most recent session of this workspace.
    #[must_use]
    pub(crate) fn with_continue(mut self) -> Self {
        self.session = PrintSession::Continue;
        self
    }

    /// Resume the session whose id starts with `id`.
    #[must_use]
    pub(crate) fn with_resume(mut self, id: impl Into<String>) -> Self {
        self.session = PrintSession::Resume(id.into());
        self
    }
}

/// Drives one turn of the selected backend and writes its events to stdio.
/// - `AgentEvent::Text` deltas go to stdout;
/// - errors and tool markers go to stderr, so stdout stays pipeable.
/// # Errors
/// Propagates config, model resolution and turn errors. A run that fails
/// still writes its summary record.
pub(crate) async fn run_print(opts: PrintOptions) -> Result<()> {
    let started = std::time::Instant::now();
    // Open the record before the run: a path that cannot be written must
    // fail before the run costs anything.
    let mut file = match &opts.summary {
        Some(path) => Some(std::fs::File::create(path).map_err(|error| {
            Error::State(format!("open the run summary {}: {error}", path.display()))
        })?),
        None => None,
    };
    // The consumer owns the only view of the run's events, and a setup
    // failure has no events at all, so the record lives outside the turn.
    // Both run on this task, so a plain cell suffices.
    let summary = std::cell::RefCell::new(PrintSummary::default());
    let result = run_print_turn(&opts, &summary).await;
    if let Some(file) = &mut file {
        let wall_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let mut record = summary.into_inner();
        // The record is written for a failed run too: its spend and stop
        // reason are what the caller most needs.
        if let Err(error) = &result {
            record.note_error(&error.to_string());
        }
        if let Err(error) = record.write(file, wall_ms) {
            // A write failure must not hide the run's own error.
            if result.is_ok() {
                return Err(error);
            }
            eprintln!("error: {error}");
        }
    }
    result
}

/// The transcript one print run writes, and the history it resumed.
struct PrintTranscript {
    /// Holds the store and the session file open for the whole run.
    _sink: lofi_core::session::sink::SessionSink,
    cursor: lofi_core::session::store::SessionCursor,
    /// Messages a resumed session already holds, oldest first.
    history: Vec<lofi_types::Message>,
}

impl PrintTranscript {
    fn cursor(&self) -> &lofi_core::session::store::SessionCursor {
        &self.cursor
    }

    fn id(&self) -> String {
        self.cursor.id()
    }

    /// Hand the resumed history to the turn exactly once.
    fn take_history(&mut self) -> Vec<lofi_types::Message> {
        std::mem::take(&mut self.history)
    }
}

/// Open the transcript a print run asked for. A run without one writes
/// nothing, the way every print run used to behave.
fn open_transcript(
    opts: &PrintOptions,
    model: &lofi_types::RunModel,
    system_prompt: &str,
) -> Result<Option<PrintTranscript>> {
    use lofi_core::session::{replay, sink::SessionSink};
    match &opts.session {
        PrintSession::None => Ok(None),
        PrintSession::Resume(id) => {
            let (sink, cursor, snapshot) = SessionSink::resume_by_id(&opts.root, id)?;
            let history =
                replay::history_from_index(&cursor, &snapshot.index, snapshot.history_start)?;
            Ok(Some(PrintTranscript {
                _sink: sink,
                cursor,
                history,
            }))
        }
        PrintSession::Continue => match SessionSink::resume_last(&opts.root)? {
            Some((sink, cursor, snapshot)) => {
                let history =
                    replay::history_from_index(&cursor, &snapshot.index, snapshot.history_start)?;
                Ok(Some(PrintTranscript {
                    _sink: sink,
                    cursor,
                    history,
                }))
            }
            // --continue with nothing to continue starts a session, the way
            // it starts the TUI on a fresh transcript.
            None => Ok(Some(fresh_transcript(&opts.root, model, system_prompt)?)),
        },
        PrintSession::Fresh => Ok(Some(fresh_transcript(&opts.root, model, system_prompt)?)),
    }
}

fn fresh_transcript(
    root: &std::path::Path,
    model: &lofi_types::RunModel,
    system_prompt: &str,
) -> Result<PrintTranscript> {
    let mut sink = lofi_core::session::sink::SessionSink::open(root)?;
    let cursor = sink.cursor_or_create(model, system_prompt)?;
    Ok(PrintTranscript {
        _sink: sink,
        cursor,
        history: Vec::new(),
    })
}

/// Runs one non-interactive turn and folds its events into `summary`.
#[allow(clippy::too_many_lines)]
async fn run_print_turn(
    opts: &PrintOptions,
    summary: &std::cell::RefCell<PrintSummary>,
) -> Result<()> {
    let env = lofi_core::config_loader::parse_env_specs(&opts.env)?;
    // A subagent started by this run keeps no transcript of its own: the
    // parent's record carries its spend, and no one resumes a child.
    let bootstrap =
        lofi_core::backend::SessionBootstrap::prepare(None, &opts.root, env, false).await?;
    let agent = bootstrap
        .open(opts.backend.as_deref(), opts.model.as_deref())
        .await?;
    // The record names the model the agent settled on even when the run
    // fails before its first round reports one.
    summary.borrow_mut().set_model(agent.run_model());

    // A run in a session keeps a transcript, so a driver that kills it and
    // resumes the id gets every completed round back.
    let mut transcript = open_transcript(opts, &agent.run_model(), agent.system_prompt())?;
    let mut messages = transcript
        .as_mut()
        .map(PrintTranscript::take_history)
        .unwrap_or_default();
    if let Some(transcript) = &transcript {
        // The driver that kills this run needs the id to resume it, and the
        // record carries the same id for a driver that reads it afterwards.
        let id = transcript.id();
        eprintln!("session: {id}");
        summary.borrow_mut().set_session(id);
    }

    let (tx, mut rx) = tokio::sync::mpsc::channel::<AgentEvent>(64);

    let prompt = opts.prompt.clone();
    let agent_run = agent.run_turn(
        &mut messages,
        lofi_core::TurnRequest {
            input: lofi_core::TurnInput::Prompt {
                text: prompt,
                kind: lofi_types::PromptKind::User,
                attachments: Vec::new(),
            },
            notices: Vec::new(),
        },
        tx,
        transcript.as_ref().map(PrintTranscript::cursor),
        lofi_core::RunControls {
            cancel: None,
            preempt: None,
        },
    );

    // The agent future is not `Send` (the QuickJS `AsyncContext` is not
    // `Sync`), so it cannot be `tokio::spawn`'d. Drive it concurrently with
    // the event consumer on the same task: when the agent finishes it drops
    // `tx`, `rx.recv()` returns `None`, and the consumer drains.
    let consumer = async {
        let mut stdout = std::io::stdout().lock();
        let mut stderr = std::io::stderr().lock();
        let mut context_pressure = false;
        loop {
            let Some(ev) = rx.recv().await else {
                break;
            };
            summary.borrow_mut().note(&ev);
            let write_res: std::io::Result<()> = match ev {
                AgentEvent::Text(delta) => stdout.write_all(delta.as_bytes()),
                AgentEvent::Thinking(_)
                | AgentEvent::ThinkingEnd { .. }
                | AgentEvent::ToolEnd { .. }
                | AgentEvent::NativeToolStart { .. }
                | AgentEvent::NativeToolEnd { .. }
                | AgentEvent::ToolInputDelta { .. }
                | AgentEvent::RoundCommitted { .. }
                | AgentEvent::TurnCommitted { .. }
                | AgentEvent::RoundUsage { .. }
                | AgentEvent::TurnUsage { .. }
                | AgentEvent::SubagentUsage { .. }
                | AgentEvent::ContextReported { .. }
                | AgentEvent::ModelReported { .. }
                | AgentEvent::RetryEnd { .. }
                | AgentEvent::RunStart
                | AgentEvent::Prompt { .. }
                | AgentEvent::TurnContinue
                | AgentEvent::Compaction { .. }
                | AgentEvent::UserShellStart { .. }
                | AgentEvent::UserShellDelta(_)
                | AgentEvent::UserShell { .. } => Ok(()),
                AgentEvent::ContextPressure { resume, .. } => {
                    if resume {
                        context_pressure = true;
                        let _ = stdout.write_all(b"\n");
                        writeln!(
                            stderr,
                            "error: context limit reached; use the interactive TUI to compact and continue"
                        )
                    } else {
                        // The turn already stopped on its own: a one-shot
                        // run has nothing to continue.
                        Ok(())
                    }
                }
                AgentEvent::RetryStart {
                    attempt,
                    max_attempts,
                    delay_ms,
                    error,
                } => writeln!(
                    stderr,
                    "warning: provider retry {attempt}/{max_attempts} in {delay_ms}ms: {error}"
                ),
                AgentEvent::Notice(msg) => writeln!(stderr, "warning: {msg}"),
                AgentEvent::Error(msg) => writeln!(stderr, "error: {msg}"),
                AgentEvent::ToolStart { name, .. } => writeln!(stderr, "[{name}]"),
                AgentEvent::ToolInput { code, .. } => writeln!(stderr, "{code}"),
                AgentEvent::TurnEnd { .. } => stdout.write_all(b"\n"),
                AgentEvent::TurnFailed { error, .. } => {
                    let _ = stdout.write_all(b"\n");
                    writeln!(stderr, "error: {error}")
                }
                AgentEvent::TurnCancelled { .. } => {
                    let _ = stdout.write_all(b"\n");
                    writeln!(stderr, "Operation aborted")
                }
            };
            if let Err(e) = write_res {
                // Close the receive side so agent sends fail immediately
                // instead of filling the bounded channel and blocking
                // forever on a consumer whose output pipe is gone.
                rx.close();
                return Err(Error::Io(e));
            }
        }
        stdout.flush()?;
        stderr.flush()?;
        if context_pressure {
            return Err(Error::State(
                "context limit reached in non-interactive mode".to_string(),
            ));
        }
        Ok::<(), Error>(())
    };

    // Drive agent and consumer concurrently. The agent future is not `Send`,
    // so both run on this task. If the consumer fails (output pipe gone),
    // return its error immediately without waiting for the agent to time
    // out; the agent future is dropped and its in-flight tool (if any) is
    // left to the `spawn_blocking` cap. On normal agent completion, drain
    // the consumer so buffered events flush before returning.
    tokio::pin!(agent_run, consumer);
    // Neither branch returns early: a failed run still writes its record.
    let result = tokio::select! {
        biased;
        consumer_res = &mut consumer => match consumer_res {
            Ok(()) => (&mut agent_run).await.map(|_| ()),
            Err(e) => Err(e),
        },
        agent_res = &mut agent_run => {
            // Agent finished and dropped tx; drain remaining events + flush.
            // Propagate the consumer result so a failed terminal write or
            // flush is not silently swallowed.
            match (&mut consumer).await {
                Ok(()) => agent_res.map(|_| ()),
                // A run that failed after the drain is reported with its own
                // error: the consumer's latch states the stop that came first,
                // and would hide the failure that ended the run behind it.
                Err(e) => match agent_res {
                    Err(error) => Err(error),
                    Ok(_) => Err(e),
                },
            }
        }
    };
    result
}
