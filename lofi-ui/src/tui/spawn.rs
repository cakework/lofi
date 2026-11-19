#![allow(clippy::wildcard_imports)]

use super::*;

pub(super) fn parse_user_shell(prompt: &str) -> Option<(String, bool)> {
    if let Some(command) = prompt.strip_prefix("!!") {
        let command = command.trim_start();
        return (!command.is_empty()).then(|| (command.to_string(), true));
    }
    let command = prompt.strip_prefix('!')?.trim_start();
    (!command.is_empty()).then(|| (command.to_string(), false))
}

pub(super) fn persist_unsent_prompts(app: &mut App) {
    if app.prompt_queue.is_empty() {
        return;
    }
    let run_model = app.run_model();
    let system_prompt = app.system_prompt.clone();
    for queued in std::mem::take(&mut app.prompt_queue) {
        if queued.text.trim().is_empty() {
            continue;
        }
        let Some(sink) = app.session.sink_mut() else {
            continue;
        };
        let recorded = sink
            .cursor_or_create(&run_model, &system_prompt)
            .and_then(|cursor| cursor.record_unrun_prompt(&queued.text, queued.kind));
        if let Err(error) = recorded {
            app.notify(
                NotifyKind::Error,
                format!("could not save queued prompt: {error}"),
            );
        }
    }
    app.session.refresh_cursor();
}

pub(super) fn finish_user_shell(
    app: &mut App,
    result: lofi_core::UserShellResult,
    exclude_from_context: bool,
) {
    if !exclude_from_context {
        if let Err(error) = app.lifecycle.push_message(Message {
            origin: None,
            role: Role::User,
            blocks: vec![ContentBlock::Text {
                text: result.context_text(),
            }],
            kind: PromptKind::default(),
        }) {
            app.notify(NotifyKind::Error, format!("update agent history: {error}"));
        }
    }
    // Recording the command is a core-owned session write — the UI hands the
    // finished result to the sink rather than assembling/appending an event.
    let run_model = app.run_model();
    let mut byte_range = None;
    if let Some(sink) = app.session.sink_mut() {
        match sink.record_user_shell(
            &result,
            exclude_from_context,
            &run_model,
            &app.system_prompt,
        ) {
            Ok(range) => byte_range = Some(range),
            Err(error) => app.notify(
                NotifyKind::Error,
                format!("transcript write failed: {error}"),
            ),
        }
    }
    app.session.refresh_cursor();
    app.apply_event(AgentEvent::UserShell {
        command: result.command,
        output: result.output,
        exit_code: result.exit_code,
        signal: result.signal,
        duration_ms: result.duration_ms,
        truncated: result.truncated,
        cancelled: result.cancelled,
        exclude_from_context,
    });
    if let (Some(range), Some(slot)) = (byte_range, app.transcript.turn_byte_ranges.last_mut()) {
        *slot = Some(range);
    }
}

fn install_run(app: &mut App, current_run: &mut Option<RunHandle>, run: RunHandle) {
    *current_run = Some(run);
    app.begin_active_run();
    // A scrolled-up view must follow the turn this submit starts.
    app.pin_to_latest();
}

pub(super) fn spawn_agent_run(
    app: &mut App,
    current_run: &mut Option<RunHandle>,
    agent: &BackendRef,
    prompt: Option<String>,
    prompt_kind: lofi_types::PromptKind,
) {
    let run_model = app.run_model();
    let system_prompt = app.system_prompt.clone();
    let mut session_sink = app.session.sink.clone();
    let session_cursor = Arc::new(std::sync::Mutex::new(app.session.cursor.clone()));
    let worker_cursor = Arc::clone(&session_cursor);
    // Mirror the lineage-birth pin into the live history so the first request
    // carries the system prompt; resume has already rebuilt history from the
    // log, so seed_system is a no-op there.
    let _ = app.lifecycle.seed_system(&app.system_prompt);
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    let lifecycle = app.lifecycle.clone();
    let agent = agent.clone();
    let err_tx = tx.clone();
    let cancel = lofi_core::new_cancel_flag();
    let cancel_clone = cancel.clone();
    let preempt = Arc::new(AtomicBool::new(false));
    let preempt_clone = preempt.clone();
    let continuation = prompt.is_none();
    let prompt = prompt.unwrap_or_default();
    // Handover notices (e.g. stale job ids) ride the next agent run so the
    // model sees them inline before the user's text. Continuations skip them:
    // mid-run context is already shaped. The app keeps its copy until this
    // run publishes that notice, which proves it is recorded.
    let startup_notices = if continuation {
        Vec::new()
    } else {
        app.startup_notices.clone()
    };
    app.notices_in_flight = !startup_notices.is_empty();
    // Pre-render only when the submitted prompt is the first durable row.
    // Otherwise the core event stream must establish notice-before-user
    // ordering without deduplicating the wrong last row.
    if !continuation && startup_notices.is_empty() {
        app.begin_prompt_turn(prompt.clone(), prompt_kind);
    }
    // The agent run loop performs durable transcript syncs. Its own runtime
    // keeps those blocking writes off the current-thread TUI runtime.
    let handle = tokio::task::spawn_blocking(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build();
        let result = runtime.map_err(|error| format!("start agent runtime: {error}"));
        let result = result.and_then(|runtime| {
            runtime.block_on(async move {
                let cursor = session_sink
                    .as_mut()
                    .map(|sink| sink.cursor_or_create(&run_model, &system_prompt))
                    .transpose()
                    .map_err(|error| format!("create session: {error}"))?;
                if let Some(cursor) = cursor.as_ref() {
                    *worker_cursor
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(cursor.clone());
                }
                let mut messages = lifecycle.lease_history();
                let result = std::panic::AssertUnwindSafe(agent.run_turn(
                    &mut messages,
                    lofi_core::TurnRequest {
                        input: if continuation {
                            lofi_core::TurnInput::Continue
                        } else {
                            lofi_core::TurnInput::Prompt {
                                text: prompt,
                                kind: prompt_kind,
                                attachments: Vec::new(),
                            }
                        },
                        notices: startup_notices,
                    },
                    tx,
                    cursor.as_ref(),
                    lofi_core::RunControls {
                        cancel: Some(cancel_clone),
                        preempt: Some(preempt_clone),
                    },
                ))
                .catch_unwind()
                .await;
                drop(messages);
                match result {
                    Ok(Ok(_)) => Ok(()),
                    Ok(Err(error)) => Err(error.to_string()),
                    Err(payload) => Err(format!(
                        "agent task panicked: {}",
                        panic_message(payload.as_ref())
                    )),
                }
            })
        });
        if let Err(error) = result {
            let _ = err_tx.blocking_send(AgentEvent::Error(error));
        }
    });
    install_run(
        app,
        current_run,
        RunHandle {
            handle,
            rx,
            cancel,
            preempt,
            session_cursor,
            user_shell: None,
        },
    );
}

pub(super) fn spawn_user_shell(
    app: &mut App,
    current_run: &mut Option<RunHandle>,
    command: String,
    exclude_from_context: bool,
) {
    // Ensure a session exists (core-owned) so the finished command can be
    // recorded; mirror the cursor for reads.
    let run_model = app.run_model();
    if let Some(sink) = app.session.sink_mut() {
        let _ = sink.cursor_or_create(&run_model, &app.system_prompt);
    }
    app.session.refresh_cursor();
    // Deltas can arrive per pipe drain; 1 would turn every chunk into a
    // round trip against the event loop.
    let (tx, rx) = tokio::sync::mpsc::channel(256);
    let cwd = app.session.cwd.clone();
    let command_for_run = command.clone();
    let cancel = lofi_core::new_cancel_flag();
    let cancel_for_run = Arc::clone(&cancel);
    let handle = tokio::task::spawn_local(async move {
        let _ = tx
            .send(AgentEvent::UserShellStart {
                command: command_for_run.clone(),
                exclude_from_context,
            })
            .await;
        let (delta_tx, mut delta_rx) = tokio::sync::mpsc::channel::<String>(256);
        let delta_forward = tx.clone();
        let forwarder = tokio::task::spawn_local(async move {
            while let Some(delta) = delta_rx.recv().await {
                if delta_forward
                    .send(AgentEvent::UserShellDelta(delta))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });
        let event = match Box::pin(lofi_core::run_user_shell_command(
            &cwd,
            command_for_run.clone(),
            cancel_for_run,
            Some(delta_tx),
        ))
        .await
        {
            Ok(result) => AgentEvent::UserShell {
                command: result.command,
                output: result.output,
                exit_code: result.exit_code,
                signal: result.signal,
                duration_ms: result.duration_ms,
                truncated: result.truncated,
                cancelled: result.cancelled,
                exclude_from_context,
            },
            Err(error) => AgentEvent::UserShell {
                command: command_for_run,
                output: error.to_string(),
                exit_code: Some(1),
                signal: None,
                duration_ms: 0,
                truncated: false,
                cancelled: false,
                exclude_from_context,
            },
        };
        // Drain every queued delta before the finishing event so the block
        // finalizes the accumulated stream, not the other way around.
        let _ = forwarder.await;
        let _ = tx.send(event).await;
    });
    install_run(
        app,
        current_run,
        RunHandle {
            handle,
            rx,
            cancel,
            preempt: Arc::new(AtomicBool::new(false)),
            session_cursor: Arc::new(std::sync::Mutex::new(app.session.cursor.clone())),
            user_shell: Some((command, exclude_from_context)),
        },
    );
}

pub(super) fn spawn_prompt(
    app: &mut App,
    agent: Option<&BackendRef>,
    current_run: &mut Option<RunHandle>,
    prompt: String,
    kind: lofi_types::PromptKind,
) {
    if let Some((command, exclude_from_context)) = parse_user_shell(&prompt) {
        spawn_user_shell(app, current_run, command, exclude_from_context);
        return;
    }
    let Some(agent) = agent else {
        app.transcript.push_turn(Turn {
            prompt,
            kind,
            blocks: app
                .model
                .no_models_hint
                .clone()
                .map_or_else(Vec::new, |hint| vec![Block::Error(hint)]),
        });
        app.pin_to_latest();
        return;
    };
    spawn_agent_run(app, current_run, agent, Some(prompt), kind);
}

pub(super) fn spawn_continue(
    app: &mut App,
    agent: Option<&BackendRef>,
    current_run: &mut Option<RunHandle>,
) {
    if let Some(agent) = agent {
        spawn_agent_run(app, current_run, agent, None, lofi_types::PromptKind::User);
    }
}

pub(super) fn restore_queued_prompts(app: &mut App) {
    if app.prompt_queue.is_empty() {
        return;
    }
    // User-typed follow-ups belong back in the editor on interrupt. System-
    // injected Notices stay queued for the next round. They must not enter
    // `app.input`, but dropping them here loses a job completion that raced
    // with cancellation of the foreground run.
    let mut user_prompts = Vec::new();
    app.prompt_queue.retain(|queued| {
        if queued.kind == lofi_types::PromptKind::User {
            user_prompts.push(queued.text.clone());
            false
        } else {
            true
        }
    });
    let queued = user_prompts.join("\n\n");
    let draft = std::mem::take(&mut app.prompt.text);
    app.prompt.text = [queued, draft]
        .into_iter()
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    app.prompt.cursor = app.prompt.text.len();
    app.prompt.history_idx = None;
    app.refresh_slash_complete();
}

pub(super) fn interrupt_run(app: &mut App, current_run: &mut Option<RunHandle>) {
    if let Some(r) = current_run.as_mut() {
        if r.user_shell.is_none() {
            // Restore steering/follow-up messages to the editor when a
            // stream is aborted instead of submitting them automatically.
            restore_queued_prompts(app);
        }
        // All runs must settle cooperatively: cancel only raises the flag.
        // A user shell then finishes by returning what its pipes captured so
        // far — aborting here would discard the streamed output — and an
        // agent run checkpoints each completed round, flushes the partial
        // current response, and writes a TurnCancelled marker before its
        // channel closes. Both paths finish through the event loop.
        //
        // The task must not be aborted from here: it is a blocking task (an
        // abort only detaches a running closure), and a turn that resumed
        // later would overwrite the shared history with its own snapshot and
        // append to the transcript off an old leaf.
        lofi_core::signal_cancel(&r.cancel);
        if let Some(active) = app.active_run.as_mut() {
            active.cancel_at.get_or_insert_with(Instant::now);
        }
        app.ctrl_c_at = None;
    }
}

pub(super) fn request_quit(app: &mut App, current_run: &mut Option<RunHandle>) {
    app.should_quit = true;
    if let Some(r) = current_run.as_mut() {
        // Signal cancel now. The event loop waits for flush and does not abort.
        lofi_core::signal_cancel(&r.cancel);
    }
}

#[cfg(test)]
mod user_shell_tests {
    use super::parse_user_shell;

    #[test]
    fn parses_context_modes() {
        assert_eq!(
            parse_user_shell("!  printf ok"),
            Some(("printf ok".to_string(), false))
        );
        assert_eq!(
            parse_user_shell("!! printf ok"),
            Some(("printf ok".to_string(), true))
        );
        assert_eq!(parse_user_shell("!   "), None);
        assert_eq!(parse_user_shell("ordinary prompt"), None);
    }
}
