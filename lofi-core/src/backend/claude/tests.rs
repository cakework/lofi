#![allow(clippy::unwrap_used)]

use serde_json::{json, Value};

use super::*;

fn init_frame(tools: &Value, server_status: &str) -> Value {
    json!({
        "tools": tools,
        "mcp_servers": [{"name": "lofi", "status": server_status}],
    })
}

#[test]
fn validate_init_accepts_the_exec_tool_and_the_lofi_server() {
    let frame = init_frame(&json!([EXEC]), "connected");
    assert!(validate_init(&frame).is_ok());
}

#[test]
fn validate_init_names_an_extra_tool() {
    let frame = init_frame(&json!([EXEC, "ToolSearch"]), "connected");
    let error = validate_init(&frame).unwrap_err().to_string();
    assert!(error.contains("ToolSearch"), "{error}");
}

#[test]
fn validate_init_names_a_missing_tool_list() {
    let frame = json!({"mcp_servers": []});
    let error = validate_init(&frame).unwrap_err().to_string();
    assert!(error.contains("null"), "{error}");
}

#[test]
fn validate_init_names_a_server_that_did_not_connect() {
    let frame = init_frame(&json!([EXEC]), "failed");
    let error = validate_init(&frame).unwrap_err().to_string();
    assert!(error.contains("failed"), "{error}");
}

#[test]
fn observed_cuts_a_long_list() {
    let tools = Value::Array((0..50).map(|i| json!(format!("tool-{i}"))).collect());
    let text = observed(&tools);
    assert!(text.ends_with('…'));
    assert_eq!(text.chars().count(), 201);
}

#[test]
fn effort_levels_include_the_cli_maximum() {
    let levels = effort_levels();
    let names: Vec<&str> = levels
        .iter()
        .map(lofi_types::ThinkingLevel::as_str)
        .collect();
    assert_eq!(names, ["low", "medium", "high", "xhigh", "max"]);
}

#[test]
fn effort_flag_omits_off_and_passes_max() {
    assert_eq!(effort_flag(&lofi_types::ThinkingLevel::Off), None);
    assert_eq!(
        effort_flag(&lofi_types::ThinkingLevel::Custom("max".into())),
        Some("--effort=max".into())
    );
}

#[test]
fn content_value_stays_a_string_without_images() {
    assert_eq!(content_value("hello", &[]), Value::String("hello".into()));
}

#[test]
fn content_value_sends_images_as_base64_blocks() {
    let block = lofi_types::ContentBlock::Image {
        bytes: vec![1, 2, 3].into(),
        media_type: "image/png".into(),
    };
    let value = content_value("look", &[block]);
    assert_eq!(value[0], json!({"type": "text", "text": "look"}));
    assert_eq!(value[1]["type"], "image");
    assert_eq!(value[1]["source"]["type"], "base64");
    assert_eq!(value[1]["source"]["media_type"], "image/png");
    assert_eq!(value[1]["source"]["data"], "AQID");
}

// A turn has no length limit; only the silence between frames ends it.
#[tokio::test(start_paused = true)]
async fn a_silent_cli_ends_the_turn_at_the_frame_idle_deadline() {
    let error = frame_within(
        FRAME_IDLE_TIMEOUT,
        "Claude Code sent no frame before the idle deadline",
        std::future::pending::<Result<Value>>(),
    )
    .await
    .unwrap_err()
    .to_string();

    assert!(error.contains("no frame"), "{error}");
}

#[test]
fn an_unset_frame_idle_uses_the_wide_default() {
    assert_eq!(resolve_frame_idle(None).unwrap(), FRAME_IDLE_TIMEOUT);
    assert_eq!(
        resolve_frame_idle(Some(Duration::from_millis(500))).unwrap(),
        Duration::from_millis(500)
    );
}

#[test]
fn a_zero_frame_idle_is_rejected() {
    let error = resolve_frame_idle(Some(Duration::ZERO)).unwrap_err();
    assert!(error.to_string().contains("greater than zero"), "{error}");
}

#[test]
fn a_running_call_pauses_the_frame_clock() {
    let idle = Duration::from_secs(10);
    let start = Instant::now();
    let since = |gap: Duration| start.checked_sub(gap).unwrap();

    // Nothing running and nothing counted yet: the whole gap is left.
    let fresh = frame_budget(start, idle, (0, None)).unwrap();
    assert!(
        fresh <= idle && fresh > idle.checked_sub(Duration::from_secs(1)).unwrap(),
        "{fresh:?}"
    );
    // A call of ours is what the child waits on, so the clock waits too.
    assert_eq!(
        frame_budget(since(Duration::from_secs(30)), idle, (1, None)),
        Some(BUSY_POLL)
    );
    // The call ended two seconds ago, and only that silence counts.
    let left = frame_budget(
        since(Duration::from_secs(30)),
        idle,
        (0, Some(Duration::from_secs(2))),
    )
    .unwrap();
    assert!(
        left <= Duration::from_secs(8) && left > Duration::from_secs(7),
        "{left:?}"
    );
    // Spent, with nothing in flight: the turn fails.
    assert_eq!(
        frame_budget(since(Duration::from_secs(30)), idle, (0, None)),
        None
    );
    assert_eq!(
        frame_budget(
            since(Duration::from_secs(30)),
            idle,
            (0, Some(Duration::from_secs(30)))
        ),
        None
    );
}

#[tokio::test]
async fn a_frame_that_arrives_is_the_turn_continuing() {
    let frame = frame_within(
        FRAME_IDLE_TIMEOUT,
        "Claude Code sent no frame before the idle deadline",
        async { Ok(json!({"type": "assistant"})) },
    )
    .await
    .unwrap();

    assert_eq!(frame["type"], "assistant");
}

#[test]
fn result_error_message_surfaces_subtype_and_detail() {
    let frame = json!({
        "type": "result",
        "subtype": "error_max_turns",
        "is_error": true,
        "result": "rate limit exceeded on upstream"
    });
    let msg = result_error_message(&frame);
    assert!(msg.contains("error_max_turns"), "{msg}");
    assert!(msg.contains("rate limit exceeded on upstream"), "{msg}");

    let errors_frame = json!({
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": true,
        "errors": ["invalid model alias", "check config"]
    });
    let msg = result_error_message(&errors_frame);
    assert!(msg.contains("error_during_execution"), "{msg}");
    assert!(msg.contains("invalid model alias"), "{msg}");
}

#[tokio::test]
async fn oneshot_surfaces_child_stderr_on_early_exit() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("failing-claude");
    std::fs::write(
        &script,
        "#!/bin/sh\necho 'fatal: bad cli flag --foo' >&2\nexit 2\n",
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();

    let err = super::super::oneshot::complete(
        &script.display().to_string(),
        dir.path(),
        None,
        &lofi_types::ThinkingLevel::Off,
        &super::super::contract::CompletionRequest {
            system: "sys".into(),
            prompt: "hi".into(),
            max_tokens: None,
        },
    )
    .await
    .unwrap_err()
    .to_string();

    assert!(err.contains("fatal: bad cli flag --foo"), "{err}");
}

#[tokio::test]
async fn session_prompt_surfaces_child_stderr_on_early_exit() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("failing-claude");
    std::fs::write(
        &script,
        "#!/bin/sh\necho 'Error: auth token expired' >&2\nexit 1\n",
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();

    let exe = std::env::current_exe().unwrap();
    let mut session = ClaudeSession::open(
        Launch {
            command: &script.display().to_string(),
            model: None,
            effort: ThinkingLevel::Off,
            resume: None,
            system_prompt: "You are lofi",
            frame_idle: Some(Duration::from_secs(5)),
        },
        dir.path(),
        &exe,
    )
    .await
    .unwrap();
    let executor = ToolExecutor::in_memory(dir.path());
    let err = session
        .prompt(
            Value::String("hi".into()),
            executor,
            ToolCallContext::default(),
            PromptControls::default(),
        )
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains("Error: auth token expired"), "{err}");
}
