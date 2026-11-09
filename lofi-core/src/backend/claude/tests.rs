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
        bytes: vec![1, 2, 3],
        media_type: "image/png".into(),
    };
    let value = content_value("look", &[block]);
    assert_eq!(value[0], json!({"type": "text", "text": "look"}));
    assert_eq!(value[1]["type"], "image");
    assert_eq!(value[1]["source"]["type"], "base64");
    assert_eq!(value[1]["source"]["media_type"], "image/png");
    assert_eq!(value[1]["source"]["data"], "AQID");
}

// A turn used to end at a fixed wall clock, which killed runs that were
// still working. The term is the silence between frames.
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
