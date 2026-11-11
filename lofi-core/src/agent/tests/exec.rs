#![allow(clippy::unwrap_used)]
#![allow(clippy::wildcard_imports)]

use super::*;

#[test]
fn cap_tool_result_keeps_short_unchanged() {
    assert_eq!(cap_tool_result("hello"), "hello");
}

#[test]
fn cap_tool_result_truncates_long_with_marker() {
    let big = "x".repeat(MAX_TOOL_RESULT_BYTES + 1000);
    let capped = cap_tool_result(&big);
    assert!(capped.len() < big.len());
    assert!(capped.contains("[output truncated:"));
    assert!(capped.contains(&format!("{} bytes total", big.len())));
    assert!(capped.starts_with("xxxx"));
}

#[test]
fn cap_tool_result_respects_char_boundary() {
    // Fill to just past the cap with multibyte chars so the boundary cut
    // would land mid-char if unhandled.
    let unit = "é"; // 2 bytes
    let n = MAX_TOOL_RESULT_BYTES / 2 + 50;
    let big = unit.repeat(n);
    let capped = cap_tool_result(&big);
    assert!(capped.contains("[output truncated:"));
}

#[test]
fn cap_exec_result_uses_larger_outer_limit() {
    // The exec-level cap is larger than the per-tool cap, so a payload
    // bigger than MAX_TOOL_RESULT_BYTES but under MAX_EXEC_RESULT_BYTES is
    // preserved by cap_exec_result but would be truncated by cap_tool_result.
    let mid = "x".repeat(MAX_TOOL_RESULT_BYTES + 1000);
    assert!(mid.len() < MAX_EXEC_RESULT_BYTES);
    assert_eq!(cap_exec_result(&mid), mid);
    assert_ne!(cap_tool_result(&mid), mid);
    let huge = "x".repeat(MAX_EXEC_RESULT_BYTES + 5000);
    let capped = cap_exec_result(&huge);
    assert!(capped.contains("[output truncated:"));
    assert!(capped.contains(&format!("{} bytes total", huge.len())));
}

#[test]
fn exec_schema_uses_code_contract() {
    let schema = exec_tool_schema();
    assert_eq!(schema.name, lofi_code::EXEC_TOOL_NAME);
    assert_eq!(schema.description, lofi_code::EXEC_TOOL_DESCRIPTION);
    assert_eq!(schema.input_schema, lofi_code::exec_tool_input_schema());
}

#[test]
fn parse_exec_input_extracts_fields() {
    let input = serde_json::json!({
        "code": "return 1",
        "strings": { "a": "x", "b": 2 },
        "description": "count things"
    });
    let (code, strings, description) = parse_exec_input(&input);
    assert_eq!(code, "return 1");
    assert_eq!(strings.get("a").unwrap(), "x");
    assert_eq!(strings.get("b").unwrap(), "2");
    assert_eq!(description, "count things");
}

#[test]
fn parse_exec_input_missing_code_is_empty() {
    let (code, strings, description) = parse_exec_input(&serde_json::Value::Null);
    assert!(code.is_empty());
    assert!(strings.is_empty());
    assert!(description.is_empty());
}

#[test]
fn parse_exec_input_trims_surrounding_newlines() {
    let input = serde_json::json!({ "code": "\nreturn 1\n" });
    let (code, _, _) = parse_exec_input(&input);
    assert_eq!(code, "return 1");
}

#[test]
fn exec_input_code_and_label_reads_the_description() {
    let input = serde_json::json!({ "code": "return 1", "description": "count things" });
    assert_eq!(
        exec_input_code_and_label(&input),
        ("return 1".to_string(), Some("count things".to_string()))
    );
}

#[test]
fn a_long_description_is_capped_with_its_head_kept() {
    let long = format!("{} end", "d".repeat(400));
    let input = serde_json::json!({ "code": "return 1", "description": long });
    let (_, _, description) = parse_exec_input(&input);
    assert!(description.ends_with(" [...truncated]"));
    assert!(description.starts_with("ddd"));
    assert!(description.chars().count() < 230);
}

#[test]
fn normalize_exec_input_caps_a_long_description() {
    let long = format!("{} end", "d".repeat(400));
    let input = serde_json::json!({ "code": "return 1", "description": long });
    let normalized = normalize_exec_input(&input);
    let description = normalized["description"].as_str().unwrap();
    assert!(description.starts_with("ddd"));
    assert!(description.ends_with(" [...truncated]"));
}

#[test]
fn normalize_exec_input_drops_a_non_string_description() {
    let input = serde_json::json!({
        "code": "return 1",
        "description": serde_json::json!(["not", "a", "string"]),
    });
    let normalized = normalize_exec_input(&input);
    assert!(normalized.get("description").is_none());
}

#[tokio::test]
async fn the_recorded_tool_use_carries_the_capped_description() {
    let dir = tempdir().unwrap();
    let executor = crate::agent::tool_executor::ToolExecutor::new(
        dir.path(),
        &lofi_types::Config::default(),
        &[],
    )
    .unwrap();
    let recorded = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink: std::sync::Arc<dyn Fn(ContentBlock) -> lofi_error::Result<()> + Send + Sync> = {
        let recorded = recorded.clone();
        std::sync::Arc::new(move |block| {
            recorded.lock().unwrap().push(block);
            Ok(())
        })
    };
    let context = crate::agent::tool_executor::ToolCallContext {
        record_tool: Some(sink),
        ..Default::default()
    };
    let long = format!("{} end", "d".repeat(400));
    executor
        .execute(
            lofi_code::EXEC_TOOL_NAME,
            &serde_json::json!({ "code": "return 1", "description": long }),
            context,
        )
        .await
        .unwrap();
    let recorded = recorded.lock().unwrap().clone();
    let [ContentBlock::ToolUse { input, .. }, _] = &recorded[..] else {
        panic!("expected the tool use then the result: {recorded:?}");
    };
    let description = input["description"].as_str().unwrap();
    assert!(description.starts_with("ddd"));
    assert!(description.ends_with(" [...truncated]"));
}

/// A thrown message can outgrow the result cap on its own: it is trimmed
/// like the logs, so the body still parses and still names its status.
#[tokio::test]
async fn a_thrown_message_larger_than_the_result_cap_keeps_its_status() {
    let dir = tempdir().unwrap();
    let executor = crate::agent::tool_executor::ToolExecutor::new(
        dir.path(),
        &lofi_types::Config::default(),
        &[],
    )
    .unwrap();

    let result = executor
        .execute(
            lofi_code::EXEC_TOOL_NAME,
            &serde_json::json!({
                "code": "throw new Error('\\\\n'.repeat(300 * 1024));",
                "description": "throw a message past the result cap",
            }),
            crate::agent::tool_executor::ToolCallContext::default(),
        )
        .await
        .unwrap();

    let text = match &result {
        ContentBlock::ToolResult { content, .. } => content.clone(),
        other => panic!("expected a tool result: {other:?}"),
    };
    assert!(
        text.len() <= MAX_EXEC_RESULT_BYTES,
        "the body stays within the limit: {} bytes",
        text.len()
    );
    let body: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("the body is still JSON: {e}"));
    assert_eq!(
        body["status"],
        serde_json::json!("failed"),
        "{}",
        &text[..400.min(text.len())]
    );
    assert!(
        body["errors"][0].as_str().is_some_and(|e| !e.is_empty()),
        "the message it threw survives the trim"
    );
}
// A stopped script that printed near the log cap still reports `status`: the
// body is measured as it is built, so the cap cannot cut the keys that sort
// after `logs`.
#[tokio::test]
async fn a_stopped_exec_keeps_its_status_when_its_logs_are_large() {
    let dir = tempdir().unwrap();
    let executor = crate::agent::tool_executor::ToolExecutor::new(
        dir.path(),
        &lofi_types::Config::default(),
        &[],
    )
    .unwrap();

    let result = executor
        .execute(
            lofi_code::EXEC_TOOL_NAME,
            &serde_json::json!({
                "code": "print('x'.repeat(300 * 1024)); await lofi.sleep(30); return 1",
                "description": "print a lot then sleep past the budget",
                "timeoutMs": 300,
            }),
            crate::agent::tool_executor::ToolCallContext::default(),
        )
        .await
        .unwrap();

    let text = match &result {
        ContentBlock::ToolResult { content, .. } => content.clone(),
        other => panic!("expected a tool result: {other:?}"),
    };
    assert!(
        text.len() <= MAX_EXEC_RESULT_BYTES,
        "the body stays within the limit: {} bytes",
        text.len()
    );
    let body: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("the body is still JSON: {e}"));
    assert_eq!(body["status"], serde_json::json!("timed_out"), "{text}");
    assert!(
        body["logs"].as_str().is_some_and(|logs| !logs.is_empty()),
        "the logs it reached survive the trim"
    );
    assert_eq!(
        body["truncated"],
        serde_json::json!(true),
        "a trimmed body says so: {text}"
    );
}

// The call's own wall budget reaches the guest, which reads it and plans with
// it: a script that sleeps past its budget is cut rather than run to its end.
#[tokio::test]
async fn an_exec_budget_cuts_a_guest_that_would_outlive_it() {
    let dir = tempdir().unwrap();
    let executor = crate::agent::tool_executor::ToolExecutor::new(
        dir.path(),
        &lofi_types::Config::default(),
        &[],
    )
    .unwrap();

    let started = std::time::Instant::now();
    let error = executor
        .execute(
            lofi_code::EXEC_TOOL_NAME,
            &serde_json::json!({
                "code": "await lofi.sleep(30); return 1",
                "description": "sleep past the budget",
                "timeoutMs": 300,
            }),
            crate::agent::tool_executor::ToolCallContext::default(),
        )
        .await;

    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "the guest outlived the budget it was given"
    );
    let text = format!("{error:?}");
    assert!(
        text.contains("wall-clock"),
        "the report names the budget that ended it: {text}"
    );
    assert!(
        text.contains("timed_out"),
        "the result says what stopped it in the shared vocabulary: {text}"
    );
}

#[test]
fn exec_label_trims_and_rejects_blank() {
    assert_eq!(exec_label("  do thing  "), Some("do thing".to_string()));
    assert_eq!(exec_label("   "), None);
    assert_eq!(exec_label(""), None);
}

#[test]
fn exec_result_display_handles_structured_success() {
    assert_eq!(
        exec_result_display(r#"{"ok":true,"value":null}"#, false),
        "null"
    );
    assert_eq!(
        exec_result_display(r#"{"ok":true,"value":"done"}"#, false),
        "done"
    );
    assert_eq!(
        exec_result_display(r#"{"ok":true,"logs":"worked\n"}"#, false),
        "worked\n"
    );
    assert_eq!(exec_result_display(r#"{"ok":true}"#, false), "");
}

#[test]
fn extract_code_prefix_streams_and_trims_leading_newline() {
    let cases = [
        (r#"{"code":"\nle"#, "le"),
        (r#"{"code":"\nlet x"#, "let x"),
        (r#"{"code":"\nlet x\nlet y"#, "let x\nlet y"),
        (r#"{"description":"z","code":"\nlet x"#, "let x"),
    ];
    for (raw, want) in cases {
        assert_eq!(extract_code_prefix(raw), want, "raw: {raw}");
    }
}

#[test]
fn code_prefix_decoder_handles_split_escape_incrementally() {
    let mut decoder = CodePrefixDecoder::default();
    assert_eq!(decoder.update(r#"{"code":"\nlet x\u"#), "let x");
    assert_eq!(decoder.update(r#"{"code":"\nlet x\u0061"}"#), "let xa");
}
