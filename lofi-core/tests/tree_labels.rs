//! /tree row labels for tool calls: a sandboxed call records one
//! `native_tool` event per nested tool inside the turn, so its tool result
//! hangs off the last record rather than off the assistant message.

use lofi_core::session::store::{SessionCursor, SessionStore};
use lofi_core::session::tree::{build_tree_row_skeletons, build_tree_rows, hydrate_tree_rows};
use lofi_core::Result;
use lofi_types::{
    ContentBlock, Message, NativeToolRecord, PromptKind, Role, SessionEvent, SessionEventKind,
    Usage,
};
use std::path::Path;
use tempfile::tempdir;

fn message(role: Role, blocks: Vec<ContentBlock>) -> SessionEvent {
    SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::Message(Message {
            origin: None,
            role,
            blocks,
            kind: PromptKind::default(),
        }),
    }
}

fn user(text: &str) -> SessionEvent {
    message(
        Role::User,
        vec![ContentBlock::Text {
            text: text.to_string(),
        }],
    )
}

fn call(id: &str, name: &str) -> SessionEvent {
    message(
        Role::Assistant,
        vec![ContentBlock::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input: serde_json::json!({ "code": "1+1" }),
        }],
    )
}

fn result(tool_use_id: &str, content: &str, is_error: bool) -> SessionEvent {
    message(
        Role::Tool,
        vec![ContentBlock::ToolResult {
            tool_use_id: tool_use_id.to_string(),
            content: content.to_string(),
            is_error,
            images: Vec::new(),
        }],
    )
}

fn native(parent: &str, call_id: u64, name: &str, args: &str) -> SessionEvent {
    SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::NativeTool(NativeToolRecord {
            parent: parent.to_string(),
            call_id,
            name: name.to_string(),
            args: args.to_string(),
            result: "{}".to_string(),
            is_error: false,
        }),
    }
}

/// The recorder writes the turn's spend between the call and its nested
/// tools, so the fixtures keep it there too.
fn usage() -> SessionEvent {
    SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::TurnUsage {
            elapsed_ms: 10,
            cost: 0.0,
            usage: Usage::default(),
        },
    }
}

fn labels(cursor: &SessionCursor) -> Result<Vec<String>> {
    let index = cursor.tree_index()?;
    Ok(build_tree_rows(&index.index, index.leaf_pos, cursor)
        .into_iter()
        .map(|row| row.label.to_string())
        .collect())
}

/// The picker hydrates skeleton rows instead of building every row, so it
/// resolves tool names through the lazy path. Both paths must agree.
fn hydrated_labels(cursor: &SessionCursor) -> Result<Vec<String>> {
    let index = cursor.tree_index()?;
    let skeletons = build_tree_row_skeletons(&index.index, index.leaf_pos, cursor);
    let requested: Vec<_> = skeletons.iter().cloned().enumerate().collect();
    let mut hydrated: Vec<(usize, String)> = Vec::new();
    hydrate_tree_rows(
        &index.index,
        cursor,
        &requested,
        |rows| {
            hydrated.extend(
                rows.into_iter()
                    .map(|(position, row)| (position, row.label.to_string())),
            );
            true
        },
        || false,
    );
    hydrated.sort_by_key(|(position, _)| *position);
    Ok(hydrated.into_iter().map(|(_, label)| label).collect())
}

fn cursor_for(mut events: Vec<SessionEvent>) -> Result<(tempfile::TempDir, SessionCursor)> {
    let dir = tempdir()?;
    let store = SessionStore::new(dir.path().join("sessions"));
    let cursor = store.create_cursor(Path::new("/x"), &"m".into())?;
    cursor.append_events(&mut events)?;
    Ok((dir, cursor))
}

#[test]
fn a_sandboxed_call_lists_its_nested_tools_and_errors() -> Result<()> {
    let (_dir, cursor) = cursor_for(vec![
        user("p1"),
        call("call_1", "exec"),
        usage(),
        native("call_1", 0, "skill", "code-review"),
        native("call_1", 1, "bash", "jj status"),
        result("call_1", "bash: command not found", true),
    ])?;

    let expected = vec![
        "user: p1",
        "exec: \u{2717} skill code-review, bash jj status",
    ];
    assert_eq!(labels(&cursor)?, expected);
    assert_eq!(hydrated_labels(&cursor)?, expected);
    Ok(())
}

#[test]
fn a_direct_call_still_uses_its_own_name() -> Result<()> {
    let (_dir, cursor) = cursor_for(vec![
        user("p1"),
        call("call_2", "read"),
        result("call_2", "fn main() {}", false),
    ])?;

    let expected = vec!["user: p1", "tool: read: fn main() {}"];
    assert_eq!(labels(&cursor)?, expected);
    assert_eq!(hydrated_labels(&cursor)?, expected);
    Ok(())
}

#[test]
fn a_result_past_the_turn_start_finds_no_name() -> Result<()> {
    let (_dir, cursor) = cursor_for(vec![
        user("p1"),
        call("call_0", "read"),
        result("call_0", "x", false),
        user("p2"),
        usage(),
        result("call_0", "orphan", false),
    ])?;

    let expected = vec!["user: p1", "tool: read: x", "user: p2", "tool: ?: orphan"];
    assert_eq!(labels(&cursor)?, expected);
    assert_eq!(hydrated_labels(&cursor)?, expected);
    Ok(())
}

#[test]
fn a_result_never_borrows_a_name_from_an_earlier_turn() -> Result<()> {
    let (_dir, cursor) = cursor_for(vec![
        user("p1"),
        call("call_0", "read"),
        result("call_0", "x", false),
        user("p2"),
        message(
            Role::Assistant,
            vec![ContentBlock::Text {
                text: "no call here".to_string(),
            }],
        ),
        result("call_0", "orphan", false),
    ])?;

    let expected = vec!["user: p1", "tool: read: x", "user: p2", "tool: ?: orphan"];
    assert_eq!(labels(&cursor)?, expected);
    assert_eq!(hydrated_labels(&cursor)?, expected);
    Ok(())
}
