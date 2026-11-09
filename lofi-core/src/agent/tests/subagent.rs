#![allow(clippy::await_holding_lock, clippy::expect_used)]
// The lock serializes tests against the process-global LOFI_STATE_HOME, not
// against shared runtime state, so holding it across awaits is the point.

use super::*;
use crate::session::store::IndexKind;
use lofi_code::SubagentOp;
use serde_json::json;

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    crate::state::STATE_ENV_LOCK.lock().unwrap()
}

fn support(cwd: &std::path::Path) -> SubagentSupport {
    SubagentSupport::new(
        crate::models::ModelRegistry::default(),
        lofi_types::Config::default(),
        cwd.to_path_buf(),
    )
}

// Point LOFI_STATE_HOME at a temp dir so spawning tests never write to the
// real state directory; restores the previous value on drop.
struct StateEnv {
    prev: Option<std::ffi::OsString>,
    _dir: tempfile::TempDir,
}

impl StateEnv {
    fn new() -> (Self, std::path::PathBuf) {
        let dir = tempdir().unwrap();
        let prev = std::env::var_os("LOFI_STATE_HOME");
        std::env::set_var("LOFI_STATE_HOME", dir.path());
        let path = dir.path().to_path_buf();
        (Self { prev, _dir: dir }, path)
    }
}

impl Drop for StateEnv {
    fn drop(&mut self) {
        match &self.prev {
            Some(v) => std::env::set_var("LOFI_STATE_HOME", v),
            None => std::env::remove_var("LOFI_STATE_HOME"),
        }
    }
}

fn text_round(text: &str) -> Vec<StreamingEvent> {
    vec![
        StreamingEvent::TextDelta(text.to_string()),
        StreamingEvent::Done {
            usage: Usage::default(),
            stop_reason: None,
        },
    ]
}

#[tokio::test]
async fn spawn_returns_id_status_and_result() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![text_round("sub result")], dir.path())
        .with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "do it" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["result"], json!("sub result"), "{out}");
    // The handle space starts at the clock, so a run names a handle, not `1`.
    let id = out["id"].as_str().unwrap_or_default();
    assert!(
        lofi_types::parse_short_id(id).is_some(),
        "the run names a handle: {out}"
    );
}

#[tokio::test]
async fn hook_is_none_without_support() {
    let _env = env_lock();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![], dir.path());
    assert!(agent.subagent_hook().is_none());
}

#[tokio::test]
async fn resume_appends_to_child_history() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let provider = Arc::new(MockProvider::recording(vec![
        text_round("first"),
        text_round("second"),
    ]));
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = provider.clone();
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "a" })).await;
    let id = out["id"].as_str().unwrap().to_string();
    let out = hook(SubagentOp::Run, json!({ "id": id, "prompt": "b" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["result"], json!("second"), "{out}");

    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    let roles: Vec<_> = requests[1]
        .iter()
        .map(|m| (m.role, first_text(m)))
        .collect();
    let a = roles
        .iter()
        .position(|(r, t)| *r == Role::User && t == "a")
        .unwrap();
    assert_eq!(roles[a + 1], (Role::Assistant, "first".to_string()));
    assert_eq!(roles[a + 2], (Role::User, "b".to_string()));
}

#[tokio::test]
async fn resume_rejects_model_override() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![text_round("first")], dir.path())
        .with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "a" })).await;
    let id = out["id"].as_str().unwrap().to_string();
    let out = hook(
        SubagentOp::Run,
        json!({ "id": id, "prompt": "b", "model": "p/m" }),
    )
    .await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    assert!(
        out["error"]
            .as_str()
            .unwrap()
            .contains("only apply when spawning"),
        "{out}"
    );
}

#[tokio::test]
async fn unknown_id_reports_failed() {
    let _env = env_lock();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "id": "deadbeef", "prompt": "b" })).await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    assert!(
        out["error"]
            .as_str()
            .unwrap()
            .contains("unknown subagent id"),
        "{out}"
    );
    assert_eq!(out["id"], json!("deadbeef"));
}

#[tokio::test]
async fn output_limit_truncates_result() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![text_round("abcdefghijklmnopqrstuvwxyz")], dir.path())
        .with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "a", "output_limit": 5 })).await;
    assert_eq!(out["result"], json!("abcde"), "{out}");
}

#[tokio::test]
async fn spawn_persists_a_transcript() {
    let _env = env_lock();
    let (_state, state_path) = StateEnv::new();

    let dir = tempdir().unwrap();
    let agent = agent_with(vec![text_round("sub result")], dir.path())
        .with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();
    let out = hook(SubagentOp::Run, json!({ "prompt": "do it" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");

    let files = find_jsonl(&state_path);
    assert!(
        !files.is_empty(),
        "no transcript under {}",
        state_path.display()
    );
}

#[tokio::test]
async fn spawn_without_persistence_writes_no_transcript() {
    let _env = env_lock();
    let (_state, state_path) = StateEnv::new();

    let dir = tempdir().unwrap();
    let agent = agent_with(vec![text_round("sub result")], dir.path())
        .with_subagent_support(support(dir.path()).with_persistence(false));
    let hook = agent.subagent_hook().unwrap();
    let out = hook(SubagentOp::Run, json!({ "prompt": "do it" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert!(
        find_jsonl(&state_path).is_empty(),
        "transcript written under {}",
        state_path.display()
    );
}

#[tokio::test]
async fn invalid_arg_types_fail_instead_of_defaulting() {
    let _env = env_lock();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![text_round("sub result")], dir.path())
        .with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "id": 123, "prompt": "a" })).await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    assert!(
        out["error"].as_str().unwrap().contains("must be a string"),
        "{out}"
    );

    let out = hook(SubagentOp::Run, json!({ "prompt": "a", "timeout_ms": 0 })).await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    assert!(
        out["error"].as_str().unwrap().contains("positive integer"),
        "{out}"
    );

    let out = hook(SubagentOp::Run, json!({})).await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    assert!(
        out["error"]
            .as_str()
            .unwrap()
            .contains("prompt is required"),
        "{out}"
    );

    let out = hook(SubagentOp::Run, json!({ "prompt": "   " })).await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    assert!(
        out["error"]
            .as_str()
            .unwrap()
            .contains("prompt is required"),
        "{out}"
    );

    let out = hook(SubagentOp::Run, json!({ "id": "", "prompt": "a" })).await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    assert!(
        out["error"].as_str().unwrap().contains("non-empty string"),
        "{out}"
    );
}

#[tokio::test]
async fn output_limit_counts_bytes_at_char_boundary() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent = agent_with(
        vec![text_round("ab\u{1f600}cd"), text_round("ab\u{1f600}cd")],
        dir.path(),
    )
    .with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    // The emoji is 4 bytes: limit 4 would split it, so it drops to "ab".
    let out = hook(SubagentOp::Run, json!({ "prompt": "a", "output_limit": 4 })).await;
    assert_eq!(out["result"].as_str().unwrap(), "ab", "{out}");
    // Limit 6 fits the emoji exactly.
    let out = hook(SubagentOp::Run, json!({ "prompt": "b", "output_limit": 6 })).await;
    assert_eq!(out["result"].as_str().unwrap(), "ab\u{1f600}", "{out}");
}

/// Whether a request is the wrap-up turn a stopped run is given. These
/// providers stall it too, so a stopped run keeps its deadline status.
fn is_wrap_up(messages: &[Message]) -> bool {
    messages
        .last()
        .is_some_and(|m| is_wrap_up_text(&first_text(m)))
}

fn is_wrap_up_text(text: &str) -> bool {
    text.starts_with("Your run was stopped at its deadline")
}

// The answer depends on the request content, not on a call counter, so the
// tests stay deterministic no matter when the stop fires: a prompt "a" asks
// for a tool call, the follow-up after a tool result pends forever (the run
// can be stopped mid-turn), and anything else completes with plain text.
struct TimeoutResumeProvider {
    requests: std::sync::Mutex<Vec<Vec<Message>>>,
    tool_round: Vec<StreamingEvent>,
    text_round: Vec<StreamingEvent>,
}

#[async_trait]
impl Provider for TimeoutResumeProvider {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        self.requests.lock().unwrap().push(messages.to_vec());
        let last = messages.last();
        let round = match last {
            _ if is_wrap_up(messages) => std::future::pending().await,
            Some(m) if m.role == Role::Tool => std::future::pending().await,
            Some(m) if m.role == Role::User && first_text(m) == "a" => self.tool_round.clone(),
            _ => self.text_round.clone(),
        };
        Ok(Box::pin(stream::iter(round.into_iter().map(Ok))))
    }
}

#[tokio::test]
async fn timeout_keeps_the_partial_history() {
    let _env = env_lock();
    let (_state, state_path) = StateEnv::new();
    let dir = tempdir().unwrap();
    let provider = Arc::new(TimeoutResumeProvider {
        requests: std::sync::Mutex::new(Vec::new()),
        tool_round: exec_round("return 1"),
        text_round: text_round("resumed"),
    });
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = provider.clone();
    let hook = agent.subagent_hook().unwrap();

    // The model round after the tool call never answers, so the deadline can
    // only fire once the tool exchange is complete. A deadline long enough to
    // outlast the sandbox startup keeps the fixture off a race.
    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "a", "timeout_ms": 2000 }),
    )
    .await;
    assert_eq!(out["status"], json!("timed_out"), "{out}");
    let id = out["id"].as_str().unwrap().to_string();
    // The parent reads the error, not the docs: it must name the resume path
    // so the work already paid for is continued instead of rerun.
    assert!(
        out["error"]
            .as_str()
            .unwrap()
            .contains(&format!("resume with id {id}")),
        "{out}"
    );

    let out = hook(
        SubagentOp::Run,
        json!({ "id": id, "prompt": "b", "timeout_ms": 5000 }),
    )
    .await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["result"], json!("resumed"), "{out}");

    // The resumed turn continues the timed-out one: it replays the prompt
    // that run was given, the tool call it made, and the result of that call.
    // The wrap-up prompt the stopped run was given sits between them.
    let requests = provider.requests.lock().unwrap();
    let last = requests.last().unwrap();
    let roles: Vec<_> = last
        .iter()
        .filter(|m| !is_wrap_up_text(&first_text(m)))
        .map(|m| m.role)
        .collect();
    assert_eq!(
        roles,
        vec![
            Role::System,
            Role::User,
            Role::Assistant,
            Role::Tool,
            Role::User
        ],
        "{roles:?}"
    );
    assert_tool_calls_are_answered(last);
    let calls = last
        .iter()
        .flat_map(|m| m.blocks.iter())
        .filter(|block| matches!(block, ContentBlock::ToolUse { .. }))
        .count();
    assert_eq!(calls, 1, "the run's one tool call survives: {last:?}");
    drop(requests);

    // Transcript: the timed-out run stays on the lineage up to its own settle
    // marker, so a replay after a restart resumes from the same point.
    let files = find_jsonl(&state_path);
    assert_eq!(files.len(), 1, "{files:?}");
    let (_cursor, snapshot) =
        crate::session::store::SessionCursor::open_snapshot(files[0].clone()).unwrap();
    let kinds: Vec<_> = snapshot.index.iter().map(|e| e.kind).collect();
    let first_settle = kinds
        .iter()
        .position(|kind| *kind == IndexKind::RunSettled)
        .expect("the timed-out run leaves a settle marker");
    assert_eq!(kinds.first(), Some(&IndexKind::SystemMessage), "{kinds:?}");
    assert!(
        kinds[..first_settle].contains(&IndexKind::ToolResult),
        "the timed-out run's tool exchange is off its lineage: {kinds:?}"
    );
    // Both task prompts and the wrap-up prompt the stopped run was given.
    assert_eq!(
        kinds
            .iter()
            .filter(|kind| **kind == IndexKind::UserPrompt)
            .count(),
        3,
        "both prompts stay on the lineage: {kinds:?}"
    );
}

// A panic in the provider stream unwinds the run's task, so SettleOnDrop —
// not the normal settle — records the run. The history the guard keeps has
// to reach the next resume: the run already paid for that tool call.
struct PanicAfterToolProvider {
    requests: std::sync::Mutex<Vec<Vec<Message>>>,
    tool_round: Vec<StreamingEvent>,
    text_round: Vec<StreamingEvent>,
}

#[async_trait]
impl Provider for PanicAfterToolProvider {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        self.requests.lock().unwrap().push(messages.to_vec());
        let last = messages.last();
        let round = match last {
            // The tool result is in: die before the round that answers it.
            Some(m) if m.role == Role::Tool => panic!("provider stream failed"),
            Some(m) if m.role == Role::User && first_text(m) == "a" => self.tool_round.clone(),
            _ => self.text_round.clone(),
        };
        Ok(Box::pin(stream::iter(round.into_iter().map(Ok))))
    }
}

fn panic_after_tool_provider() -> Arc<PanicAfterToolProvider> {
    Arc::new(PanicAfterToolProvider {
        requests: std::sync::Mutex::new(Vec::new()),
        tool_round: exec_round("return 1"),
        text_round: text_round("resumed"),
    })
}

/// A resumed request continues the aborted run: the prompt that run was
/// given, the tool call it made, and the result of that call.
fn assert_resumed_tool_exchange(request: &[Message]) {
    let roles: Vec<_> = request.iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        vec![
            Role::System,
            Role::User,
            Role::Assistant,
            Role::Tool,
            Role::User
        ],
        "{roles:?}"
    );
    assert_tool_calls_are_answered(request);
    let calls = request
        .iter()
        .flat_map(|m| m.blocks.iter())
        .filter(|block| matches!(block, ContentBlock::ToolUse { .. }))
        .count();
    assert_eq!(
        calls, 1,
        "the aborted run's one tool call survives: {request:?}"
    );
}

#[tokio::test]
async fn a_panicked_run_keeps_the_history_it_reached() {
    let _env = env_lock();
    let (_state, state_path) = StateEnv::new();
    let dir = tempdir().unwrap();
    let provider = panic_after_tool_provider();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = provider.clone();
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "a" })).await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    assert_eq!(
        out["error"],
        json!("subagent task ended unexpectedly"),
        "{out}"
    );
    let id = out["id"].as_str().unwrap().to_string();

    let out = hook(SubagentOp::Run, json!({ "id": id, "prompt": "b" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["result"], json!("resumed"), "{out}");
    assert_resumed_tool_exchange(provider.requests.lock().unwrap().last().unwrap());

    // The aborted run stays on the lineage up to its own settle marker, so a
    // replay after a restart resumes from the same point.
    let files = find_jsonl(&state_path);
    assert_eq!(files.len(), 1, "{files:?}");
    let (_cursor, snapshot) =
        crate::session::store::SessionCursor::open_snapshot(files[0].clone()).unwrap();
    let kinds: Vec<_> = snapshot.index.iter().map(|e| e.kind).collect();
    let first_settle = kinds
        .iter()
        .position(|kind| *kind == IndexKind::RunSettled)
        .expect("the aborted run leaves a settle marker");
    assert!(
        kinds[..first_settle].contains(&IndexKind::ToolResult),
        "the aborted run's tool exchange is off its lineage: {kinds:?}"
    );
}

#[tokio::test]
async fn a_panicked_run_keeps_the_history_without_a_transcript() {
    let _env = env_lock();
    let (_state, state_path) = StateEnv::new();
    let dir = tempdir().unwrap();
    let provider = panic_after_tool_provider();
    let mut agent = agent_with(vec![], dir.path())
        .with_subagent_support(support(dir.path()).with_persistence(false));
    agent.provider = provider.clone();
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "a" })).await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    let id = out["id"].as_str().unwrap().to_string();

    // No transcript holds the run, so the resume reads the entry's copy.
    let out = hook(SubagentOp::Run, json!({ "id": id, "prompt": "b" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_resumed_tool_exchange(provider.requests.lock().unwrap().last().unwrap());
    assert!(
        find_jsonl(&state_path).is_empty(),
        "transcript written under {}",
        state_path.display()
    );
}

/// Every tool call in a request must be answered before the next message of
/// another role: a provider rejects an orphaned call.
fn assert_tool_calls_are_answered(request: &[Message]) {
    for (index, message) in request.iter().enumerate() {
        if message.role != Role::Assistant {
            continue;
        }
        let pending: Vec<&str> = message
            .blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        if pending.is_empty() {
            continue;
        }
        let answered: Vec<&str> = request[index + 1..]
            .iter()
            .take_while(|message| message.role == Role::Tool)
            .flat_map(|message| message.blocks.iter())
            .filter_map(|block| match block {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
                _ => None,
            })
            .collect();
        for id in pending {
            assert!(
                answered.contains(&id),
                "call {id} has no result: {request:?}"
            );
        }
    }
}

// Answers the first prompt with a tool call whose tool outlives the run's
// timeout, so the run stops with that call still open; a later prompt gets
// plain text.
struct SlowToolProvider {
    requests: std::sync::Mutex<Vec<Vec<Message>>>,
}

#[async_trait]
impl Provider for SlowToolProvider {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        self.requests.lock().unwrap().push(messages.to_vec());
        let round = match messages.last() {
            _ if is_wrap_up(messages) => std::future::pending().await,
            Some(message) if message.role == Role::User && first_text(message) == "a" => {
                exec_round("await lofi.sleep(5)")
            }
            _ => text_round("resumed"),
        };
        Ok(Box::pin(stream::iter(round.into_iter().map(Ok))))
    }
}

#[tokio::test]
async fn timeout_inside_a_tool_call_stays_resumable() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let provider = Arc::new(SlowToolProvider {
        requests: std::sync::Mutex::new(Vec::new()),
    });
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = provider.clone();
    let hook = agent.subagent_hook().unwrap();

    // The tool outlives the run's timeout, so the run stops with its call open.
    let out = hook(SubagentOp::Run, json!({ "prompt": "a", "timeout_ms": 200 })).await;
    assert_eq!(out["status"], json!("timed_out"), "{out}");
    let id = out["id"].as_str().unwrap().to_string();

    let out = hook(
        SubagentOp::Run,
        json!({ "id": id, "prompt": "b", "timeout_ms": 5000 }),
    )
    .await;
    assert_eq!(out["status"], json!("completed"), "{out}");

    // The second resume replays the history that open call now sits inside.
    let out = hook(
        SubagentOp::Run,
        json!({ "id": id, "prompt": "c", "timeout_ms": 5000 }),
    )
    .await;
    assert_eq!(out["status"], json!("completed"), "{out}");

    // The stopped run also asks for a wrap-up; the task requests are what
    // the replay history is read from.
    let requests: Vec<_> = provider
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter(|request| !is_wrap_up(request))
        .cloned()
        .collect();
    assert_eq!(requests.len(), 3);
    for request in &requests {
        assert_tool_calls_are_answered(request);
    }
    let texts: Vec<_> = requests
        .last()
        .unwrap()
        .iter()
        .filter(|message| message.role == Role::User)
        .map(first_text)
        // The wrap-up prompt the stopped run was given stays in its history.
        .filter(|text| !is_wrap_up_text(text))
        .collect();
    assert_eq!(
        texts,
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        "{texts:?}"
    );
}

#[tokio::test]
async fn timeout_before_any_work_promises_no_resume() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(NeverProvider);
    let hook = agent.subagent_hook().unwrap();

    // The provider never answers, so the history holds only the task prompt
    // the run was handed: a resume would replay that prompt and nothing else.
    let out = hook(SubagentOp::Run, json!({ "prompt": "a", "timeout_ms": 100 })).await;
    assert_eq!(out["status"], json!("timed_out"), "{out}");
    assert_eq!(
        out["error"],
        json!("timed out after 100 ms; the run kept no work, so a resume would start over"),
        "{out}"
    );
}

// Answers the first prompt with plain text and hangs on every request after
// it, so a resumed run can be stopped before it writes anything of its own.
struct FirstTextThenStallProvider;

#[async_trait]
impl Provider for FirstTextThenStallProvider {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        match messages.last() {
            Some(message) if message.role == Role::User && first_text(message) == "a" => Ok(
                Box::pin(stream::iter(text_round("first").into_iter().map(Ok))),
            ),
            _ => std::future::pending().await,
        }
    }
}

#[tokio::test]
async fn resume_without_progress_still_names_the_resume() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(FirstTextThenStallProvider);
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "a" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    let id = out["id"].as_str().unwrap().to_string();

    // This leg appends only its prompt before the deadline fires, but the
    // history it would replay still holds the first run's answer, so the
    // entry kept work and the hint must name it.
    let out = hook(
        SubagentOp::Run,
        json!({ "id": id, "prompt": "b", "timeout_ms": 100 }),
    )
    .await;
    assert_eq!(out["status"], json!("timed_out"), "{out}");
    assert!(
        out["error"]
            .as_str()
            .unwrap()
            .contains(&format!("resume with id {id}")),
        "{out}"
    );
}

#[tokio::test]
async fn timeout_notice_names_the_resume_path() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let provider = Arc::new(TimeoutResumeProvider {
        requests: std::sync::Mutex::new(Vec::new()),
        tool_round: exec_round("return 1"),
        text_round: text_round("resumed"),
    });
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = provider;
    let hook = agent.subagent_hook().unwrap();
    let mut notices = agent.subagents().subscribe_notices();

    let out = hook(
        SubagentOp::Spawn,
        json!({ "prompt": "a", "timeout_ms": 2000 }),
    )
    .await;
    let id = out["id"].as_str().unwrap().to_string();
    let done = hook(SubagentOp::Wait, json!({ "id": id, "timeout_ms": 30_000 })).await;
    assert_eq!(done["status"], json!("timed_out"), "{done}");

    // The parent reads this notice as its prompt, so the hint must survive
    // both the first-line rule and the 200-byte detail cap.
    let notice = notices.try_recv().unwrap();
    assert!(
        notice.text.contains(&format!("resume with id {id}")),
        "{notice:?}"
    );
}

struct PendingProvider;

#[async_trait]
impl Provider for PendingProvider {
    async fn stream(
        &self,
        _model: &Model,
        _messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        std::future::pending().await
    }
}

// Poll the registry until an entry reaches `want`, returning its id.
async fn wait_for_status(registry: &SubagentRegistry, want: &str) -> String {
    for _ in 0..1000 {
        let hit = registry
            .ids()
            .into_iter()
            .find(|id| registry.status_of(id).as_deref() == Some(want));
        if let Some(id) = hit {
            return id;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("no subagent reached status {want}");
}

#[tokio::test]
async fn resume_while_running_is_rejected() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(PendingProvider);
    let registry = agent.subagents();
    let hook = agent.subagent_hook().unwrap();

    let run = tokio::spawn({
        let hook = hook.clone();
        async move { hook(SubagentOp::Run, json!({ "prompt": "a" })).await }
    });
    let id = wait_for_status(&registry, "running").await;
    let out = hook(SubagentOp::Run, json!({ "id": id, "prompt": "b" })).await;
    // A busy id reports `running`, not `failed`: the run is still going and
    // the id stays valid for a later resume.
    assert_eq!(out["status"], json!("running"), "{out}");
    assert_eq!(out["ok"], json!(false), "{out}");
    assert!(out["error"].as_str().unwrap().contains("running"), "{out}");
    assert_eq!(out["id"], json!(id), "{out}");
    run.abort();
}

#[tokio::test]
async fn dropped_hook_future_cancels_and_stays_resumable() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let provider = Arc::new(TimeoutResumeProvider {
        requests: std::sync::Mutex::new(Vec::new()),
        tool_round: exec_round("return 1"),
        text_round: text_round("resumed"),
    });
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = provider;
    let registry = agent.subagents();
    let hook = agent.subagent_hook().unwrap();

    // Dropping the hook future (parent turn aborted) must cancel the child:
    // the entry turns terminal and the same id stays resumable.
    let run = tokio::spawn({
        let hook = hook.clone();
        async move { hook(SubagentOp::Run, json!({ "prompt": "a" })).await }
    });
    let id = wait_for_status(&registry, "running").await;
    run.abort();
    let _ = run.await;
    let cancelled = wait_for_status(&registry, "cancelled").await;
    assert_eq!(cancelled, id);

    let out = hook(
        SubagentOp::Run,
        json!({ "id": id, "prompt": "b", "timeout_ms": 5000 }),
    )
    .await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["result"], json!("resumed"), "{out}");
}

#[tokio::test]
async fn registry_evicts_terminal_entries_at_capacity() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let rounds: Vec<_> = (0..67).map(|i| text_round(&format!("r{i}"))).collect();
    let agent = agent_with(rounds, dir.path()).with_subagent_support(support(dir.path()));
    let registry = agent.subagents();
    let hook = agent.subagent_hook().unwrap();

    let mut ids = Vec::new();
    for i in 0..64 {
        let out = hook(SubagentOp::Run, json!({ "prompt": format!("p{i}") })).await;
        assert_eq!(out["status"], json!("completed"), "{out}");
        ids.push(out["id"].as_str().unwrap().to_string());
    }
    assert_eq!(registry.ids().len(), 64);
    // The 65th spawn evicts the least recently used finished entry (p0)
    // instead of failing or dropping a run the guest may still resume.
    let out = hook(SubagentOp::Run, json!({ "prompt": "overflow" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(registry.ids().len(), 64);
    let overflow = out["id"].as_str().unwrap().to_string();

    let out = hook(SubagentOp::Run, json!({ "id": ids[0], "prompt": "again" })).await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    assert!(
        out["error"]
            .as_str()
            .unwrap()
            .contains("unknown subagent id"),
        "{out}"
    );
    let out = hook(SubagentOp::Run, json!({ "id": ids[63], "prompt": "again" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["result"], json!("r65"), "{out}");
    let out = hook(
        SubagentOp::Run,
        json!({ "id": overflow, "prompt": "again" }),
    )
    .await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["result"], json!("r66"), "{out}");
}

#[tokio::test]
async fn running_entries_hit_the_concurrency_cap() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(PendingProvider);
    let registry = agent.subagents();
    let hook = agent.subagent_hook().unwrap();

    let mut runs = Vec::new();
    for _ in 0..8 {
        runs.push(tokio::spawn({
            let hook = hook.clone();
            async move { hook(SubagentOp::Run, json!({ "prompt": "a" })).await }
        }));
    }
    for _ in 0..1000 {
        if registry.ids().len() == 8 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(registry.ids().len(), 8, "entries never filled");

    let out = hook(SubagentOp::Run, json!({ "prompt": "overflow" })).await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    assert!(
        out["error"].as_str().unwrap().contains("concurrency cap"),
        "{out}"
    );

    for run in runs {
        run.abort();
    }
}

#[tokio::test]
async fn spawn_applies_the_system_override() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let provider = Arc::new(TimeoutResumeProvider {
        requests: std::sync::Mutex::new(Vec::new()),
        tool_round: exec_round("return 1"),
        text_round: text_round("done"),
    });
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = provider.clone();
    let hook = agent.subagent_hook().unwrap();

    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "b", "system": "custom sys" }),
    )
    .await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    // Scope the guard: it must not live across the second spawn, whose
    // provider call takes the same lock.
    {
        let requests = provider.requests.lock().unwrap();
        let first = &requests[0][0];
        assert_eq!(first.role, Role::System, "{first:?}");
        assert_eq!(first_text(first), "custom sys");
    }

    // Without an override the child keeps the session system prompt.
    let out = hook(SubagentOp::Run, json!({ "prompt": "b" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    let requests = provider.requests.lock().unwrap();
    assert_eq!(first_text(&requests[1][0]), "sys");
}

#[tokio::test]
async fn completed_run_without_text_is_a_failure() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![text_round(""), text_round("fine")], dir.path())
        .with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    // An empty final answer must not surface as a successful empty result.
    let out = hook(SubagentOp::Run, json!({ "prompt": "a" })).await;
    assert_eq!(out["ok"], json!(false), "{out}");
    assert_eq!(out["status"], json!("failed"), "{out}");
    assert!(
        out["error"]
            .as_str()
            .unwrap()
            .contains("without a text result"),
        "{out}"
    );

    // A failed entry stays resumable.
    let id = out["id"].as_str().unwrap().to_string();
    let out = hook(SubagentOp::Run, json!({ "id": id, "prompt": "b" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["result"], json!("fine"), "{out}");
}

// The target checks a level, so an override at the level the child already
// runs at returns the child unchanged and needs no rebuild. The mock session
// model declares no levels, and it runs at off.
#[tokio::test]
async fn thinking_override_at_the_childs_level_keeps_the_child() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent =
        agent_with(vec![text_round("ok")], dir.path()).with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "a", "thinking": "off" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
}

// Observable through the sandbox, not the fields: after `without_subagent`
// a guest that calls `lofi.models()` gets the unavailable status.
#[tokio::test]
async fn without_subagent_makes_models_unavailable_in_the_sandbox() {
    let dir = tempdir().unwrap();
    let mut executor = ToolExecutor::new(dir.path(), &lofi_types::Config::default(), &[]).unwrap();
    executor.subagent = Some(Arc::new(
        |_, _| -> std::pin::Pin<Box<dyn std::future::Future<Output = serde_json::Value> + Send>> {
            Box::pin(async { json!(null) })
        },
    ));
    executor.models = Some(Arc::new(|| json!([])));
    let executor = executor.without_subagent();
    let context = ToolCallContext {
        events: None,
        session: None,
        recall: None,
        result: None,
        cancel: None,
        on_job_acquired: None,
        record_tool: None,
        alive: None,
    };
    let out = executor
        .execute(
            lofi_code::EXEC_TOOL_NAME,
            &json!({ "code": "return lofi.models()", "description": "list models" }),
            context,
        )
        .await
        .unwrap();
    let ContentBlock::ToolResult { content, .. } = out else {
        panic!("expected a tool result: {out:?}");
    };
    assert!(content.contains("\"status\":\"unavailable\""), "{content}");
}

fn find_jsonl(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                out.extend(find_jsonl(&path));
            } else if path.extension().is_some_and(|e| e == "jsonl") {
                out.push(path);
            }
        }
    }
    out
}

fn count_blocks(messages: &[Message], want: impl Fn(&ContentBlock) -> bool) -> usize {
    messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .filter(|block| want(block))
        .count()
}

fn first_text(m: &Message) -> String {
    m.blocks
        .iter()
        .find_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .unwrap_or("")
        .to_string()
}

#[tokio::test]
async fn settled_run_spend_folds_into_session_totals() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let child = vec![
        StreamingEvent::TextDelta("sub result".to_string()),
        StreamingEvent::Done {
            usage: Usage {
                input_tokens: 11,
                output_tokens: 7,
                ..Usage::default()
            },
            stop_reason: None,
        },
    ];
    let agent = agent_with(vec![child], dir.path()).with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "do it" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    let registry = agent.subagents();
    let (cost, usage) = registry.take_unreported().unwrap();
    assert_eq!(usage.input_tokens, 11);
    assert_eq!(usage.output_tokens, 7);
    assert!(cost >= 0.0);
    assert!(registry.take_unreported().is_none(), "spend is taken once");
}

#[tokio::test]
async fn usage_rollup_folds_each_retained_entry() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let child = |input_tokens: u64, output_tokens: u64| {
        vec![
            StreamingEvent::TextDelta("sub result".to_string()),
            StreamingEvent::Done {
                usage: Usage {
                    input_tokens,
                    output_tokens,
                    ..Usage::default()
                },
                stop_reason: None,
            },
        ]
    };
    let provider = Arc::new(MockProvider::recording(vec![
        child(11, 7),
        child(5, 3),
        child(9, 4),
    ]));
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = provider;
    let hook = agent.subagent_hook().unwrap();

    let first = hook(SubagentOp::Run, json!({ "prompt": "a" })).await;
    let first_id = first["id"].as_str().unwrap().to_string();
    let _ = hook(SubagentOp::Run, json!({ "prompt": "b" })).await;

    let all = hook(SubagentOp::Usage, json!({})).await;
    assert_eq!(all["ok"], json!(true), "{all}");
    assert_eq!(all["runs"], json!(2), "{all}");
    assert_eq!(all["input_tokens"], json!(16), "{all}");
    assert_eq!(all["output_tokens"], json!(10), "{all}");
    assert_eq!(all["cache_read_tokens"], json!(0), "{all}");
    assert!(all["cost"].as_f64().is_some(), "{all}");

    // A named id narrows the fold; one that no entry holds is reported, not
    // failed: entries are evicted at capacity.
    let one = hook(
        SubagentOp::Usage,
        json!({ "ids": [first_id, first_id, "no-such-run"] }),
    )
    .await;
    assert_eq!(one["runs"], json!(1), "a repeated id counts once: {one}");
    assert_eq!(one["input_tokens"], json!(11), "{one}");
    assert_eq!(one["missing"], json!(["no-such-run"]), "{one}");

    // A resume adds the run it replaces to the entry's earlier runs, so the
    // fold prices the child session and not just the run in flight.
    let again = hook(SubagentOp::Run, json!({ "id": first_id, "prompt": "more" })).await;
    assert_eq!(again["ok"], json!(true), "{again}");
    let resumed = hook(SubagentOp::Usage, json!({ "ids": [first_id] })).await;
    assert_eq!(resumed["runs"], json!(2), "{resumed}");
    assert_eq!(resumed["input_tokens"], json!(20), "{resumed}");
    let all = hook(SubagentOp::Usage, json!({})).await;
    assert_eq!(all["input_tokens"], json!(25), "{all}");
    assert_eq!(all["runs"], json!(3), "{all}");

    let bad = hook(SubagentOp::Usage, json!({ "ids": "x" })).await;
    assert_eq!(bad["ok"], json!(false), "{bad}");
    assert_eq!(bad["status"], json!("failed"), "{bad}");
}

// The child's transcript is the child session: its settle marker prices the
// run it records plus the runs before it, so a dialog row and a restart both
// see what the session spent, not what its last run did.
#[tokio::test]
async fn a_resumed_child_transcript_totals_both_runs() {
    let _env = env_lock();
    let (_state, state_path) = StateEnv::new();
    let dir = tempdir().unwrap();
    let round = |input: u64| {
        vec![
            StreamingEvent::TextDelta("sub result".to_string()),
            StreamingEvent::Done {
                usage: Usage {
                    input_tokens: input,
                    ..Usage::default()
                },
                stop_reason: None,
            },
        ]
    };
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    // A priced model: one million input tokens bill at the input price.
    agent.model.input_price = Some(3.0);
    agent.model.output_price = Some(15.0);
    agent.provider = Arc::new(MockProvider::recording(vec![
        round(1_000_000),
        round(500_000),
    ]));
    agent.subagents().set_parent_session("parent-x");
    let hook = agent.subagent_hook().unwrap();

    let first = hook(SubagentOp::Run, json!({ "prompt": "a" })).await;
    let id = first["id"].as_str().unwrap().to_string();
    let again = hook(SubagentOp::Run, json!({ "id": id, "prompt": "b" })).await;
    assert_eq!(again["status"], json!("completed"), "{again}");

    let files = find_jsonl(&state_path);
    assert_eq!(files.len(), 1, "{files:?}");

    // The fold of the live entry prices both runs.
    let folded = hook(SubagentOp::Usage, json!({ "ids": [id] })).await;
    assert_eq!(folded["runs"], json!(2), "{folded}");
    let folded_cost = folded["cost"].as_f64().unwrap();
    assert!((folded_cost - 4.5).abs() < 1e-9, "{folded}");

    // So does the fold of an entry a restart restored: the marker's earlier
    // runs belong to the new entry too.
    let restarted = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    restarted.subagents().set_parent_session("parent-x");
    restarted
        .subagents()
        .rehydrate(&restarted, &support(dir.path()));
    let hook = restarted.subagent_hook().unwrap();
    let restored = hook(SubagentOp::Usage, json!({ "ids": [id] })).await;
    assert_eq!(restored["runs"], json!(2), "{restored}");
    let restored_cost = restored["cost"].as_f64().unwrap();
    assert!((restored_cost - 4.5).abs() < 1e-9, "{restored}");

    // The dialog row takes a settled entry's cost from the entry, so it must
    // agree with the marker instead of showing the earlier runs alone.
    let row = restarted
        .subagents()
        .live()
        .into_iter()
        .find(|run| run.id == id)
        .expect("the restart restored the entry");
    let row_cost = row.cost.expect("the restored row keeps a cost");
    assert!(
        (row_cost - 4.5).abs() < 1e-9,
        "the row prices both runs: {row_cost}"
    );
}

#[tokio::test]
async fn background_run_survives_a_per_turn_runtime() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![text_round("bg done")], dir.path())
        .with_subagent_support(support(dir.path()));
    // The TUI's turn loop runs on its own worker runtime that dies when
    // the turn ends; recording it as the host runtime must not win over
    // the construction-time capture.
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let turn = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                turn.block_on(async {
                    agent
                        .subagents()
                        .set_host_handle(tokio::runtime::Handle::current());
                });
            })
            .join()
            .unwrap();
    });
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Spawn, json!({ "prompt": "a" })).await;
    let id = out["id"].as_str().unwrap().to_string();
    let done = hook(SubagentOp::Wait, json!({ "id": id, "timeout_ms": 30_000 })).await;
    assert_eq!(done["status"], json!("completed"), "{done}");
    assert_eq!(done["result"], json!("bg done"), "{done}");
}

#[tokio::test]
async fn cache_only_spend_is_still_reported() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    // A cache hit with no billable tokens and no cost still counts: the
    // predicate must not wait for another run to carry it.
    let child = vec![
        StreamingEvent::TextDelta("sub result".to_string()),
        StreamingEvent::Done {
            usage: Usage {
                cache_read_tokens: 500,
                ..Usage::default()
            },
            stop_reason: None,
        },
    ];
    let agent = agent_with(vec![child], dir.path()).with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "do it" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    let (_, usage) = agent.subagents().take_unreported().unwrap();
    assert_eq!(usage.cache_read_tokens, 500);
}

#[tokio::test]
async fn parent_turn_records_and_reports_subagent_spend() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let tool_round = vec![
        StreamingEvent::ToolUseStart {
            id: "t1".into(),
            name: "exec".into(),
        },
        StreamingEvent::ToolUseInputDelta {
            id: "t1".into(),
            delta: r#"{"description":"run subagent","code":"const r = await lofi.subagent({ prompt: 'x' }); return r.status;"}"#
                .into(),
        },
        StreamingEvent::ToolUseInputComplete {
            id: "t1".into(),
            input: r#"{"description":"run subagent","code":"const r = await lofi.subagent({ prompt: 'x' }); return r.status;"}"#
                .into(),
        },
        StreamingEvent::ToolUseEnd { id: "t1".into() },
        StreamingEvent::Done {
            usage: Usage::default(),
            stop_reason: Some(lofi_types::StopReason::ToolUse),
        },
    ];
    let child_round = vec![
        StreamingEvent::TextDelta("child answer".into()),
        StreamingEvent::Done {
            usage: Usage {
                input_tokens: 13,
                output_tokens: 5,
                ..Usage::default()
            },
            stop_reason: None,
        },
    ];
    let final_round = vec![
        StreamingEvent::TextDelta("done".into()),
        StreamingEvent::Done {
            usage: Usage::default(),
            stop_reason: Some(lofi_types::StopReason::EndTurn),
        },
    ];
    let agent = agent_with(vec![tool_round, child_round, final_round], dir.path())
        .with_subagent_support(support(dir.path()));
    let path = dir.path().join("s.jsonl");
    std::fs::write(
        &path,
        b"{\"type\":\"meta\",\"version\":1,\"created\":0,\"cwd\":\"\",\"model\":\"m\"}\n",
    )
    .unwrap();
    let cursor = crate::session::store::SessionCursor::new(path, None);
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let mut messages = Vec::new();
    agent
        .run_turn(
            &mut messages,
            TurnRequest::prompt("go".into(), lofi_types::PromptKind::User),
            tx,
            Some(&cursor),
            RunControls {
                cancel: None,
                preempt: None,
            },
        )
        .await
        .unwrap();

    let mut reported = None;
    while let Ok(ev) = rx.try_recv() {
        if let AgentEvent::SubagentUsage { usage, .. } = ev {
            reported = Some(usage);
        }
    }
    let usage = reported.unwrap();
    assert_eq!(usage.input_tokens, 13);
    assert_eq!(usage.output_tokens, 5);
    let events = cursor.load_tree_events().unwrap();
    assert!(
        events.iter().any(|event| matches!(
            &event.kind,
            SessionEventKind::SubagentUsage { usage, .. }
                if usage.input_tokens == 13 && usage.output_tokens == 5
        )),
        "transcript carries the subagent spend"
    );
}

#[tokio::test]
async fn spawn_returns_running_and_wait_collects_the_outcome() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![text_round("bg result")], dir.path())
        .with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Spawn, json!({ "prompt": "do it" })).await;
    assert_eq!(out["ok"], json!(true), "{out}");
    assert_eq!(out["status"], json!("running"), "{out}");
    let id = out["id"].as_str().unwrap().to_string();

    let done = hook(SubagentOp::Wait, json!({ "id": id, "timeout_ms": 30_000 })).await;
    assert_eq!(done["ok"], json!(true), "{done}");
    assert_eq!(done["status"], json!("completed"), "{done}");
    assert_eq!(done["result"], json!("bg result"), "{done}");
}

struct BigUsageProvider;

#[async_trait]
impl Provider for BigUsageProvider {
    async fn stream(
        &self,
        _model: &Model,
        _messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        Ok(Box::pin(stream::iter([
            Ok(StreamingEvent::TextDelta("done".to_string())),
            Ok(StreamingEvent::Done {
                usage: Usage {
                    input_tokens: 100,
                    output_tokens: 50,
                    ..Usage::default()
                },
                stop_reason: None,
            }),
        ])))
    }
}

#[tokio::test]
async fn token_budget_exceeded_fails_the_run() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(BigUsageProvider);
    let hook = agent.subagent_hook().unwrap();

    let out = hook(
        SubagentOp::Spawn,
        json!({ "prompt": "spend", "token_budget": 10 }),
    )
    .await;
    assert_eq!(out["ok"], json!(true), "{out}");
    let id = out["id"].as_str().unwrap().to_string();

    let done = hook(SubagentOp::Wait, json!({ "id": id, "timeout_ms": 30_000 })).await;
    assert_eq!(done["status"], json!("failed"), "{done}");
    // The budget stop keeps the history too, so its error names the resume
    // path for the same reason a timeout does.
    assert_eq!(
        done["error"],
        json!(format!(
            "token budget exceeded (150 > 10); resume with id {id} to continue from the work so far"
        )),
        "{done}"
    );
}

#[tokio::test]
async fn token_budget_zero_is_rejected() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "x", "token_budget": 0 })).await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    assert!(
        out["error"].as_str().unwrap().contains("token_budget"),
        "{out}"
    );
    let out = hook(
        SubagentOp::Spawn,
        json!({ "prompt": "x", "token_budget": 0 }),
    )
    .await;
    assert_eq!(out["ok"], json!(false), "{out}");
}

#[tokio::test]
async fn read_pages_do_not_repeat_events() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![text_round("paged")], dir.path())
        .with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Spawn, json!({ "prompt": "page me" })).await;
    let id = out["id"].as_str().unwrap().to_string();
    let done = hook(SubagentOp::Wait, json!({ "id": id, "timeout_ms": 30_000 })).await;
    assert_eq!(done["status"], json!("completed"), "{done}");

    let page1 = hook(SubagentOp::Read, json!({ "id": id, "limit": 2 })).await;
    let events1 = page1["events"].as_array().unwrap();
    assert_eq!(events1.len(), 2, "{page1}");
    let cursor = page1["cursor"].as_u64().unwrap();
    let page2 = hook(SubagentOp::Read, json!({ "id": id, "cursor": cursor })).await;
    let events2 = page2["events"].as_array().unwrap();
    assert!(!events2.is_empty(), "{page2}");
    let first = events2[0]["seq"].as_u64().unwrap();
    assert!(first >= cursor, "page 2 repeats page 1: {page2}");
    assert_eq!(page2["done"], json!(true), "{page2}");
}

#[tokio::test]
async fn wait_after_settle_returns_the_stored_outcome() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![text_round("late waiter")], dir.path())
        .with_subagent_support(support(dir.path()));
    let registry = agent.subagents();
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Spawn, json!({ "prompt": "quick" })).await;
    let id = out["id"].as_str().unwrap().to_string();
    // Poll status until terminal: no waiter is subscribed when the run
    // settles, so the watch channel must retain the terminal state.
    for _ in 0..1000 {
        let status = hook(SubagentOp::Status, json!({ "id": id })).await;
        if status["status"] != json!("running") {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(registry.ids().contains(&id));
    let done = hook(SubagentOp::Wait, json!({ "id": id, "timeout_ms": 30_000 })).await;
    assert_eq!(done["status"], json!("completed"), "{done}");
    assert_eq!(done["result"], json!("late waiter"), "{done}");
}

#[tokio::test]
async fn resume_is_rejected_at_the_concurrency_cap() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![text_round("settled")], dir.path())
        .with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    // One settled run to resume later.
    let out = hook(SubagentOp::Run, json!({ "prompt": "first" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    let resume_id = out["id"].as_str().unwrap().to_string();

    // Fill the running set with eight pending runs.
    agent.provider = Arc::new(NeverProvider);
    let registry = agent.subagents();
    let mut running = Vec::new();
    for _ in 0..8 {
        let out = hook(SubagentOp::Spawn, json!({ "prompt": "hang" })).await;
        assert_eq!(out["ok"], json!(true), "{out}");
        running.push(out["id"].as_str().unwrap().to_string());
    }
    assert_eq!(registry.ids().len(), 9);

    let out = hook(
        SubagentOp::Run,
        json!({ "id": resume_id, "prompt": "again" }),
    )
    .await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    assert!(
        out["error"].as_str().unwrap().contains("concurrency cap"),
        "{out}"
    );

    for id in running {
        hook(SubagentOp::Kill, json!({ "id": id })).await;
    }
}

#[tokio::test]
async fn foreground_run_publishes_no_terminal_notice() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![text_round("quiet")], dir.path())
        .with_subagent_support(support(dir.path()));
    let mut notices = agent.subagents().subscribe_notices();
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "quiet" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert!(
        notices.try_recv().is_err(),
        "foreground run published a notice"
    );
}

struct CacheUsageProvider;

#[async_trait]
impl Provider for CacheUsageProvider {
    async fn stream(
        &self,
        _model: &Model,
        _messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        Ok(Box::pin(stream::iter([
            Ok(StreamingEvent::TextDelta("done".to_string())),
            Ok(StreamingEvent::Done {
                usage: Usage {
                    cache_read_tokens: 100,
                    ..Usage::default()
                },
                stop_reason: None,
            }),
        ])))
    }
}

#[tokio::test]
async fn token_budget_counts_cache_tokens() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(CacheUsageProvider);
    let hook = agent.subagent_hook().unwrap();

    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "cache", "token_budget": 10 }),
    )
    .await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    assert!(
        out["error"]
            .as_str()
            .unwrap()
            .contains("token budget exceeded"),
        "{out}"
    );
}

#[tokio::test]
async fn resume_of_a_background_run_publishes_no_notice() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![text_round("first"), text_round("second")], dir.path())
        .with_subagent_support(support(dir.path()));
    let mut notices = agent.subagents().subscribe_notices();
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Spawn, json!({ "prompt": "a" })).await;
    let id = out["id"].as_str().unwrap().to_string();
    let done = hook(SubagentOp::Wait, json!({ "id": id, "timeout_ms": 30_000 })).await;
    assert_eq!(done["status"], json!("completed"), "{done}");
    // The background run published its terminal notice; drain it.
    while notices.try_recv().is_ok() {}

    // Resume is always a foreground op: its outcome goes to the caller, so
    // no second notice may be queued as a parent prompt.
    let out = hook(SubagentOp::Run, json!({ "id": id, "prompt": "b" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["result"], json!("second"), "{out}");
    assert!(
        notices.try_recv().is_err(),
        "foreground resume published a notice"
    );
}

#[tokio::test]
async fn multibyte_prompt_at_the_event_cap_does_not_panic() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent =
        agent_with(vec![text_round("ok")], dir.path()).with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    // Byte 8192 lands inside the two-byte character.
    let mut prompt = "x".repeat(8191);
    prompt.push('\u{00e9}');
    prompt.push_str(&"y".repeat(64));
    let out = hook(SubagentOp::Run, json!({ "prompt": prompt })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
}

#[tokio::test]
async fn read_rejects_non_string_types_items() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent =
        agent_with(vec![text_round("ok")], dir.path()).with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "a" })).await;
    let id = out["id"].as_str().unwrap().to_string();
    let out = hook(SubagentOp::Read, json!({ "id": id, "types": ["text", 3] })).await;
    assert_eq!(out["ok"], json!(false), "{out}");
    assert!(out["error"].as_str().unwrap().contains("types"), "{out}");
}

#[tokio::test]
async fn spawn_rejects_resume_and_output_limit_args() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(
        SubagentOp::Spawn,
        json!({ "prompt": "x", "id": "whatever" }),
    )
    .await;
    assert_eq!(out["ok"], json!(false), "{out}");
    assert_eq!(out["status"], json!("failed"), "{out}");
    let out = hook(
        SubagentOp::Spawn,
        json!({ "prompt": "x", "output_limit": 10 }),
    )
    .await;
    assert_eq!(out["ok"], json!(false), "{out}");
}

#[tokio::test]
async fn wait_timeout_reports_running_and_never_cancels() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(NeverProvider);
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Spawn, json!({ "prompt": "hang" })).await;
    let id = out["id"].as_str().unwrap().to_string();
    let out = hook(SubagentOp::Wait, json!({ "id": id, "timeout_ms": 50 })).await;
    assert_eq!(out["ok"], json!(true), "{out}");
    assert_eq!(out["status"], json!("running"), "{out}");

    // The run survived the timed-out wait.
    let status = hook(SubagentOp::Status, json!({ "id": id })).await;
    assert_eq!(status["status"], json!("running"), "{status}");

    // Clean up so the test runtime does not drop a live run.
    let killed = hook(SubagentOp::Kill, json!({ "id": id })).await;
    assert_eq!(killed["ok"], json!(true), "{killed}");
    let done = hook(SubagentOp::Wait, json!({ "id": id, "timeout_ms": 30_000 })).await;
    assert_eq!(done["status"], json!("cancelled"), "{done}");
}

#[tokio::test]
async fn kill_is_idempotent_and_reports_terminal_state() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(NeverProvider);
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Spawn, json!({ "prompt": "hang" })).await;
    let id = out["id"].as_str().unwrap().to_string();
    let first = hook(SubagentOp::Kill, json!({ "id": id, "reason": "test" })).await;
    assert_eq!(first["ok"], json!(true), "{first}");
    let done = hook(SubagentOp::Wait, json!({ "id": id, "timeout_ms": 30_000 })).await;
    assert_eq!(done["status"], json!("cancelled"), "{done}");
    // The kill backstop settles through the drop guard, which names the
    // model like the normal settle does.
    assert_eq!(done["model"], json!("p/m"), "{done}");
    let second = hook(SubagentOp::Kill, json!({ "id": id })).await;
    assert_eq!(second["ok"], json!(true), "{second}");
    assert_eq!(second["status"], json!("cancelled"), "{second}");
}

#[tokio::test]
async fn killing_a_run_drops_the_timers_it_armed() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(NeverProvider);
    let timers = agent.tool_executor().timers();
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Spawn, json!({ "prompt": "hang" })).await;
    let id = out["id"].as_str().unwrap().to_string();
    // A child's timers live in the session's registry, tagged with its run.
    timers.schedule_every(Some(&id), std::time::Duration::from_hours(1), None);
    assert_eq!(timers.pending_for(Some(&id)), 1);

    let killed = hook(SubagentOp::Kill, json!({ "id": id })).await;
    assert_eq!(killed["ok"], json!(true), "{killed}");
    let done = hook(SubagentOp::Wait, json!({ "id": id, "timeout_ms": 30_000 })).await;
    assert_eq!(done["status"], json!("cancelled"), "{done}");

    assert_eq!(timers.pending_for(Some(&id)), 0);
}

#[tokio::test]
async fn replacing_the_session_drops_the_timers_of_stopped_runs() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(NeverProvider);
    let timers = agent.tool_executor().timers();
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Spawn, json!({ "prompt": "hang" })).await;
    let id = out["id"].as_str().unwrap().to_string();
    timers.schedule_at(
        Some(&id),
        std::time::SystemTime::now() + std::time::Duration::from_hours(1),
        None,
    );
    assert_eq!(timers.pending_for(Some(&id)), 1);

    let _ = agent.subagents().clear();

    assert_eq!(timers.pending_for(Some(&id)), 0);
}

#[tokio::test]
async fn read_pages_events_and_reports_gaps() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![text_round("page me")], dir.path())
        .with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Spawn, json!({ "prompt": "do it" })).await;
    let id = out["id"].as_str().unwrap().to_string();
    let done = hook(SubagentOp::Wait, json!({ "id": id, "timeout_ms": 30_000 })).await;
    assert_eq!(done["status"], json!("completed"), "{done}");

    let page = hook(SubagentOp::Read, json!({ "id": id })).await;
    assert_eq!(page["ok"], json!(true), "{page}");
    assert_eq!(page["done"], json!(true), "{page}");
    assert_eq!(page["dropped"], json!(0), "{page}");
    let events = page["events"].as_array().unwrap();
    let types: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();
    assert!(types.contains(&"prompt"), "{types:?}");
    assert!(types.contains(&"text"), "{types:?}");
    assert_eq!(types.last(), Some(&"state"), "{types:?}");
    // Sequence numbers are consecutive from 0; the cursor is past the end.
    let seqs: Vec<u64> = events.iter().filter_map(|e| e["seq"].as_u64()).collect();
    assert!(
        seqs.iter().enumerate().all(|(i, seq)| *seq == i as u64),
        "{seqs:?}"
    );
    assert_eq!(page["cursor"].as_u64().unwrap(), seqs.len() as u64);

    // A filtered page advances the cursor over filtered-out events.
    let filtered = hook(SubagentOp::Read, json!({ "id": id, "types": ["state"] })).await;
    let fseq: Vec<u64> = filtered["events"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|e| e["seq"].as_u64())
        .collect();
    assert_eq!(fseq.len(), 1, "{filtered}");
    assert_eq!(filtered["cursor"], page["cursor"]);
}

#[tokio::test]
async fn list_filters_state_and_pages_by_creation_order() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent =
        agent_with(vec![text_round("done")], dir.path()).with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let first = hook(SubagentOp::Spawn, json!({ "prompt": "one" })).await;
    let first_id = first["id"].as_str().unwrap().to_string();
    let done = hook(
        SubagentOp::Wait,
        json!({ "id": first_id, "timeout_ms": 30_000 }),
    )
    .await;
    assert_eq!(done["status"], json!("completed"), "{done}");

    // The second run never finishes: it stays in the running filter.
    agent.provider = Arc::new(NeverProvider);
    let hook = agent.subagent_hook().unwrap();
    let second = hook(SubagentOp::Spawn, json!({ "prompt": "two" })).await;
    let second_id = second["id"].as_str().unwrap().to_string();

    let all = hook(SubagentOp::List, json!({})).await;
    assert_eq!(all["entries"].as_array().unwrap().len(), 2, "{all}");
    let running = hook(SubagentOp::List, json!({ "state": "running" })).await;
    let rids: Vec<&str> = running["entries"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|e| e["id"].as_str())
        .collect();
    assert_eq!(rids, vec![second_id.as_str()], "{running}");

    let page1 = hook(SubagentOp::List, json!({ "limit": 1 })).await;
    assert_eq!(page1["entries"].as_array().unwrap().len(), 1, "{page1}");
    let cursor = page1["cursor"].as_str().unwrap().to_string();
    let page2 = hook(SubagentOp::List, json!({ "limit": 1, "cursor": cursor })).await;
    assert_eq!(page2["entries"].as_array().unwrap().len(), 1, "{page2}");
    assert!(page2.get("cursor").is_none(), "{page2}");
    let id1 = page1["entries"][0]["id"].as_str().unwrap();
    let id2 = page2["entries"][0]["id"].as_str().unwrap();
    assert_ne!(id1, id2);

    let _ = hook(SubagentOp::Kill, json!({ "id": second_id })).await;
}

#[tokio::test]
async fn concurrency_cap_rejects_spawn_over_the_running_limit() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(NeverProvider);
    let hook = agent.subagent_hook().unwrap();

    let mut ids = Vec::new();
    for _ in 0..8 {
        let out = hook(SubagentOp::Spawn, json!({ "prompt": "hang" })).await;
        assert_eq!(out["ok"], json!(true), "{out}");
        ids.push(out["id"].as_str().unwrap().to_string());
    }
    let over = hook(SubagentOp::Spawn, json!({ "prompt": "once more" })).await;
    assert_eq!(over["ok"], json!(false), "{over}");
    assert_eq!(over["status"], json!("failed"), "{over}");
    assert!(
        over["error"].as_str().unwrap().contains("concurrency cap"),
        "{over}"
    );

    for id in ids {
        let _ = hook(SubagentOp::Kill, json!({ "id": id })).await;
    }
}

#[tokio::test]
async fn spawned_entry_resumes_through_the_foreground_op() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let provider = Arc::new(MockProvider::recording(vec![
        text_round("first"),
        text_round("second"),
    ]));
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = provider.clone();
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Spawn, json!({ "prompt": "a" })).await;
    let id = out["id"].as_str().unwrap().to_string();
    let done = hook(SubagentOp::Wait, json!({ "id": id, "timeout_ms": 30_000 })).await;
    assert_eq!(done["result"], json!("first"), "{done}");

    let resumed = hook(SubagentOp::Run, json!({ "id": id, "prompt": "b" })).await;
    assert_eq!(resumed["status"], json!("completed"), "{resumed}");
    assert_eq!(resumed["result"], json!("second"), "{resumed}");
    assert_eq!(provider.requests().len(), 2);
}

// Restart rehydration: entries are restored from transcripts so a run can be
// resumed after the process that spawned it exited. Each test builds a fresh
// registry over the same state dir to stand in for the restart.

fn user_event(text: &str) -> lofi_types::SessionEvent {
    lofi_types::SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: lofi_types::SessionEventKind::Message(Message {
            origin: None,
            role: Role::User,
            blocks: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            kind: PromptKind::default(),
        }),
    }
}

#[tokio::test]
async fn rehydrated_run_resumes_with_replayed_history() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let provider = Arc::new(MockProvider::recording(vec![
        text_round("first"),
        text_round("second"),
    ]));
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = provider.clone();
    agent.subagents().set_parent_session("parent-x");
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "a" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    let id = out["id"].as_str().unwrap().to_string();
    let session_id = out["session_id"].as_str().unwrap().to_string();

    // The restart: a fresh agent and registry over the same state dir.
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = provider.clone();
    agent.subagents().set_parent_session("parent-x");
    agent.subagents().rehydrate(&agent, &support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let list = hook(SubagentOp::List, json!({})).await;
    let entries = list["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 1, "{list}");
    assert_eq!(entries[0]["id"], json!(id), "{list}");
    assert_eq!(entries[0]["status"], json!("completed"), "{list}");
    assert_eq!(entries[0]["session_id"], json!(session_id), "{list}");

    let status = hook(SubagentOp::Status, json!({ "id": id })).await;
    assert_eq!(status["status"], json!("completed"), "{status}");

    let out = hook(SubagentOp::Run, json!({ "id": id, "prompt": "b" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["result"], json!("second"), "{out}");

    // The resume replayed the transcript: the provider saw the original
    // exchange before the new prompt.
    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    let roles: Vec<_> = requests[1]
        .iter()
        .map(|m| (m.role, first_text(m)))
        .collect();
    let a = roles
        .iter()
        .position(|(r, t)| *r == Role::User && t == "a")
        .unwrap();
    assert_eq!(roles[a + 1], (Role::Assistant, "first".to_string()));
    assert_eq!(roles[a + 2], (Role::User, "b".to_string()));
}

#[tokio::test]
async fn rehydrate_without_settle_marker_is_cancelled() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    // A transcript whose run never settled: prompt recorded, no marker.
    let store = crate::session::store::SessionStore::open().unwrap();
    let cursor = store
        .create_subagent_cursor(dir.path(), &"p/m".into(), Some("parent-x"), "runid001")
        .unwrap();
    cursor.append_events(&mut [user_event("a")]).unwrap();

    let agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.subagents().set_parent_session("parent-x");
    agent.subagents().rehydrate(&agent, &support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let status = hook(SubagentOp::Status, json!({ "id": "runid001" })).await;
    assert_eq!(status["status"], json!("cancelled"), "{status}");
    assert!(
        status["error"]
            .as_str()
            .unwrap()
            .contains("exited before the run settled"),
        "{status}"
    );
}

#[tokio::test]
async fn a_restored_run_keeps_its_mailbox_and_no_run_management() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let store = crate::session::store::SessionStore::open().unwrap();
    let cursor = store
        .create_subagent_cursor(dir.path(), &"p/m".into(), Some("parent-x"), "runid004")
        .unwrap();
    cursor.append_events(&mut [user_event("a")]).unwrap();

    let agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.subagents().set_parent_session("parent-x");
    agent.subagents().rehydrate(&agent, &support(dir.path()));
    // The restored entry is a child, not a second manager: its hook answers
    // mailbox ops and reports run management as unavailable.
    let restored = agent.subagents().backend_of("runid004").unwrap();
    let hook = restored.subagent_hook().unwrap();
    let managed = hook(SubagentOp::List, json!({})).await;
    assert_eq!(managed["status"], json!("unavailable"), "{managed}");
    let mailbox = hook(SubagentOp::MailboxList, json!({})).await;
    assert_eq!(mailbox["agent_id"], json!("runid004"), "{mailbox}");
    // A restored run is terminal until a resume claims it: the address takes
    // no new messages.
    let parent = agent.subagent_hook().unwrap();
    let sent = parent(
        SubagentOp::MailboxSend,
        json!({ "to": "runid004", "body": "late" }),
    )
    .await;
    assert_eq!(sent["status"], json!("recipient_ended"), "{sent}");
}

#[tokio::test]
async fn a_failed_resume_closes_the_inbox_again() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let store = crate::session::store::SessionStore::open().unwrap();
    let cursor = store
        .create_subagent_cursor(dir.path(), &"p/m".into(), Some("parent-x"), "runid005")
        .unwrap();
    cursor.append_events(&mut [user_event("a")]).unwrap();
    let path = cursor.path().to_path_buf();

    let agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.subagents().set_parent_session("parent-x");
    agent.subagents().rehydrate(&agent, &support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    // The claim reopens the address; the load then fails, so the claim must
    // close it again instead of leaving a run that no one will start.
    std::fs::remove_file(&path).unwrap();
    let resumed = hook(
        SubagentOp::Run,
        json!({ "id": "runid005", "prompt": "continue" }),
    )
    .await;
    assert_eq!(resumed["ok"], json!(false), "{resumed}");

    let sent = hook(
        SubagentOp::MailboxSend,
        json!({ "to": "runid005", "body": "late" }),
    )
    .await;
    assert_eq!(sent["status"], json!("recipient_ended"), "{sent}");
}

#[tokio::test]
async fn rehydrate_legacy_transcript_falls_back_to_file_stem() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let store = crate::session::store::SessionStore::open().unwrap();
    let cursor = store
        .create_subagent_cursor(dir.path(), &"p/m".into(), Some("parent-x"), "runid002")
        .unwrap();
    cursor.append_events(&mut [user_event("a")]).unwrap();
    // Transcripts from before rehydration carry no run_id: strip it from
    // the header to reproduce one.
    let path = cursor.path().to_path_buf();
    let stem = cursor.id();
    let text = std::fs::read_to_string(&path).unwrap();
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    let mut header: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    header.as_object_mut().unwrap().remove("run_id");
    lines[0] = serde_json::to_string(&header).unwrap();
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();

    let agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.subagents().set_parent_session("parent-x");
    agent.subagents().rehydrate(&agent, &support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let list = hook(SubagentOp::List, json!({})).await;
    let entries = list["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 1, "{list}");
    assert_eq!(entries[0]["id"], json!(stem), "{list}");
}

#[tokio::test]
async fn a_restored_outcome_names_the_transcript_model() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let store = crate::session::store::SessionStore::open().unwrap();
    let cursor = store
        .create_subagent_cursor(dir.path(), &"gone/m".into(), Some("parent-x"), "runid006")
        .unwrap();
    cursor.append_events(&mut [user_event("a")]).unwrap();

    let agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.subagents().set_parent_session("parent-x");
    agent.subagents().rehydrate(&agent, &support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    // The resume may fall back to the session model when the transcript
    // model is gone; the settled outcome still describes the run that
    // produced it.
    let out = hook(SubagentOp::Wait, json!({ "id": "runid006" })).await;
    assert_eq!(out["model"], json!("gone/m"), "{out}");
}

#[tokio::test]
async fn rehydrate_runs_once_per_parent_and_after_clear() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let store = crate::session::store::SessionStore::open().unwrap();
    let cursor = store
        .create_subagent_cursor(dir.path(), &"p/m".into(), Some("parent-x"), "runid003")
        .unwrap();
    cursor.append_events(&mut [user_event("a")]).unwrap();

    let agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    let registry = agent.subagents();
    registry.set_parent_session("parent-x");
    registry.rehydrate(&agent, &support(dir.path()));
    registry.rehydrate(&agent, &support(dir.path()));
    let hook = agent.subagent_hook().unwrap();
    let list = hook(SubagentOp::List, json!({})).await;
    assert_eq!(list["entries"].as_array().unwrap().len(), 1, "{list}");

    // A session switch drops the entries; rehydrate for the same parent
    // restores them.
    let _ = registry.clear();
    let list = hook(SubagentOp::List, json!({})).await;
    assert_eq!(list["entries"].as_array().unwrap().len(), 0, "{list}");
    registry.rehydrate(&agent, &support(dir.path()));
    let list = hook(SubagentOp::List, json!({})).await;
    assert_eq!(list["entries"].as_array().unwrap().len(), 1, "{list}");
}

#[tokio::test]
async fn resume_after_rehydrate_closes_dangling_tool_calls() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let store = crate::session::store::SessionStore::open().unwrap();
    let cursor = store
        .create_subagent_cursor(dir.path(), &"p/m".into(), Some("parent-x"), "runid004")
        .unwrap();
    // The process died between the assistant's tool call and its result.
    let mut events = vec![
        user_event("a"),
        lofi_types::SessionEvent {
            id: String::new(),
            parent_id: None,
            kind: lofi_types::SessionEventKind::Message(Message {
                origin: None,
                role: Role::Assistant,
                blocks: vec![ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "exec".to_string(),
                    input: json!({}),
                }],
                kind: PromptKind::default(),
            }),
        },
    ];
    cursor.append_events(&mut events).unwrap();

    let provider = Arc::new(MockProvider::recording(vec![text_round("done")]));
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = provider.clone();
    agent.subagents().set_parent_session("parent-x");
    agent.subagents().rehydrate(&agent, &support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let status = hook(SubagentOp::Status, json!({ "id": "runid004" })).await;
    assert_eq!(status["status"], json!("cancelled"), "{status}");

    let out = hook(SubagentOp::Run, json!({ "id": "runid004", "prompt": "b" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    let requests = provider.requests();
    assert_eq!(requests.len(), 1);
    // The stored call has no result, so the resume closes it: the model learns
    // the call did not finish instead of losing the record of it.
    let request = &requests[0];
    let call_at = request
        .iter()
        .position(|m| {
            m.blocks
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
        })
        .expect("the interrupted call reaches the request");
    assert_eq!(request[call_at + 1].role, Role::Tool, "{request:?}");
    let closed = request[call_at + 1].blocks.iter().any(|b| {
        matches!(b, ContentBlock::ToolResult { tool_use_id, is_error: true, .. } if tool_use_id == "t1")
    });
    assert!(closed, "{request:?}");
    let texts: Vec<_> = requests[0]
        .iter()
        .filter(|m| m.role == Role::User)
        .map(first_text)
        .collect();
    assert!(texts.contains(&"a".to_string()), "{texts:?}");
    assert!(texts.contains(&"b".to_string()), "{texts:?}");
}

// Reports usage on the tool round, then pends on the follow-up: the run
// stays live with a spent amount, the state the status line and the
// `/subagent` rows read.
struct SpendThenPendProvider;

#[async_trait]
impl Provider for SpendThenPendProvider {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        let round = match messages.last() {
            // The wrap-up asks for a summary, not a round: this child never
            // answers it, and nothing is billed for a request that pends.
            _ if is_wrap_up(messages) => std::future::pending().await,
            Some(m) if m.role == Role::Tool => std::future::pending().await,
            _ => {
                let mut round = exec_round("return 1");
                if let Some(StreamingEvent::Done { usage, .. }) = round.last_mut() {
                    usage.input_tokens = 1_000_000;
                }
                round
            }
        };
        Ok(Box::pin(stream::iter(round.into_iter().map(Ok))))
    }
}

#[tokio::test]
async fn a_live_run_reports_its_model_and_spend() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    // A priced model: one million input tokens bill at the input price.
    agent.model.input_price = Some(3.0);
    agent.model.output_price = Some(15.0);
    agent.provider = Arc::new(SpendThenPendProvider);
    let hook = agent.subagent_hook().unwrap();
    let registry = agent.subagents();

    let out = hook(SubagentOp::Spawn, json!({ "prompt": "spend one" })).await;
    assert_eq!(out["ok"], json!(true), "{out}");
    let id = out["id"].as_str().unwrap().to_string();

    let mut observed = None;
    for _ in 0..200 {
        if let Some(run) = registry
            .running()
            .into_iter()
            .find(|run| run.id == id && run.cost.is_some())
        {
            observed = Some(run);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let Some(run) = observed else {
        panic!("the live spend never reached the registry");
    };
    assert_eq!(run.model.as_deref(), Some("p/m"));
    assert_eq!(run.cost, Some(3.0));
    // The dialog reads the same fields through `live`.
    let Some(live) = registry.live().into_iter().find(|run| run.id == id) else {
        panic!("live row");
    };
    assert!(live.running);
    assert_eq!(live.cost, Some(3.0));

    hook(SubagentOp::Kill, json!({ "id": id })).await;
}

#[tokio::test]
async fn the_settled_outcome_names_the_model() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(SpendThenPendProvider);
    let hook = agent.subagent_hook().unwrap();

    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "spend two", "timeout_ms": 100 }),
    )
    .await;
    assert_eq!(out["model"], json!("p/m"), "{out}");
}

// A resume installs the new run's totals over the entry's. The row names the
// child session, so it must show both runs and not fall back to what the
// resumed run alone billed.
#[tokio::test]
async fn a_resume_adds_to_the_child_spend_its_row_shows() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.model.input_price = Some(3.0);
    agent.model.output_price = Some(15.0);
    agent.provider = Arc::new(SpendThenPendProvider);
    let hook = agent.subagent_hook().unwrap();
    let registry = agent.subagents();

    // The first run times out inside its tool call, having billed a round.
    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "spend one", "timeout_ms": 100 }),
    )
    .await;
    assert_eq!(out["status"], json!("timed_out"), "{out}");
    let id = out["id"].as_str().unwrap().to_string();
    let spent = registry
        .live()
        .into_iter()
        .find(|run| run.id == id)
        .unwrap()
        .cost;
    assert_eq!(spent, Some(3.0), "the first run's round is priced");

    // The resume replays the rolled-back round, bills another one, and stops
    // the same way.
    let out = hook(
        SubagentOp::Run,
        json!({ "id": id, "prompt": "spend two", "timeout_ms": 100 }),
    )
    .await;
    assert_eq!(out["status"], json!("timed_out"), "{out}");
    let run = out["usage"]["cost"].as_f64().unwrap();
    assert!(
        (run - 3.0).abs() < 1e-9,
        "the resumed run prices itself: {out}"
    );
    let kept = registry
        .live()
        .into_iter()
        .find(|run| run.id == id)
        .unwrap()
        .cost;
    let kept = kept.expect("the row keeps a cost");
    assert!(
        (kept - 6.0).abs() < 1e-9,
        "both runs are on the row: {kept}"
    );
}

#[tokio::test]
async fn running_reports_in_flight_runs_with_a_bounded_task_preview() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    // The hook clones the agent as its template, so the blocking provider
    // must be in place first; otherwise the run can finish before the
    // assertions see it.
    agent.provider = Arc::new(NeverProvider);
    let hook = agent.subagent_hook().unwrap();
    let registry = agent.subagents();

    let prompt = format!("{}\nsecond line", "x".repeat(100));
    let out = hook(SubagentOp::Spawn, json!({ "prompt": prompt })).await;
    assert_eq!(out["ok"], json!(true), "{out}");
    let id = out["id"].as_str().unwrap().to_string();

    assert!(registry.is_running(&id));
    let running = registry.running();
    assert_eq!(running.len(), 1);
    assert_eq!(running[0].id, id);
    // The first line only, cut to 80 chars plus an ellipsis.
    assert_eq!(running[0].task.chars().count(), 81);
    assert!(running[0].task.ends_with('\u{2026}'));
    // Nothing has run or streamed yet, so the row names only the handle.
    assert!(running[0].activity.is_empty());

    hook(SubagentOp::Kill, json!({ "id": id })).await;
}

struct SpeakThenPendProvider;

#[async_trait]
impl Provider for SpeakThenPendProvider {
    async fn stream(
        &self,
        _model: &Model,
        _messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        let said = vec![Ok(StreamingEvent::TextDelta(
            "planning the change\nchecking the parser".to_string(),
        ))];
        Ok(Box::pin(stream::iter(said).chain(stream::pending())))
    }
}

#[tokio::test]
async fn running_names_the_start_of_the_response_not_its_tail() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    // The hook clones the agent as its template, so the speaking provider
    // must be in place first.
    agent.provider = Arc::new(SpeakThenPendProvider);
    let hook = agent.subagent_hook().unwrap();
    let registry = agent.subagents();

    let out = hook(SubagentOp::Spawn, json!({ "prompt": "survey the parser" })).await;
    assert_eq!(out["ok"], json!(true), "{out}");
    let id = out["id"].as_str().unwrap().to_string();

    // The row holds the first line of the answer, not the line the child has
    // reached: a row that followed the tail would scroll as the child writes.
    let mut activity = String::new();
    for _ in 0..200 {
        activity = registry
            .running()
            .first()
            .map_or_else(String::new, |run| run.activity.clone());
        if activity == "planning the change" {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(activity, "planning the change");

    hook(SubagentOp::Kill, json!({ "id": id })).await;
}

#[tokio::test]
async fn running_lists_background_runs_only() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    // The hook clones the agent as its template, so the blocking provider
    // must be in place first.
    agent.provider = Arc::new(NeverProvider);
    let hook = agent.subagent_hook().unwrap();
    let registry = agent.subagents();

    // A foreground run in flight is not listed: the parent's exec block
    // already shows it.
    let foreground = tokio::spawn({
        let hook = hook.clone();
        async move { hook(SubagentOp::Run, json!({ "prompt": "fg" })).await }
    });
    let mut foreground_id = None;
    for _ in 0..200 {
        if let Some(run) = registry.live().into_iter().find(|run| run.running) {
            let id = run.id;
            foreground_id = Some(id);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let Some(foreground_id) = foreground_id else {
        panic!("foreground run never started");
    };
    assert!(registry.is_running(&foreground_id));
    assert!(registry.running().is_empty());
    assert!(!registry.has_running_background());

    // A background run is listed.
    let out = hook(SubagentOp::Spawn, json!({ "prompt": "bg" })).await;
    assert_eq!(out["ok"], json!(true), "{out}");
    let background_id = out["id"].as_str().unwrap().to_string();
    let running = registry.running();
    assert_eq!(running.len(), 1);
    assert_eq!(running[0].id, background_id);
    assert!(registry.has_running_background());

    hook(SubagentOp::Kill, json!({ "id": background_id })).await;
    hook(SubagentOp::Kill, json!({ "id": foreground_id })).await;
    foreground.abort();
}

// The child sends to the parent inside its own run; the parent then reads its
// inbox in the same exec, so the round trip crosses both sandboxes.
// Answers by prompt content, never by a call counter: the round order then
// cannot make the test flaky. Prompt "a" pends so a spawn stays live, prompt
// "b" reads the mailbox, and a tool result is answered with text.
struct MailboxResumeProvider {
    read_round: Vec<StreamingEvent>,
    text_round: Vec<StreamingEvent>,
}

#[async_trait]
impl Provider for MailboxResumeProvider {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        let last = messages.last();
        let round = match last {
            Some(m) if m.role == Role::Tool => self.text_round.clone(),
            Some(m) if m.role == Role::User && first_text(m) == "a" => std::future::pending().await,
            _ => self.read_round.clone(),
        };
        Ok(Box::pin(stream::iter(round.into_iter().map(Ok))))
    }
}

#[tokio::test]
async fn a_resumed_run_takes_back_what_the_killed_run_never_read() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let provider = Arc::new(MailboxResumeProvider {
        read_round: exec_round(
            "const got = await lofi.subagentMailboxReceive({}); return got.messages.length;",
        ),
        text_round: text_round("resumed done"),
    });
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = provider.clone();
    let hook = agent.subagent_hook().unwrap();

    let started = hook(SubagentOp::Spawn, json!({ "prompt": "a" })).await;
    let id = started["id"].as_str().unwrap().to_string();
    // The spawned run is live, so its inbox is open and the send is accepted.
    let sent = hook(
        SubagentOp::MailboxSend,
        json!({ "to": id, "body": "instruction" }),
    )
    .await;
    assert_eq!(sent["ok"], json!(true), "{sent}");
    hook(SubagentOp::Kill, json!({ "id": id })).await;

    // The kill is observed by the run; wait for it to settle.
    let ended = hook(SubagentOp::Wait, json!({ "id": id, "timeout_ms": 10_000 })).await;
    let status = ended["status"].as_str().unwrap().to_string();
    assert_ne!(status, "running", "{ended}");

    // The ended run never read it, so the parent sees it as undelivered.
    let listed = hook(SubagentOp::MailboxList, json!({ "agent_id": id })).await;
    assert_eq!(
        listed["messages"][0]["undelivered"],
        json!(status),
        "{listed}"
    );

    // The resume claim reopens the inbox, and the new run reads the message.
    let resumed = hook(SubagentOp::Run, json!({ "id": id, "prompt": "b" })).await;
    assert_eq!(resumed["status"], json!("completed"), "{resumed}");

    let after = hook(SubagentOp::MailboxList, json!({ "agent_id": id })).await;
    assert!(
        after["messages"].as_array().unwrap().is_empty(),
        "the resumed run read the message: {after}"
    );
}

#[tokio::test]
async fn a_message_to_the_parent_inbox_wakes_the_host() {
    let registry = SubagentRegistry::default();
    let mut notices = registry.subscribe_notices();
    registry.mailbox.open("child-a");
    let sent = registry.mailbox.send(
        "child-a",
        false,
        &json!({ "to": "parent", "body": "secret" }),
    );
    assert_eq!(sent["ok"], true, "{sent}");
    let notice = notices.try_recv().unwrap();
    assert_eq!(notice.kind, SubagentNoticeKind::Mailbox);
    assert_eq!(notice.id, "child-a");
    assert!(
        notice.text.contains("lofi.subagentMailboxReceive"),
        "{}",
        notice.text
    );
    // The body stays in the inbox: the notice becomes a prompt, and a child's
    // words must reach the parent as tool output, never as an instruction.
    assert!(!notice.text.contains("secret"), "{}", notice.text);
    assert!(registry.mailbox.parent_inbox_pending());
    // One wake per non-empty inbox, however many messages arrive.
    registry.mailbox.send(
        "child-a",
        false,
        &json!({ "to": "parent", "body": "again" }),
    );
    assert!(notices.try_recv().is_err());
    // Draining the inbox arms the next wake.
    let read = registry.mailbox.receive("parent", &json!({})).await;
    assert_eq!(read["messages"].as_array().map(Vec::len), Some(2), "{read}");
    assert!(!registry.mailbox.parent_inbox_pending());
    registry.mailbox.send(
        "child-a",
        false,
        &json!({ "to": "parent", "body": "third" }),
    );
    assert_eq!(
        notices.try_recv().unwrap().kind,
        SubagentNoticeKind::Mailbox
    );
    // A notice lost to lag is rebuilt from the inbox while the wake is
    // untaken, and never after the host claims it: the turn that took the
    // wake may have been cancelled with the mail unread.
    assert!(registry.lost_mailbox_wake().is_some());
    assert!(registry.claim_mailbox_wake());
    assert!(registry.lost_mailbox_wake().is_none());
    // A session switch drops the inbox the stored wake points at, and the
    // wake with it, even while it is unclaimed.
    let read = registry.mailbox.receive("parent", &json!({})).await;
    assert_eq!(read["messages"].as_array().map(Vec::len), Some(1), "{read}");
    assert_eq!(
        registry.mailbox.send(
            "child-a",
            false,
            &json!({ "to": "parent", "body": "fourth" }),
        )["ok"],
        true
    );
    assert!(registry.mailbox_wake_unclaimed());
    let _ = registry.clear();
    assert!(!registry.mailbox_wake_unclaimed());
    assert!(registry.lost_mailbox_wake().is_none());
}

#[tokio::test]
async fn a_child_reaches_the_parent_mailbox_from_its_own_sandbox() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let parent_round = vec![
        StreamingEvent::ToolUseStart {
            id: "p1".into(),
            name: "exec".into(),
        },
        StreamingEvent::ToolUseInputComplete {
            id: "p1".into(),
            input: json!({ "code": "const run = await lofi.subagent({ prompt: 'x' }); const got = await lofi.subagentMailboxReceive({ wait_ms: 500 }); return { status: run.status, messages: got.messages };", "description": "run subagent" }).to_string(),
        },
        StreamingEvent::ToolUseEnd { id: "p1".into() },
        StreamingEvent::Done {
            usage: Usage::default(),
            stop_reason: Some(lofi_types::StopReason::ToolUse),
        },
    ];
    let child_round = vec![
        StreamingEvent::ToolUseStart {
            id: "c1".into(),
            name: "exec".into(),
        },
        StreamingEvent::ToolUseInputComplete {
            id: "c1".into(),
            input: json!({ "code": "const r = await lofi.subagentMailboxSend({ to: 'parent', body: 'from child' }); return r.ok;", "description": "send mailbox message" }).to_string(),
        },
        StreamingEvent::ToolUseEnd { id: "c1".into() },
        StreamingEvent::Done {
            usage: Usage::default(),
            stop_reason: Some(lofi_types::StopReason::ToolUse),
        },
    ];

    let child_final = vec![
        StreamingEvent::TextDelta("child done".into()),
        StreamingEvent::Done {
            usage: Usage::default(),
            stop_reason: Some(lofi_types::StopReason::EndTurn),
        },
    ];
    let parent_final = vec![
        StreamingEvent::TextDelta("done".into()),
        StreamingEvent::Done {
            usage: Usage::default(),
            stop_reason: Some(lofi_types::StopReason::EndTurn),
        },
    ];
    // Rounds are drawn in consumption order: the parent's tool round, the
    // child's tool round and closing text, then the parent's closing text.
    let agent = agent_with(
        vec![parent_round, child_round, child_final, parent_final],
        dir.path(),
    )
    .with_subagent_support(support(dir.path()));
    let mut notices = agent.subagents().subscribe_notices();
    let mut messages = vec![user_msg("go")];
    // The first round asks for the exec; the second ends the turn.
    assert!(!agent.run_once(&mut messages).await.unwrap());
    assert!(agent.run_once(&mut messages).await.unwrap());

    // The send from the child's own sandbox wakes the host through the
    // scope it shares with the registry.
    let notice = notices.try_recv().unwrap();
    assert_eq!(notice.kind, SubagentNoticeKind::Mailbox);
    assert!(
        notice.text.contains("lofi.subagentMailboxReceive"),
        "{}",
        notice.text
    );

    let result = messages
        .iter()
        .filter(|message| message.role == Role::Tool)
        .flat_map(|message| message.blocks.iter())
        .find_map(|block| match block {
            ContentBlock::ToolResult { content, .. } if content.contains("from child") => {
                Some(content.clone())
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("the parent read the child's message: {messages:#?}"));
    // The reply carries the child's id and its completed status.
    assert!(result.contains("\"status\":\"completed\""), "{result}");
}

// The deadline work: an idle run is stopped and asks for a wrap-up, so the
// parent gets a summary instead of a bare timeout.
struct PendThenSummarizeProvider;

#[async_trait]
impl Provider for PendThenSummarizeProvider {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        let round = if is_wrap_up(messages) {
            text_round("reached the third step")
        } else {
            std::future::pending().await
        };
        Ok(Box::pin(stream::iter(round.into_iter().map(Ok))))
    }
}

#[tokio::test]
async fn a_stalled_run_is_stopped_and_wraps_up() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(PendThenSummarizeProvider);
    let hook = agent.subagent_hook().unwrap();

    // One short idle deadline and a far cap, so only the idle deadline can
    // stop this run, and the wrap-up then answers.
    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "a", "idle_ms": 100, "timeout_ms": 30_000 }),
    )
    .await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["result"], json!("reached the third step"), "{out}");
    // The settle report says where the run was when its deadline fired, and
    // which deadline it was: a summary is not the task finishing.
    assert!(out["elapsed_ms"].as_u64().is_some(), "{out}");
    assert_eq!(out["stopped"], json!("idle"), "{out}");
}

// A child that keeps producing events is working: only the hard cap stops it,
// so a run whose gaps stay under the idle deadline finishes on its own.
struct TrickleProvider;

#[async_trait]
impl Provider for TrickleProvider {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        if is_wrap_up(messages) {
            return Ok(Box::pin(stream::iter(
                text_round("wrapped up").into_iter().map(Ok),
            )));
        }
        let events: Vec<StreamingEvent> = (0..40)
            .map(|n| StreamingEvent::TextDelta(format!("{n} ")))
            .chain(std::iter::once(StreamingEvent::Done {
                usage: Usage::default(),
                stop_reason: None,
            }))
            .collect();
        // A delta every 5 ms: 200 ms of work against an 80 ms idle deadline,
        // so no single gap reaches it while the whole run is far longer.
        let stream = futures::stream::unfold(events.into_iter(), |mut events| async move {
            let event = events.next()?;
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            Some((Ok(event), events))
        });
        Ok(Box::pin(stream))
    }
}

#[tokio::test]
async fn a_run_that_keeps_producing_events_is_not_idle() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(TrickleProvider);
    let hook = agent.subagent_hook().unwrap();

    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "a", "idle_ms": 80, "timeout_ms": 30_000 }),
    )
    .await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert!(
        out["result"].as_str().unwrap().starts_with("0 1 2"),
        "the run finished its own work: {out}"
    );
}

// A run waiting on the user is not idle: only its hard cap stops it, and it
// settles blocked naming the command nobody answered.
struct AskThenPendProvider;

#[async_trait]
impl Provider for AskThenPendProvider {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        let round = match messages.last() {
            Some(_) if is_wrap_up(messages) => text_round("wrapped up"),
            _ => exec_round("await lofi.bash({ cmd: \"echo blocked\" })"),
        };
        Ok(Box::pin(stream::iter(round.into_iter().map(Ok))))
    }
}

#[tokio::test]
async fn a_run_waiting_on_a_prompt_settles_blocked() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let (tx, mut prompts) = tokio::sync::mpsc::unbounded_channel();
    let mut agent = agent_with(vec![], dir.path())
        .with_confirm_tx(tx)
        .with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(AskThenPendProvider);
    let hook = agent.subagent_hook().unwrap();

    // The idle deadline is shorter than the cap, and the run is waiting on
    // the prompt for longer than that: only the cap may stop it.
    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "a", "idle_ms": 300, "timeout_ms": 2_000 }),
    )
    .await;
    assert_eq!(out["status"], json!("blocked"), "{out}");
    let error = out["error"].as_str().unwrap();
    assert!(error.contains("echo blocked"), "{out}");
    // The prompt reached the session: it was never answered.
    assert!(prompts.try_recv().is_ok(), "the child asked for approval");
}

#[tokio::test]
async fn a_settled_run_reports_where_it_worked() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(MockProvider::recording(vec![
        exec_round("return 1"),
        text_round("done"),
    ]));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "a" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["last_tool"], json!("exec"), "{out}");
    assert!(out["rounds"].as_u64().unwrap() >= 2, "{out}");
    assert!(out["elapsed_ms"].as_u64().is_some(), "{out}");
}

#[tokio::test]
async fn a_background_spawn_resumes_by_id() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(MockProvider::recording(vec![
        text_round("first"),
        text_round("second"),
    ]));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "a" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    let id = out["id"].as_str().unwrap().to_string();

    // A background spawn continues a settled run without a foreground call,
    // so the run survives the exec that resumed it.
    let out = hook(
        SubagentOp::Spawn,
        json!({ "id": id, "prompt": "b", "timeout_ms": 30_000 }),
    )
    .await;
    assert_eq!(out["ok"], json!(true), "{out}");
    assert_eq!(out["status"], json!("running"), "{out}");
    assert_eq!(out["id"], json!(id), "{out}");
    let done = hook(SubagentOp::Wait, json!({ "id": id, "timeout_ms": 30_000 })).await;
    assert_eq!(done["status"], json!("completed"), "{done}");
    assert_eq!(done["result"], json!("second"), "{done}");
}
// A call in flight reports nothing until it ends, so a long one is work: the
// idle deadline must not stop a child that is inside a single tool call.
struct SlowCallProvider {
    rounds: std::sync::atomic::AtomicUsize,
}

#[async_trait]
impl Provider for SlowCallProvider {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        let round = if is_wrap_up(messages) {
            text_round("wrapped up")
        } else if self.rounds.fetch_add(1, Ordering::SeqCst) == 0 {
            exec_round("await lofi.sleep(0.4); return 1")
        } else {
            text_round("done")
        };
        Ok(Box::pin(stream::iter(round.into_iter().map(Ok))))
    }
}

#[tokio::test]
async fn a_run_inside_one_long_tool_call_is_not_idle() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(SlowCallProvider {
        rounds: std::sync::atomic::AtomicUsize::new(0),
    });
    let hook = agent.subagent_hook().unwrap();

    // The call runs four times longer than the idle deadline and emits
    // nothing meanwhile.
    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "a", "idle_ms": 100, "timeout_ms": 30_000 }),
    )
    .await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(
        out["result"],
        json!("done"),
        "the run finished its own work: {out}"
    );
}

// A stopped run whose summary came back empty keeps the deadline status: the
// parent must not read an empty success.
struct PendThenSayNothingProvider;

#[async_trait]
impl Provider for PendThenSayNothingProvider {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        if is_wrap_up(messages) {
            let round = vec![StreamingEvent::Done {
                usage: Usage::default(),
                stop_reason: None,
            }];
            return Ok(Box::pin(stream::iter(round.into_iter().map(Ok))));
        }
        std::future::pending().await
    }
}

#[tokio::test]
async fn an_empty_wrap_up_keeps_the_deadline_status() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(PendThenSayNothingProvider);
    let hook = agent.subagent_hook().unwrap();

    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "a", "idle_ms": 100, "timeout_ms": 30_000 }),
    )
    .await;
    assert_eq!(out["status"], json!("timed_out"), "{out}");
    let error = out["error"].as_str().unwrap();
    assert!(error.contains("idle for 100 ms"), "{out}");
}

// The wrap-up is a turn of its own, so its cumulative cost adds to the run's
// instead of replacing it.
struct PaidStallProvider;

fn bill_input(round: &mut [StreamingEvent], input: u64) {
    if let Some(StreamingEvent::Done { usage, .. }) = round.last_mut() {
        usage.input_tokens = input;
    }
}

#[async_trait]
impl Provider for PaidStallProvider {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        let round = if is_wrap_up(messages) {
            let mut round = text_round("wrapped up");
            bill_input(&mut round, 500_000);
            round
        } else {
            let mut round = exec_round("await lofi.sleep(10)");
            bill_input(&mut round, 1_000_000);
            round
        };
        Ok(Box::pin(stream::iter(round.into_iter().map(Ok))))
    }
}

#[tokio::test]
async fn a_wrap_up_adds_its_spend_to_the_run() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    // One million input tokens bill at the input price.
    agent.model.input_price = Some(3.0);
    agent.model.output_price = Some(15.0);
    agent.provider = Arc::new(PaidStallProvider);
    let hook = agent.subagent_hook().unwrap();

    // The call in flight keeps the idle deadline away, so only the cap stops
    // this run, and the summary then answers.
    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "a", "idle_ms": 60_000, "timeout_ms": 300 }),
    )
    .await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["result"], json!("wrapped up"), "{out}");
    let cost = out["usage"]["cost"].as_f64().unwrap();
    assert!((cost - 4.5).abs() < 1e-9, "both turns are billed: {out}");
}

// Starts an answer and hangs inside the round, so a deadline lands on work the
// child had already begun but never finished.
struct StartedThenStallProvider {
    requests: std::sync::Mutex<Vec<Vec<Message>>>,
}

#[async_trait]
impl Provider for StartedThenStallProvider {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        self.requests.lock().unwrap().push(messages.to_vec());
        if is_wrap_up(messages) {
            return Ok(Box::pin(stream::iter(
                text_round("the summary").into_iter().map(Ok),
            )));
        }
        let started = stream::iter(vec![Ok(StreamingEvent::TextDelta(
            "half an answer".to_string(),
        ))]);
        Ok(Box::pin(futures::stream::StreamExt::chain(
            started,
            stream::pending::<Result<StreamingEvent>>(),
        )))
    }
}

#[tokio::test]
async fn a_deadline_keeps_the_round_it_stopped() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let provider = Arc::new(StartedThenStallProvider {
        requests: std::sync::Mutex::new(Vec::new()),
    });
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = provider.clone();
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "a", "idle_ms": 100 })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["stopped"], json!("idle"), "{out}");

    // The round the deadline stopped is in the history the summary reads,
    // rather than dropped with the turn that was streaming it.
    let requests = provider.requests.lock().unwrap();
    let summary = requests
        .iter()
        .find(|request| is_wrap_up(request))
        .expect("a wrap-up request");
    assert!(
        summary
            .iter()
            .any(|message| first_text(message) == "half an answer"),
        "{summary:?}"
    );
}

// Streams half a summary and then stalls, so the wrap-up's own deadline cuts
// a reply an earlier turn would have kept.
struct HalfASummaryThenStall;

#[async_trait]
impl Provider for HalfASummaryThenStall {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        if !is_wrap_up(messages) {
            std::future::pending::<()>().await;
        }
        let started = stream::iter(vec![Ok(StreamingEvent::TextDelta(
            "half a summary".to_string(),
        ))]);
        Ok(Box::pin(futures::stream::StreamExt::chain(
            started,
            stream::pending::<Result<StreamingEvent>>(),
        )))
    }
}

#[tokio::test]
async fn a_cut_summary_keeps_the_reply_it_was_streaming() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(HalfASummaryThenStall);
    let hook = agent.subagent_hook().unwrap();

    // The task turn goes idle; the summary then streams a reply and stalls
    // past the deadline that bounds it here.
    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "a", "idle_ms": 100, "timeout_ms": 400 }),
    )
    .await;
    assert_eq!(out["status"], json!("timed_out"), "{out}");
    // The half-written summary is work a resume continues from, so the hint
    // names the id instead of saying the run kept nothing.
    let id = out["id"].as_str().unwrap();
    assert_eq!(
        out["error"],
        json!(format!(
            "idle for 100 ms; resume with id {id} to continue from the work so far"
        )),
        "{out}"
    );
}

// Runs one tool exchange in the wrap-up and bills past the budget, so the
// summary is cut after completing work a resume would keep.
struct ToolThenOverBudgetSummary;

#[async_trait]
impl Provider for ToolThenOverBudgetSummary {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        let round = if is_wrap_up(messages) {
            let mut round = exec_round("return 1");
            bill_input(&mut round, 500_000);
            round
        } else {
            std::future::pending().await
        };
        Ok(Box::pin(stream::iter(round.into_iter().map(Ok))))
    }
}

#[tokio::test]
async fn a_wrap_up_cut_after_its_tool_stays_resumable() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(ToolThenOverBudgetSummary);
    let hook = agent.subagent_hook().unwrap();

    // The task turn is stopped by its deadline; the summary then runs a tool
    // and bills past the budget before it can answer.
    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "a", "idle_ms": 100, "token_budget": 10 }),
    )
    .await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    // The tool exchange it completed is work a resume continues from, even
    // though the summary itself never answered.
    let id = out["id"].as_str().unwrap();
    assert_eq!(
        out["error"],
        json!(format!(
            "token budget exceeded (500000 > 10); resume with id {id} to continue from the work so far"
        )),
        "{out}"
    );
}

// Only the summary bills here: a wrap-up round that crosses the budget is a
// failure, not the deadline that asked for it.
struct SummarizeOverBudgetProvider;

#[async_trait]
impl Provider for SummarizeOverBudgetProvider {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        if !is_wrap_up(messages) {
            std::future::pending::<()>().await;
        }
        let mut round = text_round("wrapped up");
        bill_input(&mut round, 500_000);
        Ok(Box::pin(stream::iter(round.into_iter().map(Ok))))
    }
}

#[tokio::test]
async fn a_wrap_up_that_crosses_the_budget_fails_the_run() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(SummarizeOverBudgetProvider);
    let hook = agent.subagent_hook().unwrap();

    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "a", "idle_ms": 100, "token_budget": 10 }),
    )
    .await;
    assert_eq!(out["status"], json!("failed"), "{out}");
    // The summary answered, so its reply is work a resume continues from.
    let id = out["id"].as_str().unwrap();
    assert_eq!(
        out["error"],
        json!(format!(
            "token budget exceeded (500000 > 10); resume with id {id} to continue from the work so far"
        )),
        "{out}"
    );
    assert_eq!(out["stopped"], json!("idle"), "{out}");
    // The breach ends the run; it does not wait out the wrap-up deadline.
    assert!(out["elapsed_ms"].as_u64().unwrap() < 10_000, "{out}");
}

// A summary request is a real request: it must survive the watchdog ticks that
// follow the deadline it answers, and only a caller may cut it short.
struct PendThenSlowSummarizeProvider;

#[async_trait]
impl Provider for PendThenSlowSummarizeProvider {
    async fn stream(
        &self,
        _model: &Model,
        messages: &[Message],
        _tools: &[ToolSchema],
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamingEvent>>> {
        if !is_wrap_up(messages) {
            std::future::pending::<()>().await;
        }
        let round = text_round("reached step two");
        let stream = futures::stream::unfold(round.into_iter(), |mut events| async move {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            let event = events.next()?;
            Some((Ok(event), events))
        });
        Ok(Box::pin(stream))
    }
}

#[tokio::test]
async fn a_slow_wrap_up_still_reports_where_the_run_stopped() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(PendThenSlowSummarizeProvider);
    let hook = agent.subagent_hook().unwrap();

    // The summary answers over several watchdog ticks after the deadline.
    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "a", "idle_ms": 100, "timeout_ms": 30_000 }),
    )
    .await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["result"], json!("reached step two"), "{out}");
}

// The status of a settled stopped run carries the same progress the settled
// result does, so polling the public API reads where it stopped.
#[tokio::test]
async fn a_settled_status_reports_where_the_run_worked() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = Arc::new(MockProvider::recording(vec![
        exec_round("return 1"),
        text_round("done"),
    ]));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "a" })).await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    let id = out["id"].as_str().unwrap().to_string();
    let status = hook(SubagentOp::Status, json!({ "id": id })).await;
    assert_eq!(status["status"], json!("completed"), "{status}");
    assert_eq!(status["last_tool"], json!("exec"), "{status}");
    assert!(status["rounds"].as_u64().unwrap() >= 2, "{status}");
    assert!(status["elapsed_ms"].as_u64().is_some(), "{status}");
}

// A call an earlier run left running belongs to that run: a resumed run must
// count only its own, or its idle deadline never fires.
#[tokio::test]
async fn a_resumed_run_counts_only_its_own_calls() {
    let watch = crate::agent::tool_executor::ActivityWatch::default();
    let old = watch.enter();
    watch.reset();
    let new = watch.enter();
    assert_eq!(watch.idle_mark(), (1, None));
    drop(old);
    assert_eq!(watch.idle_mark().0, 1);
    // A call that has ended keeps the idle clock young, so a tick racing its
    // event does not read the run as idle.
    let dropping = std::time::Instant::now();
    drop(new);
    assert_eq!(watch.idle_mark().0, 0);
    assert!(watch.idle_mark().1.unwrap() <= dropping.elapsed());
    watch.reset();
    assert_eq!(watch.idle_mark(), (0, None));
}

#[tokio::test]
async fn a_zero_idle_deadline_is_rejected() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let agent = agent_with(vec![text_round("never")], dir.path())
        .with_subagent_support(support(dir.path()));
    let hook = agent.subagent_hook().unwrap();

    let out = hook(SubagentOp::Run, json!({ "prompt": "a", "idle_ms": 0 })).await;
    assert_eq!(out["ok"], json!(false), "{out}");
    assert_eq!(out["status"], json!("failed"), "{out}");
    assert!(out["error"].as_str().unwrap().contains("idle_ms"), "{out}");
}

// A restart rebuilds the entry from the settle marker: a run a deadline
// stopped must not come back indistinguishable from one that finished.
#[tokio::test]
async fn a_restored_run_keeps_the_deadline_that_stopped_it() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.subagents().set_parent_session("parent-x");
    agent.provider = Arc::new(PendThenSummarizeProvider);
    let hook = agent.subagent_hook().unwrap();

    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "a", "idle_ms": 100, "timeout_ms": 30_000 }),
    )
    .await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["stopped"], json!("idle"), "{out}");
    let id = out["id"].as_str().unwrap().to_string();

    let restored = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    restored.subagents().set_parent_session("parent-x");
    restored
        .subagents()
        .rehydrate(&restored, &support(dir.path()));
    let hook = restored.subagent_hook().unwrap();
    let status = hook(SubagentOp::Status, json!({ "id": id })).await;
    assert_eq!(status["status"], json!("completed"), "{status}");
    assert_eq!(status["stopped"], json!("idle"), "{status}");
}

// The notice is what a background run's parent reads first, so a wrapped run
// says that a deadline stopped it.
#[test]
fn a_wrapped_notice_names_the_deadline_and_where_it_stopped() {
    let out = json!({
        "status": "completed",
        "result": "reached step two\nmore",
        "stopped": "idle",
        "rounds": 4,
        "elapsed_ms": 312_500,
        "last_tool": "bash",
    });
    let notice = crate::agent::subagent::terminal_notice(
        "id1",
        crate::agent::subagent::Status::Completed,
        &out,
    );
    assert_eq!(
        notice,
        "subagent id1 completed (stopped: idle; round 4 after 312 s, bash): reached step two"
    );
}

#[tokio::test]
async fn a_prompt_watch_names_the_oldest_open_prompt() {
    let watch = crate::agent::tool_executor::ConfirmWatch::default();
    let first = watch.open("echo first");
    let second = watch.open("echo second");
    assert_eq!(watch.pending(), 2);
    // Answering the newer prompt first leaves the older one blocking.
    drop(second);
    assert_eq!(watch.pending(), 1);
    assert_eq!(watch.command().as_deref(), Some("echo first"));
    drop(first);
    assert_eq!(watch.pending(), 0);
    assert_eq!(watch.command(), None);
}

// Prompts an earlier run left open answer late, because their tool work can
// outlive the reset a resume does: they must not take the new run's prompt.
#[tokio::test]
async fn a_resumed_run_keeps_its_own_prompt() {
    let watch = crate::agent::tool_executor::ConfirmWatch::default();
    let old = watch.open("echo old");
    watch.reset();
    let new = watch.open("echo new");
    drop(old);
    assert_eq!(watch.pending(), 1);
    assert_eq!(watch.command().as_deref(), Some("echo new"));
    drop(new);
    assert_eq!(watch.pending(), 0);
}

// A turn dropped mid-call leaves a tool call nothing answered. Providers
// reject that, so the history is repaired before it settles and the resumed
// run replays a request they accept.
#[tokio::test]
async fn a_stopped_run_answers_the_call_it_dropped() {
    let _env = env_lock();
    let _state = StateEnv::new();
    let dir = tempdir().unwrap();
    let provider = Arc::new(MockProvider::recording(vec![
        exec_round("await new Promise(() => {});"),
        text_round("reached the call"),
        text_round("resumed done"),
    ]));
    let mut agent = agent_with(vec![], dir.path()).with_subagent_support(support(dir.path()));
    agent.provider = provider.clone();
    let hook = agent.subagent_hook().unwrap();

    // The call in flight keeps the idle deadline away, so the cap stops this
    // run while the exec is still running.
    let out = hook(
        SubagentOp::Run,
        json!({ "prompt": "a", "idle_ms": 60_000, "timeout_ms": 300 }),
    )
    .await;
    assert_eq!(out["status"], json!("completed"), "{out}");
    assert_eq!(out["result"], json!("reached the call"), "{out}");
    let id = out["id"].as_str().unwrap().to_string();

    // The resumed run replays a history whose call is answered, so a
    // provider that pairs calls on the wire accepts it.
    let resumed = hook(SubagentOp::Run, json!({ "id": id, "prompt": "b" })).await;
    assert_eq!(resumed["status"], json!("completed"), "{resumed}");
    // The last request is the resumed run's, replaying the stopped history.
    let requests = provider.requests();
    let replay = requests.last().unwrap();
    let calls = count_blocks(replay, |block| {
        matches!(block, ContentBlock::ToolUse { .. })
    });
    let results = count_blocks(replay, |block| {
        matches!(block, ContentBlock::ToolResult { .. })
    });
    assert!(calls > 0, "the stopped run called a tool: {replay:?}");
    assert_eq!(calls, results, "every call is answered: {replay:?}");
}

#[tokio::test]
async fn a_prompt_watch_reports_the_time_it_waited() {
    let watch = crate::agent::tool_executor::ConfirmWatch::default();
    assert_eq!(watch.waited(), std::time::Duration::ZERO);
    let open = watch.open("echo x");
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert!(watch.waited() >= std::time::Duration::from_millis(20));
    drop(open);
    let closed = watch.waited();
    assert!(closed >= std::time::Duration::from_millis(20));
}
