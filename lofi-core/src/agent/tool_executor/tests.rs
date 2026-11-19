#![allow(clippy::unwrap_used, clippy::expect_used, clippy::wildcard_imports)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::approvals::approve_outside_dir;
use super::spend::{clamp_child, clock};
use super::*;
use crate::agent::{cap_exec_envelope, MAX_EXEC_RESULT_BYTES};
use crate::state;

/// The guest's own return value escapes too: a value that fits the raw
/// budget can outgrow the limit once serialized, and it used to pass
/// through uncut.
#[test]
fn an_oversized_exec_value_is_cut_and_says_so() {
    let mut output = serde_json::Map::new();
    output.insert("ok".into(), true.into());
    output.insert("status".into(), "completed".into());
    output.insert(
        "value".into(),
        "\0".repeat(MAX_EXEC_RESULT_BYTES / 2).into(),
    );

    let text = cap_exec_envelope(output);

    assert!(text.len() <= MAX_EXEC_RESULT_BYTES, "{} bytes", text.len());
    let value: serde_json::Value =
        serde_json::from_str(&text).expect("a trimmed envelope stays JSON");
    assert_eq!(value["status"], "completed");
    assert_eq!(value["truncated"], true, "a cut body has to say so");
}

/// An escaped object key is payload too: it gets cut rather than the
/// status giving up its room for it.
#[test]
fn an_oversized_object_key_is_cut_before_the_status() {
    let mut payload = serde_json::Map::new();
    payload.insert("\0".repeat(MAX_EXEC_RESULT_BYTES / 2), 1.into());
    let mut output = serde_json::Map::new();
    output.insert("ok".into(), true.into());
    output.insert("status".into(), "completed".into());
    output.insert("value".into(), serde_json::Value::Object(payload));

    let text = cap_exec_envelope(output);

    assert!(text.len() <= MAX_EXEC_RESULT_BYTES, "{} bytes", text.len());
    let value: serde_json::Value =
        serde_json::from_str(&text).expect("a trimmed envelope stays JSON");
    assert_eq!(value["status"], "completed", "the status keeps its word");
    assert_eq!(value["truncated"], true, "a cut body has to say so");
}

/// Many short strings outgrow the limit together, each escaping to six
/// times its length. Trimming them one per pass would serialize the whole
/// envelope once per string, so the top half-length band is cut in one
/// pass.
#[test]
fn many_short_escaped_strings_are_trimmed_together() {
    let mut output = serde_json::Map::new();
    output.insert("ok".into(), true.into());
    output.insert("status".into(), "completed".into());
    output.insert(
        "value".into(),
        serde_json::Value::Array(
            (0..9000)
                .map(|_| serde_json::Value::String("\0".repeat(20)))
                .collect(),
        ),
    );

    let started = std::time::Instant::now();
    let text = cap_exec_envelope(output);

    assert!(text.len() <= MAX_EXEC_RESULT_BYTES, "{} bytes", text.len());
    let value: serde_json::Value =
        serde_json::from_str(&text).expect("a trimmed envelope stays JSON");
    assert_eq!(value["status"], "completed");
    assert_eq!(value["truncated"], true, "a cut body has to say so");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "trimming took {:?}",
        started.elapsed()
    );
}

/// The logs escape when serialized, so text that fits the sandbox's raw
/// budget can outgrow the envelope. Trimming must keep the JSON and the
/// keys that follow the logs, `status` among them.
#[test]
fn an_oversized_exec_envelope_keeps_its_status() {
    let mut output = serde_json::Map::new();
    output.insert("ok".into(), false.into());
    output.insert("status".into(), "failed".into());
    output.insert("logs".into(), "\n".repeat(MAX_EXEC_RESULT_BYTES / 2).into());

    let text = cap_exec_envelope(output);

    assert!(text.len() <= MAX_EXEC_RESULT_BYTES, "{} bytes", text.len());
    let value: serde_json::Value =
        serde_json::from_str(&text).expect("a trimmed envelope stays JSON");
    assert_eq!(value["status"], "failed");
}

fn state_env() -> tempfile::TempDir {
    let state_home = tempfile::tempdir().unwrap();
    std::env::set_var("LOFI_STATE_HOME", state_home.path());
    state_home
}

#[tokio::test]
async fn a_dropped_directory_ask_closes_its_prompt() {
    let ws = tempfile::tempdir().unwrap();
    let counter = Arc::new(AtomicU64::new(0));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
    let mut ask = Box::pin(approve_outside_dir(
        ws.path(),
        dir_approval("/data/dir", lofi_types::Access::Read),
        &[],
        &tx,
        &counter,
        None,
    ));
    let req = tokio::select! {
        req = rx.recv() => req.unwrap(),
        answered = &mut ask => panic!("the ask answered itself: {answered:?}"),
    };
    assert!(req.active.load(Ordering::Relaxed));
    drop(ask);
    assert!(
        !req.active.load(Ordering::Relaxed),
        "a dropped ask must close its modal"
    );
}

fn dir_approval(dir: &str, requested: lofi_types::Access) -> lofi_code::DirApproval {
    lofi_code::DirApproval {
        dir: PathBuf::from(dir),
        narrow: None,
        marker: None,
        requested,
    }
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn grants_are_remembered_across_sessions() {
    let _lock = state::STATE_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _state_home = state_env();
    let ws = tempfile::tempdir().unwrap();
    let counter = Arc::new(AtomicU64::new(0));
    let write = lofi_types::Access::Write;

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
    let responder = tokio::spawn(async move {
        let req = rx.recv().await.unwrap();
        assert_eq!(req.command, "/data/dir");
        let reason = req
            .reason
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            matches!(*reason, lofi_code::ConfirmReason::OutsideRoot { requested, current: None, .. } if requested == write)
        );
        req.respond
            .send(ConfirmResponse::Grant {
                access: write,
                narrow: false,
            })
            .unwrap();
    });
    let granted = approve_outside_dir(
        ws.path(),
        dir_approval("/data/dir", write),
        &[],
        &tx,
        &counter,
        None,
    )
    .await;
    responder.await.unwrap();
    assert_eq!(
        granted,
        Some(lofi_code::GrantedDir {
            dir: PathBuf::from("/data/dir"),
            access: write
        })
    );

    // A later session reads the state file: the call returns at once and
    // no prompt opens on the channel.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
    let second = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        approve_outside_dir(
            ws.path(),
            dir_approval("/data/dir/sub", write),
            &[],
            &tx,
            &counter,
            None,
        ),
    )
    .await
    .expect("remembered grant must not wait for a prompt");
    assert_eq!(
        second,
        Some(lofi_code::GrantedDir {
            dir: PathBuf::from("/data/dir"),
            access: write
        })
    );
    assert!(rx.try_recv().is_err());

    std::env::remove_var("LOFI_STATE_HOME");
}

fn project_approval(root: &str, narrow: Option<&str>) -> lofi_code::DirApproval {
    lofi_code::DirApproval {
        dir: PathBuf::from(root),
        narrow: narrow.map(PathBuf::from),
        marker: Some(".jj"),
        requested: lofi_types::Access::Write,
    }
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn a_project_grant_covers_a_sibling_without_asking() {
    // The prompt offers the project root; answering it once must cover
    // every sibling directory, not only the one that asked.
    let _lock = state::STATE_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _state_home = state_env();
    let ws = tempfile::tempdir().unwrap();
    let counter = Arc::new(AtomicU64::new(0));
    let write = lofi_types::Access::Write;

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
    let responder = tokio::spawn(async move {
        let req = rx.recv().await.unwrap();
        assert_eq!(req.command, "/data/repo");
        let reason = req
            .reason
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            matches!(&*reason, lofi_code::ConfirmReason::OutsideRoot { narrow: Some(narrow), marker: Some(".jj"), .. } if narrow == "/data/repo/src"),
            "the prompt names both units and the marker"
        );
        req.respond
            .send(ConfirmResponse::Grant {
                access: write,
                narrow: false,
            })
            .unwrap();
    });
    let granted = approve_outside_dir(
        ws.path(),
        project_approval("/data/repo", Some("/data/repo/src")),
        &[],
        &tx,
        &counter,
        None,
    )
    .await;
    responder.await.unwrap();
    assert_eq!(
        granted,
        Some(lofi_code::GrantedDir {
            dir: PathBuf::from("/data/repo"),
            access: write
        })
    );

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
    let sibling = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        approve_outside_dir(
            ws.path(),
            project_approval("/data/repo", Some("/data/repo/lib")),
            &[],
            &tx,
            &counter,
            None,
        ),
    )
    .await
    .expect("a project grant must not ask again for a sibling");
    assert_eq!(
        sibling,
        Some(lofi_code::GrantedDir {
            dir: PathBuf::from("/data/repo"),
            access: write
        })
    );
    assert!(rx.try_recv().is_err());

    std::env::remove_var("LOFI_STATE_HOME");
}

/// An answer that lands after the call's clock closed the prompt is
/// reported so the tool can refuse the work, but the access it asked for
/// is not kept past the call that asked.
#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn a_grant_answered_after_the_wall_is_not_remembered() {
    let _lock = state::STATE_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _state_home = state_env();
    let ws = tempfile::tempdir().unwrap();
    let counter = Arc::new(AtomicU64::new(0));
    let write = lofi_types::Access::Write;
    let past = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_millis(1))
        .unwrap_or_else(std::time::Instant::now);

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
    let responder = tokio::spawn(async move {
        let req = rx.recv().await.unwrap();
        req.respond
            .send(ConfirmResponse::Grant {
                access: write,
                narrow: false,
            })
            .unwrap();
    });
    let granted = approve_outside_dir(
        ws.path(),
        project_approval("/data/repo", None),
        &[],
        &tx,
        &counter,
        Some(past),
    )
    .await;
    responder.await.unwrap();
    assert!(
        granted.is_some(),
        "the answer reaches the tool, which refuses"
    );

    // Nothing was kept: the next request asks again.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
    let asked = tokio::time::timeout(
        std::time::Duration::from_millis(200),
        approve_outside_dir(
            ws.path(),
            project_approval("/data/repo", None),
            &[],
            &tx,
            &counter,
            None,
        ),
    )
    .await;
    assert!(asked.is_err(), "a late grant is not remembered");
    assert!(rx.try_recv().is_ok(), "the next request asks");

    std::env::remove_var("LOFI_STATE_HOME");
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn a_narrow_grant_remembers_the_requested_directory() {
    let _lock = state::STATE_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _state_home = state_env();
    let ws = tempfile::tempdir().unwrap();
    let counter = Arc::new(AtomicU64::new(0));
    let write = lofi_types::Access::Write;

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
    let responder = tokio::spawn(async move {
        let req = rx.recv().await.unwrap();
        req.respond
            .send(ConfirmResponse::Grant {
                access: write,
                narrow: true,
            })
            .unwrap();
    });
    let granted = approve_outside_dir(
        ws.path(),
        project_approval("/data/repo", Some("/data/repo/src")),
        &[],
        &tx,
        &counter,
        None,
    )
    .await;
    responder.await.unwrap();
    assert_eq!(
        granted,
        Some(lofi_code::GrantedDir {
            dir: PathBuf::from("/data/repo/src"),
            access: write
        })
    );

    // The narrower grant answers inside itself and asks again beside it.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
    let inside = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        approve_outside_dir(
            ws.path(),
            project_approval("/data/repo", Some("/data/repo/src/deep")),
            &[],
            &tx,
            &counter,
            None,
        ),
    )
    .await
    .expect("a remembered grant covers its own subdirectory");
    assert_eq!(
        inside,
        Some(lofi_code::GrantedDir {
            dir: PathBuf::from("/data/repo/src"),
            access: write
        })
    );
    assert!(rx.try_recv().is_err());

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
    let responder = tokio::spawn(async move {
        let req = rx.recv().await.unwrap();
        req.respond
            .send(ConfirmResponse::Grant {
                access: write,
                narrow: false,
            })
            .unwrap();
    });
    let sibling = approve_outside_dir(
        ws.path(),
        project_approval("/data/repo", Some("/data/repo/lib")),
        &[],
        &tx,
        &counter,
        None,
    )
    .await;
    responder.await.unwrap();
    assert_eq!(
        sibling,
        Some(lofi_code::GrantedDir {
            dir: PathBuf::from("/data/repo"),
            access: write
        }),
        "a sibling is not covered by the narrower grant"
    );

    std::env::remove_var("LOFI_STATE_HOME");
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn read_request_under_readonly_grant_never_prompts() {
    let _lock = state::STATE_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _state_home = state_env();
    let ws = tempfile::tempdir().unwrap();
    let counter = Arc::new(AtomicU64::new(0));

    // A config read grant answers at once even though the workspace
    // state remembers nothing.
    let config = vec![lofi_code::GrantedDir {
        dir: PathBuf::from("/data/dir"),
        access: lofi_types::Access::Read,
    }];
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
    let hit = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        approve_outside_dir(
            ws.path(),
            dir_approval("/data/dir/sub", lofi_types::Access::Read),
            &config,
            &tx,
            &counter,
            None,
        ),
    )
    .await
    .expect("config grant must not wait for a prompt");
    assert_eq!(
        hit,
        Some(lofi_code::GrantedDir {
            dir: PathBuf::from("/data/dir"),
            access: lofi_types::Access::Read
        })
    );
    assert!(rx.try_recv().is_err());

    // A write under the same read-only grant offers the widening prompt.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
    let responder = tokio::spawn(async move {
        let req = rx.recv().await.unwrap();
        let reason = req
            .reason
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            matches!(&*reason, lofi_code::ConfirmReason::OutsideRoot { requested, current: Some(c), .. } if *requested == lofi_types::Access::Write && c.access == lofi_types::Access::Read && c.dir == std::path::Path::new("/data/dir"))
        );
        req.respond
            .send(ConfirmResponse::Grant {
                access: lofi_types::Access::Write,
                narrow: false,
            })
            .unwrap();
    });
    let widened = approve_outside_dir(
        ws.path(),
        dir_approval("/data/dir", lofi_types::Access::Write),
        &config,
        &tx,
        &counter,
        None,
    )
    .await;
    responder.await.unwrap();
    assert_eq!(
        widened,
        Some(lofi_code::GrantedDir {
            dir: PathBuf::from("/data/dir"),
            access: lofi_types::Access::Write
        })
    );

    std::env::remove_var("LOFI_STATE_HOME");
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn refusal_is_not_remembered() {
    let _lock = state::STATE_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _state_home = state_env();
    let ws = tempfile::tempdir().unwrap();
    let counter = Arc::new(AtomicU64::new(0));

    // A refusal is not cached: both calls ask again and both are denied.
    for _ in 0..2 {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfirmRequest>();
        let responder = tokio::spawn(async move {
            let req = rx.recv().await.unwrap();
            req.respond.send(ConfirmResponse::Deny).unwrap();
        });
        let granted = approve_outside_dir(
            ws.path(),
            dir_approval("/data/dir", lofi_types::Access::Write),
            &[],
            &tx,
            &counter,
            None,
        )
        .await;
        responder.await.unwrap();
        assert_eq!(granted, None);
    }
    assert_eq!(
        state::WorkspaceState::open(ws.path())
            .unwrap()
            .grant_allowing(std::path::Path::new("/data/dir"), lofi_types::Access::Read),
        None
    );

    std::env::remove_var("LOFI_STATE_HOME");
}

#[test]
fn a_wait_takes_the_time_the_run_has_left() {
    let (args, cut) = clamp_child(
        lofi_code::SubagentOp::Wait,
        serde_json::json!({ "id": "a" }),
        clock(Duration::from_secs(30)),
    );

    let wait = args["timeout_ms"].as_u64().unwrap();
    assert!(
        (29_000..=30_000).contains(&wait),
        "the wait takes what the run has left: {args}"
    );
    assert_eq!(cut, None, "a wait the caller did not bound was not cut");
}

#[test]
fn a_mailbox_receive_waits_no_longer_than_the_call() {
    let clock = clock(Duration::from_secs(3));
    let (args, cut) = clamp_child(
        lofi_code::SubagentOp::MailboxReceive,
        serde_json::json!({ "limit": 5, "wait_ms": 10_000 }),
        clock,
    );

    let wait = args["wait_ms"].as_u64().unwrap();
    assert!(
        (2_000..=3_000).contains(&wait),
        "the receive must end with the call, not block past it: {args}"
    );
    assert_eq!(cut, None, "a receive names no cap of its own");
    assert_eq!(args["limit"], serde_json::json!(5));

    // A receive that named no wait is a poll: filling the bound in would
    // make it block for the call instead.
    let (poll, _) = clamp_child(
        lofi_code::SubagentOp::MailboxReceive,
        serde_json::json!({ "limit": 5 }),
        clock,
    );
    assert_eq!(poll["wait_ms"], serde_json::Value::Null);
}

#[test]
fn a_spent_clock_still_asks_for_a_short_wait() {
    let (args, _) = clamp_child(
        lofi_code::SubagentOp::Wait,
        serde_json::json!({ "id": "a" }),
        clock(Duration::ZERO),
    );

    assert_eq!(
        args["timeout_ms"],
        serde_json::json!(1),
        "zero would be read as an invalid argument: {args}"
    );
}

#[test]
fn a_wait_that_is_not_a_number_is_left_to_the_hook() {
    let clock = clock(Duration::from_secs(30));

    for asked in [
        serde_json::json!("bad"),
        serde_json::json!(-1),
        serde_json::json!(1.5),
        serde_json::json!(0),
    ] {
        let (args, _) = clamp_child(
            lofi_code::SubagentOp::Wait,
            serde_json::json!({ "id": "a", "timeout_ms": asked }),
            clock,
        );
        assert_eq!(args["timeout_ms"], asked, "{args}");
    }
}

#[test]
fn a_child_reports_what_its_caller_asked_for_when_the_cap_cut_it() {
    let clock = clock(Duration::from_mins(1));

    let (args, cut) = clamp_child(
        lofi_code::SubagentOp::Run,
        serde_json::json!({ "prompt": "x", "timeout_ms": 600_000 }),
        clock,
    );
    let (unset, _) = clamp_child(
        lofi_code::SubagentOp::Run,
        serde_json::json!({ "prompt": "x" }),
        clock,
    );

    assert_eq!(cut, Some(600_000));
    assert_eq!(
        args["requested_timeout_ms"],
        serde_json::json!(600_000),
        "the run states the allowance it was cut from, not only what it was left: {args}"
    );
    assert!(
        unset.get("requested_timeout_ms").is_none(),
        "a caller that set no timeout has none to report: {unset}"
    );
}

#[test]
fn an_uncapped_run_leaves_a_wait_alone() {
    let (args, _) = clamp_child(
        lofi_code::SubagentOp::Wait,
        serde_json::json!({ "id": "a" }),
        lofi_types::Deadline::NONE,
    );

    assert_eq!(args, serde_json::json!({ "id": "a" }), "{args}");
}

#[test]
fn a_wait_is_clamped_to_the_run_and_not_stretched_by_it() {
    let clock = clock(Duration::from_secs(2));

    let (asked, cut) = clamp_child(
        lofi_code::SubagentOp::Wait,
        serde_json::json!({ "id": "a", "timeout_ms": 900_000 }),
        clock,
    );
    let (short, _) = clamp_child(
        lofi_code::SubagentOp::Wait,
        serde_json::json!({ "id": "a", "timeout_ms": 500 }),
        clock,
    );

    assert!(
        asked["timeout_ms"].as_u64().unwrap() <= 2_000,
        "a wait longer than the run has would block past it: {asked}"
    );
    assert_eq!(
        cut, None,
        "a wait returns the run's own shape, so its timeout is not the run's cap"
    );
    assert_eq!(short["timeout_ms"], serde_json::json!(500));
}

#[test]
fn a_spawned_child_keeps_its_own_timeout() {
    let clock = clock(Duration::from_mins(1));

    let (asked, cut) = clamp_child(
        lofi_code::SubagentOp::Spawn,
        serde_json::json!({ "prompt": "x", "timeout_ms": 600_000 }),
        clock,
    );
    let (unset, uncut) = clamp_child(
        lofi_code::SubagentOp::Spawn,
        serde_json::json!({ "prompt": "x" }),
        clock,
    );

    assert_eq!(
        asked["timeout_ms"],
        serde_json::json!(600_000),
        "a spawned run outlives the call that started it: {asked}"
    );
    assert_eq!(cut, None, "the call's bound is not the child's");
    assert!(
        unset.get("timeout_ms").is_none(),
        "a spawn that set no timeout stays unbounded: {unset}"
    );
    assert_eq!(uncut, None, "a child that set no timeout was not cut");
}

#[test]
fn a_foreground_child_takes_what_the_call_has_left() {
    let clock = clock(Duration::from_mins(1));

    let (run, cut) = clamp_child(
        lofi_code::SubagentOp::Run,
        serde_json::json!({ "prompt": "x" }),
        clock,
    );
    let (asked, asked_cut) = clamp_child(
        lofi_code::SubagentOp::Run,
        serde_json::json!({ "prompt": "x", "timeout_ms": 600_000 }),
        clock,
    );

    let run_ms = run["timeout_ms"].as_u64().unwrap();
    assert!(
        run_ms > 0 && run_ms <= 60_000,
        "a foreground run ends with the call that awaits it: {run}"
    );
    assert_eq!(cut, None, "the call's bound is not a timeout to report");
    assert_eq!(
        asked["timeout_ms"],
        serde_json::json!(60_000),
        "a timeout longer than the call is cut to it: {asked}"
    );
    assert_eq!(
        asked_cut,
        Some(600_000),
        "the allowance it asked for is reported"
    );
}

#[test]
fn a_call_that_runs_nothing_is_left_alone() {
    let clock = clock(Duration::from_secs(2));

    for op in [
        lofi_code::SubagentOp::Kill,
        lofi_code::SubagentOp::Status,
        lofi_code::SubagentOp::List,
    ] {
        let (args, cut) = clamp_child(
            op,
            serde_json::json!({ "id": "a", "timeout_ms": 900_000 }),
            clock,
        );
        assert_eq!(
            args,
            serde_json::json!({ "id": "a", "timeout_ms": 900_000 })
        );
        assert_eq!(cut, None);
    }
}
