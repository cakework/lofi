#![allow(clippy::unwrap_used, clippy::wildcard_imports)]

use lofi_types::{ContentBlock, Message, PromptKind, Role, SessionEvent, SessionEventKind, Usage};

use super::*;
use crate::agent::AgentEvent;

fn user_msg(t: &str) -> SessionEvent {
    SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::Message(Message {
            origin: None,
            role: Role::User,
            blocks: vec![ContentBlock::Text { text: t.into() }],
            kind: PromptKind::default(),
        }),
    }
}

fn user_msg_with_kind(t: &str, kind: lofi_types::PromptKind) -> SessionEvent {
    SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::Message(Message {
            origin: None,
            role: Role::User,
            blocks: vec![ContentBlock::Text { text: t.into() }],
            kind,
        }),
    }
}

fn chain(events: &mut [SessionEvent]) {
    for (i, ev) in events.iter_mut().enumerate() {
        ev.id = format!("e{i}");
        ev.parent_id = (i > 0).then(|| format!("e{}", i - 1));
    }
}

fn timer(id: u64) -> lofi_types::TimerRecord {
    lofi_types::TimerRecord {
        id,
        due_ms: 1_700_000_000_000,
        every_ms: None,
        message: Some(format!("timer {id}")),
        owner: None,
    }
}

fn timer_state(timers: &[lofi_types::TimerRecord]) -> SessionEvent {
    SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::TimerState {
            timers: timers.to_vec(),
        },
    }
}

fn job_started(id: u64) -> SessionEvent {
    SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::JobStarted {
            job_id: id,
            cmd: None,
        },
    }
}

fn job_finished(id: u64) -> SessionEvent {
    SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::JobFinished { job_id: id },
    }
}

fn assistant_msg(t: &str) -> SessionEvent {
    SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::Message(Message {
            origin: None,
            role: Role::Assistant,
            blocks: vec![ContentBlock::Text { text: t.into() }],
            kind: PromptKind::default(),
        }),
    }
}

fn round_discarded(detail: &str) -> SessionEvent {
    SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::RoundDiscarded {
            detail: detail.to_string(),
        },
    }
}

fn turn_end() -> SessionEvent {
    SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::TurnEnd {
            model: "m".into(),
            elapsed_ms: 0,
            cost: 0.0,
            usage: Usage::default(),
            stop_reason: None,
        },
    }
}

fn turn_failed(keep_context: bool) -> SessionEvent {
    SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::TurnFailed {
            model: "m".into(),
            elapsed_ms: 0,
            error: "boom".into(),
            cost: 0.0,
            usage: Usage::default(),
            keep_context,
        },
    }
}

fn turn_cancelled() -> SessionEvent {
    SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::TurnCancelled {
            model: "m".into(),
            elapsed_ms: 0,
            cost: 0.0,
            usage: Usage::default(),
        },
    }
}

fn turn_started() -> SessionEvent {
    SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::TurnStarted,
    }
}

fn turn_usage(cost: f64, input_tokens: u64) -> SessionEvent {
    turn_usage_at(5, cost, input_tokens)
}

fn turn_usage_at(elapsed_ms: u64, cost: f64, input_tokens: u64) -> SessionEvent {
    SessionEvent {
        id: String::new(),
        parent_id: None,
        kind: SessionEventKind::TurnUsage {
            elapsed_ms,
            cost,
            usage: Usage {
                input_tokens,
                ..Usage::default()
            },
        },
    }
}

#[test]
fn a_turn_no_marker_ended_reports_the_spend_its_checkpoint_kept() {
    let mut events = vec![
        user_msg("go"),
        assistant_msg("round one"),
        turn_usage(0.5, 40),
    ];
    chain(&mut events);
    let mut replayed = Vec::new();
    replay_session_events(&events, |event| replayed.push(event));

    assert!(
        replayed.iter().any(|event| matches!(
            event,
            AgentEvent::TurnUsage { cost, usage }
                if (*cost - 0.5).abs() < 1e-9 && usage.input_tokens == 40
        )),
        "a killed turn keeps its spend: {replayed:?}"
    );
}

#[test]
fn a_terminal_marker_supersedes_the_checkpoint_spend() {
    let mut events = vec![
        user_msg("go"),
        assistant_msg("round one"),
        turn_usage(0.5, 40),
        turn_end(),
    ];
    chain(&mut events);
    let mut replayed = Vec::new();
    replay_session_events(&events, |event| replayed.push(event));

    assert!(
        replayed
            .iter()
            .all(|event| !matches!(event, AgentEvent::TurnUsage { .. })),
        "the terminal marker already carries the same spend: {replayed:?}"
    );
    assert_eq!(
        replayed
            .iter()
            .filter(|event| matches!(event, AgentEvent::TurnEnd { .. }))
            .count(),
        1
    );
}

#[test]
fn the_next_turn_does_not_erase_the_spend_of_a_killed_one() {
    // Run A checkpointed $0.50 and was killed. Run B resumed the session:
    // its first checkpoint reports the new turn's own spend, not A's.
    let mut events = vec![
        user_msg("go"),
        assistant_msg("round one"),
        turn_usage(0.5, 40),
        user_msg("resumed"),
        assistant_msg("round two"),
        turn_usage(0.2, 10),
        SessionEvent {
            id: String::new(),
            parent_id: None,
            kind: SessionEventKind::TurnEnd {
                model: "m".into(),
                elapsed_ms: 0,
                cost: 0.2,
                usage: Usage::default(),
                stop_reason: None,
            },
        },
    ];
    chain(&mut events);
    let mut replayed = Vec::new();
    replay_session_events(&events, |event| replayed.push(event));

    let killed: Vec<f64> = replayed
        .iter()
        .filter_map(|event| match event {
            AgentEvent::TurnUsage { cost, .. } => Some(*cost),
            _ => None,
        })
        .collect();
    assert_eq!(killed.len(), 1, "only the killed turn reports a snapshot");
    assert!((killed[0] - 0.5).abs() < 1e-9, "{}", killed[0]);
    assert!(
        replayed.iter().any(|event| matches!(
            event,
            AgentEvent::TurnEnd { cost, .. } if (*cost - 0.2).abs() < 1e-9
        )),
        "the settled turn reports itself: {replayed:?}"
    );
}

#[test]
fn two_killed_turns_each_report_their_own_spend() {
    let mut events = vec![
        user_msg("go"),
        assistant_msg("round one"),
        turn_usage(0.5, 40),
        user_msg("resumed"),
        assistant_msg("round two"),
        turn_usage(0.2, 10),
    ];
    chain(&mut events);
    let mut replayed = Vec::new();
    replay_session_events(&events, |event| replayed.push(event));

    let killed: Vec<f64> = replayed
        .iter()
        .filter_map(|event| match event {
            AgentEvent::TurnUsage { cost, .. } => Some(*cost),
            _ => None,
        })
        .collect();
    assert_eq!(killed.len(), 2, "{replayed:?}");
    assert!((killed[0] - 0.5).abs() < 1e-9 && (killed[1] - 0.2).abs() < 1e-9);
}

#[test]
fn a_resumed_history_closes_the_tool_call_a_kill_left_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.jsonl");
    std::fs::write(
        &path,
        b"{\"type\":\"meta\",\"version\":1,\"created\":0,\"cwd\":\"\",\"model\":\"m\"}\n",
    )
    .unwrap();
    let cursor = crate::session::store::SessionCursor::new(path, None);
    let mut events = vec![
        user_msg("go"),
        SessionEvent {
            id: String::new(),
            parent_id: None,
            kind: SessionEventKind::Message(Message {
                origin: None,
                role: Role::Assistant,
                blocks: vec![ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: serde_json::json!({ "cmd": "sleep 30" }),
                }],
                kind: PromptKind::default(),
            }),
        },
    ];
    cursor.append_events(&mut events).unwrap();
    let snapshot = cursor.snapshot().unwrap();

    let history = history_from_index(&cursor, &snapshot.index, snapshot.history_start).unwrap();

    // No provider accepts a tool call with no result, so the restore pairs
    // the call the killed process left open.
    let last = history.last().unwrap();
    assert_eq!(last.role, Role::Tool);
    assert!(
        matches!(
            &last.blocks[0],
            ContentBlock::ToolResult { tool_use_id, is_error: true, .. } if tool_use_id == "t1"
        ),
        "{:?}",
        last.blocks
    );
}

#[test]
fn a_turn_start_settles_the_spend_of_a_turn_with_no_marker() {
    // A capped turn stops with no marker. The turn that follows has no
    // prompt of its own, so the durable turn start is its only boundary.
    let mut events = vec![
        user_msg("go"),
        assistant_msg("round one"),
        turn_usage_at(9_000, 0.5, 40),
        turn_started(),
        assistant_msg("round two"),
        turn_usage_at(300, 0.1, 10),
        turn_end(),
    ];
    chain(&mut events);
    let mut replayed = Vec::new();
    replay_session_events(&events, |event| replayed.push(event));

    let costs: Vec<f64> = replayed
        .iter()
        .filter_map(|event| match event {
            AgentEvent::TurnUsage { cost, .. } => Some(*cost),
            _ => None,
        })
        .collect();
    assert_eq!(costs.len(), 1, "{replayed:?}");
    assert!((costs[0] - 0.5).abs() < 1e-9, "{}", costs[0]);
}

#[test]
fn a_continuation_after_the_context_cap_keeps_the_capped_spend() {
    // The capped turn stopped with no marker and wrote $0.50. The UI then
    // compacts and continues: that turn has no prompt of its own, so only
    // the turn clock shows it is a new one.
    let mut events = vec![
        user_msg("go"),
        assistant_msg("round one"),
        turn_usage_at(9_000, 0.5, 40),
        SessionEvent {
            id: String::new(),
            parent_id: None,
            kind: SessionEventKind::Compaction {
                summary: "so far".into(),
                first_kept_entry_id: String::new(),
                summarized_range: [String::new(), String::new()],
                checkpointed_tail: false,
                summarized: 2,
                kept: 0,
                represented: 0,
            },
        },
        assistant_msg("round two"),
        turn_usage_at(400, 0.1, 10),
        turn_end(),
    ];
    chain(&mut events);
    let mut replayed = Vec::new();
    replay_session_events(&events, |event| replayed.push(event));

    let costs: Vec<f64> = replayed
        .iter()
        .filter_map(|event| match event {
            AgentEvent::TurnUsage { cost, .. } => Some(*cost),
            _ => None,
        })
        .collect();
    assert_eq!(
        costs.len(),
        1,
        "the capped turn reports its spend: {replayed:?}"
    );
    assert!((costs[0] - 0.5).abs() < 1e-9, "{}", costs[0]);
}

#[test]
fn an_auto_continue_notice_does_not_split_a_turn() {
    // The notice is a display row of the same turn, so the turn's own
    // marker must still supersede the snapshots before and after it.
    let mut events = vec![
        user_msg("go"),
        assistant_msg("round one"),
        turn_usage(0.5, 40),
        user_msg_with_kind("token limit reached", lofi_types::PromptKind::Notice),
        assistant_msg("round two"),
        turn_usage(0.2, 10),
        turn_end(),
    ];
    chain(&mut events);
    let mut replayed = Vec::new();
    replay_session_events(&events, |event| replayed.push(event));

    assert!(
        replayed
            .iter()
            .all(|event| !matches!(event, AgentEvent::TurnUsage { .. })),
        "the marker carries the whole turn: {replayed:?}"
    );
}

#[test]
fn round_discarded_keeps_content_visible_but_excludes_it_from_context() {
    let mut events = vec![
        user_msg("go"),
        assistant_msg("repeating reasoning"),
        round_discarded("potential agent loop detected"),
        user_msg_with_kind("try again differently", lofi_types::PromptKind::Notice),
        assistant_msg("recovery answer"),
        turn_end(),
    ];
    chain(&mut events);
    let visible = visible_event_indices(&events);
    assert_eq!(
        visible.len(),
        events.len(),
        "the discarded round stays visible"
    );

    let messages = messages_from_events(&events);
    let texts: Vec<&str> = messages
        .iter()
        .map(|message| match &message.blocks[0] {
            ContentBlock::Text { text } => text.as_str(),
            _ => "",
        })
        .collect();
    assert_eq!(texts, ["go", "try again differently", "recovery answer"]);

    let mut replayed = Vec::new();
    replay_session_events(&events, |event| replayed.push(event));
    assert!(replayed.iter().any(|event| matches!(
        event,
        AgentEvent::Prompt { prompt, kind }
            if *kind == PromptKind::Notice && prompt == "try again differently"
    )));
    assert!(
        replayed
            .iter()
            .all(|event| !matches!(event, AgentEvent::Notice(_))),
        "discard metadata must not become a transient notification"
    );
}

#[test]
fn consecutive_round_discards_chain() {
    let mut events = vec![
        user_msg("go"),
        assistant_msg("first loop"),
        round_discarded("first discard"),
        user_msg_with_kind("notice", lofi_types::PromptKind::Notice),
        assistant_msg("second loop"),
        assistant_msg("still looping"),
        round_discarded("second discard"),
        user_msg_with_kind("notice2", lofi_types::PromptKind::Notice),
        assistant_msg("done answer"),
        turn_end(),
    ];
    chain(&mut events);
    let messages = messages_from_events(&events);
    let texts: Vec<&str> = messages
        .iter()
        .map(|message| match &message.blocks[0] {
            ContentBlock::Text { text } => text.as_str(),
            _ => "",
        })
        .collect();
    assert_eq!(
        texts,
        ["go", "notice", "notice2", "done answer"],
        "each marker drops the assistant messages it ends with"
    );
}

#[test]
fn failed_turn_keeps_prompt_and_completed_rounds_in_context() {
    let mut events = vec![
        user_msg("previous prompt"),
        assistant_msg("previous reply"),
        turn_end(),
        user_msg("failed prompt"),
        assistant_msg("discarded attempt"),
        round_discarded("loop"),
        user_msg_with_kind("retry notice", lofi_types::PromptKind::Notice),
        assistant_msg("failed partial"),
        round_discarded("discarding partial response after failure: boom"),
        turn_failed(false),
    ];
    chain(&mut events);

    let messages = messages_from_events(&events);
    let texts: Vec<&str> = messages
        .iter()
        .map(|message| match &message.blocks[0] {
            ContentBlock::Text { text } => text.as_str(),
            _ => "",
        })
        .collect();

    // Each discard marker drops the assistant round it ends; the failed
    // turn's prompt and notices stay in context for the next prompt.
    assert_eq!(
        texts,
        [
            "previous prompt",
            "previous reply",
            "failed prompt",
            "retry notice"
        ]
    );
}

#[test]
fn failed_turn_that_kept_context_keeps_its_prompt() {
    // A failure that never ran a round writes a kept turn and no discard
    // marker. Its prompt stays in context, as it did live.
    let mut events = vec![
        user_msg("previous prompt"),
        assistant_msg("previous reply"),
        turn_end(),
        user_msg("failed prompt"),
        turn_failed(true),
        user_msg("next prompt"),
        assistant_msg("next reply"),
        turn_end(),
    ];
    chain(&mut events);

    let messages = messages_from_events(&events);
    let texts: Vec<&str> = messages
        .iter()
        .map(|message| match &message.blocks[0] {
            ContentBlock::Text { text } => text.as_str(),
            _ => "",
        })
        .collect();

    assert_eq!(
        texts,
        [
            "previous prompt",
            "previous reply",
            "failed prompt",
            "next prompt",
            "next reply"
        ]
    );
}

#[test]
fn failed_turn_without_discard_marker_hides_the_whole_turn() {
    // Logs from before the failure arm wrote a discard marker have no
    // boundary inside the failed turn, so the skip runs to the previous
    // turn boundary as before.
    let mut events = vec![
        user_msg("previous prompt"),
        assistant_msg("previous reply"),
        turn_end(),
        user_msg("failed prompt"),
        assistant_msg("failed partial"),
        turn_failed(false),
    ];
    chain(&mut events);

    let messages = messages_from_events(&events);
    let texts: Vec<&str> = messages
        .iter()
        .map(|message| match &message.blocks[0] {
            ContentBlock::Text { text } => text.as_str(),
            _ => "",
        })
        .collect();

    assert_eq!(texts, ["previous prompt", "previous reply"]);
}

#[test]
fn failed_turn_does_not_hide_prior_cancelled_turn() {
    let mut events = vec![
        user_msg("cancelled prompt"),
        assistant_msg("cancelled partial"),
        turn_cancelled(),
        user_msg("failed prompt"),
        assistant_msg("failed partial"),
        turn_failed(false),
    ];
    chain(&mut events);

    let messages = messages_from_events(&events);
    let texts: Vec<&str> = messages
        .iter()
        .map(|message| match &message.blocks[0] {
            ContentBlock::Text { text } => text.as_str(),
            _ => "",
        })
        .collect();

    assert_eq!(texts, ["cancelled prompt", "cancelled partial"]);
}

#[test]
fn outstanding_job_ids_returns_empty_on_no_markers() {
    let mut events = vec![user_msg("hello")];
    chain(&mut events);
    assert!(outstanding_job_ids(&events).is_empty());
}

#[test]
fn outstanding_job_ids_returns_unfinished_starts() {
    let mut events = vec![
        user_msg("a"),
        job_started(1),
        job_started(2),
        job_finished(1),
        user_msg("b"),
    ];
    chain(&mut events);
    assert_eq!(outstanding_job_ids(&events), vec![2]);
}

#[test]
fn outstanding_job_ids_returns_all_when_none_finished() {
    let mut events = vec![job_started(7), user_msg("x"), job_started(8)];
    chain(&mut events);
    assert_eq!(outstanding_job_ids(&events), vec![7, 8]);
}

#[test]
fn outstanding_job_ids_returns_empty_when_all_finished() {
    let mut events = vec![job_started(3), job_finished(3), user_msg("done")];
    chain(&mut events);
    assert!(outstanding_job_ids(&events).is_empty());
}

#[test]
fn outstanding_jobs_at_ignores_a_job_finished_on_a_pruned_branch() {
    let dir = tempfile::tempdir().unwrap();
    let store = crate::session::store::SessionStore::new(dir.path().join("sessions"));
    let cursor = store.create_cursor(dir.path(), &"p/m".into()).unwrap();
    let owner = cursor.record_job_started(42, "cargo test").unwrap();
    let started_id = cursor.leaf_id();
    cursor
        .record_job_finished(&owner, 42, lofi_code::JobRelease::Exited)
        .unwrap();
    // Roll back to the JobStarted event so the JobFinished marker lands on
    // the pruned branch while JobStarted stays on the selected lineage.
    cursor.restore_branch(started_id).unwrap();
    let _live = cursor.record_job_started(43, "sleep 60").unwrap();

    let snapshot = cursor.snapshot().unwrap();
    let outstanding = outstanding_jobs_at(&cursor, &snapshot.index).unwrap();
    let ids = outstanding.iter().map(|job| job.id).collect::<Vec<_>>();
    assert_eq!(
        ids,
        vec![43],
        "only the job released on another branch is dropped: {outstanding:?}"
    );
}

#[test]
fn job_lifecycle_markers_do_not_map_to_ir() {
    assert!(agent_message_for_event(&SessionEventKind::JobStarted {
        job_id: 1,
        cmd: None,
    })
    .is_none());
    assert!(agent_message_for_event(&SessionEventKind::JobFinished { job_id: 1 }).is_none());
}

#[test]
fn replay_suppresses_job_lifecycle_markers() {
    let events = vec![
        user_msg("a"),
        job_started(1),
        job_finished(1),
        user_msg("b"),
    ];
    let mut prompts: Vec<String> = Vec::new();
    replay_selected_session_events(&events, |ev| {
        if let AgentEvent::Prompt { prompt, .. } = ev {
            prompts.push(prompt);
        }
    });
    assert_eq!(prompts, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn timer_records_returns_the_last_state_on_the_lineage() {
    let mut events = vec![
        user_msg("a"),
        timer_state(&[timer(3)]),
        user_msg("b"),
        timer_state(&[timer(4), timer(5)]),
    ];
    chain(&mut events);
    assert_eq!(timer_records(&events), vec![timer(4), timer(5)]);
}

#[test]
fn timer_records_returns_empty_when_a_later_state_cleared_them() {
    let mut events = vec![timer_state(&[timer(1)]), timer_state(&[])];
    chain(&mut events);
    assert!(timer_records(&events).is_empty());
}

#[test]
fn timer_state_does_not_map_to_ir() {
    assert!(agent_message_for_event(&SessionEventKind::TimerState { timers: vec![] }).is_none());
}

#[test]
fn replay_restores_turn_kind_from_message_kind_field() {
    let events = vec![
        user_msg("typed"),
        user_msg_with_kind("job 1 completed", lofi_types::PromptKind::Notice),
        user_msg("typed again"),
    ];
    let mut kinds: Vec<lofi_types::PromptKind> = Vec::new();
    replay_selected_session_events(&events, |ev| {
        if let AgentEvent::Prompt { kind, .. } = ev {
            kinds.push(kind);
        }
    });
    assert_eq!(
        kinds,
        vec![
            lofi_types::PromptKind::User,
            lofi_types::PromptKind::Notice,
            lofi_types::PromptKind::User,
        ],
        "kind travels with each user message and resets to User"
    );
}

#[test]
fn replay_does_not_guess_live_run_boundaries() {
    let events = vec![
        user_msg("first"),
        assistant_msg("partial"),
        user_msg("second"),
    ];
    let mut prompts = Vec::new();
    replay_selected_session_events(&events, |event| match event {
        AgentEvent::RunStart => panic!("run starts are live-only"),
        AgentEvent::Prompt { prompt, .. } => prompts.push(prompt),
        _ => {}
    });
    assert_eq!(prompts, ["first", "second"]);
}
