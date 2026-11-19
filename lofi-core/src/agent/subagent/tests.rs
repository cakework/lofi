#![allow(clippy::unwrap_used, clippy::wildcard_imports)]

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use lofi_types::{ContentBlock, Message, PromptKind, Role};
use serde_json::{json, Value};

use super::ops::{mailbox_hook, release_op, usage_op};
use super::registry::Activity;
use super::runner::{holds_work, Deadlines, Expiry, RunProgress};
use super::*;
use crate::agent::mailbox::{Mailbox, MailboxScope, PARENT_ADDRESS};
use crate::agent::tool_executor::{ActivityWatch, ConfirmWatch};
use crate::agent::{lock, AgentEvent};

mod mailbox_capability {
    use super::*;

    #[tokio::test]
    async fn a_child_hook_answers_mailbox_ops_but_not_management() {
        let mailbox = Mailbox::default();
        mailbox.open("child");
        let hook = mailbox_hook(MailboxScope {
            mailbox: mailbox.clone(),
            me: "child".to_string(),
        });

        let run = hook(SubagentOp::Run, json!({ "prompt": "x" })).await;
        assert_eq!(run["status"], "unavailable", "{run}");

        let sent = hook(
            SubagentOp::MailboxSend,
            json!({ "to": PARENT_ADDRESS, "body": "hi" }),
        )
        .await;
        assert_eq!(sent["ok"], true, "{sent}");

        let sibling = hook(
            SubagentOp::MailboxSend,
            json!({ "to": "other", "body": "hi" }),
        )
        .await;
        assert_eq!(sibling["status"], "forbidden", "{sibling}");

        let received = mailbox.receive(PARENT_ADDRESS, &json!({})).await;
        assert_eq!(received["messages"][0]["body"], "hi", "{received}");
    }
}

mod retired_spend {
    use super::*;

    // Eviction keeps a run's spend for a by-id fold and adds it to the
    // session total, so a fan-out larger than the retention cap still prices
    // every child and no fold changes what another one sees.
    #[test]
    fn an_evicted_run_is_priced_until_its_exec_releases_it() {
        let registry = SubagentRegistry::default();
        let ids: Vec<String> = (0..2000).map(|i| format!("child-{i}")).collect();
        {
            let mut retired = lock(&registry.retired);
            for id in &ids {
                retired.claim(id);
                retired.keep(
                    id.clone(),
                    json!({ "runs": 1, "input_tokens": 40, "output_tokens": 3 }),
                );
            }
        }

        let by_id = usage_op(&registry, &json!({ "ids": ids.clone() }));
        assert_eq!(by_id["runs"].as_u64(), Some(2000), "{by_id}");
        assert_eq!(by_id["input_tokens"].as_u64(), Some(80_000), "{by_id}");
        assert_eq!(by_id["output_tokens"].as_u64(), Some(6_000), "{by_id}");
        assert!(by_id.get("missing").is_none(), "{by_id}");

        // A fold is a read: the next one sees the same spend.
        let again = usage_op(&registry, &json!({ "ids": ids.clone() }));
        assert_eq!(again, by_id);

        let all = usage_op(&registry, &json!({ "ids": Value::Null }));
        assert_eq!(all["runs"].as_u64(), Some(2000), "{all}");
        assert_eq!(all["input_tokens"].as_u64(), Some(80_000), "{all}");

        // The exec that started them releases their records at its end.
        release_op(&registry, &json!({ "ids": ids.clone() }));
        assert!(lock(&registry.retired).spend.is_empty());
        let after = usage_op(&registry, &json!({ "ids": ids }));
        assert_eq!(after["runs"].as_u64(), Some(0), "{after}");
        let session = usage_op(&registry, &json!({ "ids": Value::Null }));
        assert_eq!(
            session["runs"].as_u64(),
            Some(2000),
            "the session total keeps every evicted run: {session}"
        );

        // A run no exec claims needs no record: its spend is already in that
        // exec's result, and only the session total still counts it.
        {
            let mut retired = lock(&registry.retired);
            retired.keep(
                "foreground".to_string(),
                json!({ "runs": 1, "input_tokens": 5 }),
            );
            assert!(!retired.spend.contains_key("foreground"));
        }
        let session = usage_op(&registry, &json!({ "ids": Value::Null }));
        assert_eq!(session["runs"].as_u64(), Some(2001), "{session}");
        let by_id = usage_op(&registry, &json!({ "ids": ["foreground"] }));
        assert_eq!(by_id["missing"], json!(["foreground"]), "{by_id}");
    }
}

mod activity_line {
    use super::*;

    fn exec(description: Option<&str>) -> AgentEvent {
        AgentEvent::ToolInput {
            id: "t1".to_string(),
            code: "lofi.bash({ cmd: \"cargo test\" })".to_string(),
            label: description.map(str::to_string),
        }
    }

    fn tool_start() -> AgentEvent {
        AgentEvent::ToolStart {
            id: "t1".to_string(),
            name: "exec".to_string(),
        }
    }

    #[test]
    fn a_fresh_run_names_nothing() {
        assert_eq!(Activity::default().line(), "");
    }

    #[test]
    fn an_exec_shows_its_description() {
        let mut activity = Activity::default();
        activity.note(&exec(Some("Map the parser")));
        assert_eq!(activity.line(), "Exec: Map the parser");
    }

    #[test]
    fn the_description_collapses_onto_one_row() {
        let mut activity = Activity::default();
        activity.note(&exec(Some("Map  the\n parser")));
        assert_eq!(activity.line(), "Exec: Map the parser");
    }

    #[test]
    fn an_exec_without_a_description_shows_the_verb() {
        let mut activity = Activity::default();
        activity.note(&exec(None));
        assert_eq!(activity.line(), "Exec");
    }

    #[test]
    fn a_native_tool_leaves_the_description_alone() {
        let mut activity = Activity::default();
        activity.note(&exec(Some("Map the parser")));
        activity.note(&AgentEvent::NativeToolStart {
            parent: "t1".to_string(),
            id: 1,
            name: "read".to_string(),
            args: "{\"path\":\"parser.rs\"}".to_string(),
        });
        assert_eq!(activity.line(), "Exec: Map the parser");
    }

    #[test]
    fn the_response_is_named_by_its_first_line() {
        let mut activity = Activity::default();
        activity.note(&AgentEvent::Text(
            "\n  Surveying the  parser,\n".to_string(),
        ));
        activity.note(&AgentEvent::Text("then the lexer".to_string()));
        assert_eq!(activity.line(), "Surveying the parser,");
    }

    #[test]
    fn a_thinking_round_keeps_the_last_line() {
        let mut activity = Activity::default();
        activity.note(&AgentEvent::Text("the first answer".to_string()));
        activity.note(&tool_start());
        assert_eq!(activity.line(), "the first answer");
        activity.note(&exec(Some("Map the parser")));
        assert_eq!(activity.line(), "Exec: Map the parser");
        activity.note(&AgentEvent::Thinking("weighing the lexer".to_string()));
        assert_eq!(activity.line(), "Exec: Map the parser");
        activity.note(&AgentEvent::Text("the second answer".to_string()));
        assert_eq!(activity.line(), "the second answer");
    }

    #[test]
    fn a_blank_first_chunk_keeps_the_last_line() {
        let mut activity = Activity::default();
        activity.note(&exec(Some("Map the parser")));
        activity.note(&AgentEvent::Text("\n  ".to_string()));
        assert_eq!(activity.line(), "Exec: Map the parser");
    }
}

mod deadlines {
    use super::*;

    fn limits(args: &Value) -> Deadlines {
        match Deadlines::from_args(args) {
            Ok(limits) => limits,
            Err(error) => panic!("{error}"),
        }
    }

    fn progress() -> RunProgress {
        RunProgress::new(
            std::time::Instant::now(),
            Arc::new(ConfirmWatch::default()),
            Arc::new(ActivityWatch::default()),
        )
    }

    // A cap is the caller's bound, never a default: a fixed clock cuts exactly
    // the long-horizon work a child exists for.
    #[test]
    fn a_run_has_no_cap_unless_the_caller_sets_one() {
        let limits = limits(&json!({ "prompt": "a" }));

        assert!(limits.cap.is_none(), "a run was capped by default");
        assert_eq!(limits.idle, Duration::from_millis(DEFAULT_IDLE_MS));
    }

    #[test]
    fn timeout_ms_is_the_only_way_to_cap_a_run() {
        let limits = limits(&json!({ "prompt": "a", "timeout_ms": 1_500 }));

        assert_eq!(limits.cap, Some(Duration::from_millis(1_500)));
    }

    #[test]
    fn an_uncapped_run_never_expires_on_the_clock() {
        let limits = Deadlines {
            idle: Duration::from_hours(1),
            cap: None,
        };
        // Thirty hours of work, with an event a second ago: long, not stuck.
        let mut progress = progress();
        let Some(far_past) = std::time::Instant::now().checked_sub(Duration::from_hours(30)) else {
            panic!("thirty hours fits in an instant");
        };
        progress.started = far_past;
        progress.last_event.store(
            (Duration::from_hours(30) - Duration::from_secs(1)).as_millis() as u64,
            Ordering::SeqCst,
        );

        assert_eq!(progress.expired(&limits), None);

        let capped = Deadlines {
            idle: limits.idle,
            cap: Some(Duration::from_hours(2)),
        };
        assert_eq!(
            progress.expired(&capped),
            Some(Expiry::Cap(Duration::from_hours(2)))
        );
    }

    #[test]
    fn run_progress_counts_from_the_instant_it_is_given() {
        let limits = Deadlines {
            idle: Duration::from_hours(1),
            cap: Some(Duration::from_secs(30)),
        };
        // Armed a minute ago, though this task only started now: the cap is
        // spent, because the run began when its clock did.
        let armed = std::time::Instant::now()
            .checked_sub(Duration::from_mins(1))
            .unwrap_or_else(std::time::Instant::now);

        let progress = RunProgress::new(
            armed,
            Arc::new(ConfirmWatch::default()),
            Arc::new(ActivityWatch::default()),
        );

        assert_eq!(
            progress.expired(&limits),
            Some(Expiry::Cap(Duration::from_secs(30)))
        );
    }

    #[test]
    fn a_capped_run_expires_at_its_cap_and_names_it() {
        let limits = Deadlines {
            idle: Duration::from_hours(1),
            cap: Some(Duration::ZERO),
        };

        assert_eq!(
            progress().expired(&limits),
            Some(Expiry::Cap(Duration::ZERO))
        );
    }
}

mod kept_work {
    use super::*;

    fn message(role: Role, blocks: Vec<ContentBlock>) -> Message {
        Message {
            origin: None,
            role,
            blocks,
            kind: PromptKind::default(),
        }
    }

    fn text(text: &str) -> ContentBlock {
        ContentBlock::Text {
            text: text.to_string(),
        }
    }

    fn thinking(text: &str, signature: Option<&str>) -> ContentBlock {
        ContentBlock::Thinking {
            text: text.to_string(),
            signature: signature.map(str::to_string),
            redacted: false,
        }
    }

    #[test]
    fn a_reply_or_a_tool_result_is_work() {
        assert!(holds_work(&[message(Role::Assistant, vec![text("hi")])]));
        assert!(holds_work(&[message(
            Role::Assistant,
            vec![ContentBlock::ToolUse {
                id: "t1".to_string(),
                name: "exec".to_string(),
                input: json!({}),
            }],
        )]));
        assert!(holds_work(&[message(
            Role::Tool,
            vec![ContentBlock::ToolResult {
                tool_use_id: "t1".to_string(),
                content: "ok".to_string(),
                is_error: false,
                images: Vec::new(),
            }],
        )]));
    }

    // A request replays neither an empty message nor a thinking block cut
    // before its signature, and the task prompt is replayed by any resume.
    #[test]
    fn a_message_a_request_would_drop_is_not_work() {
        assert!(!holds_work(&[]));
        assert!(!holds_work(&[message(Role::Assistant, vec![])]));
        assert!(!holds_work(&[message(Role::Assistant, vec![text("")])]));
        assert!(!holds_work(&[message(Role::Assistant, vec![text(" \n ")])]));
        assert!(!holds_work(&[message(
            Role::Assistant,
            vec![thinking("half a thought", None)]
        )]));
        assert!(!holds_work(&[message(Role::User, vec![text("a")])]));
        assert!(holds_work(&[message(
            Role::Assistant,
            vec![thinking("a thought", Some("sig"))]
        )]));
    }
}
