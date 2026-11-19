use super::*;

#[test]
fn insert_str_normalizes_line_endings() {
    let mut a = app();
    a.insert_str("a\r\nb\rc");
    assert_eq!(a.prompt.text, "a\nb\nc");
}

#[test]
fn model_picker_open_preselects_current() {
    let mut a = app(); // model.label = "openai/gpt-4o"
    a.model.choices = vec![
        lofi_types::ModelChoice {
            provider: "anthropic".into(),
            id: "claude".into(),
            name: "Claude".into(),
            thinking_levels: vec![ThinkingLevel::Medium],
            service_tiers: vec![],
            supports_image: false,
            context_window: Some(200_000),
        },
        lofi_types::ModelChoice {
            provider: "openai".into(),
            id: "gpt-4o".into(),
            name: String::new(),
            thinking_levels: vec![],
            service_tiers: vec![],
            supports_image: false,
            context_window: Some(128_000),
        },
    ];
    a.open_model_picker();
    let Some(ActiveModal::Model(picker)) = &a.modal else {
        panic!("expected model picker");
    };
    assert_eq!(picker.len(), 2);
    assert_eq!(picker.selected, 1);
    assert!(a.modal_open());
}

#[test]
fn model_picker_open_empty_notifies() {
    let mut a = app();
    a.model.choices = Vec::new();
    a.open_model_picker();
    assert!(!matches!(a.modal, Some(ActiveModal::Model(_))));
    assert!(a.notify.is_some());
}

#[test]
fn model_picker_confirm_sets_pending_switch() {
    let mut a = app();
    a.model.choices = vec![
        lofi_types::ModelChoice {
            provider: "anthropic".into(),
            id: "claude".into(),
            name: "Claude".into(),
            thinking_levels: vec![],
            service_tiers: vec![],
            supports_image: false,
            context_window: None,
        },
        lofi_types::ModelChoice {
            provider: "openai".into(),
            id: "gpt-4o".into(),
            name: String::new(),
            thinking_levels: vec![],
            service_tiers: vec![],
            supports_image: false,
            context_window: None,
        },
    ];
    a.open_model_picker();
    let Some(ActiveModal::Model(picker)) = &mut a.modal else {
        panic!("expected model picker");
    };
    picker.selected = 0;
    a.model_picker_confirm();
    assert_eq!(a.model.pending_switch.as_deref(), Some("anthropic/claude"));
    assert!(!matches!(a.modal, Some(ActiveModal::Model(_))));
}

#[test]
fn apply_model_switch_updates_label_and_ctx_limit() {
    let mut a = app();
    a.model.ctx_limit = 0; // falls back to DEFAULT_CTX_LIMIT until a model reports one
    let model = lofi_types::Model {
        id: "claude".into(),
        name: "Claude".into(),
        provider: "anthropic".into(),
        api: lofi_types::Api::AnthropicMessages,
        reasoning: true,
        thinking: ThinkingLevel::XHigh,
        service_tier: ServiceTier::Auto,
        supports_image: true,
        context_window: Some(200_000),
        max_tokens: None,
        base_url: None,
        input_price: None,
        output_price: None,
        cache_read_price: None,
        cache_write_price: None,
        per_request_price: None,
    };
    a.apply_model_switch(
        &format!("{}/{}", model.provider, model.id),
        ThinkingLevel::XHigh,
        ServiceTier::Auto,
        model.context_window,
    );
    assert_eq!(a.model.label, "anthropic/claude");
    assert_eq!(a.model.thinking_label.as_deref(), Some(":xhigh"));
    assert_eq!(a.model.ctx_limit, 200_000);
    assert_eq!(a.model.thinking, ThinkingLevel::XHigh);
}

#[test]
fn apply_model_switch_with_no_context_window_uses_default() {
    let mut a = app();
    a.model.ctx_limit = 100_000;
    let model = lofi_types::Model {
        id: "local".into(),
        name: "Local".into(),
        provider: "ollama".into(),
        api: lofi_types::Api::OpenAiResponses,
        reasoning: false,
        thinking: ThinkingLevel::Off,
        service_tier: ServiceTier::Auto,
        supports_image: false,
        context_window: None,
        max_tokens: None,
        base_url: None,
        input_price: None,
        output_price: None,
        cache_read_price: None,
        cache_write_price: None,
        per_request_price: None,
    };
    a.apply_model_switch(
        &format!("{}/{}", model.provider, model.id),
        ThinkingLevel::Off,
        ServiceTier::Auto,
        model.context_window,
    );
    assert_eq!(a.model.ctx_limit, DEFAULT_CTX_LIMIT);
}

#[test]
fn thinking_picker_open_preselects_current() {
    let mut a = app(); // model.label = "openai/gpt-4o", thinking = Medium
    a.model.choices = vec![lofi_types::ModelChoice {
        provider: "openai".into(),
        id: "gpt-4o".into(),
        name: String::new(),
        thinking_levels: vec![ThinkingLevel::Medium, ThinkingLevel::High],
        service_tiers: vec![],
        supports_image: false,
        context_window: None,
    }];
    a.open_thinking_picker();
    let Some(ActiveModal::Thinking(picker)) = &a.modal else {
        panic!("expected thinking picker");
    };
    assert_eq!(
        picker.items,
        vec![
            ThinkingLevel::Off,
            ThinkingLevel::Medium,
            ThinkingLevel::High
        ]
    );
    assert_eq!(picker.selected, 1);
    assert!(a.modal_open());
}

#[test]
fn thinking_picker_open_no_levels_notifies() {
    let mut a = app();
    a.model.choices = vec![lofi_types::ModelChoice {
        provider: "openai".into(),
        id: "gpt-4o".into(),
        name: String::new(),
        thinking_levels: vec![],
        service_tiers: vec![],
        supports_image: false,
        context_window: None,
    }];
    a.open_thinking_picker();
    assert!(!matches!(a.modal, Some(ActiveModal::Thinking(_))));
    assert!(a.notify.is_some());
}

#[test]
fn thinking_picker_confirm_sets_pending_switch() {
    let mut a = app(); // model.label = "openai/gpt-4o"
    a.model.choices = vec![lofi_types::ModelChoice {
        provider: "openai".into(),
        id: "gpt-4o".into(),
        name: String::new(),
        thinking_levels: vec![ThinkingLevel::Medium, ThinkingLevel::High],
        service_tiers: vec![],
        supports_image: false,
        context_window: None,
    }];
    a.open_thinking_picker();
    let Some(ActiveModal::Thinking(picker)) = &mut a.modal else {
        panic!("expected thinking picker");
    };
    picker.selected = 2;
    a.thinking_picker_confirm();
    assert_eq!(
        a.model.pending_switch.as_deref(),
        Some("openai/gpt-4o:high")
    );
    assert!(!matches!(a.modal, Some(ActiveModal::Thinking(_))));
}

#[test]
fn thinking_picker_confirm_preserves_active_service_tier() {
    let mut a = App::new(
        "openai/gpt-4o".to_string(),
        ThinkingLevel::Medium,
        ServiceTier::Flex,
        0,
        lofi_types::CompactionConfig::default(),
        String::new(),
    );
    a.model.choices = vec![lofi_types::ModelChoice {
        provider: "openai".into(),
        id: "gpt-4o".into(),
        name: String::new(),
        thinking_levels: vec![ThinkingLevel::Medium, ThinkingLevel::High],
        service_tiers: vec![ServiceTier::Auto, ServiceTier::Flex],
        supports_image: false,
        context_window: None,
    }];
    a.open_thinking_picker();
    let Some(ActiveModal::Thinking(picker)) = &mut a.modal else {
        panic!("expected thinking picker");
    };
    picker.selected = 2;
    a.thinking_picker_confirm();
    assert_eq!(
        a.model.pending_switch.as_deref(),
        Some("openai/gpt-4o:high@flex")
    );
}

#[test]
fn policy_picker_defaults_to_ask_manual_without_auto_mode() {
    let mut a = app();
    a.executor = Some(lofi_core::agent::ToolExecutor::in_memory("/ws"));
    a.open_policy_picker();
    let Some(ActiveModal::Policy(picker)) = &a.modal else {
        panic!("expected policy picker");
    };
    assert_eq!(
        picker.items,
        vec![
            lofi_types::BashApprovalMode::AllowAll,
            lofi_types::BashApprovalMode::AskManual,
            lofi_types::BashApprovalMode::DenyAll
        ]
    );
    assert_eq!(picker.selected, 1);
    assert!(a.modal_open());
}

#[test]
fn policy_picker_includes_and_defaults_to_ask_auto_when_configured() {
    let mut a = app();
    a.executor = Some(
        lofi_core::agent::ToolExecutor::in_memory("/ws").with_auto_mode(Some(std::sync::Arc::new(
            |_| Box::pin(async { unreachable!() }),
        ))),
    );
    a.open_policy_picker();
    let Some(ActiveModal::Policy(picker)) = &a.modal else {
        panic!("expected policy picker");
    };
    assert_eq!(
        picker.items,
        vec![
            lofi_types::BashApprovalMode::AllowAll,
            lofi_types::BashApprovalMode::AskManual,
            lofi_types::BashApprovalMode::AskAuto,
            lofi_types::BashApprovalMode::DenyAll
        ]
    );
    assert_eq!(picker.selected, 2);
}

#[test]
fn policy_picker_confirm_writes_the_shared_override() {
    let mut a = app();
    let override_handle = lofi_core::PolicyOverride::default();
    a.executor = Some(
        lofi_core::agent::ToolExecutor::in_memory("/ws")
            .with_policy_override(override_handle.clone()),
    );
    a.open_policy_picker();
    let Some(ActiveModal::Policy(picker)) = &mut a.modal else {
        panic!("expected policy picker");
    };
    picker.selected = 2; // deny all
    a.policy_picker_confirm();
    assert!(!matches!(a.modal, Some(ActiveModal::Policy(_))));
    assert_eq!(
        override_handle.current(),
        Some(lofi_types::BashApprovalMode::DenyAll)
    );
}

#[test]
fn policy_badge_hidden_at_default_and_for_a_default_pick() {
    let mut a = app();
    a.executor = Some(lofi_core::agent::ToolExecutor::in_memory("/ws"));
    assert_eq!(a.policy_badge(), None);

    a.open_policy_picker();
    a.policy_picker_confirm();
    assert_eq!(a.policy_badge(), None);
}

#[test]
fn policy_badge_shows_a_non_default_pick() {
    let mut a = app();
    a.executor = Some(lofi_core::agent::ToolExecutor::in_memory("/ws"));
    a.open_policy_picker();
    let Some(ActiveModal::Policy(picker)) = &mut a.modal else {
        panic!("expected policy picker");
    };
    picker.selected = 2; // deny all
    a.policy_picker_confirm();
    assert_eq!(a.policy_badge().as_deref(), Some("deny all"));
}

#[test]
fn policy_picker_requires_an_agent() {
    let mut a = app();
    assert!(a.slash_command("/policy"));
    assert!(!matches!(a.modal, Some(ActiveModal::Policy(_))));
    let (msg, kind) = a.notify_badge().expect("no-agent /policy notified");
    assert_eq!(kind, NotifyKind::Error);
    assert!(msg.contains("/policy"));
}

#[test]
fn service_picker_open_lists_auto_plus_declared() {
    let mut a = app(); // model.label = "openai/gpt-4o", thinking = Medium
    a.model.choices = vec![lofi_types::ModelChoice {
        provider: "openai".into(),
        id: "gpt-4o".into(),
        name: String::new(),
        thinking_levels: vec![],
        service_tiers: vec![
            ServiceTier::Flex,
            ServiceTier::Priority,
            ServiceTier::Flex, // deduped
        ],
        supports_image: false,
        context_window: None,
    }];
    a.open_service_picker();
    let Some(ActiveModal::Service(picker)) = &a.modal else {
        panic!("expected service picker");
    };
    assert_eq!(
        picker.items,
        vec![ServiceTier::Auto, ServiceTier::Flex, ServiceTier::Priority]
    );
    assert_eq!(picker.selected, 0);
    assert!(a.modal_open());
}

#[test]
fn service_picker_uses_selected_model_when_display_label_differs() {
    let mut a = app();
    a.model.selected_label = Some("openai/gpt-4o".into());
    a.model.label = "openai/gpt-4o-2024-11-20".into();
    a.model.choices = vec![lofi_types::ModelChoice {
        provider: "openai".into(),
        id: "gpt-4o".into(),
        name: String::new(),
        thinking_levels: vec![],
        service_tiers: vec![ServiceTier::Flex, ServiceTier::Priority],
        supports_image: false,
        context_window: None,
    }];
    a.open_service_picker();
    let Some(ActiveModal::Service(picker)) = &a.modal else {
        panic!("expected service picker on routed display model");
    };
    assert_eq!(
        picker.items,
        vec![ServiceTier::Auto, ServiceTier::Flex, ServiceTier::Priority]
    );
}

#[test]
fn service_picker_open_no_tiers_notifies() {
    let mut a = app();
    a.model.choices = vec![lofi_types::ModelChoice {
        provider: "openai".into(),
        id: "gpt-4o".into(),
        name: String::new(),
        thinking_levels: vec![],
        service_tiers: vec![],
        supports_image: false,
        context_window: None,
    }];
    a.open_service_picker();
    assert!(!matches!(a.modal, Some(ActiveModal::Service(_))));
    assert!(a.notify.is_some());
}

#[test]
fn service_picker_confirm_sets_pending_switch() {
    let mut a = app(); // model.label = "openai/gpt-4o", thinking = Medium
    a.model.choices = vec![lofi_types::ModelChoice {
        provider: "openai".into(),
        id: "gpt-4o".into(),
        name: String::new(),
        thinking_levels: vec![],
        service_tiers: vec![ServiceTier::Auto, ServiceTier::Flex, ServiceTier::Priority],
        supports_image: false,
        context_window: None,
    }];
    a.open_service_picker();
    let Some(ActiveModal::Service(picker)) = &mut a.modal else {
        panic!("expected service picker");
    };
    picker.selected = 2;
    a.service_picker_confirm();
    assert_eq!(
        a.model.pending_switch.as_deref(),
        Some("openai/gpt-4o:medium@priority")
    );
    assert!(!matches!(a.modal, Some(ActiveModal::Service(_))));

    // Selecting "auto" while already on auto omits the @tier suffix.
    let mut a2 = app();
    a2.model.choices = vec![lofi_types::ModelChoice {
        provider: "openai".into(),
        id: "gpt-4o".into(),
        name: String::new(),
        thinking_levels: vec![],
        service_tiers: vec![ServiceTier::Auto, ServiceTier::Flex],
        supports_image: false,
        context_window: None,
    }];
    a2.open_service_picker();
    let Some(ActiveModal::Service(picker2)) = &mut a2.modal else {
        panic!("expected service picker");
    };
    picker2.selected = 0;
    a2.service_picker_confirm();
    assert_eq!(
        a2.model.pending_switch.as_deref(),
        Some("openai/gpt-4o:medium")
    );
}

#[test]
fn service_picker_confirm_switching_back_to_auto_emits_explicit_auto() {
    let mut a = App::new(
        "openai/gpt-4o".to_string(),
        ThinkingLevel::Medium,
        ServiceTier::Flex,
        0,
        lofi_types::CompactionConfig::default(),
        String::new(),
    );
    a.model.choices = vec![lofi_types::ModelChoice {
        provider: "openai".into(),
        id: "gpt-4o".into(),
        name: String::new(),
        thinking_levels: vec![],
        service_tiers: vec![ServiceTier::Auto, ServiceTier::Flex],
        supports_image: false,
        context_window: None,
    }];
    a.open_service_picker();
    let Some(ActiveModal::Service(picker)) = &mut a.modal else {
        panic!("expected service picker");
    };
    picker.selected = 0; // auto
    a.service_picker_confirm();
    assert_eq!(
        a.model.pending_switch.as_deref(),
        Some("openai/gpt-4o:medium@auto")
    );
}

#[test]
fn footer_shows_service_tier_when_not_auto() {
    let a = App::new(
        "openai/gpt-4o".to_string(),
        ThinkingLevel::High,
        ServiceTier::Flex,
        0,
        lofi_types::CompactionConfig::default(),
        String::new(),
    );
    let r: String = a
        .render_footer_right()
        .spans
        .iter()
        .map(|s| s.content.as_ref().to_string())
        .collect();
    assert!(r.contains("openai/gpt-4o:high@flex"));

    let b = App::new(
        "openai/gpt-4o".to_string(),
        ThinkingLevel::High,
        ServiceTier::Auto,
        0,
        lofi_types::CompactionConfig::default(),
        String::new(),
    );
    let r2: String = b
        .render_footer_right()
        .spans
        .iter()
        .map(|s| s.content.as_ref().to_string())
        .collect();
    assert!(r2.contains("openai/gpt-4o:high"));
    assert!(!r2.contains('@'));
}

#[test]
fn resume_model_switch_when_model_differs() {
    let mut a = app(); // model.label = "openai/gpt-4o", thinking = Medium
    a.model.choices = vec![lofi_types::ModelChoice {
        provider: "anthropic".into(),
        id: "claude".into(),
        name: String::new(),
        thinking_levels: vec![],
        service_tiers: vec![],
        supports_image: false,
        context_window: None,
    }];
    let events = vec![SessionEvent {
        id: "1".into(),
        parent_id: None,
        kind: SessionEventKind::TurnEnd {
            model: "anthropic/claude:high".into(),
            elapsed_ms: 0,
            cost: 0.0,
            usage: Usage::default(),
            stop_reason: None,
        },
    }];
    assert_eq!(
        a.resume_model_switch(&events).as_deref(),
        Some("anthropic/claude:high")
    );
}

#[test]
fn resume_model_switch_preserves_service_tier() {
    let mut a = app();
    a.model.choices = vec![lofi_types::ModelChoice {
        provider: "openai".into(),
        id: "gpt-4o".into(),
        name: String::new(),
        thinking_levels: vec![ThinkingLevel::Medium],
        service_tiers: vec![ServiceTier::Auto, ServiceTier::Flex],
        supports_image: false,
        context_window: None,
    }];
    let flex_events = vec![SessionEvent {
        id: "1".into(),
        parent_id: None,
        kind: SessionEventKind::TurnEnd {
            model: "openai/gpt-4o:medium@flex".into(),
            elapsed_ms: 0,
            cost: 0.0,
            usage: Usage::default(),
            stop_reason: None,
        },
    }];
    assert_eq!(
        a.resume_model_switch(&flex_events).as_deref(),
        Some("openai/gpt-4o:medium@flex")
    );

    // Resuming an auto-tier session from a flex-tier app emits explicit @auto.
    a.model.service_tier = ServiceTier::Flex;
    let auto_events = vec![SessionEvent {
        id: "2".into(),
        parent_id: None,
        kind: SessionEventKind::TurnEnd {
            model: "openai/gpt-4o:medium".into(),
            elapsed_ms: 0,
            cost: 0.0,
            usage: Usage::default(),
            stop_reason: None,
        },
    }];
    assert_eq!(
        a.resume_model_switch(&auto_events).as_deref(),
        Some("openai/gpt-4o:medium@auto")
    );
}

#[test]
fn resume_model_switch_none_when_same_model() {
    let mut a = app();
    a.model.choices = vec![lofi_types::ModelChoice {
        provider: "openai".into(),
        id: "gpt-4o".into(),
        name: String::new(),
        thinking_levels: vec![],
        service_tiers: vec![],
        supports_image: false,
        context_window: None,
    }];
    let events = vec![SessionEvent {
        id: "1".into(),
        parent_id: None,
        kind: SessionEventKind::TurnEnd {
            model: "openai/gpt-4o:medium".into(),
            elapsed_ms: 0,
            cost: 0.0,
            usage: Usage::default(),
            stop_reason: None,
        },
    }];
    assert!(a.resume_model_switch(&events).is_none());
}

#[test]
fn resume_model_switch_none_when_no_choices_or_no_turn() {
    let mut a = app();
    a.model.choices = Vec::new();
    let events = vec![SessionEvent {
        id: "1".into(),
        parent_id: None,
        kind: SessionEventKind::TurnEnd {
            model: "anthropic/claude:high".into(),
            elapsed_ms: 0,
            cost: 0.0,
            usage: Usage::default(),
            stop_reason: None,
        },
    }];
    assert!(a.resume_model_switch(&events).is_none());
    a.model.choices = vec![lofi_types::ModelChoice {
        provider: "x".into(),
        id: "y".into(),
        name: String::new(),
        thinking_levels: vec![],
        service_tiers: vec![],
        supports_image: false,
        context_window: None,
    }];
    let no_turn = vec![SessionEvent {
        id: "1".into(),
        parent_id: None,
        kind: SessionEventKind::Message(Message {
            origin: None,
            role: Role::User,
            blocks: vec![ContentBlock::Text { text: "hi".into() }],
            kind: PromptKind::default(),
        }),
    }];
    assert!(a.resume_model_switch(&no_turn).is_none());
}
