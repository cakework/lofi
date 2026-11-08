use lofi_types::{CommandEntry, MatchMode, WrapperKind, WrapperRuleConfig};

use super::engine::ResolvedPolicy;
use super::extract::build_wrapper_map;

fn standard_wrappers() -> Vec<WrapperRuleConfig> {
    vec![
        WrapperRuleConfig {
            name: "bash".into(),
            kind: WrapperKind::ShellC,
        },
        WrapperRuleConfig {
            name: "sh".into(),
            kind: WrapperKind::ShellC,
        },
        WrapperRuleConfig {
            name: "zsh".into(),
            kind: WrapperKind::ShellC,
        },
        WrapperRuleConfig {
            name: "dash".into(),
            kind: WrapperKind::ShellC,
        },
        WrapperRuleConfig {
            name: "ksh".into(),
            kind: WrapperKind::ShellC,
        },
        WrapperRuleConfig {
            name: "sudo".into(),
            kind: WrapperKind::UtilityOperand,
        },
        WrapperRuleConfig {
            name: "doas".into(),
            kind: WrapperKind::UtilityOperand,
        },
        WrapperRuleConfig {
            name: "time".into(),
            kind: WrapperKind::UtilityOperand,
        },
        WrapperRuleConfig {
            name: "nohup".into(),
            kind: WrapperKind::UtilityOperand,
        },
        WrapperRuleConfig {
            name: "nice".into(),
            kind: WrapperKind::UtilityOperand,
        },
        WrapperRuleConfig {
            name: "chroot".into(),
            kind: WrapperKind::UtilityOperand,
        },
        WrapperRuleConfig {
            name: "timeout".into(),
            kind: WrapperKind::UtilityOperand,
        },
        WrapperRuleConfig {
            name: "setsid".into(),
            kind: WrapperKind::UtilityOperand,
        },
        WrapperRuleConfig {
            name: "command".into(),
            kind: WrapperKind::UtilityOperand,
        },
        WrapperRuleConfig {
            name: "env".into(),
            kind: WrapperKind::Env,
        },
        WrapperRuleConfig {
            name: "xargs".into(),
            kind: WrapperKind::Xargs,
        },
        WrapperRuleConfig {
            name: "docker".into(),
            kind: WrapperKind::DockerRun,
        },
        WrapperRuleConfig {
            name: "podman".into(),
            kind: WrapperKind::DockerRun,
        },
    ]
}

fn universal_deny() -> Vec<CommandEntry> {
    vec![
        entry("sudo", MatchMode::Prefix),
        entry("doas", MatchMode::Prefix),
        entry("kill", MatchMode::Prefix),
        entry("killall", MatchMode::Prefix),
        entry("pkill", MatchMode::Prefix),
        entry("find /", MatchMode::Prefix),
        entry("find /nix", MatchMode::Prefix),
        entry("find /home", MatchMode::Prefix),
        entry("find /etc", MatchMode::Prefix),
        entry("find /var", MatchMode::Prefix),
        entry("find /usr", MatchMode::Prefix),
        entry("find /proc", MatchMode::Prefix),
        entry("find /sys", MatchMode::Prefix),
    ]
}

fn entry(match_str: &str, mode: MatchMode) -> CommandEntry {
    CommandEntry {
        match_str: match_str.into(),
        mode,
    }
}

/// The only built-in rules are the universal denials. A command that no
/// `[[allow]]` rule matches fails closed to `ask`.
#[must_use]
pub fn resolve(config: &lofi_types::ShellPolicyConfig) -> ResolvedPolicy {
    let mut deny = universal_deny();
    deny.extend(config.deny.clone());

    let mut wrappers = standard_wrappers();
    wrappers.extend(config.wrappers.clone());

    ResolvedPolicy {
        allow: config.allow.clone(),
        ask: config.ask.clone(),
        deny,
        wrappers: build_wrapper_map(&wrappers),
        redirects: config.redirects.clone(),
        heredocs: config.heredocs.clone(),
        yolo: config.yolo,
    }
}
