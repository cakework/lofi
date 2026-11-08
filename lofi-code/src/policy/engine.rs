use lofi_types::{CommandEntry, HeredocPolicy, MatchMode, PolicyAction, RedirectPolicy};

use super::extract::{extract_commands, CommandSource, ExtractedCommand, WrapperRuleMap};
use super::token::{tokenize, Token};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    pub action: PolicyAction,
    pub reason: String,
    pub matched_command: Option<String>,
    /// An `[[ask]]` rule from `policy.toml` matched. Such a rule reserves
    /// the decision for the user, so the tool layer prompts and never
    /// consults auto mode. A redirect or heredoc action, an unparseable
    /// command, and a fail-closed miss leave it false.
    pub ask_rule: bool,
    /// The steering suggestion folded into `reason` after `"; "`. Only the
    /// backgrounding deny sets it; callers drop it when steering does not
    /// apply (`jobSpawn`, and the bash tool for `steer: false`).
    pub steer: Option<String>,
}

impl Decision {
    /// The reason with the steering suggestion dropped.
    #[must_use]
    pub fn reason_without_steering(&self) -> &str {
        self.steer
            .as_deref()
            .and_then(|h| self.reason.strip_suffix(h))
            .and_then(|r| r.strip_suffix("; "))
            .unwrap_or(&self.reason)
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedPolicy {
    pub allow: Vec<CommandEntry>,
    /// The `[[ask]]` rules from `policy.toml`. A match marks the decision
    /// `ask_rule`, which keeps auto mode out of it.
    pub ask: Vec<CommandEntry>,
    pub deny: Vec<CommandEntry>,
    pub wrappers: WrapperRuleMap,
    pub redirects: RedirectPolicy,
    pub heredocs: HeredocPolicy,
    pub yolo: bool,
}

impl ResolvedPolicy {
    #[must_use]
    pub fn evaluate(&self, command: &str) -> Decision {
        let tokens = match tokenize(command) {
            Ok(t) => t,
            Err(e) => {
                return Decision {
                    action: PolicyAction::Ask,
                    reason: format!("unparseable: {e}"),
                    matched_command: Some(command.to_string()),
                    ask_rule: false,
                    steer: None,
                };
            }
        };

        // Hard-coded (not a per-rule deny): no opt-out and no config knob
        // turn this off, because a detached child outlives the Bash call and
        // the model never sees its output.
        if let Some((base, suggestion)) = detect_backgrounding(&tokens) {
            return Decision {
                action: PolicyAction::Deny,
                reason: format!("{base}; {suggestion}"),
                matched_command: Some(command.to_string()),
                ask_rule: false,
                steer: Some(suggestion),
            };
        }

        let cmds = extract_commands(&tokens, CommandSource::Direct, &self.wrappers);
        if cmds.is_empty() {
            return Decision {
                action: PolicyAction::Ask,
                reason: "empty or unrecognized command".into(),
                matched_command: Some(command.to_string()),
                ask_rule: false,
                steer: None,
            };
        }

        let cmd_decision = self.evaluate_commands(&cmds);
        if cmd_decision.action == PolicyAction::Deny {
            return cmd_decision;
        }

        let redir_decision = self.evaluate_redirects(&cmds);
        if redir_decision.action == PolicyAction::Deny {
            return redir_decision;
        }

        let here_decision = self.evaluate_heredocs(&cmds);
        if here_decision.action == PolicyAction::Deny {
            return here_decision;
        }

        let winner = [&cmd_decision, &redir_decision, &here_decision]
            .into_iter()
            .max_by_key(|d| action_rank(d.action))
            .unwrap_or(&cmd_decision);

        // `yolo` skips an explicit ask rule too: it is a session-wide
        // choice to run without prompts, and it still never bypasses a deny.
        let action = if winner.action == PolicyAction::Allow
            || (self.yolo && winner.action != PolicyAction::Deny)
        {
            PolicyAction::Allow
        } else {
            PolicyAction::Ask
        };

        // A rule that reserves the answer names itself, even when a redirect
        // or heredoc asks as well: the user answers the rule, not the
        // redirect, and a tie would otherwise hand the decision to the last
        // source in the merge order.
        let named = if action == PolicyAction::Ask && cmd_decision.ask_rule {
            &cmd_decision
        } else {
            winner
        };

        Decision {
            action,
            reason: if action == PolicyAction::Ask && named.reason.is_empty() {
                "no policy match".into()
            } else {
                named.reason.clone()
            },
            matched_command: named.matched_command.clone(),
            // The rule that matched the command itself decides whether the
            // user answers this prompt, whatever source merged the action.
            ask_rule: cmd_decision.ask_rule,
            // Only the backgrounding deny (an early return above) carries a
            // steering suggestion, so a merged winner never does.
            steer: None,
        }
    }

    fn evaluate_commands(&self, cmds: &[ExtractedCommand]) -> Decision {
        let mut result = PolicyAction::Allow;
        let mut reason = String::new();
        let mut matched = None;
        let mut explicit = false;
        let mut saw_direct_unmatched = false;

        for cmd in cmds {
            let deny_match = self.deny.iter().find(|e| match_entry(cmd, e));
            if deny_match.is_some() {
                return Decision {
                    action: PolicyAction::Deny,
                    reason: format!("denied: {}", cmd.name),
                    matched_command: Some(cmd.full_text.clone()),
                    ask_rule: false,
                    steer: None,
                };
            }

            // A configured rule reserves the answer for the user: auto
            // mode never answers it.
            let rule_match = self.ask.iter().find(|e| match_entry(cmd, e));
            if let Some(rule) = rule_match {
                result = PolicyAction::Ask;
                reason = format!(
                    "requires confirmation by a configured rule: {}",
                    rule.match_str
                );
                matched = Some(cmd.full_text.clone());
                explicit = true;
            }

            let allow_match = self.allow.iter().find(|e| match_entry(cmd, e));
            if allow_match.is_some() {
                if result == PolicyAction::Allow
                    && (cmd.source != CommandSource::Direct || !saw_direct_unmatched)
                {
                    reason = format!("allowed: {}", cmd.name);
                    matched = Some(cmd.full_text.clone());
                }
            } else if rule_match.is_none() {
                if cmd.source == CommandSource::Direct {
                    saw_direct_unmatched = true;
                }
                // A wrapper exists to expose its inner command for matching.
                // Once the inner command has been extracted, the wrapper's own
                // name is not another unmatched command that can veto it.
                let is_wrapper = cmd.source == CommandSource::Direct
                    && self.wrappers.contains_key(&cmd.name.to_ascii_lowercase());
                if result == PolicyAction::Allow && !is_wrapper {
                    result = PolicyAction::Ask;
                    reason = format!("no policy match: {}", cmd.name);
                    matched = Some(cmd.full_text.clone());
                }
            }
        }

        Decision {
            action: result,
            reason,
            matched_command: matched,
            ask_rule: explicit,
            steer: None,
        }
    }

    fn evaluate_redirects(&self, cmds: &[ExtractedCommand]) -> Decision {
        for cmd in cmds {
            for (op, target) in &cmd.redirects {
                let base = op.trim_start_matches(|c: char| c.is_ascii_digit());
                if matches!(base, "<" | "<<" | "<<-" | "<<<") {
                    continue;
                }
                if op.ends_with('&') && self.redirects.allow_fd_dup {
                    continue;
                }
                if self.redirects.safe_targets.iter().any(|t| t == target) {
                    continue;
                }
                if self.redirects.action != PolicyAction::Allow {
                    return Decision {
                        action: self.redirects.action,
                        reason: format!("redirect {op} {target}"),
                        matched_command: Some(cmd.full_text.clone()),
                        ask_rule: false,
                        steer: None,
                    };
                }
            }
        }
        Decision {
            action: PolicyAction::Allow,
            reason: String::new(),
            matched_command: None,
            ask_rule: false,
            steer: None,
        }
    }

    fn evaluate_heredocs(&self, cmds: &[ExtractedCommand]) -> Decision {
        for cmd in cmds {
            if cmd
                .redirects
                .iter()
                .any(|(op, _)| op == "<<" || op == "<<-")
                && self.heredocs.action != PolicyAction::Allow
            {
                return Decision {
                    action: self.heredocs.action,
                    reason: "heredoc detected".into(),
                    matched_command: Some(cmd.full_text.clone()),
                    ask_rule: false,
                    steer: None,
                };
            }
        }
        Decision {
            action: PolicyAction::Allow,
            reason: String::new(),
            matched_command: None,
            ask_rule: false,
            steer: None,
        }
    }
}

/// Detect a shell idiom that detaches work from the Bash call. Returns the
/// deny reason and its steering suggestion on match, None when the input is
/// clean.
///
/// Recurses into subshells and command substitution so "sh -c 'sleep 1 &'"
/// cannot sneak past the top-level scan.
fn detect_backgrounding(tokens: &[Token]) -> Option<(String, String)> {
    const DETACH_WORDS: &[&str] = &["nohup", "setsid", "disown"];
    const JOBSPAWN: &str = "use lofi.jobSpawn for background jobs";

    let mut start_of_segment = true;
    for tok in tokens {
        match tok {
            Token::Operator(op) if op == "&" => {
                return Some(("bare & is not allowed".into(), JOBSPAWN.into()));
            }
            Token::Operator(_) => {
                start_of_segment = true;
            }
            Token::Word(w) => {
                if start_of_segment && DETACH_WORDS.contains(&w.as_str()) {
                    return Some((format!("{w} is not allowed"), JOBSPAWN.into()));
                }
                start_of_segment = false;
            }
            Token::Group { tokens, .. } => {
                if let Some(found) = detect_backgrounding(tokens) {
                    return Some(found);
                }
            }
            Token::Redirect { .. } => {}
        }
    }
    None
}

fn action_rank(a: PolicyAction) -> u8 {
    match a {
        PolicyAction::Allow => 1,
        PolicyAction::Ask => 2,
        PolicyAction::Deny => 3,
    }
}

fn match_entry(cmd: &ExtractedCommand, entry: &CommandEntry) -> bool {
    match entry.mode {
        MatchMode::Exact => cmd.full_text.trim().eq_ignore_ascii_case(&entry.match_str),
        MatchMode::Prefix => {
            let lower = cmd.full_text.trim_start().to_ascii_lowercase();
            let m = entry.match_str.to_ascii_lowercase();
            lower == m || lower.starts_with(&format!("{m} "))
        }
        MatchMode::Substring => {
            let match_tokens: Vec<&str> = entry.match_str.split_whitespace().collect();
            if match_tokens.is_empty() || match_tokens.len() > cmd.words.len() {
                return false;
            }
            let cmd_lower: Vec<String> = cmd.words.iter().map(|w| w.to_ascii_lowercase()).collect();
            let match_lower: Vec<String> = match_tokens
                .iter()
                .map(|t| t.to_ascii_lowercase())
                .collect();
            (0..=cmd_lower.len().saturating_sub(match_lower.len())).any(|i| {
                cmd_lower[i..i + match_lower.len()]
                    .iter()
                    .zip(&match_lower)
                    .all(|(a, b)| a == b)
            })
        }
        MatchMode::Args => {
            let (prefix, required) = parse_args_pattern(&entry.match_str);
            if prefix != "*"
                && !cmd
                    .full_text
                    .trim_start()
                    .to_ascii_lowercase()
                    .starts_with(&prefix.to_ascii_lowercase())
            {
                return false;
            }
            let prefix_words = if prefix == "*" {
                0
            } else {
                prefix.split_whitespace().count()
            };
            let cmd_args: Vec<String> = cmd.words[prefix_words..]
                .iter()
                .map(|w| w.to_ascii_lowercase())
                .collect();
            required.iter().all(|r| cmd_args.iter().any(|a| a == r))
        }
    }
}

fn parse_args_pattern(pattern: &str) -> (String, Vec<String>) {
    match pattern.find(':') {
        None => (pattern.to_string(), Vec::new()),
        Some(idx) => {
            let prefix = pattern[..idx].to_string();
            let required: Vec<String> = pattern[idx + 1..]
                .split_whitespace()
                .map(str::to_ascii_lowercase)
                .collect();
            (prefix, required)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::super::defaults;
    use super::*;
    use lofi_types::*;

    fn default_policy() -> ResolvedPolicy {
        defaults::resolve(&ShellPolicyConfig::default())
    }

    /// A config that allows the listed commands by prefix.
    fn allowing(entries: &[&str]) -> ShellPolicyConfig {
        ShellPolicyConfig {
            allow: entries
                .iter()
                .map(|match_str| CommandEntry {
                    match_str: (*match_str).into(),
                    mode: MatchMode::Prefix,
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn the_default_policy_asks_for_every_non_denied_command() {
        let p = default_policy();
        for command in [
            "ls",
            "cargo build",
            "cat ~/.ssh/id_rsa",
            "curl https://example.com",
            "rm file.txt",
            "git commit -m test",
            "totally-unknown-binary --flag",
        ] {
            assert_eq!(p.evaluate(command).action, PolicyAction::Ask, "{command}");
        }
        assert_eq!(p.evaluate("sudo ls").action, PolicyAction::Deny);
    }

    #[test]
    fn a_redirect_asks_even_when_the_command_is_allowed() {
        let p = defaults::resolve(&allowing(&["echo"]));
        let d = p.evaluate("echo x > /tmp/outside-workspace");
        assert_eq!(d.action, PolicyAction::Ask);
        // A redirect asks on its own, without a rule reserving the answer.
        assert!(!d.ask_rule);
    }

    #[test]
    fn network_clients_are_not_builtin_allows() {
        let p = default_policy();
        assert_eq!(
            p.evaluate("curl https://example.com").action,
            PolicyAction::Ask
        );
        assert_eq!(
            p.evaluate("wget https://example.com").action,
            PolicyAction::Ask
        );
    }

    #[test]
    fn an_allow_rule_allows_the_commands_it_names() {
        let p = defaults::resolve(&allowing(&["ls", "cargo"]));
        assert_eq!(p.evaluate("ls -la").action, PolicyAction::Allow);
        assert_eq!(p.evaluate("cargo build").action, PolicyAction::Allow);
    }

    #[test]
    fn deny_sudo() {
        let p = default_policy();
        let d = p.evaluate("sudo rm -rf /");
        assert_eq!(d.action, PolicyAction::Deny);
    }

    #[test]
    fn an_ask_rule_is_reserved() {
        let p = defaults::resolve(&ShellPolicyConfig {
            ask: vec![CommandEntry {
                match_str: "git push".into(),
                mode: MatchMode::Prefix,
            }],
            ..Default::default()
        });
        let d = p.evaluate("git push origin main");
        assert_eq!(d.action, PolicyAction::Ask);
        assert!(d.ask_rule);
        assert_eq!(
            d.reason,
            "requires confirmation by a configured rule: git push"
        );
    }

    #[test]
    fn a_fail_closed_ask_is_not_reserved() {
        let p = default_policy();
        let d = p.evaluate("special-cmd");
        assert_eq!(d.action, PolicyAction::Ask);
        assert!(!d.ask_rule);
    }

    #[test]
    fn deny_bare_ampersand_backgrounds() {
        let p = default_policy();
        let d = p.evaluate("sleep 30 &");
        assert_eq!(d.action, PolicyAction::Deny);
        assert!(d.reason.contains("jobSpawn"), "reason: {}", d.reason);
        // The suggestion is steering: the full reason carries it and
        // reason_without_steering drops only that part.
        assert_eq!(
            d.steer.as_deref(),
            Some("use lofi.jobSpawn for background jobs")
        );
        let plain = d.reason_without_steering();
        assert!(plain.contains("bare & is not allowed"), "reason: {plain}");
        assert!(!plain.contains("jobSpawn"), "reason: {plain}");
    }

    #[test]
    fn deny_ampersand_after_otherwise_allowed_command() {
        let p = defaults::resolve(&allowing(&["cargo"]));
        // Even when every segment is allowed, the bare "&" is a hard deny.
        let d = p.evaluate("cargo build & cargo test");
        assert_eq!(d.action, PolicyAction::Deny);
    }

    #[test]
    fn deny_nohup_setsid_disown() {
        let p = default_policy();
        for cmd in ["nohup sleep 5", "setsid cargo watch", "disown %1"] {
            let d = p.evaluate(cmd);
            assert_eq!(d.action, PolicyAction::Deny, "cmd: {cmd}");
            // The suggestion rides in `reason` and drops out of
            // `reason_without_steering`.
            assert!(d.reason.contains("jobSpawn"), "cmd: {cmd}");
            assert!(
                !d.reason_without_steering().contains("jobSpawn"),
                "cmd: {cmd}"
            );
        }
    }

    #[test]
    fn deny_backgrounding_inside_subshell_and_substitution() {
        let p = default_policy();
        for cmd in ["(sleep 5 &)", "echo $(sleep 5 &)", "echo `sleep 5 &`"] {
            let d = p.evaluate(cmd);
            assert_eq!(d.action, PolicyAction::Deny, "cmd: {cmd}");
        }
    }

    #[test]
    fn allow_double_ampersand_and_pipeline() {
        let p = defaults::resolve(&allowing(&["cargo", "tail"]));
        // "&&" and "|" are sequencing, not backgrounding.
        let d = p.evaluate("cargo build && cargo test");
        assert_eq!(d.action, PolicyAction::Allow);
        let d = p.evaluate("cargo build | tail -3");
        assert_eq!(d.action, PolicyAction::Allow);
    }

    #[test]
    fn deny_find_root() {
        let p = default_policy();
        let d = p.evaluate("find / -name foo");
        assert_eq!(d.action, PolicyAction::Deny);
    }

    fn yolo_policy(config: ShellPolicyConfig) -> ResolvedPolicy {
        defaults::resolve(&ShellPolicyConfig {
            yolo: true,
            ..config
        })
    }

    #[test]
    fn yolo_allows_a_fail_closed_ask() {
        let p = yolo_policy(ShellPolicyConfig::default());
        let d = p.evaluate("rm file.txt");
        assert_eq!(d.action, PolicyAction::Allow);
    }

    #[test]
    fn yolo_still_denies_sudo() {
        let p = yolo_policy(ShellPolicyConfig::default());
        let d = p.evaluate("sudo rm -rf /");
        assert_eq!(d.action, PolicyAction::Deny);
    }

    #[test]
    fn pipeline_both_checked() {
        // `ls` alone is allowed, so only the second segment can ask.
        let p = defaults::resolve(&allowing(&["ls"]));
        let d = p.evaluate("ls | rm -rf /tmp");
        assert_eq!(d.action, PolicyAction::Ask);
    }

    #[test]
    fn pipeline_deny_dominates() {
        let p = default_policy();
        let d = p.evaluate("ls | sudo rm /etc/passwd");
        assert_eq!(d.action, PolicyAction::Deny);
    }

    #[test]
    fn wrapper_bash_c_unwrapped() {
        let p = default_policy();
        let d = p.evaluate("bash -c 'sudo rm -rf /'");
        assert_eq!(d.action, PolicyAction::Deny);
    }

    #[test]
    fn wrapper_sudo_unwrapped() {
        let p = default_policy();
        let d = p.evaluate("sudo ls");
        assert_eq!(d.action, PolicyAction::Deny);
    }

    #[test]
    fn subshell_checked() {
        let p = default_policy();
        let d = p.evaluate("(sudo rm /etc/passwd)");
        assert_eq!(d.action, PolicyAction::Deny);
    }

    #[test]
    fn cmd_substitution_checked() {
        let p = default_policy();
        let d = p.evaluate("echo $(sudo ls)");
        assert_eq!(d.action, PolicyAction::Deny);
    }

    #[test]
    fn unparseable_fails_closed() {
        let p = default_policy();
        let d = p.evaluate("echo 'unclosed");
        assert_eq!(d.action, PolicyAction::Ask);
    }

    #[test]
    fn unmatched_fails_closed() {
        let p = default_policy();
        let d = p.evaluate("totally-unknown-binary --flag");
        assert_eq!(d.action, PolicyAction::Ask);
    }

    #[test]
    fn custom_allow_rule() {
        let p = defaults::resolve(&allowing(&["my-tool"]));
        let d = p.evaluate("my-tool --check");
        assert_eq!(d.action, PolicyAction::Allow);
    }

    #[test]
    fn a_deny_rule_beats_an_allow_rule() {
        let p = defaults::resolve(&ShellPolicyConfig {
            deny: vec![CommandEntry {
                match_str: "cargo".into(),
                mode: MatchMode::Prefix,
            }],
            ..allowing(&["cargo"])
        });
        let d = p.evaluate("cargo build");
        assert_eq!(d.action, PolicyAction::Deny);
    }

    #[test]
    fn an_ask_rule_names_itself_over_a_redirect() {
        let p = defaults::resolve(&ShellPolicyConfig {
            ask: vec![CommandEntry {
                match_str: "printf".into(),
                mode: MatchMode::Prefix,
            }],
            ..allowing(&["printf"])
        });
        let d = p.evaluate("printf x > out.txt");
        assert_eq!(d.action, PolicyAction::Ask);
        assert!(d.ask_rule);
        assert_eq!(
            d.reason,
            "requires confirmation by a configured rule: printf"
        );
    }

    #[test]
    fn args_mode_match() {
        let p = defaults::resolve(&ShellPolicyConfig {
            ask: vec![CommandEntry {
                match_str: "*:-X POST".into(),
                mode: MatchMode::Args,
            }],
            ..allowing(&["curl"])
        });
        let d = p.evaluate("curl -X POST http://example.com");
        assert_eq!(d.action, PolicyAction::Ask);
        assert!(d.ask_rule);
        assert_eq!(
            d.reason,
            "requires confirmation by a configured rule: *:-X POST"
        );
    }

    #[test]
    fn heredoc_ask_by_default() {
        let p = defaults::resolve(&allowing(&["cat"]));
        let d = p.evaluate("cat <<EOF\nhello\nEOF");
        assert_eq!(d.action, PolicyAction::Ask);
    }

    #[test]
    fn env_wrapper_unwrapped() {
        let p = default_policy();
        let d = p.evaluate("env FOO=bar sudo rm /etc");
        assert_eq!(d.action, PolicyAction::Deny);
    }

    #[test]
    fn process_substitution_checked() {
        let p = default_policy();
        let d = p.evaluate("diff <(sudo rm /etc/passwd) <(ls)");
        assert_eq!(d.action, PolicyAction::Deny);
    }

    #[test]
    fn quoted_cmd_substitution_checked() {
        let p = default_policy();
        let d = p.evaluate("echo \"$(sudo rm -rf /)\"");
        assert_eq!(d.action, PolicyAction::Deny);
    }
}
