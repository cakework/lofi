use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::process::Command;

use lofi_error::{Error, Result};
use lofi_types::ThinkingLevel;

use super::contract::{Completion, CompletionRequest};
use super::rpc;

/// Ceiling on the one-shot exchange. A healthy evaluator answers in seconds,
/// but the caller gates a tool call on it: a child that hangs on auth or the
/// network must fail the call instead of blocking the turn that waits for the
/// approval.
const EVALUATOR_TIMEOUT: Duration = Duration::from_mins(2);

/// Answer one prompt with text only.
///
/// The child runs without the exec bridge, without tools and without a saved
/// session: it cannot touch the workspace, it shares nothing with a running
/// conversation, and it dies with the call. That is what makes the shell
/// evaluator safe to run on the same backend as the turn it is judging.
pub(super) async fn complete(
    command: &str,
    root: &Path,
    model: Option<&str>,
    effort: &ThinkingLevel,
    request: &CompletionRequest,
) -> Result<Completion> {
    if command.trim().is_empty() {
        return Err(Error::Config("empty Claude Code command".into()));
    }
    if request.system.trim().is_empty() {
        return Err(Error::Config("empty Claude Code system prompt".into()));
    }
    let dir = crate::state::create_session_tmp_dir(root)?;
    dir.ensure()?;
    let prompt_file = dir.path().join("system-prompt.md");
    std::fs::write(&prompt_file, &request.system)?;
    crate::state::ensure_private_file(&prompt_file)?;
    let id = uuid::Uuid::new_v4().to_string();
    let mut process = Command::new(command);
    process.args([
        "--print",
        "--verbose",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--tools",
        "",
        "--permission-mode",
        "dontAsk",
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--settings",
        r#"{"disableAllHooks":true}"#,
        "--disable-slash-commands",
        "--no-chrome",
    ]);
    process
        .arg("--system-prompt-file")
        .arg(&prompt_file)
        .arg(format!("--session-id={id}"));
    if let Some(flag) = super::claude::effort_flag(effort) {
        process.arg(flag);
    }
    if let Some(model) = model {
        process.arg(format!("--model={model}"));
    }
    // The CLI has no flag for the output cap, only the environment variable;
    // without it a caller's budget would be silently unbounded here while the
    // provider path honors it.
    if let Some(max) = request.max_tokens {
        process.env("CLAUDE_CODE_MAX_OUTPUT_TOKENS", max.to_string());
    }
    let mut child = process
        .env("ENABLE_TOOL_SEARCH", "false")
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .kill_on_drop(true)
        .spawn()?;
    // Guard the whole process group so timeouts and dropped futures terminate
    // descendants spawned by the CLI, not only the direct child PID.
    let group = lofi_code::tools::PgrpKillGuard::new(child.id());
    let mut input = child
        .stdin
        .take()
        .ok_or_else(|| Error::State("missing Claude Code stdin".into()))?;
    let mut output = BufReader::new(
        child
            .stdout
            .take()
            .ok_or_else(|| Error::State("missing Claude Code stdout".into()))?,
    );
    let mut stderr = rpc::StderrTail::spawn(
        child
            .stderr
            .take()
            .ok_or_else(|| Error::State("missing Claude Code stderr".into()))?,
    );
    // The ceiling covers the whole exchange: a child that hangs before it
    // reads stdin blocks the write just as surely as one that never answers.
    let exchange = async {
        rpc::write(
            &mut input,
            &json!({"type":"user","session_id":id,
                "parent_tool_use_id":null,"message":{"role":"user","content":request.prompt}}),
        )
        .await?;
        let _ = input.flush().await;
        collect(&mut output).await
    };
    let text = match tokio::time::timeout(EVALUATOR_TIMEOUT, exchange).await {
        Ok(text) => text,
        Err(_) => Err(Error::State("Claude Code evaluator timed out".into())),
    };
    drop(group);
    let _ = child.wait().await;
    match text {
        Ok(completion) => Ok(completion),
        Err(error) => {
            let tail = stderr.finish().await;
            Err(rpc::with_stderr(error, &tail))
        }
    }
}

async fn collect(output: &mut BufReader<tokio::process::ChildStdout>) -> Result<Completion> {
    let mut text = String::new();
    loop {
        let message = rpc::read_json(output).await?;
        match message["type"].as_str() {
            Some("system") if message["subtype"] == "init" => validate_init(&message)?,
            Some("assistant") => {
                let content = &message["message"]["content"];
                if let Some(blocks) = content.as_array() {
                    for block in blocks {
                        if block["type"] == "text" {
                            if let Some(delta) = block["text"].as_str() {
                                text.push_str(delta);
                            }
                        }
                        // A tool-free child cannot answer by calling a tool. A
                        // tool request means the isolation flags did not hold.
                        if block["type"] == "tool_use" {
                            return Err(Error::State(
                                "the evaluator requested a tool; it must run without tools".into(),
                            ));
                        }
                    }
                }
            }
            Some("result") => {
                if message["is_error"] == true {
                    let detail = message["result"]
                        .as_str()
                        .unwrap_or("Claude Code reported an error");
                    return Err(Error::State(detail.to_string()));
                }
                if text.trim().is_empty() {
                    if let Some(result) = message["result"].as_str() {
                        text.push_str(result);
                    }
                }
                return Ok(Completion {
                    text,
                    usage: usage_of(&message["usage"]),
                    cost: cost_of(&message),
                });
            }
            _ => {}
        }
    }
}

fn validate_init(message: &Value) -> Result<()> {
    if message["tools"] != json!([]) {
        return Err(Error::State(format!(
            "Claude Code evaluator must expose no tools, but reported {}",
            super::claude::observed(&message["tools"]),
        )));
    }
    if message["mcp_servers"] != json!([]) {
        return Err(Error::State(format!(
            "Claude Code evaluator must connect to no MCP servers, but reported {}",
            super::claude::observed(&message["mcp_servers"]),
        )));
    }
    Ok(())
}

fn usage_of(value: &Value) -> lofi_types::Usage {
    let number = |key: &str| value[key].as_u64().unwrap_or(0);
    lofi_types::Usage {
        input_tokens: number("input_tokens"),
        output_tokens: number("output_tokens"),
        cache_read_tokens: number("cache_read_input_tokens"),
        cache_write_tokens: number("cache_creation_input_tokens"),
    }
}

fn cost_of(message: &Value) -> Option<f64> {
    message["total_cost_usd"]
        .as_f64()
        .filter(|cost| cost.is_finite() && *cost >= 0.0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    fn write_script(dir: &Path, name: &str, body: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        path
    }

    fn pid_alive(pid: u32) -> bool {
        let stat_path = format!("/proc/{pid}/stat");
        if let Ok(stat) = std::fs::read_to_string(&stat_path) {
            if stat.contains(") Z ") {
                return false;
            }
            return true;
        }
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }

    fn request() -> CompletionRequest {
        CompletionRequest {
            system: "evaluator prompt".into(),
            prompt: "ls".into(),
            max_tokens: None,
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn oneshot_drop_kills_descendant_process_group() {
        let _env = crate::state::STATE_ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("descendant.pid");
        let script = write_script(
            dir.path(),
            "spawn-descendant",
            &format!(
                "#!/bin/sh\nsleep 30 &\necho $! > '{}'\nwait\n",
                pid_file.display()
            ),
        );

        let cmd = script.display().to_string();
        let root = dir.path().to_path_buf();
        let handle = tokio::spawn(async move {
            let _ = complete(&cmd, &root, None, &ThinkingLevel::Off, &request()).await;
        });

        let descendant_pid: u32 = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(text) = std::fs::read_to_string(&pid_file) {
                    if let Ok(pid) = text.trim().parse::<u32>() {
                        break pid;
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();

        assert!(pid_alive(descendant_pid), "descendant should be running");
        handle.abort();
        let _ = handle.await;

        let exited = tokio::time::timeout(Duration::from_secs(2), async {
            while pid_alive(descendant_pid) {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .is_ok();

        if !exited {
            let _ = std::process::Command::new("kill")
                .args(["-9", &descendant_pid.to_string()])
                .status();
        }
        assert!(exited, "descendant process {descendant_pid} survived drop");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn oneshot_rejects_init_frame_advertising_tools_or_mcp_servers() {
        let _env = crate::state::STATE_ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let bad_tools = write_script(
            dir.path(),
            "bad-tools",
            "#!/bin/sh\nread -r _\nprintf '%s\\n' '{\"type\":\"system\",\"subtype\":\"init\",\"tools\":[\"Bash\"],\"mcp_servers\":[]}'\nsleep 30\n",
        );
        let err = tokio::time::timeout(
            Duration::from_secs(5),
            complete(
                &bad_tools.display().to_string(),
                dir.path(),
                None,
                &ThinkingLevel::Off,
                &request(),
            ),
        )
        .await
        .unwrap()
        .unwrap_err()
        .to_string();
        assert!(err.contains("Bash"), "{err}");

        let bad_mcp = write_script(
            dir.path(),
            "bad-mcp",
            "#!/bin/sh\nread -r _\nprintf '%s\\n' '{\"type\":\"system\",\"subtype\":\"init\",\"tools\":[],\"mcp_servers\":[{\"name\":\"extra\",\"status\":\"connected\"}]}'\nsleep 30\n",
        );
        let err = tokio::time::timeout(
            Duration::from_secs(5),
            complete(
                &bad_mcp.display().to_string(),
                dir.path(),
                None,
                &ThinkingLevel::Off,
                &request(),
            ),
        )
        .await
        .unwrap()
        .unwrap_err()
        .to_string();
        assert!(err.contains("extra"), "{err}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn oneshot_accepts_empty_init_frame() {
        let _env = crate::state::STATE_ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let ok_script = write_script(
            dir.path(),
            "ok-init",
            "#!/bin/sh\nread -r _\nprintf '%s\\n' '{\"type\":\"system\",\"subtype\":\"init\",\"tools\":[],\"mcp_servers\":[]}'\nprintf '%s\\n' '{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":false,\"result\":\"allow\"}'\n",
        );
        let completion = tokio::time::timeout(
            Duration::from_secs(5),
            complete(
                &ok_script.display().to_string(),
                dir.path(),
                None,
                &ThinkingLevel::Off,
                &request(),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(completion.text, "allow");
    }
}
