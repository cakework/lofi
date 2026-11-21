use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use base64::Engine;
use lofi_error::{Error, Result};
use lofi_types::ContentBlock;
use serde_json::{json, Value};
use tokio::io::BufReader;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{mpsc, Notify};

use super::rpc;
use crate::agent::{ToolCallContext, ToolExecutor};

/// Maximum concurrent in-flight MCP requests per bridge connection.
const MAX_IN_FLIGHT: usize = 64;
/// Safe Unix socket path byte ceiling across Linux (108) and macOS (104).
const MAX_UNIX_SOCKET_BYTES: usize = 96;

struct SocketLease {
    path: PathBuf,
    owned_dir: Option<PathBuf>,
}

impl Drop for SocketLease {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        if let Some(dir) = &self.owned_dir {
            let _ = std::fs::remove_dir(dir);
        }
    }
}

fn bind_unix_socket(dir: &Path) -> Result<(UnixListener, SocketLease)> {
    let candidate = dir.join("exec.sock");
    let (path, owned_dir) = if candidate.as_os_str().len() <= MAX_UNIX_SOCKET_BYTES {
        let _ = std::fs::remove_file(&candidate);
        (candidate, None)
    } else {
        let short = uuid::Uuid::new_v4().simple().to_string();
        let fallback_dir = std::env::temp_dir().join(format!("lofi-mcp-{}", &short[..12]));
        crate::state::ensure_private_dir(&fallback_dir)?;
        let path = fallback_dir.join("s");
        (path, Some(fallback_dir))
    };
    let listener = UnixListener::bind(&path)?;
    crate::state::ensure_private_file(&path)?;
    Ok((listener, SocketLease { path, owned_dir }))
}

struct AbortOnDrop(Vec<tokio::task::AbortHandle>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        for handle in &self.0 {
            handle.abort();
        }
    }
}

struct InFlightGuard {
    in_flight: Arc<Mutex<HashSet<String>>>,
    key: String,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.in_flight
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.key);
    }
}

pub(super) struct BridgeWatch {
    finished: Arc<AtomicBool>,
    done: Arc<Notify>,
}

impl BridgeWatch {
    pub(super) async fn wait(&self) {
        loop {
            let notified = self.done.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.finished.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }
}

struct FinishOnDrop {
    finished: Arc<AtomicBool>,
    done: Arc<Notify>,
}

impl Drop for FinishOnDrop {
    fn drop(&mut self) {
        self.finished.store(true, Ordering::Release);
        self.done.notify_waiters();
    }
}

pub(super) struct Bridge {
    pub config: Value,
    active: Arc<Mutex<Option<(ToolExecutor, ToolCallContext)>>>,
    finished: Arc<AtomicBool>,
    done: Arc<Notify>,
    task: tokio::task::JoinHandle<Result<()>>,
    _socket: SocketLease,
}

impl Bridge {
    pub fn start(executable: &Path, dir: &Path) -> Result<Self> {
        ensure_bridge_executable(executable)?;
        let (listener, socket) = bind_unix_socket(dir)?;
        let token = uuid::Uuid::new_v4().to_string();
        let config = json!({"name":"lofi", "command":executable, "args":["mcp", "serve"],
            "env":[{"name":"LOFI_EXEC_SOCKET","value":socket.path}, {"name":"LOFI_EXEC_TOKEN","value":token}]});
        let active = Arc::new(Mutex::new(None));
        let finished = Arc::new(AtomicBool::new(false));
        let done = Arc::new(Notify::new());
        let state = active.clone();
        let task_finished = finished.clone();
        let task_done = done.clone();
        let task = tokio::spawn(async move {
            let _guard = FinishOnDrop {
                finished: task_finished,
                done: task_done,
            };
            let (mut reader, writer) = loop {
                let (stream, _) = listener.accept().await?;
                let (reader, writer) = stream.into_split();
                let mut reader = BufReader::new(reader);
                let hello = tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    rpc::read(&mut reader),
                )
                .await;
                let Ok(Ok(hello)) = hello else {
                    continue;
                };
                if hello["method"] == "_lofi/connect" && hello["params"]["token"] == token {
                    break (reader, writer);
                }
            };
            drop(listener);
            serve(&mut reader, writer, state).await
        });
        Ok(Self {
            config,
            active,
            finished,
            done,
            task,
            _socket: socket,
        })
    }

    pub fn activate(&self, executor: ToolExecutor, context: ToolCallContext) {
        *self
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((executor, context));
    }

    pub fn deactivate(&self) {
        if let Some(context) = self
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            drop(context);
        }
    }

    pub fn watcher(&self) -> BridgeWatch {
        BridgeWatch {
            finished: self.finished.clone(),
            done: self.done.clone(),
        }
    }

    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }
}

/// The executable the child spawns: a path to the pinned image inside `dir`,
/// which no later rebuild can take away or re-point. The caller's own path is
/// kept only where nothing was pinned, or where `dir` cannot execute files.
pub(super) fn bridge_executable(
    executable: &std::path::Path,
    dir: &std::path::Path,
) -> Result<std::path::PathBuf> {
    let path = crate::state::running_image_path(executable, dir).map_err(|error| {
        let message = if executable.exists() {
            format!(
                "could not prepare the running lofi image in {} for the exec \
                 bridge: {error}",
                dir.display()
            )
        } else {
            format!(
                "the running lofi binary {} was replaced by a rebuild, and no \
                 path to it could be prepared: {error}; restart the session to \
                 run the new build",
                executable.display()
            )
        };
        Error::State(message)
    })?;
    Ok(path.unwrap_or_else(|| executable.to_path_buf()))
}

/// Refuse a path that a rebuild took away. The kernel reports a replaced
/// binary as `... (deleted)`, which no child can spawn: the agent backend
/// would come up with no tools instead of a clear error.
fn ensure_bridge_executable(executable: &std::path::Path) -> Result<()> {
    if executable.exists() {
        return Ok(());
    }
    let message = format!(
        "the running lofi binary {} was replaced by a rebuild, so the agent \
         backend cannot start its exec bridge; restart the session to run the \
         new build",
        executable.display()
    );
    Err(Error::State(message))
}

impl Drop for Bridge {
    fn drop(&mut self) {
        self.deactivate();
        self.task.abort();
    }
}

async fn serve(
    reader: &mut (impl tokio::io::AsyncBufRead + Unpin),
    mut writer: impl tokio::io::AsyncWrite + Unpin + Send + 'static,
    active: Arc<Mutex<Option<(ToolExecutor, ToolCallContext)>>>,
) -> Result<()> {
    let (write_tx, mut write_rx) = mpsc::channel::<Value>(64);
    let writer_task = tokio::spawn(async move {
        while let Some(frame) = write_rx.recv().await {
            rpc::write(&mut writer, &frame).await?;
        }
        Ok::<(), Error>(())
    });
    let mut children = AbortOnDrop(vec![writer_task.abort_handle()]);
    let in_flight: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
    let mut initialized = false;
    let mut ready = false;
    loop {
        let request = rpc::read(reader).await?;
        let method = request["method"].as_str().unwrap_or_default();
        let Some(id) = request.get("id").cloned() else {
            if method == "notifications/initialized" && initialized {
                ready = true;
            }
            continue;
        };
        if !(id.is_string() || id.is_number()) {
            return Err(Error::State("invalid MCP request ID".into()));
        }
        let id_key = id.to_string();
        let accepted = {
            let mut guard = in_flight
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.len() < MAX_IN_FLIGHT && guard.insert(id_key.clone())
        };
        if !accepted {
            send_frame(
                &write_tx,
                rpc::error(&id, -32600, "duplicate or excessive in-flight MCP request"),
            )
            .await?;
            continue;
        }
        let id_guard = InFlightGuard {
            in_flight: Arc::clone(&in_flight),
            key: id_key,
        };
        match method {
            "initialize" if !initialized => {
                initialized = true;
                let result = json!({"protocolVersion":"2025-03-26","capabilities":{"tools":{}},
                    "serverInfo":{"name":"lofi-exec","version":env!("CARGO_PKG_VERSION")}});
                send_frame(&write_tx, json!({"jsonrpc":"2.0","id":id,"result":result})).await?;
                drop(id_guard);
            }
            "ping" => {
                send_frame(&write_tx, json!({"jsonrpc":"2.0","id":id,"result":{}})).await?;
                drop(id_guard);
            }
            "tools/list" if ready => {
                let result = json!({"tools":[{"name":lofi_code::EXEC_TOOL_NAME,
                    "description":lofi_code::EXEC_TOOL_DESCRIPTION, "inputSchema":lofi_code::exec_tool_input_schema()}]});
                send_frame(&write_tx, json!({"jsonrpc":"2.0","id":id,"result":result})).await?;
                drop(id_guard);
            }
            "tools/call" if ready => {
                if let Some(handle) =
                    dispatch_tool_call(id, request["params"].clone(), &active, &write_tx, id_guard)
                        .await?
                {
                    children.0.retain(|h| !h.is_finished());
                    children.0.push(handle);
                }
            }
            _ => {
                send_frame(&write_tx, rpc::error(&id, -32601, "MCP method unavailable")).await?;
                drop(id_guard);
            }
        }
    }
}

async fn dispatch_tool_call(
    id: Value,
    params: Value,
    active: &Arc<Mutex<Option<(ToolExecutor, ToolCallContext)>>>,
    write_tx: &mpsc::Sender<Value>,
    id_guard: InFlightGuard,
) -> Result<Option<tokio::task::AbortHandle>> {
    if params["name"] != "exec" || !params["arguments"]["code"].is_string() {
        send_frame(
            write_tx,
            rpc::error(&id, -32602, "expected exec with a code string"),
        )
        .await?;
        drop(id_guard);
        return Ok(None);
    }
    let active_turn = active
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let Some((executor, context)) = active_turn else {
        send_frame(write_tx, rpc::error(&id, -32600, "no active Lofi turn")).await?;
        drop(id_guard);
        return Ok(None);
    };
    let reply_tx = write_tx.clone();
    let handle = tokio::spawn(async move {
        let _guard = id_guard;
        let call_id = format!("exec-{}", uuid::Uuid::new_v4());
        let response = match executor
            .execute(&call_id, &params["arguments"], context)
            .await
            .and_then(tool_result)
        {
            Ok(result) => json!({"jsonrpc":"2.0","id":id,"result":result}),
            Err(error) => rpc::error(&id, -32603, &error.to_string()),
        };
        let _ = reply_tx.send(response).await;
    });
    Ok(Some(handle.abort_handle()))
}

async fn send_frame(tx: &mpsc::Sender<Value>, frame: Value) -> Result<()> {
    tx.send(frame)
        .await
        .map_err(|_| Error::State("exec bridge writer stopped".into()))
}

fn tool_result(block: ContentBlock) -> Result<Value> {
    let ContentBlock::ToolResult {
        content,
        is_error,
        images,
        ..
    } = block
    else {
        return Err(Error::State("exec returned an invalid tool result".into()));
    };
    let mut blocks = vec![json!({"type":"text","text":content})];
    for image in images {
        blocks.push(json!({"type":"image", "mimeType":image.media_type,
            "data":base64::engine::general_purpose::STANDARD.encode(image.bytes)}));
    }
    Ok(json!({"content":blocks,"isError":is_error}))
}

/// Relay MCP stdio to the running Lofi session. This process never owns tools.
/// # Errors
/// Returns errors for missing bridge credentials, invalid socket paths or broken pipes.
pub async fn run_stdio_proxy() -> Result<()> {
    let socket = std::env::var_os("LOFI_EXEC_SOCKET")
        .filter(|path| !path.is_empty())
        .ok_or_else(|| Error::Config("missing exec bridge socket".into()))?;
    let token = std::env::var("LOFI_EXEC_TOKEN")
        .map_err(|_| Error::Config("missing exec bridge token".into()))?;
    let mut stream = UnixStream::connect(&socket).await?;
    rpc::write(
        &mut stream,
        &json!({"jsonrpc":"2.0","method":"_lofi/connect","params":{"token":token}}),
    )
    .await?;
    let (mut reader, mut writer) = stream.into_split();
    let mut stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    tokio::select! {
        result = tokio::io::copy(&mut stdin, &mut writer) => { result?; }
        result = tokio::io::copy(&mut reader, &mut stdout) => { result?; }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::{ensure_bridge_executable, Bridge};

    #[tokio::test]
    async fn start_refuses_a_replaced_executable() {
        let dir = tempfile::tempdir().unwrap();
        let gone = dir.path().join("lofi (deleted)");
        let error = match Bridge::start(&gone, dir.path()) {
            Ok(_) => panic!("a replaced executable must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("was replaced by a rebuild"), "{error}");
    }

    #[test]
    fn an_unpinned_process_keeps_its_own_path() {
        // Nothing pins an image in this binary, so the child keeps the path it
        // was given. The pinned ladder is covered in state.rs.
        let dir = tempfile::tempdir().unwrap();
        let exe = std::env::current_exe().unwrap();
        assert_eq!(super::bridge_executable(&exe, dir.path()).unwrap(), exe);
        assert!(!dir.path().join("lofi").exists());
    }

    #[test]
    fn an_existing_executable_is_accepted() {
        ensure_bridge_executable(&std::env::current_exe().unwrap()).unwrap();
    }

    #[test]
    fn a_replaced_executable_is_refused_by_name() {
        let dir = tempfile::tempdir().unwrap();
        // What the kernel reports after a rebuild replaced the binary.
        let gone = dir.path().join("lofi (deleted)");
        let error = ensure_bridge_executable(&gone).unwrap_err().to_string();
        assert!(error.contains("was replaced by a rebuild"), "{error}");
        assert!(error.contains("lofi (deleted)"), "{error}");
        assert!(error.contains("restart the session"), "{error}");
    }

    fn bridge_env(bridge: &Bridge, key: &str) -> Option<String> {
        bridge.config["env"]
            .as_array()?
            .iter()
            .find(|entry| entry["name"] == key)
            .and_then(|entry| entry["value"].as_str().map(str::to_owned))
    }

    #[tokio::test]
    async fn bridge_uses_unix_socket_and_survives_rejected_auth() {
        let dir = tempfile::tempdir().unwrap();
        let exe = std::env::current_exe().unwrap();
        let bridge = Bridge::start(&exe, dir.path()).unwrap();
        let socket_path = bridge_env(&bridge, "LOFI_EXEC_SOCKET").unwrap();
        let token = bridge_env(&bridge, "LOFI_EXEC_TOKEN").unwrap();

        // A bogus client with a wrong token is disconnected without killing the bridge.
        let mut bogus = tokio::net::UnixStream::connect(&socket_path).await.unwrap();
        super::rpc::write(
            &mut bogus,
            &serde_json::json!({"jsonrpc":"2.0","method":"_lofi/connect","params":{"token":"wrong"}}),
        )
        .await
        .unwrap();
        drop(bogus);

        // The legitimate client still connects and initializes cleanly.
        let stream = tokio::net::UnixStream::connect(&socket_path).await.unwrap();
        let (read_half, mut write_half) = stream.into_split();
        let mut reader = tokio::io::BufReader::new(read_half);
        super::rpc::write(
            &mut write_half,
            &serde_json::json!({"jsonrpc":"2.0","method":"_lofi/connect","params":{"token":token}}),
        )
        .await
        .unwrap();
        super::rpc::write(
            &mut write_half,
            &serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        )
        .await
        .unwrap();
        let resp = super::rpc::read(&mut reader).await.unwrap();
        assert_eq!(resp["id"], 1);
        assert!(resp.get("result").is_some(), "{resp}");
    }

    #[tokio::test]
    async fn bridge_answers_ping_while_exec_call_is_blocked() {
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            let dir = tempfile::tempdir().unwrap();
            let exe = std::env::current_exe().unwrap();
            let bridge = Bridge::start(&exe, dir.path()).unwrap();
            let socket_path = bridge_env(&bridge, "LOFI_EXEC_SOCKET").unwrap();
            let token = bridge_env(&bridge, "LOFI_EXEC_TOKEN").unwrap();

            let (confirm_send, mut confirm_recv) =
                tokio::sync::mpsc::unbounded_channel::<crate::agent::ConfirmRequest>();
            let executor =
                crate::agent::ToolExecutor::in_memory(dir.path()).with_confirm_tx(confirm_send);
            bridge.activate(executor, crate::agent::ToolCallContext::default());

            let stream = tokio::net::UnixStream::connect(&socket_path).await.unwrap();
            let (read_half, mut write_half) = stream.into_split();
            let mut reader = tokio::io::BufReader::new(read_half);
            super::rpc::write(
                &mut write_half,
                &serde_json::json!({"jsonrpc":"2.0","method":"_lofi/connect","params":{"token":token}}),
            )
            .await
            .unwrap();
            super::rpc::write(
                &mut write_half,
                &serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
            )
            .await
            .unwrap();
            let _ = super::rpc::read(&mut reader).await.unwrap();
            super::rpc::write(
                &mut write_half,
                &serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
            )
            .await
            .unwrap();

            // Start an exec call that blocks waiting for bash confirmation.
            super::rpc::write(
                &mut write_half,
                &serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "exec",
                        "arguments": {
                            "code": "return await lofi.bash({cmd:'echo hi',timeoutMs:2000})",
                            "description": "blocked bash"
                        }
                    }
                }),
            )
            .await
            .unwrap();
            let confirm = confirm_recv.recv().await.unwrap();

            // Send ping while the exec call is still blocked on confirmation.
            super::rpc::write(
                &mut write_half,
                &serde_json::json!({"jsonrpc":"2.0","id":3,"method":"ping","params":{}}),
            )
            .await
            .unwrap();
            let ping_resp = super::rpc::read(&mut reader).await.unwrap();
            assert_eq!(ping_resp["id"], 3, "ping must not wait for blocked exec");

            let _ = confirm.respond.send(crate::agent::ConfirmResponse::Allow);
            let exec_resp = super::rpc::read(&mut reader).await.unwrap();
            assert_eq!(exec_resp["id"], 2);
            assert!(exec_resp.get("result").is_some(), "{exec_resp}");
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn bridge_allows_more_than_4096_sequential_requests() {
        tokio::time::timeout(std::time::Duration::from_secs(15), async {
            let dir = tempfile::tempdir().unwrap();
            let exe = std::env::current_exe().unwrap();
            let bridge = Bridge::start(&exe, dir.path()).unwrap();
            let socket_path = bridge_env(&bridge, "LOFI_EXEC_SOCKET").unwrap();
            let token = bridge_env(&bridge, "LOFI_EXEC_TOKEN").unwrap();

            let stream = tokio::net::UnixStream::connect(&socket_path).await.unwrap();
            let (read_half, mut write_half) = stream.into_split();
            let mut reader = tokio::io::BufReader::new(read_half);
            super::rpc::write(
                &mut write_half,
                &serde_json::json!({"jsonrpc":"2.0","method":"_lofi/connect","params":{"token":token}}),
            )
            .await
            .unwrap();
            super::rpc::write(
                &mut write_half,
                &serde_json::json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{}}),
            )
            .await
            .unwrap();
            let _ = super::rpc::read(&mut reader).await.unwrap();

            for id in 1..=4100u64 {
                super::rpc::write(
                    &mut write_half,
                    &serde_json::json!({"jsonrpc":"2.0","id":id,"method":"ping","params":{}}),
                )
                .await
                .unwrap();
                let resp = super::rpc::read(&mut reader).await.unwrap();
                assert_eq!(resp["id"], id);
            }
        })
        .await
        .unwrap();
    }
}
