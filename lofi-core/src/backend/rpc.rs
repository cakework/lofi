use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lofi_error::{Error, Result};
use serde_json::Value;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::process::ChildStderr;

/// Child to Lofi. Guards a hostile or broken child.
pub(super) const MAX_READ_FRAME_BYTES: u64 = 4 * 1024 * 1024;
/// Lofi to child. Must carry the request image budget after base64 inflation.
pub(super) const MAX_WRITE_FRAME_BYTES: u64 = 16 * 1024 * 1024;
/// Maximum stderr tail retained per CLI child for failure diagnostics.
const MAX_STDERR_TAIL_BYTES: usize = 8 * 1024;
/// Maximum characters of stderr appended to a single error message.
const MAX_STDERR_ERROR_CHARS: usize = 500;

/// Drains a CLI child's stderr in the background into a bounded tail buffer so
/// unexpected exits and protocol errors carry the child's actual diagnostics.
pub(super) struct StderrTail {
    buf: Arc<Mutex<VecDeque<u8>>>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl StderrTail {
    pub(super) fn spawn(mut stderr: ChildStderr) -> Self {
        let buf = Arc::new(Mutex::new(VecDeque::new()));
        let shared = Arc::clone(&buf);
        let task = tokio::spawn(async move {
            let mut chunk = [0u8; 1024];
            loop {
                match stderr.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        let mut guard = shared
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        let slice = &chunk[..n];
                        if slice.len() >= MAX_STDERR_TAIL_BYTES {
                            guard.clear();
                            guard.extend(&slice[slice.len() - MAX_STDERR_TAIL_BYTES..]);
                        } else {
                            let overflow =
                                (guard.len() + slice.len()).saturating_sub(MAX_STDERR_TAIL_BYTES);
                            for _ in 0..overflow {
                                guard.pop_front();
                            }
                            guard.extend(slice);
                        }
                    }
                }
            }
        });
        Self {
            buf,
            task: Some(task),
        }
    }

    /// Wait briefly for the stderr drain to finish after child shutdown, then
    /// return the trimmed tail.
    pub(super) async fn finish(&mut self) -> String {
        if let Some(task) = self.task.take() {
            let _ = tokio::time::timeout(Duration::from_millis(100), task).await;
        }
        let guard = self
            .buf
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let bytes: Vec<u8> = guard.iter().copied().collect();
        let text = String::from_utf8_lossy(&bytes);
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return String::new();
        }
        match trimmed.char_indices().nth(MAX_STDERR_ERROR_CHARS) {
            Some((at, _)) => format!("{}…", &trimmed[..at]),
            None => trimmed.to_owned(),
        }
    }
}

impl Drop for StderrTail {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

pub(super) fn with_stderr(error: Error, stderr: &str) -> Error {
    if stderr.is_empty() || matches!(error, Error::Cancelled) {
        return error;
    }
    Error::State(format!("{error}; stderr: {stderr}"))
}

pub(super) async fn read(reader: &mut (impl AsyncBufRead + Unpin)) -> Result<Value> {
    let value = read_json(reader).await?;
    if value["jsonrpc"] != "2.0" {
        return Err(Error::State("invalid backend JSON-RPC envelope".into()));
    }
    Ok(value)
}

pub(super) async fn read_json(reader: &mut (impl AsyncBufRead + Unpin)) -> Result<Value> {
    let mut bytes = Vec::new();
    let count = reader
        .take(MAX_READ_FRAME_BYTES + 1)
        .read_until(b'\n', &mut bytes)
        .await?;
    if count == 0 {
        return Err(Error::State("backend disconnected".into()));
    }
    if count as u64 > MAX_READ_FRAME_BYTES || bytes.last() != Some(&b'\n') {
        return Err(Error::State(
            "backend frame exceeds limit or is incomplete".into(),
        ));
    }
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|e| Error::State(format!("invalid backend JSON: {e}")))?;
    if !value.is_object() {
        return Err(Error::State("invalid backend JSON object".into()));
    }
    Ok(value)
}

pub(super) async fn write(writer: &mut (impl AsyncWrite + Unpin), value: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec(value).map_err(|e| Error::State(e.to_string()))?;
    if bytes.len() as u64 >= MAX_WRITE_FRAME_BYTES {
        return Err(Error::State("backend frame exceeds limit".into()));
    }
    bytes.push(b'\n');
    writer.write_all(&bytes).await?;
    writer.flush().await?;
    Ok(())
}

pub(super) fn error(id: &Value, code: i32, message: &str) -> Value {
    serde_json::json!({"jsonrpc":"2.0", "id":id, "error":{"code":code,"message":message}})
}
