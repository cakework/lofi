use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncReadExt;

/// Polls `cancel` until it is set. Raced against a long-running tool's work so
/// a tool awaiting I/O (which produces no JS bytecode ticks for the `QuickJS`
/// interrupt handler) can still observe user cancellation. Mirrors
/// `agent.rs::wait_for_cancel`.
pub async fn wait_for_cancel(cancel: &Arc<AtomicBool>) {
    while !cancel.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

pub struct PgrpKillGuard {
    pid: Option<u32>,
}

impl PgrpKillGuard {
    #[allow(clippy::must_use_candidate)]
    pub fn new(pid: Option<u32>) -> Self {
        Self { pid }
    }
    pub fn disarm(&mut self) {
        self.pid = None;
    }
}

#[allow(clippy::cast_possible_wrap)]
impl Drop for PgrpKillGuard {
    fn drop(&mut self) {
        if let Some(pid) = self.pid.take() {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(-(pid as i32)),
                nix::sys::signal::Signal::SIGKILL,
            );
        }
    }
}

#[allow(clippy::missing_errors_doc)]
pub async fn read_capped<R: tokio::io::AsyncRead + Unpin>(
    r: &mut R,
    cap: usize,
) -> std::io::Result<(Vec<u8>, bool)> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        let n = r.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        let room = cap.saturating_sub(buf.len());
        if room == 0 {
            drain(r).await?;
            return Ok((buf, true));
        }
        let take = n.min(room);
        buf.extend_from_slice(&tmp[..take]);
        if take < n {
            drain(r).await?;
            return Ok((buf, true));
        }
    }
    Ok((buf, false))
}

/// Read to the pipe end, appending as the data arrives.
///
/// The bytes belong to `sink`, not to the future doing the reading: a command
/// the clock cuts drops that future, and what it read is still there to report.
#[allow(clippy::missing_errors_doc)]
pub async fn read_capped_into<R: tokio::io::AsyncRead + Unpin>(
    r: &mut R,
    sink: &std::sync::Mutex<Vec<u8>>,
    cap: usize,
) -> std::io::Result<()> {
    let mut tmp = [0u8; 8192];
    loop {
        let n = r.read(&mut tmp).await?;
        if n == 0 {
            return Ok(());
        }
        let full = {
            let mut buf = sink
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let room = cap.saturating_sub(buf.len());
            let take = n.min(room);
            buf.extend_from_slice(&tmp[..take]);
            take < n
        };
        if full {
            drain(r).await?;
            return Ok(());
        }
    }
}

async fn drain<R: tokio::io::AsyncRead + Unpin>(r: &mut R) -> std::io::Result<()> {
    let mut tmp = [0u8; 8192];
    loop {
        let n = r.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
    }
    Ok(())
}
