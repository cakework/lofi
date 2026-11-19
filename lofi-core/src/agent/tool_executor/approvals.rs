use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::oneshot;

use super::watches::ActivePrompt;
use crate::state;

#[derive(Debug)]
pub struct ConfirmRequest {
    pub id: u64,
    pub command: String,
    pub reason: Arc<Mutex<lofi_code::ConfirmReason>>,
    pub active: Arc<AtomicBool>,
    /// When auto mode denies this prompt on its own. `None` on prompts the
    /// auto-deny does not cover; the UI counts down to a set value.
    pub deny_at: Arc<Mutex<Option<std::time::Instant>>>,
    pub respond: oneshot::Sender<ConfirmResponse>,
}

/// One answer to a [`ConfirmRequest`]. Shell prompts take `Allow`/`Deny`;
/// directory grants take `Grant` with the access the user chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmResponse {
    Allow,
    Deny,
    Grant {
        access: lofi_types::Access,
        /// Grant the requested directory rather than the project root offered
        /// as the default.
        narrow: bool,
    },
}

impl ConfirmResponse {
    /// Whether this answer grants write access to a directory.
    #[must_use]
    pub fn is_grant_write(self) -> bool {
        matches!(
            self,
            Self::Grant {
                access: lofi_types::Access::Write,
                ..
            }
        )
    }

    /// Whether this answer grants read-only access to a directory.
    #[must_use]
    pub fn is_grant_read(self) -> bool {
        matches!(
            self,
            Self::Grant {
                access: lofi_types::Access::Read,
                ..
            }
        )
    }
}

/// Asks the user to grant one directory tree for use outside the workspace
/// root, and remembers the grant in the per-workspace state file so later
/// sessions do not ask again. A remembered grant that already allows the
/// request answers at once. Returns the granted directory and its access;
/// a refused prompt, or a closed confirmation channel, keeps the path
/// rejected.
pub(super) async fn approve_outside_dir(
    root: &std::path::Path,
    approval: lofi_code::DirApproval,
    config_grants: &[lofi_code::GrantedDir],
    tx: &tokio::sync::mpsc::UnboundedSender<ConfirmRequest>,
    counter: &Arc<AtomicU64>,
    deny_at: Option<std::time::Instant>,
) -> Option<lofi_code::GrantedDir> {
    // A grant that already covers the requested directory answers at once,
    // whatever its width: the prompt's unit is the project root, and a
    // remembered grant for the directory itself still allows the path.
    let requested_dir = approval
        .narrow
        .clone()
        .unwrap_or_else(|| approval.dir.clone());
    // A state file that cannot be opened only costs the prompt: the grant
    // is then not remembered for the next session.
    let state = state::WorkspaceState::open(root).ok();
    if let Some(grant) = state
        .as_ref()
        .and_then(|s| s.grant_allowing(&requested_dir, approval.requested))
    {
        return Some(lofi_code::GrantedDir {
            dir: grant.dir,
            access: grant.access,
        });
    }
    // A configured grant that already allows the request answers at once.
    if let Some(grant) = config_grants
        .iter()
        .filter(|g| requested_dir.starts_with(&g.dir))
        .max_by_key(|g| g.access)
        .filter(|g| g.access.allows(approval.requested))
    {
        return Some(grant.clone());
    }
    // The strongest covering grant, remembered or configured, names what a
    // widening prompt starts from. Its directory travels with the access, so
    // the prompt describes only the tree the grant actually holds.
    let current = state
        .as_ref()
        .and_then(|s| s.strongest_covering(&requested_dir))
        .map(|grant| lofi_code::GrantedDir {
            dir: grant.dir,
            access: grant.access,
        })
        .or_else(|| {
            config_grants
                .iter()
                .filter(|g| requested_dir.starts_with(&g.dir))
                .max_by_key(|g| g.access)
                .cloned()
        });
    let id = counter.fetch_add(1, Ordering::SeqCst);
    let (resp_tx, resp_rx) = oneshot::channel();
    let active = Arc::new(AtomicBool::new(true));
    let _closes = ActivePrompt(active.clone());
    let req = ConfirmRequest {
        id,
        command: approval.dir.display().to_string(),
        reason: Arc::new(Mutex::new(lofi_code::ConfirmReason::OutsideRoot {
            dir: approval.dir.display().to_string(),
            narrow: approval
                .narrow
                .as_ref()
                .map(|dir| dir.display().to_string()),
            marker: approval.marker,
            requested: approval.requested,
            current,
        })),
        active,
        deny_at: Arc::new(Mutex::new(deny_at)),
        respond: resp_tx,
    };
    if tx.send(req).is_err() {
        return None;
    }
    let ConfirmResponse::Grant { access, narrow } = resp_rx.await.ok()? else {
        return None;
    };
    // A grant that cannot satisfy the request is a refusal.
    if !access.allows(approval.requested) {
        return None;
    }
    let dir = if narrow {
        approval.narrow.unwrap_or(approval.dir)
    } else {
        approval.dir
    };
    // An answer that lands after the call's clock closed the prompt is
    // reported but not kept: the call refuses the work it was asked for, so
    // the access it asked for must not outlive it.
    let late = deny_at.is_some_and(|end| std::time::Instant::now() >= end);
    // The grant is reused only by this call's retry; a failed save means
    // the next request asks again.
    if let Some(state) = state.as_ref().filter(|_| !late) {
        let _ = state.remember_grant(&dir, access);
    }
    Some(lofi_code::GrantedDir { dir, access })
}
