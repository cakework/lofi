use std::path::{Path, PathBuf};

use lofi_error::{Error, Result};
use lofi_types::{ContentBlock, Message, Role, RunModel};

use crate::lifecycle::{lineage_catch_up, CatchUp};
use crate::session::store::SessionCursor;

#[derive(Default, PartialEq, Eq)]
pub(super) enum BoundModel {
    #[default]
    Unchecked,
    Known(Option<String>),
}

impl BoundModel {
    pub(super) fn matches(&self, model: Option<&str>) -> bool {
        matches!(self, Self::Known(bound) if bound.as_deref() == model)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(super) struct Binding {
    pub(super) backend: String,
    pub(super) session_id: String,
    pub(super) cwd: PathBuf,
    pub(super) model: Option<String>,
    pub(super) leaf_id: Option<String>,
}

fn binding_path(cursor: &SessionCursor) -> PathBuf {
    cursor.path().with_extension("backend.json")
}

/// Compare the saved backend checkpoint with the session lineage.
///
/// The worker runs this per exchange, not the caller: an abandoned turn can
/// still be settling when the next one is enqueued, and a checkpoint read
/// before that settlement would name a leaf the turn already moved past.
pub(super) fn catch_up(
    name: &str,
    root: &Path,
    cursor: &SessionCursor,
    frontier: Option<&str>,
) -> Result<CatchUp> {
    let binding = read_binding(cursor)?;
    let Some(binding) = binding.filter(|b| b.backend == name && b.cwd == root) else {
        return Ok(CatchUp::Diverged);
    };
    lineage_catch_up(cursor, frontier, binding.leaf_id.as_deref())
}

/// Drop a checkpoint the child cannot be resumed from. The file is kept when
/// it already names the current leaf: a failed directory sync can still leave
/// it correct, and removing it would resend the whole history.
pub(super) fn drop_checkpoint(cursor: &SessionCursor) {
    let leaf = cursor.leaf_id();
    match read_binding(cursor) {
        Ok(Some(binding)) if binding.leaf_id == leaf => {}
        Ok(_) | Err(_) => remove_checkpoint(cursor),
    }
}

pub(super) fn remove_checkpoint(cursor: &SessionCursor) {
    let path = binding_path(cursor);
    if !path.exists() {
        return;
    }
    if let Err(error) = std::fs::remove_file(&path) {
        tracing::warn!(error = %error, "backend checkpoint removal failed");
    }
}

pub(super) fn read_binding(cursor: &SessionCursor) -> Result<Option<Binding>> {
    match std::fs::read(binding_path(cursor)) {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes).ok()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

pub(super) fn write_binding(cursor: &SessionCursor, binding: &Binding) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let path = binding_path(cursor);
    let dir = path
        .parent()
        .ok_or_else(|| Error::State("session path has no parent".into()))?;
    let tmp = dir.join(format!(
        ".{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("backend.json")
    ));
    let bytes = serde_json::to_vec(binding).map_err(|e| Error::State(e.to_string()))?;
    {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, &path)?;
    crate::state::ensure_private_file(&path)?;
    std::fs::File::open(dir)?.sync_all()?;
    Ok(())
}

/// The frame a silent continuation sends. A provider round continues from the
/// same history; a stdio child needs a message to produce output.
pub(super) const CONTINUE_PROMPT: &str = "Continue the task from where you left off.";

/// Render a history for a continuation frame: each image becomes a pointer,
/// and its bytes are appended to images in reference order. Base64 never
/// enters model-visible text.
pub(super) fn render_transfer(
    history: &[Message],
    images: &mut Vec<ContentBlock>,
) -> Result<String> {
    let mut budget = crate::agent::MAX_REQUEST_IMAGE_BYTES;
    let mut rendered = Vec::with_capacity(history.len());
    for message in history {
        let mut blocks = Vec::with_capacity(message.blocks.len());
        for block in &message.blocks {
            if let ContentBlock::Image { bytes, media_type } = block {
                let size = bytes.len() as u64 * 4 / 3 + 4;
                if size <= budget {
                    budget -= size;
                    images.push(ContentBlock::Image {
                        bytes: bytes.clone(),
                        media_type: media_type.clone(),
                    });
                    blocks.push(ContentBlock::Text {
                        text: format!("[image {}]", images.len()),
                    });
                } else {
                    blocks.push(ContentBlock::Text {
                        text: "[image omitted: transfer budget]".into(),
                    });
                }
            } else {
                blocks.push(block.clone());
            }
        }
        rendered.push(Message {
            blocks,
            ..message.clone()
        });
    }
    serde_json::to_string(&rendered).map_err(|e| Error::State(e.to_string()))
}

/// The identity the terminal marker carries before the child reports one.
pub(super) fn run_model_for(
    name: &str,
    model: Option<&str>,
    effort: &lofi_types::ThinkingLevel,
) -> RunModel {
    let selected = model.unwrap_or("default").to_owned();
    RunModel {
        provider: name.into(),
        id: selected.clone(),
        selected,
        display: None,
        thinking: effort.clone(),
        service_tier: lofi_types::ServiceTier::Auto,
    }
}

/// The role a recorded block belongs to.
pub(super) fn block_role(block: &ContentBlock) -> Role {
    if matches!(block, ContentBlock::ToolUse { .. }) {
        Role::Assistant
    } else {
        Role::Tool
    }
}
