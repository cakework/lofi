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

/// Headroom reserved for the JSON-RPC user-frame wrapper around `content`.
const FRAME_ENVELOPE_HEADROOM: u64 = 4096;

struct ByteCounter(u64);

impl std::io::Write for ByteCounter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 = self.0.saturating_add(buf.len() as u64);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Reject a frame that would exceed the child write limit before sending it so
/// the turn loop can recover through hard compaction instead of failing on a
/// write error.
pub(super) fn ensure_frame_fits(frame: &serde_json::Value) -> Result<()> {
    let mut counter = ByteCounter(0);
    serde_json::to_writer(&mut counter, frame).map_err(|e| Error::State(e.to_string()))?;
    let limit = super::rpc::MAX_WRITE_FRAME_BYTES.saturating_sub(FRAME_ENVELOPE_HEADROOM);
    if counter.0 >= limit {
        return Err(Error::ContextPressure(format!(
            "backend prompt frame totals {} bytes",
            counter.0
        )));
    }
    Ok(())
}

fn extract_image(
    bytes: &std::sync::Arc<[u8]>,
    media_type: &str,
    budget: &mut u64,
    images: &mut Vec<ContentBlock>,
) -> String {
    let size = bytes.len() as u64 * 4 / 3 + 4;
    if size <= *budget {
        *budget -= size;
        images.push(ContentBlock::Image {
            bytes: bytes.clone(),
            media_type: media_type.to_owned(),
        });
        format!("[image {}]", images.len())
    } else {
        "[image omitted: transfer budget]".into()
    }
}

/// Render a history for a continuation frame: each image becomes a pointer,
/// and its bytes are appended to images in reference order. Base64 never
/// enters model-visible text, and provider-only replay blobs are omitted.
pub(super) fn render_transfer(
    history: &[Message],
    images: &mut Vec<ContentBlock>,
) -> Result<String> {
    let mut budget = crate::agent::MAX_REQUEST_IMAGE_BYTES;
    let mut rendered = Vec::with_capacity(history.len());
    for message in history {
        let mut blocks = Vec::with_capacity(message.blocks.len());
        for block in &message.blocks {
            match block {
                ContentBlock::Image { bytes, media_type } => {
                    blocks.push(ContentBlock::Text {
                        text: extract_image(bytes, media_type, &mut budget, images),
                    });
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                    images: tool_images,
                } => {
                    if tool_images.is_empty() {
                        blocks.push(block.clone());
                    } else {
                        let mut text = content.clone();
                        for image in tool_images {
                            let pointer =
                                extract_image(&image.bytes, &image.media_type, &mut budget, images);
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(&pointer);
                        }
                        blocks.push(ContentBlock::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content: text,
                            is_error: *is_error,
                            images: Vec::new(),
                        });
                    }
                }
                ContentBlock::PartSignature { .. } => {}
                ContentBlock::Thinking { text, redacted, .. } => {
                    if !*redacted && !text.is_empty() {
                        blocks.push(ContentBlock::Thinking {
                            text: text.clone(),
                            signature: None,
                            redacted: false,
                        });
                    }
                }
                ContentBlock::Text { .. } | ContentBlock::ToolUse { .. } => {
                    blocks.push(block.clone());
                }
            }
        }
        if !blocks.is_empty() {
            rendered.push(Message {
                origin: None,
                blocks,
                role: message.role,
                kind: message.kind,
            });
        }
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use lofi_types::{PartSignatureFormat, PromptKind, ToolResultImage};

    #[test]
    fn render_transfer_extracts_tool_result_images_without_base64_in_text() {
        let history = vec![Message {
            role: Role::Tool,
            kind: PromptKind::User,
            origin: None,
            blocks: vec![ContentBlock::ToolResult {
                tool_use_id: "call-1".into(),
                content: "read image".into(),
                is_error: false,
                images: vec![ToolResultImage {
                    bytes: vec![1, 2, 3, 4, 5, 6].into(),
                    media_type: "image/png".into(),
                }],
            }],
        }];
        let mut images = Vec::new();
        let json = render_transfer(&history, &mut images).unwrap();
        assert_eq!(images.len(), 1);
        assert!(matches!(
            &images[0],
            ContentBlock::Image { bytes, media_type }
                if bytes.as_ref() == [1, 2, 3, 4, 5, 6] && media_type == "image/png"
        ));
        // Base64 of [1, 2, 3, 4, 5, 6] is "AQIDBAUG"
        assert!(!json.contains("AQIDBAUG"), "raw base64 leaked: {json}");
        assert!(json.contains("[image 1]"), "missing image pointer: {json}");
    }

    #[test]
    fn render_transfer_strips_provider_replay_metadata() {
        let history = vec![Message {
            role: Role::Assistant,
            kind: PromptKind::User,
            origin: Some(lofi_types::ModelOrigin {
                provider: "google".into(),
                api: lofi_types::Api::GoogleGenerativeAi,
                model: "gemini-3".into(),
            }),
            blocks: vec![
                ContentBlock::Thinking {
                    text: "plan".into(),
                    signature: Some("opaque-thinking-sig".into()),
                    redacted: false,
                },
                ContentBlock::Thinking {
                    text: "redacted".into(),
                    signature: Some("opaque-redacted-blob".into()),
                    redacted: true,
                },
                ContentBlock::PartSignature {
                    provider: "google".into(),
                    model: "gemini-3".into(),
                    format: PartSignatureFormat::Google,
                    signature: "opaque-part-sig".into(),
                },
                ContentBlock::Text {
                    text: "answer".into(),
                },
            ],
        }];
        let mut images = Vec::new();
        let json = render_transfer(&history, &mut images).unwrap();
        assert!(json.contains("plan"), "kept thinking text: {json}");
        assert!(json.contains("answer"), "kept answer text: {json}");
        assert!(!json.contains("opaque-thinking-sig"), "sig leaked: {json}");
        assert!(
            !json.contains("opaque-redacted-blob"),
            "redacted leaked: {json}"
        );
        assert!(!json.contains("opaque-part-sig"), "part sig leaked: {json}");
        assert!(
            !json.contains("part_signature"),
            "part_signature kept: {json}"
        );
        assert!(!json.contains("\"origin\""), "origin leaked: {json}");
    }

    #[test]
    fn render_transfer_omits_tool_result_images_beyond_budget() {
        let oversized = vec![0u8; (crate::agent::MAX_REQUEST_IMAGE_BYTES as usize) + 16];
        let history = vec![Message {
            role: Role::Tool,
            kind: PromptKind::User,
            origin: None,
            blocks: vec![ContentBlock::ToolResult {
                tool_use_id: "call-1".into(),
                content: "read image".into(),
                is_error: false,
                images: vec![ToolResultImage {
                    bytes: oversized.into(),
                    media_type: "image/png".into(),
                }],
            }],
        }];
        let mut images = Vec::new();
        let json = render_transfer(&history, &mut images).unwrap();
        assert!(images.is_empty());
        assert!(json.contains("[image omitted: transfer budget]"), "{json}");
    }

    #[test]
    fn ensure_frame_fits_reports_context_pressure_over_limit() {
        let small = serde_json::Value::String("ok".into());
        assert!(ensure_frame_fits(&small).is_ok());

        let huge = serde_json::Value::String(
            "x".repeat(super::super::rpc::MAX_WRITE_FRAME_BYTES as usize),
        );
        let err = ensure_frame_fits(&huge).unwrap_err();
        assert!(matches!(err, Error::ContextPressure(_)), "{err:?}");
    }
}
