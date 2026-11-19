use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use lofi_error::{Error, Result};
#[cfg(test)]
use lofi_types::SessionEventKind;
use lofi_types::{Message, Role, SessionEvent};
use serde::Serialize;

#[cfg(test)]
use super::parse_event;
use super::{short_id, EventIndex, IndexId, IndexKind};

struct CountingWriter<W> {
    inner: W,
    pos: u64,
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.pos += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

pub(super) struct RecordWriter<'a> {
    writer: CountingWriter<BufWriter<&'a std::fs::File>>,
    spans: Vec<(u64, u64)>,
}

impl RecordWriter<'_> {
    pub(super) fn record(&mut self, record: &impl Serialize) -> Result<(u64, u64)> {
        let start = self.writer.pos;
        serde_json::to_writer(&mut self.writer, record)
            .map_err(|error| Error::State(format!("json: {error}")))?;
        self.writer.write_all(b"\n")?;
        let end = self.writer.pos;
        self.spans.push((start, end));
        Ok((start, end))
    }
}

pub(super) fn open_locked_transcript(path: &Path) -> Result<std::fs::File> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    file.lock()?;
    Ok(file)
}

pub(super) struct AppendCommit {
    pub(super) start: u64,
    pub(super) end: u64,
    pub(super) spans: Vec<(u64, u64)>,
}

// Callers hold the locked transcript file across parent resolution and record writes.
pub(super) fn append_records_locked(
    file: &mut std::fs::File,
    write: impl FnOnce(&mut RecordWriter<'_>) -> Result<()>,
) -> Result<AppendCommit> {
    let byte_start = file.seek(SeekFrom::End(0))?;
    let written = (|| -> Result<(u64, Vec<(u64, u64)>)> {
        let mut writer = RecordWriter {
            writer: CountingWriter {
                inner: BufWriter::new(&*file),
                pos: byte_start,
            },
            spans: Vec::new(),
        };
        write(&mut writer)?;
        writer.writer.flush()?;
        Ok((writer.writer.pos, writer.spans))
    })();
    match written.and_then(|(byte_end, spans)| {
        file.sync_data()?;
        Ok(AppendCommit {
            start: byte_start,
            end: byte_end,
            spans,
        })
    }) {
        Ok(committed) => Ok(committed),
        Err(error) => {
            let _ = file.set_len(byte_start);
            let _ = file.seek(SeekFrom::Start(byte_start));
            Err(error)
        }
    }
}

pub(super) enum AppendParent<'a> {
    Explicit(Option<&'a str>),
    #[cfg(test)]
    PhysicalEof,
}

impl AppendParent<'_> {
    #[cfg_attr(not(test), allow(unused_variables, clippy::unnecessary_wraps))]
    fn resolve(self, file: &mut std::fs::File) -> Result<Option<String>> {
        match self {
            Self::Explicit(parent) => Ok(parent.map(str::to_string)),
            #[cfg(test)]
            Self::PhysicalEof => last_event_id(file),
        }
    }
}

/// Append session events to a transcript file (one JSON line each). The
/// file is synchronized before returning so a crash after the turn still has
/// the data.
/// Each event is stamped with a fresh `id` (any incoming `id` is
/// overwritten) and chained to the previous one. A supplied `parent_hint`
/// selects an explicit branch parent; otherwise this low-level compatibility
/// helper continues from physical EOF. Active apps use [`SessionCursor`],
/// which always supplies its explicit logical parent instead.
/// # Errors
/// Returns [`Error::Io`] on open/write failure or [`Error::State`] on a
/// serialization failure.
#[cfg(test)]
pub(crate) fn append_events(
    path: &Path,
    events: &mut [SessionEvent],
    parent_hint: Option<&str>,
) -> Result<(u64, u64)> {
    let parent = parent_hint.map_or(AppendParent::PhysicalEof, |id| {
        AppendParent::Explicit(Some(id))
    });
    append_events_from(path, events, parent)
}

#[cfg(test)]
fn append_events_from(
    path: &Path,
    events: &mut [SessionEvent],
    parent: AppendParent<'_>,
) -> Result<(u64, u64)> {
    if events.is_empty() {
        let len = std::fs::metadata(path).map_or(0, |m| m.len());
        return Ok((len, len));
    }
    let mut lock = open_locked_transcript(path)?;
    let mut parent = parent.resolve(&mut lock)?;
    // Assign a fresh id to each event. `parent_id` is auto-filled only when
    // the event did not set one explicitly, preserving callers that construct
    // a branch inside one append batch.
    for ev in events.iter_mut() {
        ev.id = short_id();
        if ev.parent_id.is_none() {
            ev.parent_id.clone_from(&parent);
        }
        parent = Some(ev.id.clone());
    }
    let committed = append_prepared_events_locked(&mut lock, events)?;
    Ok((committed.start, committed.end))
}

#[derive(Debug, Clone, Copy)]
pub struct CompactionCounts {
    pub summarized: usize,
    pub represented: usize,
    pub kept: usize,
}

/// Append a compaction checkpoint as one batch: edited kept-tail messages
/// followed by the marker. A failed write is rolled back to the original file
/// length so the transcript cannot expose a partial checkpoint as its leaf.
/// # Errors
/// Propagates transcript read, serialization, and write failures.
#[cfg(test)]
pub(crate) fn append_compaction(
    path: &Path,
    kept_messages: &[Message],
    parent_hint: Option<&str>,
    summary: &str,
    summarized_range: &[String; 2],
    counts: CompactionCounts,
) -> Result<(u64, u64, String)> {
    let parent = parent_hint.map_or(AppendParent::PhysicalEof, |id| {
        AppendParent::Explicit(Some(id))
    });
    let (start, end, marker_id, _) = append_compaction_from(
        path,
        kept_messages,
        parent,
        summary,
        summarized_range,
        counts,
    )?;
    Ok((start, end, marker_id))
}

pub(super) fn append_compaction_from(
    path: &Path,
    kept_messages: &[Message],
    parent: AppendParent<'_>,
    summary: &str,
    summarized_range: &[String; 2],
    counts: CompactionCounts,
) -> Result<(u64, u64, String, Vec<EventIndex>)> {
    #[derive(Serialize)]
    struct MessageCheckpoint<'a> {
        id: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<&'a str>,
        #[serde(rename = "type")]
        kind: &'static str,
        role: Role,
        blocks: &'a [lofi_types::ContentBlock],
    }
    #[derive(Serialize)]
    struct CompactionCheckpoint<'a> {
        id: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<&'a str>,
        #[serde(rename = "type")]
        kind: &'static str,
        summary: &'a str,
        first_kept_entry_id: &'a str,
        summarized_range: &'a [String; 2],
        checkpointed_tail: bool,
        summarized: usize,
        represented: usize,
        kept: usize,
    }

    let mut lock = open_locked_transcript(path)?;
    let parent = parent.resolve(&mut lock)?;
    let ids: Vec<String> = (0..=kept_messages.len()).map(|_| short_id()).collect();
    let committed = append_records_locked(&mut lock, |writer| {
        for (index, message) in kept_messages.iter().enumerate() {
            let event = MessageCheckpoint {
                id: &ids[index],
                parent_id: if index == 0 {
                    parent.as_deref()
                } else {
                    Some(&ids[index - 1])
                },
                kind: "message",
                role: message.role,
                blocks: &message.blocks,
            };
            writer.record(&event)?;
        }
        let marker_index = kept_messages.len();
        let marker = CompactionCheckpoint {
            id: &ids[marker_index],
            parent_id: if marker_index == 0 {
                parent.as_deref()
            } else {
                Some(&ids[marker_index - 1])
            },
            kind: "compaction",
            summary,
            first_kept_entry_id: ids
                .first()
                .filter(|_| marker_index > 0)
                .map_or("", String::as_str),
            summarized_range,
            checkpointed_tail: true,
            summarized: counts.summarized,
            represented: counts.represented,
            kept: counts.kept,
        };
        writer.record(&marker)?;
        Ok(())
    })?;
    let suffix = compaction_suffix_index(kept_messages, parent.as_deref(), &ids, &committed.spans);
    Ok((
        committed.start,
        committed.end,
        ids[kept_messages.len()].clone(),
        suffix,
    ))
}

fn compaction_suffix_index(
    kept_messages: &[Message],
    parent: Option<&str>,
    ids: &[String],
    spans: &[(u64, u64)],
) -> Vec<EventIndex> {
    let mut suffix = Vec::with_capacity(spans.len());
    for (index, message) in kept_messages.iter().enumerate() {
        let (offset, end_offset) = spans[index];
        let parent_id = if index == 0 {
            parent.map(IndexId::borrow)
        } else {
            Some(IndexId::borrow(&ids[index - 1]))
        };
        let kind = match message.role {
            Role::User => IndexKind::UserPrompt,
            Role::Assistant => IndexKind::AssistantMessage,
            Role::Tool => IndexKind::ToolResult,
            Role::System => IndexKind::SystemMessage,
        };
        suffix.push(EventIndex {
            id: IndexId::borrow(&ids[index]),
            parent_id,
            offset,
            end_offset,
            kind,
        });
    }
    let marker_index = kept_messages.len();
    let (offset, end_offset) = spans[marker_index];
    suffix.push(EventIndex {
        id: IndexId::borrow(&ids[marker_index]),
        parent_id: if marker_index == 0 {
            parent.map(IndexId::borrow)
        } else {
            Some(IndexId::borrow(&ids[marker_index - 1]))
        },
        offset,
        end_offset,
        kind: IndexKind::Compaction,
    });
    suffix
}

pub(super) fn append_prepared_events_locked(
    file: &mut std::fs::File,
    events: &[SessionEvent],
) -> Result<AppendCommit> {
    append_records_locked(file, |writer| {
        for event in events {
            writer.record(event)?;
        }
        Ok(())
    })
}

/// # Errors
/// Returns [`Error::Io`] on a read failure.
#[cfg(test)]
fn last_event_id(file: &mut std::fs::File) -> Result<Option<String>> {
    use std::io::{BufRead, BufReader};
    file.seek(SeekFrom::Start(0))?;
    let mut last: Option<String> = None;
    let mut first = true;
    for line in BufReader::new(&*file).lines() {
        let line = line?;
        let trimmed = line.trim_end_matches(['\n', '\r']);
        if trimmed.is_empty() {
            continue;
        }
        if first {
            first = false;
            continue;
        }
        if let Ok(ev) = parse_event(trimmed) {
            if !matches!(ev.kind, SessionEventKind::Cursor { .. }) {
                last = Some(ev.id);
            }
        }
    }
    Ok(last)
}
