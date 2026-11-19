#[allow(clippy::wildcard_imports)]
use super::*;

pub(in crate::session::store) struct IndexedSkeleton {
    pub(in crate::session::store) id: IndexId,
    pub(in crate::session::store) parent_id: Option<IndexId>,
    pub(in crate::session::store) kind: IndexKind,
    pub(in crate::session::store) cursor_leaf: Option<IndexId>,
    pub(in crate::session::store) checkpointed_tail: bool,
    pub(in crate::session::store) first_kept_entry_id: IndexId,
}

pub(in crate::session::store) fn read_index_skeleton<R>(
    reader: &mut std::io::BufReader<R>,
    line: &mut Vec<u8>,
) -> Result<Option<(u64, u64, IndexedSkeleton)>>
where
    R: std::io::Read + std::io::Seek,
{
    read_index_event(reader, line, |start, end, event| {
        let kind = index_kind(event.kind_type, event.role);
        (
            start,
            end,
            IndexedSkeleton {
                id: IndexId::borrow(event.id),
                parent_id: event.parent_id.map(IndexId::borrow),
                kind,
                cursor_leaf: event
                    .leaf_id
                    .filter(|id| !id.is_empty())
                    .map(IndexId::borrow),
                checkpointed_tail: event.checkpointed_tail,
                first_kept_entry_id: IndexId::borrow(event.first_kept_entry_id),
            },
        )
    })
}

pub(in crate::session::store) fn read_jsonl_value<T, R>(
    reader: &mut std::io::BufReader<R>,
) -> Result<Option<(u64, u64, T)>>
where
    T: serde::de::DeserializeOwned,
    R: std::io::Read + std::io::Seek,
{
    use std::io::{BufRead, Seek};

    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(None);
        }
        let whitespace = available
            .iter()
            .take_while(|byte| byte.is_ascii_whitespace())
            .count();
        if whitespace == 0 {
            break;
        }
        reader.consume(whitespace);
    }
    let start = reader.stream_position()?;
    let value = serde_json::Deserializer::from_reader(&mut *reader)
        .into_iter::<T>()
        .next()
        .transpose()
        .map_err(|error| Error::State(format!("json: {error}")))?
        .ok_or_else(|| Error::State("missing JSON value".to_string()))?;

    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            break;
        }
        let consumed = available
            .iter()
            .take_while(|byte| byte.is_ascii_whitespace())
            .count();
        if consumed == 0 {
            break;
        }
        reader.consume(consumed);
    }
    let end = reader.stream_position()?;
    Ok(Some((start, end, value)))
}

pub(in crate::session::store) fn read_session_event<R>(
    reader: &mut std::io::BufReader<R>,
) -> Result<Option<(u64, u64, SessionEvent)>>
where
    R: std::io::Read + std::io::Seek,
{
    // Seek is retained in the bound because callers operate on seekable
    // files; the read itself only advances the cursor.
    read_jsonl_value::<SessionEvent, _>(reader)
}

pub(in crate::session::store) fn count_newlines(reader: &mut impl std::io::Read) -> Result<usize> {
    let mut count = 0usize;
    let mut buf = [0u8; 16 * 1024];
    loop {
        let read = reader.read(&mut buf)?;
        if read == 0 {
            return Ok(count);
        }
        count = count.saturating_add(memchr::memchr_iter(b'\n', &buf[..read]).count());
    }
}

pub(in crate::session::store) fn malformed_record_is_final<R>(
    reader: &mut std::io::BufReader<R>,
    start: u64,
) -> Result<bool>
where
    R: std::io::Read + std::io::Seek,
{
    use std::io::{Read, Seek};

    reader.seek(std::io::SeekFrom::Start(start))?;
    let mut after_line = false;
    let mut buf = [0u8; 16 * 1024];
    loop {
        let read = reader.read(&mut buf)?;
        if read == 0 {
            return Ok(true);
        }
        for &byte in &buf[..read] {
            if after_line && !byte.is_ascii_whitespace() {
                return Ok(false);
            }
            after_line |= byte == b'\n';
        }
    }
}

pub(in crate::session::store) struct LinearSessionIndex {
    pub meta: SessionMeta,
    pub entries: Vec<SessionIndexEntry>,
    pub file_size: u64,
    pub leaf_id: Option<String>,
    pub history_start: usize,
    pub compaction_suffix: Vec<EventIndex>,
}

enum TrailingCursor {
    Absent,
    DetachedRoot,
    Leaf(IndexId),
}

/// Build the normal resume projection in a single sequential pass without
/// retaining the transcript graph. Returns `None` when the selected lineage is
/// not the physical event prefix; callers then use the general graph loader for
/// branch semantics.
pub(in crate::session::store) fn load_linear_session_index(
    path: &Path,
) -> Result<Option<LinearSessionIndex>> {
    use std::io::{BufReader, Seek};

    let mut reader = BufReader::with_capacity(64 * 1024, std::fs::File::open(path)?);
    let Some((_start, event_start, header)) = read_jsonl_value::<Header, _>(&mut reader)? else {
        return Err(Error::State(format!(
            "session file has no header: {}",
            path.display()
        )));
    };
    if header.meta.version != SESSION_VERSION {
        return Err(Error::State(format!(
            "unsupported session version {} in {}",
            header.meta.version,
            path.display()
        )));
    }

    let mut entries: Vec<SessionIndexEntry> = Vec::new();
    let mut ids: Vec<IndexId> = Vec::new();
    let mut latest_cursor = TrailingCursor::Absent;
    let mut history_start = 0usize;
    let mut file_size = event_start;
    let mut line = Vec::with_capacity(4 * 1024);

    loop {
        let record_start = reader.stream_position()?;
        let event = read_index_skeleton(&mut reader, &mut line);
        let Some((line_start, line_end, skel)) = (match event {
            Ok(event) => event,
            Err(error) => {
                if malformed_record_is_final(&mut reader, record_start)? {
                    break;
                }
                return Err(Error::State(format!(
                    "parse index event in {}: {error}",
                    path.display()
                )));
            }
        }) else {
            break;
        };
        file_size = line_end;
        let kind = skel.kind;
        if kind == IndexKind::Cursor {
            latest_cursor = match skel.cursor_leaf {
                Some(leaf) => TrailingCursor::Leaf(leaf),
                None => TrailingCursor::DetachedRoot,
            };
            continue;
        }
        latest_cursor = TrailingCursor::Absent;
        let linked = match (&skel.parent_id, ids.last()) {
            (None, None) => true,
            (Some(parent), Some(previous)) => parent == previous,
            _ => false,
        };
        if !linked {
            return Ok(None);
        }
        let position = entries.len();
        if kind == IndexKind::Compaction {
            let start = if skel.first_kept_entry_id.is_empty() {
                position
            } else {
                ids.iter()
                    .position(|id| *id == skel.first_kept_entry_id)
                    .unwrap_or(position)
            };
            history_start = start;
            if skel.checkpointed_tail && start < position {
                entries[start..]
                    .iter_mut()
                    .for_each(|entry: &mut SessionIndexEntry| entry.visible = false);
            }
        }
        entries.push(SessionIndexEntry {
            offset: line_start,
            end_offset: line_end,
            kind,
            visible: true,
        });
        ids.push(skel.id);
    }

    Ok(finish_linear_session_index(
        header.meta,
        entries,
        ids,
        latest_cursor,
        history_start,
        file_size,
    ))
}

fn finish_linear_session_index(
    meta: SessionMeta,
    entries: Vec<SessionIndexEntry>,
    ids: Vec<IndexId>,
    latest_cursor: TrailingCursor,
    history_start: usize,
    file_size: u64,
) -> Option<LinearSessionIndex> {
    let leaf = match latest_cursor {
        TrailingCursor::DetachedRoot => {
            return Some(LinearSessionIndex {
                meta,
                entries: Vec::new(),
                file_size,
                leaf_id: None,
                history_start: 0,
                compaction_suffix: Vec::new(),
            });
        }
        TrailingCursor::Leaf(leaf) if ids.last() == Some(&leaf) => Some(leaf),
        TrailingCursor::Absent => ids.last().cloned(),
        TrailingCursor::Leaf(_) => return None,
    };

    let Some(leaf) = leaf else {
        return Some(LinearSessionIndex {
            meta,
            entries: Vec::new(),
            file_size,
            leaf_id: None,
            history_start: 0,
            compaction_suffix: Vec::new(),
        });
    };

    let compaction_suffix: Vec<EventIndex> = (history_start..entries.len())
        .map(|idx| {
            let entry = entries[idx];
            EventIndex {
                id: ids[idx].clone(),
                parent_id: idx.checked_sub(1).map(|prev| ids[prev].clone()),
                offset: entry.offset,
                end_offset: entry.end_offset,
                kind: entry.kind,
            }
        })
        .collect();
    drop(ids);

    Some(LinearSessionIndex {
        meta,
        entries,
        file_size,
        leaf_id: Some(leaf.to_event_id()),
        history_start,
        compaction_suffix,
    })
}
