//! Positional session-tree index.
//!
//! /tree needs the whole event graph, but that graph is a nearly linear
//! append-only chain: each event names its parent, and the parent is almost
//! always the previous event. Keeping the graph as positions holds one array
//! slot per event instead of a struct plus a heap `Vec` of children per
//! parent, which is what made the picker's peak grow with the transcript.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use lofi_error::{Error, Result};

#[allow(clippy::wildcard_imports)]
use super::*;

/// Parent slot of an event whose parent is absent or unresolvable.
pub const NO_PARENT: u32 = u32::MAX;

/// A compaction checkpoint resolved against the index: the marker position and
/// the position of the first event it keeps. `first_kept` is absent when the
/// marker names no kept event or that event is not in this index.
#[derive(Debug, Clone, Copy)]
pub struct TreeCheckpoint {
    pub marker: u32,
    /// The marker names a kept event. `first_kept` is absent when the named
    /// event is not in the index, so the two are not interchangeable.
    pub names_first_kept: bool,
    pub first_kept: Option<u32>,
    pub checkpointed_tail: bool,
}

/// Whole-tree event graph in positional form.
///
/// Positions are dense event indices in file order with cursor records
/// excluded. Each parent's children keep the physical push order the walkers
/// rely on. A parent normally sits before its child, but a forward or
/// duplicate id link can resolve to a later position; nothing depends on the
/// order.
#[derive(Debug, Clone, Default)]
pub struct TreeIndex {
    offset: Vec<u64>,
    kind: Vec<IndexKind>,
    parent: Vec<u32>,
    id: Vec<IndexId>,
    child_start: Vec<u32>,
    child: Vec<u32>,
    checkpoints: Vec<TreeCheckpoint>,
}

impl TreeIndex {
    #[must_use]
    pub fn len(&self) -> usize {
        self.offset.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.offset.is_empty()
    }

    #[must_use]
    pub fn offset(&self, position: usize) -> u64 {
        self.offset[position]
    }

    #[must_use]
    pub fn kind(&self, position: usize) -> IndexKind {
        self.kind[position]
    }

    #[must_use]
    pub fn id(&self, position: usize) -> &IndexId {
        &self.id[position]
    }

    #[must_use]
    pub fn parent(&self, position: usize) -> Option<usize> {
        let parent = self.parent[position];
        (parent != NO_PARENT).then_some(parent as usize)
    }

    /// Children of `position` in physical order.
    #[must_use]
    pub fn children(&self, position: usize) -> &[u32] {
        let start = self.child_start[position] as usize;
        let end = self.child_start[position + 1] as usize;
        &self.child[start..end]
    }

    #[must_use]
    pub fn checkpoints(&self) -> &[TreeCheckpoint] {
        &self.checkpoints
    }

    /// Position of the event carrying `id`. Later duplicates win. Callers use
    /// it for a head or a checkpoint target, never per row, so the scan stays
    /// off the hot path.
    #[must_use]
    pub fn position_of(&self, id: &str) -> Option<usize> {
        self.id.iter().rposition(|candidate| candidate.matches(id))
    }

    /// Root-first positions of the selected lineage.
    ///
    /// # Errors
    /// Returns an error for a cyclic graph.
    pub fn active_path(&self, leaf: Option<u32>) -> Result<Vec<u32>> {
        let Some(leaf) = leaf else {
            return Ok(Vec::new());
        };
        let mut selected = Vec::new();
        let mut current = Some(leaf as usize);
        // A cyclic index would otherwise loop forever; cap the walk at the
        // number of events so a malformed transcript terminates.
        while let Some(position) = current {
            if selected.len() > self.len() {
                break;
            }
            selected.push(position as u32);
            current = self.parent(position);
        }
        if selected.len() > self.len() {
            return Err(Error::State("cycle in session event lineage".to_string()));
        }
        selected.reverse();
        Ok(selected)
    }

    /// Position of the event carrying `id`, or an error naming a missing head.
    ///
    /// # Errors
    /// Returns an error when `id` is present but no event carries it.
    pub fn resolve_leaf(&self, id: Option<&str>) -> Result<Option<u32>> {
        let Some(id) = id.filter(|id| !id.is_empty()) else {
            return Ok(None);
        };
        self.position_of(id)
            .map(|position| Some(position as u32))
            .ok_or_else(|| Error::State("session branch leaf not found".to_string()))
    }
}

/// Accumulates the positional arrays in one pass over the transcript.
#[derive(Default)]
struct TreeIndexBuilder {
    offset: Vec<u64>,
    kind: Vec<IndexKind>,
    parent: Vec<u32>,
    id: Vec<IndexId>,
    checkpoints: Vec<TreeCheckpoint>,
    /// Parent ids that are not the previous event, resolved after the pass.
    pending_parent: Vec<(u32, IndexId)>,
    /// Checkpoint slots whose kept-event id still needs a position.
    pending_kept: Vec<(u32, IndexId)>,
    end_offset: u64,
}

impl TreeIndexBuilder {
    fn push(&mut self, line_start: u64, line_end: u64, skel: IndexedSkeleton) {
        self.end_offset = line_end;
        if skel.kind == IndexKind::Cursor {
            // A cursor record carries no id and no parent, so a position for
            // it would become a bogus root and shift every row's event.
            return;
        }
        let position = self.offset.len() as u32;
        let parent = match skel.parent_id.filter(|parent| !parent.is_empty()) {
            // Nearly every event is appended directly after its parent.
            Some(parent) if position > 0 && self.id[(position - 1) as usize] == parent => {
                position - 1
            }
            Some(parent) => {
                self.pending_parent.push((position, parent));
                NO_PARENT
            }
            None => NO_PARENT,
        };
        if skel.kind == IndexKind::Compaction {
            let slot = self.checkpoints.len() as u32;
            let names_first_kept = !skel.first_kept_entry_id.is_empty();
            if names_first_kept {
                self.pending_kept
                    .push((slot, skel.first_kept_entry_id.clone()));
            }
            self.checkpoints.push(TreeCheckpoint {
                marker: position,
                names_first_kept,
                first_kept: None,
                checkpointed_tail: skel.checkpointed_tail,
            });
        }
        self.offset.push(line_start);
        self.kind.push(skel.kind);
        self.parent.push(parent);
        self.id.push(skel.id);
    }

    fn finish(mut self) -> TreeIndex {
        self.resolve_pending();
        let (child_start, child) = children_csr(&self.parent);
        TreeIndex {
            offset: self.offset,
            kind: self.kind,
            parent: self.parent,
            id: self.id,
            child_start,
            child,
            checkpoints: self.checkpoints,
        }
    }

    /// Resolve the parent and kept-event ids that are not the previous event
    /// with one scan over the ids and one small table, so a graph with a few
    /// cross links never pays for a whole-tree id map.
    fn resolve_pending(&mut self) {
        if self.pending_parent.is_empty() && self.pending_kept.is_empty() {
            return;
        }
        let mut resolved: HashMap<IndexId, u32> = HashMap::new();
        {
            let mut wanted: HashSet<&IndexId> =
                HashSet::with_capacity(self.pending_parent.len() + self.pending_kept.len());
            wanted.extend(self.pending_parent.iter().map(|(_, id)| id));
            wanted.extend(self.pending_kept.iter().map(|(_, id)| id));
            for (position, id) in self.id.iter().enumerate() {
                if wanted.contains(id) {
                    // Later duplicates win. The previous-event link in `push`
                    // is tried first, so a duplicate id keeps that earlier
                    // position when the parent names its predecessor.
                    resolved.insert(id.clone(), position as u32);
                }
            }
        }
        for (position, parent) in self.pending_parent.drain(..) {
            if let Some(&resolved) = resolved.get(&parent) {
                self.parent[position as usize] = resolved;
            }
        }
        for (slot, kept) in self.pending_kept.drain(..) {
            self.checkpoints[slot as usize].first_kept = resolved.get(&kept).copied();
        }
    }
}

fn children_csr(parent: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut child_start = vec![0u32; parent.len() + 1];
    for &slot in parent {
        if slot != NO_PARENT {
            child_start[slot as usize + 1] += 1;
        }
    }
    for position in 0..parent.len() {
        child_start[position + 1] += child_start[position];
    }
    let mut cursor: Vec<u32> = child_start.iter().copied().take(parent.len()).collect();
    let mut child = vec![0u32; child_start[parent.len()] as usize];
    // Ascending positions fill each parent's slice in physical order.
    for (position, &slot) in parent.iter().enumerate() {
        if slot != NO_PARENT {
            let next = &mut cursor[slot as usize];
            child[*next as usize] = position as u32;
            *next += 1;
        }
    }
    (child_start, child)
}

/// Index the whole transcript tree. One skeleton pass, no newline pre-pass:
/// the arrays grow amortized and the parent links resolve from the pass itself.
///
/// # Errors
/// Returns an error when the header is missing or unsupported, the file cannot
/// be read, or a record before the final one is malformed.
pub(in crate::session::store) fn load_tree_index(
    path: &Path,
) -> Result<(SessionMeta, TreeIndex, u64)> {
    use std::io::{BufReader, Seek};

    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let Some((_header_start, _header_end, header)) = read_jsonl_value::<Header, _>(&mut reader)?
    else {
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
    let mut builder = TreeIndexBuilder::default();
    let mut line = Vec::with_capacity(4 * 1024);
    loop {
        let record_start = reader.stream_position()?;
        let event = read_index_skeleton(&mut reader, &mut line);
        let Some((line_start, line_end, skel)) = (match event {
            Ok(event) => event,
            Err(error) => {
                // Only the final malformed record is ignored. A process can
                // die during a write, so the committed prefix must remain
                // resumable. Earlier malformed records remain errors.
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
        builder.push(line_start, line_end, skel);
    }
    let file_size = builder.end_offset;
    Ok((header.meta, builder.finish(), file_size))
}
