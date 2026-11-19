#![allow(clippy::wildcard_imports)]

use super::*;

/// Bounds the window measure loop in `measure_visible_turns`. Each pass either
/// measures a turn (a strictly smaller set of estimates) or stops, so the loop
/// converges; the bound only protects a frame from a pathological rewrap that
/// keeps swapping one turn for another.
const MAX_WINDOW_MEASURE_PASSES: usize = 4;

/// Compaction summary attached after a turn's committed byte range so the
/// turn can release its uncompressed live blocks without losing the marker.
#[derive(Debug, Clone)]
pub(super) struct CompactionMarker {
    pub(super) after_byte: u64,
    pub(super) summarized: usize,
    pub(super) kept: usize,
    pub(super) summary: String,
}

/// Retains compressed, width-independent display models so scrolling back
/// into a resumed transcript does not rescan its JSONL file. Successful
/// native results are reduced to their collapsed previews before insertion;
/// per-turn compression avoids retaining the full Rust object graph. Entries
/// are evicted least-recently-used past a hard budget, viewport-driven
/// materializations being the only inserts — background height re-measure
/// parses without caching, so the cache stays proportional to what the user
/// has actually looked at, not to the whole transcript.
pub(super) struct CollapsedTurnCache {
    pub(super) map: HashMap<usize, (u64, Box<[u8]>)>,
    /// Monotonic access-stamp index: smallest stamp is least recently used.
    pub(super) order: std::collections::BTreeMap<u64, usize>,
    /// Hot uncompressed turns (at most 2) so measure + render in the same
    /// frame or scrolling inside a tall turn avoids repeated LZ4 + JSON parses.
    pub(super) hot: VecDeque<(usize, Arc<Turn>)>,
    next_stamp: u64,
    pub(super) retained_bytes: usize,
    pub(super) frame_hits: usize,
    pub(super) frame_misses: usize,
    pub(super) frame_materialize_us: u128,
}

impl CollapsedTurnCache {
    const MAX_RETAINED_BYTES: usize = 8 * 1024 * 1024;
    /// A single entry above this size is never retained; it reparses from the
    /// transcript on demand instead of flushing everyone else.
    const MAX_ENTRY_BYTES: usize = 1024 * 1024;
    const MAX_HOT_TURNS: usize = 2;

    pub(super) fn new() -> Self {
        Self {
            map: HashMap::new(),
            order: std::collections::BTreeMap::new(),
            hot: VecDeque::with_capacity(Self::MAX_HOT_TURNS),
            next_stamp: 0,
            retained_bytes: 0,
            frame_hits: 0,
            frame_misses: 0,
            frame_materialize_us: 0,
        }
    }

    pub(super) fn remember_hot(&mut self, idx: usize, turn: Arc<Turn>) {
        self.hot.retain(|(existing, _)| *existing != idx);
        if self.hot.len() >= Self::MAX_HOT_TURNS {
            self.hot.pop_front();
        }
        self.hot.push_back((idx, turn));
    }

    pub(super) fn get(&mut self, idx: usize) -> Option<Arc<Turn>> {
        if let Some(pos) = self.hot.iter().position(|(existing, _)| *existing == idx) {
            if let Some(entry) = self.hot.remove(pos) {
                let turn = Arc::clone(&entry.1);
                self.hot.push_back(entry);
                if let Some((stamp, _)) = self.map.get_mut(&idx) {
                    let old = *stamp;
                    let next = self.next_stamp;
                    self.next_stamp = next.wrapping_add(1);
                    *stamp = next;
                    self.order.remove(&old);
                    self.order.insert(next, idx);
                }
                self.frame_hits += 1;
                return Some(turn);
            }
        }
        let mut hit_stamp = None;
        let turn = self.map.get_mut(&idx).and_then(|(stamp, data)| {
            lz4_flex::decompress_size_prepended(data)
                .ok()
                .and_then(|json| serde_json::from_slice(&json).ok())
                .map(|mut turn| {
                    assign_detail_ids(&mut turn, idx);
                    let old = *stamp;
                    let next = self.next_stamp;
                    self.next_stamp = next.wrapping_add(1);
                    *stamp = next;
                    hit_stamp = Some((old, next));
                    Arc::new(turn)
                })
        });
        if let Some((old, next)) = hit_stamp {
            self.frame_hits += 1;
            self.order.remove(&old);
            self.order.insert(next, idx);
            if let Some(turn) = &turn {
                self.remember_hot(idx, Arc::clone(turn));
            }
        } else {
            self.frame_misses += 1;
        }
        turn
    }

    pub(super) fn finish_materialize(&mut self, elapsed_us: u128) {
        self.frame_materialize_us += elapsed_us;
    }

    pub(super) fn reset_frame_profile(&mut self) {
        self.frame_hits = 0;
        self.frame_misses = 0;
        self.frame_materialize_us = 0;
    }

    pub(super) fn insert(&mut self, idx: usize, turn: &Turn) {
        if self.map.contains_key(&idx) {
            return;
        }
        let Ok(json) = serde_json::to_vec(turn) else {
            return;
        };
        let data = lz4_flex::compress_prepend_size(&json).into_boxed_slice();
        let bytes = data.len();
        if bytes > Self::MAX_ENTRY_BYTES {
            return;
        }
        let stamp = self.next_stamp;
        self.next_stamp = stamp.wrapping_add(1);
        self.retained_bytes += bytes;
        self.order.insert(stamp, idx);
        self.map.insert(idx, (stamp, data));
        while self.retained_bytes > Self::MAX_RETAINED_BYTES {
            let Some((_, victim)) = self.order.pop_first() else {
                break;
            };
            if let Some((_, data)) = self.map.remove(&victim) {
                self.retained_bytes = self.retained_bytes.saturating_sub(data.len());
            }
            self.hot.retain(|(existing, _)| *existing != victim);
        }
    }

    pub(super) fn remove(&mut self, idx: usize) {
        if let Some((stamp, data)) = self.map.remove(&idx) {
            self.retained_bytes = self.retained_bytes.saturating_sub(data.len());
            self.order.remove(&stamp);
        }
        self.hot.retain(|(existing, _)| *existing != idx);
    }

    pub(super) fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
        self.hot.clear();
        self.next_stamp = 0;
        self.retained_bytes = 0;
        self.reset_frame_profile();
    }
}

/// Retains only viewport-adjacent rendering to avoid holding the styled
/// representation of the full session.
pub(super) struct FrozenCache {
    pub(super) order: VecDeque<usize>,
    pub(super) map: HashMap<usize, Vec<view::RenderLine>>,
    /// The visible slice of a turn too tall to keep in `map`. Rendering one
    /// window walks the whole turn, and a tall turn stays on screen for many
    /// frames, so the slice is worth keeping.
    pub(super) windows: HashMap<usize, (std::ops::Range<usize>, Vec<view::RenderLine>)>,
}

impl FrozenCache {
    pub(super) fn new() -> Self {
        Self {
            order: VecDeque::new(),
            map: HashMap::new(),
            windows: HashMap::new(),
        }
    }

    pub(super) fn contains(&self, idx: usize) -> bool {
        self.map.contains_key(&idx)
    }

    pub(super) fn get(&self, idx: usize) -> Option<&Vec<view::RenderLine>> {
        self.map.get(&idx)
    }

    pub(super) fn clear(&mut self) {
        self.order.clear();
        self.map.clear();
        self.windows.clear();
    }

    pub(super) fn window(
        &self,
        idx: usize,
        range: std::ops::Range<usize>,
    ) -> Option<&Vec<view::RenderLine>> {
        self.windows
            .get(&idx)
            .and_then(|(cached, lines)| (*cached == range).then_some(lines))
    }

    pub(super) fn insert_window(
        &mut self,
        idx: usize,
        range: std::ops::Range<usize>,
        lines: Vec<view::RenderLine>,
    ) {
        self.windows.insert(idx, (range, lines));
    }

    pub(super) fn insert(&mut self, idx: usize, entry: Vec<view::RenderLine>) {
        if self.map.insert(idx, entry).is_none() {
            self.order.push_back(idx);
        }
    }

    pub(super) fn retain_near(&mut self, visible: Option<(usize, usize)>, frozen_turns: usize) {
        let Some((first, last)) = visible else {
            self.clear();
            return;
        };
        let first = first.saturating_sub(1);
        let last = last.saturating_add(1).min(frozen_turns.saturating_sub(1));
        self.map.retain(|idx, _| (first..=last).contains(idx));
        self.order.retain(|idx| (first..=last).contains(idx));
        self.windows.retain(|idx, _| (first..=last).contains(idx));
    }
}

pub(super) struct Transcript {
    pub(super) turns: Vec<Turn>,
    /// Per-turn byte range in the transcript file (`None` for ephemeral or
    /// not-yet-committed turns). Parallel to `turns`. A frozen turn's blocks
    /// are cleared once committed; `materialize_turn` re-parses this range
    /// from the file on demand so the UI's memory stays bounded by the
    /// viewport, not the session length.
    pub(super) turn_byte_ranges: Vec<Option<(u64, u64)>>,
    pub(super) turn_event_offsets: Vec<Option<Vec<u64>>>,
    pub(super) turn_compactions: Vec<Vec<CompactionMarker>>,
    pub(super) collapsed_turns: Box<RefCell<CollapsedTurnCache>>,
    /// Frozen turns whose blocks wait for the frame that renders them from
    /// memory. Dropping them at freeze time made that frame re-read the turn
    /// from the transcript, which stalls a submit on a long previous turn.
    pending_release: Vec<usize>,
    /// First durable-transcript read failure seen while rendering. Set from
    /// interior-mutability paths that only hold `&App`; surfaced as a
    /// notification after the frame so a turn silently rendering as an
    /// empty shell gets a visible explanation instead of looking like
    /// lost transcript.
    pub(super) alert: RefCell<Option<String>>,
}

const MAX_CACHED_LIVE_BLOCK_ROWS: usize = 512;
const MAX_EAGER_LIVE_BLOCK_BYTES: usize = 32 * 1024;

#[derive(Default)]
pub(super) struct CachedLiveChild {
    pub(super) height_stamp: u64,
    pub(super) lines_stamp: u64,
    pub(super) height: usize,
    pub(super) lines: Option<Vec<view::RenderLine>>,
    pub(super) window: Option<(u64, std::ops::Range<usize>, Vec<view::RenderLine>)>,
}

#[derive(Default)]
pub(super) struct LiveTurnCache {
    pub(super) valid: bool,
    pub(super) turn_idx: usize,
    pub(super) durable: bool,
    pub(super) epoch: u64,
    pub(super) width: usize,
    pub(super) theme: Theme,
    pub(super) detail_stamp: u64,
    pub(super) prompt: Option<CachedLiveChild>,
    pub(super) blocks: Vec<CachedLiveChild>,
}

impl LiveTurnCache {
    pub(super) fn clear(&mut self) {
        self.valid = false;
        self.prompt = None;
        self.blocks.clear();
    }

    #[allow(clippy::too_many_arguments)]
    fn sync_envelope(
        &mut self,
        turn_idx: usize,
        durable: bool,
        epoch: u64,
        width: usize,
        theme: Theme,
        detail_stamp: u64,
        blocks_len: usize,
        has_prompt: bool,
    ) {
        if !self.valid
            || self.turn_idx != turn_idx
            || self.durable != durable
            || self.epoch != epoch
            || self.width != width
            || self.theme != theme
            || self.detail_stamp != detail_stamp
        {
            self.valid = true;
            self.turn_idx = turn_idx;
            self.durable = durable;
            self.epoch = epoch;
            self.width = width;
            self.theme = theme;
            self.detail_stamp = detail_stamp;
            self.prompt = None;
            self.blocks.clear();
        }
        if !has_prompt {
            self.prompt = None;
        }
        if self.blocks.len() > blocks_len {
            self.blocks.truncate(blocks_len);
        }
    }
}

pub(super) struct TranscriptLayout {
    pub(super) frozen_render: FrozenCache,
    pub(super) live_render: LiveTurnCache,
    /// Line count per frozen turn (all of them), so the viewport can be
    /// located and `total` computed without fetching rendered lines. Synced
    /// to the file-backed prefix (which may be all turns) for the active mode.
    pub(super) frozen_heights: Vec<usize>,
    /// Parallel to `frozen_heights`: true while the entry is a placeholder
    /// estimate derived from the turn byte range, still waiting for an exact
    /// re-measure by the tick loop or the viewport pass. Resumed transcripts
    /// seed estimates for every historical turn so the first frame paints
    /// without re-reading the whole session file.
    pub(super) frozen_heights_estimated: Vec<bool>,
    pub(super) render_epoch: u64,
    pub(super) frozen_epoch: u64,
    /// Viewport width the frozen cache was last built at. A resize changes
    /// the wrap width, so a mismatch discards the cache just like an epoch
    /// bump — otherwise background-padded lines keep the old (narrower)
    /// width after the terminal grows.
    pub(super) frozen_width: usize,
    /// After a width-only resize, frozen turns below this exclusive upper
    /// bound still hold heights measured at the previous width. `ensure_frozen`
    /// keeps those stale heights as the total/base source (they only differ by
    /// re-wrap, so the thumb and bottom anchor stay stable); the tick loop
    /// re-measures them back-to-front — visible bottom turns first — so a
    /// resize never stalls a frame on a long transcript. `None` once every
    /// height matches `frozen_width`.
    pub(super) height_remeasure_from: Option<usize>,
}

impl Transcript {
    pub(super) fn new() -> Self {
        Self {
            turns: Vec::new(),
            turn_byte_ranges: Vec::new(),
            turn_event_offsets: Vec::new(),
            turn_compactions: Vec::new(),
            collapsed_turns: Box::new(RefCell::new(CollapsedTurnCache::new())),
            pending_release: Vec::new(),
            alert: RefCell::new(None),
        }
    }
}

impl TranscriptLayout {
    pub(super) fn invalidate_turn(&mut self, turn: usize) {
        self.frozen_render.clear();
        if self.live_render.valid && self.live_render.turn_idx == turn {
            self.live_render.clear();
        }
        if let Some(height) = self.frozen_heights_estimated.get_mut(turn) {
            *height = true;
            self.height_remeasure_from = Some(
                self.height_remeasure_from
                    .unwrap_or_default()
                    .max(turn.saturating_add(1)),
            );
        }
    }

    pub(super) fn new() -> Self {
        Self {
            frozen_render: FrozenCache::new(),
            live_render: LiveTurnCache::default(),
            frozen_heights: Vec::new(),
            frozen_heights_estimated: Vec::new(),
            render_epoch: 0,
            frozen_epoch: 0,
            frozen_width: 0,
            height_remeasure_from: None,
        }
    }
}
impl Transcript {
    pub(super) fn freeze_previous_file_backed_turn(&mut self) {
        let Some(previous) = self.turns.len().checked_sub(1) else {
            return;
        };
        if self
            .turn_byte_ranges
            .get(previous)
            .is_some_and(Option::is_some)
            && !self.turns[previous].blocks.is_empty()
        {
            self.pending_release.push(previous);
        }
    }

    /// Release the blocks kept for the frame that froze a turn. The frame has
    /// already rendered it, so it holds the cached lines and height it needs.
    pub(super) fn release_pending_blocks(&mut self) {
        if self.pending_release.is_empty() {
            return;
        }
        for idx in std::mem::take(&mut self.pending_release) {
            if let Some(turn) = self.turns.get_mut(idx) {
                turn.blocks.clear();
            }
        }
    }

    pub(super) fn merge_turn_range(&mut self, idx: usize, byte_start: u64, byte_end: u64) {
        let Some(range) = self.turn_byte_ranges.get_mut(idx) else {
            return;
        };
        let merged = match *range {
            Some((start, end)) => (start.min(byte_start), end.max(byte_end)),
            None => (byte_start, byte_end),
        };
        if *range == Some(merged) {
            return;
        }
        *range = Some(merged);
        if let Some(markers) = self.turn_compactions.get_mut(idx) {
            markers.retain(|marker| marker.after_byte >= merged.1);
        }
        // The turn grew on disk. A materialization taken from the earlier,
        // shorter range holds only the rows committed by then, so a live turn
        // that had no blocks yet would keep rendering that stale shell.
        self.collapsed_turns.get_mut().remove(idx);
    }

    pub(super) fn push_turn(&mut self, turn: Turn) {
        self.turns.push(turn);
        self.turn_byte_ranges.push(None);
        self.turn_event_offsets.push(None);
        self.turn_compactions.push(Vec::new());
    }

    pub(super) fn insert_turn(&mut self, idx: usize, turn: Turn) {
        self.collapsed_turns.get_mut().clear();
        self.pending_release.clear();
        self.turns.insert(idx, turn);
        self.turn_byte_ranges.insert(idx, None);
        self.turn_event_offsets.insert(idx, None);
        self.turn_compactions.insert(idx, Vec::new());
    }

    pub(super) fn restore_last_turn(
        &mut self,
        start: u64,
        end: u64,
        offsets: Vec<u64>,
        contiguous: bool,
    ) {
        if let Some(range) = self.turn_byte_ranges.last_mut() {
            *range = Some((start, end));
        }
        if let Some(event_offsets) = self.turn_event_offsets.last_mut() {
            if contiguous {
                *event_offsets = None;
            } else {
                *event_offsets = Some(offsets);
            }
        }
        if let Some(shell_turn) = self.turns.last_mut() {
            shell_turn.blocks.clear();
        }
    }

    pub(super) fn clear(&mut self) {
        self.turns.clear();
        self.collapsed_turns.get_mut().clear();
        self.pending_release.clear();
        self.turn_byte_ranges.clear();
        self.turn_event_offsets.clear();
        self.turn_compactions.clear();
    }
}

impl App {
    pub(super) fn take_transcript_alert(&mut self) -> Option<String> {
        self.transcript.alert.get_mut().take()
    }

    pub(super) fn bump_render_epoch(&mut self) {
        self.layout.render_epoch = self.layout.render_epoch.wrapping_add(1);
        // Wholesale content replacement makes both the drawn scrollbar extent
        // and the viewport's content anchor meaningless: adopt the next
        // frame's.
        self.viewport.scroll_gauge = None;
        self.viewport.scroll_gauge_stamp = None;
        self.viewport.view_anchor = None;
        self.viewport.nav_anchor = None;
        self.viewport.sel_anchor = None;
    }

    /// Scrollbar geometry to draw: the offset and extent of the last frame
    /// whose extent was final. While estimated heights are still converging the
    /// total moves by an order of magnitude, so a live one makes the thumb
    /// slide and stretch with no input; the final extent is adopted once the
    /// last estimate is gone. A frame whose content changed is adopted at once,
    /// because that extent moved for real: holding it hid a submitted prompt
    /// from the thumb until the last estimate converged.
    pub(super) fn scrollbar_gauge(&mut self, off: usize, total: usize) -> (usize, usize) {
        let stamp = (self.transcript.turns.len(), self.viewport.last_turn_height);
        if self.viewport.scroll_gauge_stamp != Some(stamp)
            || self.layout.height_remeasure_from.is_none()
        {
            self.viewport.scroll_gauge = Some((off, total));
            self.viewport.scroll_gauge_stamp = Some(stamp);
        }
        self.viewport.scroll_gauge.unwrap_or((off, total))
    }

    pub(super) fn turn_has_expanded_detail(&self, idx: usize) -> bool {
        self.expanded_details
            .values()
            .any(|state| state.turn == idx)
    }

    /// True when the turn's in-memory blocks dropped released exec results
    /// but the committed transcript range can restore them. Expanded details
    /// need that durable content.
    pub(super) fn needs_durable_materialize(&self, idx: usize) -> bool {
        // A committed turn with no live blocks renders from the transcript, not
        // from the empty shell. Freezing the turn ahead of the next, or a turn
        // whose blocks were released, leaves exactly this state; rendering the
        // shell collapsed the turn to its prompt row.
        let committed = self
            .transcript
            .turn_byte_ranges
            .get(idx)
            .is_some_and(Option::is_some);
        committed
            && (self.turn_has_expanded_detail(idx)
                || self
                    .transcript
                    .turns
                    .get(idx)
                    .is_none_or(|turn| turn.blocks.is_empty()))
    }

    /// Reconstruct a frozen turn's blocks. If the turn still holds its blocks
    /// in memory (ephemeral session, or not yet frozen), clone them. Otherwise
    /// re-parse the turn's byte range from the transcript file. On any read or
    /// parse failure the turn's prompt is preserved with empty blocks.
    pub(super) fn materialize_turn(&self, idx: usize) -> Arc<Turn> {
        self.transcript.materialize_turn_caching(
            idx,
            true,
            self.turn_has_expanded_detail(idx),
            self.session.cursor.as_ref(),
        )
    }

    /// Materialize for a throwaway purpose (height measurement) that must not
    /// claim a cache slot: background re-measure walks every resumed turn, and
    /// caching its results would retain the whole transcript.
    fn peek_transcript_turn(&self, idx: usize) -> Arc<Turn> {
        self.transcript.materialize_turn_caching(
            idx,
            false,
            self.turn_has_expanded_detail(idx),
            self.session.cursor.as_ref(),
        )
    }

    pub(super) fn ensure_frozen_turn(&mut self, idx: usize, width: usize) {
        const MAX_CACHED_TURN_ROWS: usize = 4096;
        if self.layout.frozen_render.contains(idx)
            || self
                .layout
                .frozen_heights
                .get(idx)
                .copied()
                .unwrap_or(usize::MAX)
                > MAX_CACHED_TURN_ROWS
        {
            return;
        }
        let theme = self.theme;
        let turn = self.materialize_turn(idx);
        let lines = {
            let cx = view::component::Cx {
                app: self,
                theme,
                width,
                active_turn: false,
            };
            view::blocks::render_turn_lines(&cx, &turn)
        };
        self.layout.frozen_render.insert(idx, lines);
    }

    pub(super) fn frozen_turn_window(
        &self,
        idx: usize,
        width: usize,
        range: std::ops::Range<usize>,
    ) -> Vec<view::RenderLine> {
        let theme = self.theme;
        let turn = self.materialize_turn(idx);
        let cx = view::component::Cx {
            app: self,
            theme,
            width,
            active_turn: false,
        };
        view::blocks::render_turn_window(&cx, &turn, range)
    }

    /// Sync the frozen-turn cache to the current `turns`. Normally frozen turns
    /// are all but the live mutable last one; an idle fully file-backed view can
    /// freeze the final turn too.
    /// On a wholesale replacement (`bump_render_epoch`) the cache is dropped;
    /// otherwise newly-superseded turns are rendered once, their height
    /// recorded permanently in `frozen_heights`. Their temporary rendered lines
    /// are dropped here; the viewport pass caches only nearby turns. Heights
    /// are kept for every frozen turn so the viewport can be located and the
    /// scroll total computed without holding all rendered lines in memory.
    /// A viewport resize (width change) also drops the cache, since wrapping
    /// and background padding depend on width.
    pub(super) fn ensure_frozen(&mut self, width: usize) {
        let epoch_changed = self.layout.frozen_epoch != self.layout.render_epoch;
        let width_changed = self.layout.frozen_width != width;
        if epoch_changed {
            // Content changed wholesale: every cached height is invalid, so
            // drop and refill below. No incremental path here — the turns
            // themselves are different.
            self.layout.frozen_render.clear();
            self.layout.live_render.clear();
            self.layout.frozen_heights.clear();
            self.layout.frozen_heights_estimated.clear();
            self.layout.frozen_epoch = self.layout.render_epoch;
            self.layout.frozen_width = width;
            self.layout.height_remeasure_from = None;
        } else if width_changed {
            // Width-only resize: the styled lines in `frozen_render` are
            // bound to the old width and must be rebuilt, but the turn
            // content is identical — only its wrap changes. Keep the stale
            // heights as the total/base source so the bottom anchor and
            // thumb stay put, and re-measure incrementally (visible window
            // now, the rest on the tick loop) instead of stalling this frame.
            self.layout.frozen_render.clear();
            self.layout.live_render.clear();
            self.layout.frozen_width = width;
            // Stale heights are estimates at the new width: the tick loop
            // re-measures them.
            for estimated in &mut self.layout.frozen_heights_estimated {
                *estimated = true;
            }
            // Re-measure from the end: the cursor is an exclusive upper
            // bound that remeasure_heights_step walks down to zero.
            self.layout.height_remeasure_from = Some(self.layout.frozen_heights.len());
        }
        let n = self.transcript.turns.len();
        // Normally the last turn is live (or retained in memory after resume),
        // so only its prefix is frozen. /tree rollback deliberately drops the
        // selected last turn's blocks too; when it is idle and file-backed,
        // include it in the frozen set so production rendering materializes it
        // through the bounded cache instead of treating an empty shell as live.
        let target = if !self.run_active()
            && self
                .transcript
                .turns
                .last()
                .is_some_and(|turn| turn.blocks.is_empty())
            && self
                .transcript
                .turn_byte_ranges
                .last()
                .is_some_and(Option::is_some)
        {
            n
        } else {
            n.saturating_sub(1)
        };
        // A from-scratch fill (empty heights) over file-backed turns seeds
        // byte-range estimates instead of measuring: exact heights for the
        // whole transcript mean re-reading the whole session file before the
        // first frame. Estimates are good enough for viewport math off screen;
        // the viewport pass measures visible turns exactly and the tick loop
        // converges the rest.
        let seed = self.layout.frozen_heights.is_empty();
        while self.layout.frozen_heights.len() < target {
            // A frozen turn must not materialize its complete styled
            // output; the viewport pass renders only rows it needs.
            let idx = self.layout.frozen_heights.len();
            let estimate = seed
                && self
                    .transcript
                    .turn_byte_ranges
                    .get(idx)
                    .copied()
                    .flatten()
                    .is_some();
            if estimate {
                self.layout
                    .frozen_heights
                    .push(self.estimate_turn_height(idx, width));
                self.layout.frozen_heights_estimated.push(true);
            } else {
                let height = self.measure_turn_height(idx, width);
                self.layout.frozen_heights.push(height);
                self.layout.frozen_heights_estimated.push(false);
            }
        }
        if self
            .layout
            .frozen_heights_estimated
            .iter()
            .any(|&estimated| estimated)
        {
            // Exact heights converge from the bottom: the remeasure cursor is
            // an exclusive upper bound walked down by the tick loop.
            if self.layout.height_remeasure_from.is_none() {
                self.layout.height_remeasure_from = Some(self.layout.frozen_heights.len());
            }
        }
        if self.layout.frozen_heights.len() > target {
            self.layout.frozen_heights.truncate(target);
            self.layout.frozen_heights_estimated.truncate(target);
            self.layout.frozen_render.map.retain(|idx, _| *idx < target);
            self.layout.frozen_render.order.retain(|idx| *idx < target);
            // A truncated pending re-measure must not run past the new end.
            if let Some(hi) = self.layout.height_remeasure_from {
                self.layout.height_remeasure_from =
                    if hi > target { Some(target) } else { Some(hi) };
            }
        }
    }

    /// Rough line-count guess for a file-backed turn. Used as a placeholder
    /// until `remeasure_heights_step` or the viewport pass replace it with the
    /// exact measured height, so it never has to be right — only cheap. Undershoot
    /// vs overshoot merely jitter off-screen scroll offsets while they converge.
    fn estimate_turn_height(&self, idx: usize, width: usize) -> usize {
        let Some((start, end)) = self.transcript.turn_byte_ranges.get(idx).copied().flatten()
        else {
            return 4;
        };
        let bytes = usize::try_from(end.saturating_sub(start)).unwrap_or(usize::MAX);
        bytes / (width.max(1) * 2) + 3
    }

    /// Render-height of one frozen turn at `width`. Shared by the eager fill
    /// in `ensure_frozen` and the incremental resize re-measure.
    fn measure_turn_height(&self, idx: usize, width: usize) -> usize {
        self.measure_turn_height_caching(idx, width, false)
    }

    fn measure_turn_height_caching(&self, idx: usize, width: usize, cache_insert: bool) -> usize {
        let theme = self.theme;
        let turn = if cache_insert {
            self.materialize_turn(idx)
        } else {
            self.peek_transcript_turn(idx)
        };
        let cx = view::component::Cx {
            app: self,
            theme,
            width,
            active_turn: false,
        };
        view::blocks::render_turn_height(&cx, &turn)
    }

    /// Re-measure up to `budget` stale frozen heights at the current
    /// `frozen_width`, advancing `height_remeasure_from`. Runs on the tick
    /// loop after a width-only resize or an estimate-seeded resume, so no
    /// single frame pays the full cost.
    /// Returns true when this step changed a height or work remains: the
    /// caller must redraw. A final batch that only finishes the sequence
    /// changes geometry too, and skipping its redraw leaves the frame showing
    /// a total that no longer matches the heights.
    pub(super) fn remeasure_heights_step(&mut self, budget: usize) -> bool {
        let next_hi = match self.layout.height_remeasure_from {
            Some(hi) => hi.min(self.layout.frozen_heights.len()),
            None => return false,
        };
        if next_hi == 0 {
            self.layout.height_remeasure_from = None;
            return false;
        }
        let width = self.layout.frozen_width;
        // Work back-to-front: the viewport is almost always pinned to the
        // bottom, so the last frozen turns are the visible ones. Measuring
        // those first makes the on-screen content exact on the first tick;
        // the off-screen prefix converges over the following ticks. Heights
        // already exact (measured by the viewport pass, or never estimated)
        // are skipped without spending budget.
        let mut idx = next_hi;
        let mut measured = 0;
        while idx > 0 && measured < budget {
            idx -= 1;
            if self.layout.frozen_heights_estimated[idx] {
                self.layout.frozen_heights[idx] = self.measure_turn_height(idx, width);
                self.layout.frozen_heights_estimated[idx] = false;
                measured += 1;
            }
        }
        if idx == 0 {
            self.layout.height_remeasure_from = None;
        } else {
            self.layout.height_remeasure_from = Some(idx);
        }
        measured > 0 || idx > 0
    }

    /// Measure the turns the window at `off` shows, plus one either side.
    /// Their stored heights drive both the rows sliced for display and the
    /// total every offset is computed from, so they must be exact even while
    /// the off-screen prefix is still estimated. Measuring shrinks or grows a
    /// turn, which moves the window, so the selection repeats until it stops
    /// changing. Returns true when a height changed.
    pub(super) fn measure_visible_turns(
        &mut self,
        off: usize,
        height: usize,
        width: usize,
    ) -> bool {
        let frozen = self.layout.frozen_heights.len();
        if frozen == 0 {
            return false;
        }
        let mut any = false;
        for _ in 0..MAX_WINDOW_MEASURE_PASSES {
            let (lo, hi) = match self.frozen_range_at(off, height) {
                Some((first, last)) => (first.saturating_sub(1), last.saturating_add(1)),
                // The window is past the stored heights: a bottom-anchored view
                // whose off-screen prefix is over-estimated. Nothing intersects
                // it, so fill the last screenful from the end — that is what the
                // user is looking at, and it must be exact however wrong the
                // prefix is.
                None => (self.tail_window_start(height), frozen - 1),
            };
            let hi = hi.min(frozen - 1);
            let mut measured = false;
            for idx in lo..=hi {
                if self.layout.frozen_heights_estimated[idx] {
                    self.layout.frozen_heights[idx] =
                        self.measure_turn_height_caching(idx, width, true);
                    self.layout.frozen_heights_estimated[idx] = false;
                    measured = true;
                }
            }
            any |= measured;
            if !measured {
                break;
            }
        }
        any
    }

    /// Retain rendered lines only around the window, and materialize them.
    /// Never measures: the geometry published for this frame must not change
    /// once it is published.
    pub(super) fn cache_frozen_lines_for_viewport(
        &mut self,
        off: usize,
        height: usize,
        width: usize,
    ) {
        let frozen = self.layout.frozen_heights.len();
        if frozen == 0 {
            self.layout.frozen_render.clear();
            return;
        }
        let visible = self.visible_frozen_range(off, height);
        self.layout.frozen_render.retain_near(Some(visible), frozen);
        let (first, last) = visible;
        let first = first.saturating_sub(1);
        let last = last.saturating_add(1).min(frozen - 1);
        for idx in first..=last {
            self.ensure_frozen_turn(idx, width);
        }
    }

    #[cfg(test)]
    pub(super) fn sync_frozen_cache_for_viewport(
        &mut self,
        off: usize,
        height: usize,
        width: usize,
    ) {
        self.measure_visible_turns(off, height, width);
        self.cache_frozen_lines_for_viewport(off, height, width);
    }

    /// Turns whose stored height intersects the window [off, off + height).
    /// None when the window is past the end of the stored heights.
    fn frozen_range_at(&self, off: usize, height: usize) -> Option<(usize, usize)> {
        let frozen = self.layout.frozen_heights.len();
        let end = off.saturating_add(height);
        let mut pos = 0usize;
        let mut visible: Option<(usize, usize)> = None;
        for idx in 0..frozen {
            let turn_end = pos.saturating_add(self.layout.frozen_heights[idx]);
            if turn_end > off && pos < end {
                visible = Some(visible.map_or((idx, idx), |(first, _)| (first, idx)));
            }
            pos = turn_end.saturating_add(1);
        }
        visible
    }

    /// Turns whose stored height intersects the window
    /// [off, off + height). Falls back to the last turn when the window is
    /// past the end.
    fn visible_frozen_range(&self, off: usize, height: usize) -> (usize, usize) {
        let frozen = self.layout.frozen_heights.len();
        self.frozen_range_at(off, height)
            .unwrap_or((frozen - 1, frozen - 1))
    }

    /// First frozen turn that covers the last `height` rows of the stored
    /// heights, at most one either side.
    fn tail_window_start(&self, height: usize) -> usize {
        let mut rows = 0usize;
        let mut idx = self.layout.frozen_heights.len();
        while idx > 0 && rows < height {
            idx -= 1;
            rows = rows
                .saturating_add(self.layout.frozen_heights[idx])
                .saturating_add(1);
        }
        idx.saturating_sub(1)
    }
}

/// Discard successful `exec` result bodies for collapsed views while recording
/// which tool ids still have durable output available on disk.
pub(in crate::tui) fn strip_collapsed_exec_results(
    events: &mut [SessionEvent],
) -> std::collections::HashSet<String> {
    let mut exec_ids = std::collections::HashSet::new();
    let mut available_exec_results = std::collections::HashSet::new();
    for event in events {
        let SessionEventKind::Message(message) = &mut event.kind else {
            continue;
        };
        if message.role == Role::Assistant {
            exec_ids.extend(message.blocks.iter().filter_map(|block| match block {
                ContentBlock::ToolUse { id, name, .. } if name == "exec" => Some(id.clone()),
                _ => None,
            }));
        }
        for block in &mut message.blocks {
            if let ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error: false,
                ..
            } = block
            {
                if exec_ids.contains(tool_use_id) {
                    if !content.is_empty() {
                        available_exec_results.insert(tool_use_id.clone());
                    }
                    content.clear();
                    content.shrink_to_fit();
                }
            }
        }
    }
    available_exec_results
}

pub(in crate::tui) fn restore_result_availability(
    turn: &mut Turn,
    available_exec_results: &std::collections::HashSet<String>,
) {
    for block in &mut turn.blocks {
        if let Block::Tool(tool) = block {
            if available_exec_results.contains(&tool.id) {
                tool.result_availability = ResultAvailability::Available;
            }
        }
    }
}

impl Transcript {
    /// Remember the first durable-transcript read failure seen this frame;
    /// rendering holds `&App`, so interior mutability defers the notify to
    /// the event loop.
    fn note_transcript_failure(&self, message: String) {
        let mut alert = self.alert.borrow_mut();
        if alert.is_none() {
            *alert = Some(message);
        }
    }

    fn materialize_turn_caching(
        &self,
        idx: usize,
        cache_insert: bool,
        expanded: bool,
        cursor: Option<&store::SessionCursor>,
    ) -> Arc<Turn> {
        if let Some(turn) = self.turns.get(idx) {
            // A committed turn may have released its exec results from
            // memory; an expanded detail still needs the durable content,
            // so reparse the turn from the transcript instead.
            let durable = expanded && self.turn_byte_ranges.get(idx).is_some_and(Option::is_some);
            if !turn.blocks.is_empty() && !durable {
                return Arc::new(turn.clone());
            }
        }
        let cacheable = !expanded;
        if cacheable {
            if let Some(turn) = self.collapsed_turns.borrow_mut().get(idx) {
                return turn;
            }
        }
        let materialize_started = Instant::now();
        let turn = self.parse_transcript_turn(idx, expanded, cursor);
        if cacheable && cache_insert {
            let mut cache = self.collapsed_turns.borrow_mut();
            cache.insert(idx, &turn);
            cache.remember_hot(idx, Arc::clone(&turn));
            cache.finish_materialize(materialize_started.elapsed().as_micros());
        }
        turn
    }

    fn parse_transcript_turn(
        &self,
        idx: usize,
        expanded: bool,
        cursor: Option<&store::SessionCursor>,
    ) -> Arc<Turn> {
        let prompt = self
            .turns
            .get(idx)
            .map(|t| t.prompt.clone())
            .unwrap_or_default();
        let empty = Turn {
            prompt,
            kind: lofi_types::PromptKind::User,
            blocks: Vec::new(),
        };
        let Some(cursor) = cursor else {
            return Arc::new(empty);
        };
        let selected_offsets = self.turn_event_offsets.get(idx).and_then(Option::as_deref);
        let events = if let Some(offsets) = selected_offsets {
            cursor.display_events_at(offsets)
        } else {
            let Some((start, end)) = self.turn_byte_ranges.get(idx).copied().flatten() else {
                return Arc::new(empty);
            };
            cursor.display_events_in_range(start, end)
        };
        let mut events = match events {
            Ok(events) => events,
            Err(error) => {
                self.note_transcript_failure(format!(
                    "transcript read failed (turn {}): {error}",
                    idx + 1
                ));
                return Arc::new(empty);
            }
        };
        let available_exec_results = if expanded {
            std::collections::HashSet::new()
        } else {
            strip_collapsed_exec_results(&mut events)
        };
        let turns = if selected_offsets.is_some() {
            turns_from_selected_session_events(&events)
        } else {
            turns_from_session_events(&events)
        };
        let mut turn = turns.into_iter().next().unwrap_or(empty);
        if let Some(markers) = self.turn_compactions.get(idx) {
            for marker in markers {
                turn.blocks.push(Block::Compaction {
                    summarized: marker.summarized,
                    kept: marker.kept,
                    summary: marker.summary.clone(),
                });
            }
        }
        assign_detail_ids(&mut turn, idx);
        restore_result_availability(&mut turn, &available_exec_results);
        if !expanded {
            view::blocks::compact_native_previews(&mut turn);
        }
        Arc::new(turn)
    }
}

fn non_zero_stamp(stamp: u64) -> u64 {
    if stamp == 0 {
        1
    } else {
        stamp
    }
}

fn hash_sampled_bytes(hasher: &mut impl std::hash::Hasher, bytes: &[u8]) {
    use std::hash::Hash;
    if bytes.len() <= 4096 {
        bytes.hash(hasher);
        return;
    }
    let mid = bytes.len() / 2;
    bytes.len().hash(hasher);
    bytes[..64].hash(hasher);
    bytes[mid.saturating_sub(32)..(mid + 32).min(bytes.len())].hash(hasher);
    bytes[bytes.len() - 64..].hash(hasher);
}

fn prompt_stamp(turn: &Turn) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    0u8.hash(&mut h);
    (turn.kind as u8).hash(&mut h);
    hash_sampled_bytes(&mut h, turn.prompt.as_bytes());
    non_zero_stamp(h.finish())
}

fn block_stamps(block: &Block, cx: &view::component::Cx, has_fatal: bool) -> (u64, u64, bool) {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    match block {
        Block::Text(text) => {
            1u8.hash(&mut h);
            hash_sampled_bytes(&mut h, text.as_bytes());
            let s = non_zero_stamp(h.finish());
            (s, s, text.len() <= MAX_EAGER_LIVE_BLOCK_BYTES)
        }
        Block::Thinking(tb) => {
            2u8.hash(&mut h);
            let working = tb.elapsed.is_none() && cx.active_turn;
            hash_sampled_bytes(&mut h, tb.text.as_bytes());
            tb.elapsed.hash(&mut h);
            working.hash(&mut h);
            let s = non_zero_stamp(h.finish());
            (s, s, tb.text.len() <= MAX_EAGER_LIVE_BLOCK_BYTES)
        }
        Block::Tool(tool) => tool_stamps(tool, cx),
        Block::UserShell {
            id,
            command,
            output,
            exit_code,
            signal,
            duration,
            truncated,
            cancelled,
            running,
            exclude_from_context,
        } => {
            4u8.hash(&mut h);
            id.hash(&mut h);
            hash_sampled_bytes(&mut h, command.as_bytes());
            hash_sampled_bytes(&mut h, output.as_bytes());
            exit_code.hash(&mut h);
            signal.hash(&mut h);
            duration.hash(&mut h);
            truncated.hash(&mut h);
            cancelled.hash(&mut h);
            running.hash(&mut h);
            exclude_from_context.hash(&mut h);
            let hs = non_zero_stamp(h.finish());
            let ls = if *running {
                cx.spinner().hash(&mut h);
                non_zero_stamp(h.finish())
            } else {
                hs
            };
            (hs, ls, command.len() <= MAX_EAGER_LIVE_BLOCK_BYTES)
        }
        Block::Error(msg) => {
            5u8.hash(&mut h);
            msg.hash(&mut h);
            let s = non_zero_stamp(h.finish());
            (s, s, true)
        }
        Block::TurnEnd { label, elapsed } => {
            6u8.hash(&mut h);
            cx.active_turn.hash(&mut h);
            label.hash(&mut h);
            elapsed.hash(&mut h);
            let s = non_zero_stamp(h.finish());
            (s, s, true)
        }
        Block::TurnFailed {
            label,
            elapsed,
            error,
        } => {
            7u8.hash(&mut h);
            cx.active_turn.hash(&mut h);
            label.hash(&mut h);
            elapsed.hash(&mut h);
            error.hash(&mut h);
            has_fatal.hash(&mut h);
            let s = non_zero_stamp(h.finish());
            (s, s, true)
        }
        Block::TurnCancelled { label, elapsed } => {
            8u8.hash(&mut h);
            cx.active_turn.hash(&mut h);
            label.hash(&mut h);
            elapsed.hash(&mut h);
            let s = non_zero_stamp(h.finish());
            (s, s, true)
        }
        Block::Compaction {
            summarized, kept, ..
        } => {
            9u8.hash(&mut h);
            summarized.hash(&mut h);
            kept.hash(&mut h);
            let s = non_zero_stamp(h.finish());
            (s, s, true)
        }
    }
}

fn tool_stamps(tool: &ToolCall, cx: &view::component::Cx) -> (u64, u64, bool) {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    3u8.hash(&mut h);
    tool.detail_id.hash(&mut h);
    tool.id.hash(&mut h);
    tool.name.hash(&mut h);
    hash_sampled_bytes(&mut h, tool.input.as_bytes());
    tool.label.hash(&mut h);
    tool.done.hash(&mut h);
    tool.is_error.hash(&mut h);
    tool.elapsed.hash(&mut h);
    (tool.result_availability as u8).hash(&mut h);
    if let Some(res) = &tool.result {
        hash_sampled_bytes(&mut h, res.as_bytes());
    } else {
        usize::MAX.hash(&mut h);
    }
    cx.active_turn.hash(&mut h);
    tool.native.len().hash(&mut h);
    let mut any_open_native = false;
    for nt in &tool.native {
        nt.id.hash(&mut h);
        nt.name.hash(&mut h);
        hash_sampled_bytes(&mut h, nt.args.as_bytes());
        nt.done.hash(&mut h);
        nt.is_error.hash(&mut h);
        nt.result.as_ref().map(String::len).hash(&mut h);
        nt.preview.is_some().hash(&mut h);
        any_open_native |= !nt.done;
    }
    let hs = non_zero_stamp(h.finish());
    let spinning = cx.active_turn && (!tool.done || any_open_native);
    let ls = if spinning {
        cx.spinner().hash(&mut h);
        non_zero_stamp(h.finish())
    } else {
        hs
    };
    let eager = tool.native.len() <= 64 && tool.input.len() <= MAX_EAGER_LIVE_BLOCK_BYTES;
    (hs, ls, eager)
}

fn live_detail_stamp(app: &App, turn_idx: usize) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut acc = 0u64;
    for (key, state) in &app.expanded_details {
        if state.turn == turn_idx {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            key.hash(&mut h);
            state.scroll.hash(&mut h);
            acc = acc.wrapping_add(h.finish());
        }
    }
    if let Some(focus) = &app.detail_focus {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        focus.key.hash(&mut h);
        acc ^= h.finish().rotate_left(17);
    }
    acc
}

fn sync_child_height(
    slot: &mut CachedLiveChild,
    hs: u64,
    ls: u64,
    eager: bool,
    render_lines: impl FnOnce() -> Vec<view::RenderLine>,
    render_height: impl FnOnce() -> usize,
) {
    if slot.height_stamp == hs {
        return;
    }
    if eager {
        let lines = render_lines();
        slot.height = lines.len();
        slot.height_stamp = hs;
        slot.lines_stamp = ls;
        slot.lines = (lines.len() <= MAX_CACHED_LIVE_BLOCK_ROWS).then_some(lines);
        slot.window = None;
    } else {
        slot.height = render_height();
        slot.height_stamp = hs;
        slot.lines = None;
        slot.window = None;
    }
}

fn child_window_lines(
    slot: &mut CachedLiveChild,
    ls: u64,
    eager: bool,
    sub: std::ops::Range<usize>,
    render_lines: impl FnOnce() -> Vec<view::RenderLine>,
    render_window: impl FnOnce(std::ops::Range<usize>) -> Vec<view::RenderLine>,
) -> Vec<view::RenderLine> {
    if slot.lines_stamp == ls {
        if let Some(lines) = &slot.lines {
            let end = sub.end.min(lines.len());
            let start = sub.start.min(end);
            return lines[start..end].to_vec();
        }
        if let Some((win_stamp, cached_range, lines)) = &slot.window {
            if *win_stamp == ls && *cached_range == sub {
                return lines.clone();
            }
        }
    }
    if eager && slot.height <= MAX_CACHED_LIVE_BLOCK_ROWS {
        let lines = render_lines();
        slot.lines_stamp = ls;
        let end = sub.end.min(lines.len());
        let start = sub.start.min(end);
        let sliced = lines[start..end].to_vec();
        slot.lines = Some(lines);
        slot.window = None;
        sliced
    } else {
        let lines = render_window(sub.clone());
        slot.lines_stamp = ls;
        slot.window = Some((ls, sub, lines.clone()));
        lines
    }
}

impl App {
    pub(super) fn live_turn_height(
        &mut self,
        idx: usize,
        width: usize,
        active_turn: bool,
    ) -> usize {
        let durable = self.needs_durable_materialize(idx);
        let detail_stamp = live_detail_stamp(self, idx);
        let epoch = self.layout.render_epoch;
        let theme = self.theme;
        let mut cache = std::mem::take(&mut self.layout.live_render);
        let materialized = durable.then(|| self.materialize_turn(idx));
        let height = {
            let turn = materialized
                .as_deref()
                .unwrap_or(&self.transcript.turns[idx]);
            cache.sync_envelope(
                idx,
                durable,
                epoch,
                width,
                theme,
                detail_stamp,
                turn.blocks.len(),
                !turn.prompt.is_empty(),
            );
            let cx = view::component::Cx {
                app: self,
                theme,
                width,
                active_turn,
            };
            let mut total = 0usize;
            let mut first = true;
            if !turn.prompt.is_empty() {
                let ps = prompt_stamp(turn);
                let eager = turn.prompt.len() <= MAX_EAGER_LIVE_BLOCK_BYTES;
                let slot = cache.prompt.get_or_insert_with(CachedLiveChild::default);
                sync_child_height(
                    slot,
                    ps,
                    ps,
                    eager,
                    || view::blocks::render_prompt_lines(&cx, turn),
                    || view::blocks::render_prompt_height(&cx, turn),
                );
                if slot.height > 0 {
                    total = total.saturating_add(slot.height);
                    first = false;
                }
            }
            let has_fatal = turn.blocks.iter().any(|b| matches!(b, Block::Error(_)));
            for (bi, block) in turn.blocks.iter().enumerate() {
                if cache.blocks.len() <= bi {
                    cache.blocks.push(CachedLiveChild::default());
                }
                let (hs, ls, eager) = block_stamps(block, &cx, has_fatal);
                let slot = &mut cache.blocks[bi];
                sync_child_height(
                    slot,
                    hs,
                    ls,
                    eager,
                    || view::blocks::render_single_block_lines(&cx, block, has_fatal),
                    || view::blocks::render_single_block_height(&cx, block, has_fatal),
                );
                if slot.height > 0 {
                    total = total.saturating_add(slot.height + usize::from(!first));
                    first = false;
                }
            }
            total
        };
        self.layout.live_render = cache;
        height
    }

    pub(super) fn live_turn_window(
        &mut self,
        idx: usize,
        width: usize,
        active_turn: bool,
        range: std::ops::Range<usize>,
    ) -> Vec<view::RenderLine> {
        if range.start >= range.end {
            return Vec::new();
        }
        let _ = self.live_turn_height(idx, width, active_turn);
        let durable = self.needs_durable_materialize(idx);
        let theme = self.theme;
        let mut cache = std::mem::take(&mut self.layout.live_render);
        let materialized = durable.then(|| self.materialize_turn(idx));
        let out = {
            let turn = materialized
                .as_deref()
                .unwrap_or(&self.transcript.turns[idx]);
            let cx = view::component::Cx {
                app: self,
                theme,
                width,
                active_turn,
            };
            let mut out = Vec::with_capacity(range.end.saturating_sub(range.start).min(256));
            let mut pos = 0usize;
            let mut first = true;
            if !turn.prompt.is_empty() {
                if let Some(slot) = cache.prompt.as_mut() {
                    let h = slot.height;
                    if h > 0 {
                        let end = pos.saturating_add(h);
                        if end > range.start && pos < range.end {
                            let ps = prompt_stamp(turn);
                            let eager = turn.prompt.len() <= MAX_EAGER_LIVE_BLOCK_BYTES;
                            let sub = range.start.saturating_sub(pos)
                                ..range.end.saturating_sub(pos).min(h);
                            out.extend(child_window_lines(
                                slot,
                                ps,
                                eager,
                                sub,
                                || view::blocks::render_prompt_lines(&cx, turn),
                                |r| view::blocks::render_prompt_window(&cx, turn, r),
                            ));
                        }
                        pos = end;
                        first = false;
                    }
                }
            }
            let has_fatal = turn.blocks.iter().any(|b| matches!(b, Block::Error(_)));
            for (bi, block) in turn.blocks.iter().enumerate() {
                if pos >= range.end {
                    break;
                }
                let Some(slot) = cache.blocks.get_mut(bi) else {
                    break;
                };
                let h = slot.height;
                if h == 0 {
                    continue;
                }
                if !first {
                    if range.contains(&pos) {
                        out.push(view::prim::rblank());
                    }
                    pos = pos.saturating_add(1);
                }
                let end = pos.saturating_add(h);
                if end > range.start && pos < range.end {
                    let (_, ls, eager) = block_stamps(block, &cx, has_fatal);
                    let sub = range.start.saturating_sub(pos)..range.end.saturating_sub(pos).min(h);
                    out.extend(child_window_lines(
                        slot,
                        ls,
                        eager,
                        sub,
                        || view::blocks::render_single_block_lines(&cx, block, has_fatal),
                        |r| view::blocks::render_single_block_window(&cx, block, has_fatal, r),
                    ));
                }
                pos = end;
                first = false;
            }
            out
        };
        self.layout.live_render = cache;
        out
    }
}
