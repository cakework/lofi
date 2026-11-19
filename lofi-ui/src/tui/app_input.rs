#![allow(clippy::wildcard_imports)]

use super::*;

#[derive(Debug, Clone)]
pub(crate) struct PromptLayout {
    pub(crate) rows: Vec<String>,
    pub(crate) cursor_row: usize,
    pub(crate) cursor_col: usize,
}

struct CachedPromptLayout {
    text: String,
    cursor: usize,
    width: usize,
    layout: Arc<PromptLayout>,
}

/// Prompt buffer, readline kill-ring, recall history, slash-completion popover,
/// and soft-wrap layout cache.
#[derive(Default)]
pub(crate) struct PromptEditor {
    pub(crate) text: String,
    pub(crate) cursor: usize,
    pub(crate) scroll: usize,
    pub(crate) rect: Rect,
    pub(crate) history: Vec<String>,
    pub(crate) history_idx: Option<usize>,
    pub(crate) stash: String,
    pub(crate) kill_ring: String,
    /// True when the previous command was `C-k` so a consecutive `C-k`
    /// appends to the kill ring instead of replacing it.
    pub(crate) last_kill_was_kill: bool,
    pub(crate) slash_complete: Option<SlashComplete>,
    layout_cache: RefCell<Option<CachedPromptLayout>>,
}

impl App {
    pub(super) fn input_lines(&self, width: usize) -> usize {
        let content_w = width.saturating_sub(3);
        self.prompt_layout(content_w)
            .rows
            .len()
            .min(MAX_INPUT_LINES)
    }

    /// Keep `prompt.scroll` within bounds and clamp it so the cursor's select
    /// row stays inside the visible `[scroll, scroll + vis_h)` window.
    pub(super) fn sync_input_scroll(&mut self, content_w: usize, vis_h: usize) {
        let layout = self.prompt_layout(content_w);
        let total = layout.rows.len();
        let max_top = total.saturating_sub(vis_h);
        let mut scroll = self.prompt.scroll.min(max_top);
        let vrow = layout.cursor_row;
        if vrow < scroll {
            scroll = vrow;
        } else if vis_h > 0 && vrow >= scroll + vis_h {
            scroll = vrow.saturating_sub(vis_h) + 1;
        }
        self.prompt.scroll = scroll.min(max_top);
    }

    /// Compute and cache soft-wrapped prompt rows and visual cursor position in
    /// one pass so frame layout, scroll sync, input rendering, and slash
    /// completion reuse the same allocation across ticks.
    pub(super) fn prompt_layout(&self, content_w: usize) -> Arc<PromptLayout> {
        if let Some(cached) = self.prompt.layout_cache.borrow().as_ref() {
            if cached.width == content_w
                && cached.cursor == self.prompt.cursor
                && cached.text == self.prompt.text
            {
                return Arc::clone(&cached.layout);
            }
        }
        let (lrow, lcol) = self.cursor_row_col();
        let mut rows = Vec::new();
        let mut cursor_row = 0usize;
        let mut cursor_col = 0usize;
        for (i, line) in self.prompt.text.split('\n').enumerate() {
            let chars: Vec<char> = line.chars().collect();
            let ranges = wrap_input_ranges(&chars, content_w);
            if i == lrow {
                let (sub, x) = cursor_pos_in_ranges(&chars, &ranges, lcol, content_w);
                cursor_row = rows.len() + sub;
                cursor_col = x;
            }
            for (s, e) in ranges {
                rows.push(chars[s..e].iter().collect());
            }
        }
        if rows.is_empty() {
            rows.push(String::new());
        }
        let layout = Arc::new(PromptLayout {
            rows,
            cursor_row,
            cursor_col,
        });
        *self.prompt.layout_cache.borrow_mut() = Some(CachedPromptLayout {
            text: self.prompt.text.clone(),
            cursor: self.prompt.cursor,
            width: content_w,
            layout: Arc::clone(&layout),
        });
        layout
    }

    #[cfg(test)]
    pub(super) fn input_select_rows(&self, content_w: usize) -> Vec<String> {
        self.prompt_layout(content_w).rows.clone()
    }

    pub(super) fn input_cursor_pos(&self, content_w: usize) -> (usize, usize) {
        let layout = self.prompt_layout(content_w);
        (layout.cursor_row, layout.cursor_col)
    }

    pub(super) fn insert_char(&mut self, c: char) {
        self.prompt.text.insert(self.prompt.cursor, c);
        self.prompt.cursor += c.len_utf8();
        self.prompt.history_idx = None;
    }

    /// Insert a (possibly multi-line) string at the cursor in one operation.
    /// Used by bracketed-paste so a paste of any size lands atomically
    /// instead of character-by-character. Line endings are normalized to
    /// `\n` so a paste carrying `\r\n` or bare `\r` (which varies by terminal)
    /// renders and submits as real newlines, not a single garbled line.
    pub(super) fn insert_str(&mut self, s: &str) {
        let normalized: String = s.replace("\r\n", "\n").replace('\r', "\n");
        self.prompt.text.insert_str(self.prompt.cursor, &normalized);
        self.prompt.cursor += normalized.len();
        self.prompt.history_idx = None;
    }

    pub(super) fn insert_newline(&mut self) {
        self.insert_char('\n');
    }

    pub(super) fn backspace(&mut self) {
        if self.prompt.cursor == 0 {
            return;
        }
        let i = self.prompt.text[..self.prompt.cursor]
            .char_indices()
            .last()
            .map_or(0, |(i, _)| i);
        self.prompt.text.replace_range(i..self.prompt.cursor, "");
        self.prompt.cursor = i;
        self.prompt.history_idx = None;
    }

    pub(super) fn move_left(&mut self) {
        if let Some((i, _)) = self.prompt.text[..self.prompt.cursor].char_indices().last() {
            self.prompt.cursor = i;
        }
    }

    pub(super) fn move_right(&mut self) {
        if let Some((_, c)) = self.prompt.text[self.prompt.cursor..].char_indices().next() {
            self.prompt.cursor += c.len_utf8();
        }
    }

    pub(super) fn move_up(&mut self) {
        let (row, col) = self.cursor_row_col();
        if row == 0 {
            return;
        }
        self.prompt.cursor = line_start_byte(&self.prompt.text, row - 1)
            + char_index_to_byte(&self.prompt.text, row - 1, col);
    }

    pub(super) fn move_down(&mut self) {
        let (row, col) = self.cursor_row_col();
        let last_row = self.prompt.text.matches('\n').count();
        if row >= last_row {
            return;
        }
        self.prompt.cursor = line_start_byte(&self.prompt.text, row + 1)
            + char_index_to_byte(&self.prompt.text, row + 1, col);
    }

    pub(super) fn cursor_row_col(&self) -> (usize, usize) {
        let before = &self.prompt.text[..self.prompt.cursor];
        let row = before.matches('\n').count();
        let col = before
            .rsplit_once('\n')
            .map_or(before.chars().count(), |(_, last)| last.chars().count());
        (row, col)
    }

    pub(super) fn move_line_start(&mut self) {
        let before = &self.prompt.text[..self.prompt.cursor];
        self.prompt.cursor = before.rsplit_once('\n').map_or(0, |(b, _)| b.len() + 1);
    }

    pub(super) fn move_line_end(&mut self) {
        let after = &self.prompt.text[self.prompt.cursor..];
        self.prompt.cursor += after.find('\n').unwrap_or(after.len());
    }

    pub(super) fn move_word_back(&mut self) {
        self.prompt.cursor = prev_word_start(&self.prompt.text, self.prompt.cursor);
    }

    pub(super) fn move_word_fwd(&mut self) {
        self.prompt.cursor = next_word_end(&self.prompt.text, self.prompt.cursor);
    }

    pub(super) fn delete_forward_char(&mut self) {
        if let Some((i, c)) = self.prompt.text[self.prompt.cursor..].char_indices().next() {
            let end = self.prompt.cursor + i + c.len_utf8();
            self.prompt.text.replace_range(self.prompt.cursor..end, "");
        }
        self.prompt.history_idx = None;
    }

    /// Remove `text[start..end]` into the kill ring. When `append` is true the
    /// removed text is appended (consecutive `C-k`); otherwise it replaces.
    pub(super) fn kill_range(&mut self, start: usize, end: usize, append: bool) {
        let (start, end) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        if start >= end {
            return;
        }
        let removed = self.prompt.text[start..end].to_string();
        if append {
            self.prompt.kill_ring.push_str(&removed);
        } else {
            self.prompt.kill_ring = removed;
        }
        self.prompt.text.replace_range(start..end, "");
        self.prompt.cursor = start;
        self.prompt.history_idx = None;
    }

    pub(super) fn kill_line_end(&mut self, append: bool) {
        let after = &self.prompt.text[self.prompt.cursor..];
        if after.is_empty() {
            return;
        }
        let nl = after.find('\n').unwrap_or(after.len());
        let end = if nl == 0 {
            self.prompt.cursor + 1
        } else {
            self.prompt.cursor + nl
        };
        self.kill_range(self.prompt.cursor, end, append);
        self.prompt.last_kill_was_kill = true;
    }

    pub(super) fn kill_line_start(&mut self) {
        let before = &self.prompt.text[..self.prompt.cursor];
        let start = before.rsplit_once('\n').map_or(0, |(b, _)| b.len() + 1);
        self.kill_range(start, self.prompt.cursor, false);
    }

    pub(super) fn kill_word_back(&mut self) {
        let start = prev_word_start(&self.prompt.text, self.prompt.cursor);
        self.kill_range(start, self.prompt.cursor, false);
    }

    pub(super) fn kill_word_fwd(&mut self) {
        let end = next_word_end(&self.prompt.text, self.prompt.cursor);
        self.kill_range(self.prompt.cursor, end, false);
    }

    pub(super) fn yank(&mut self) {
        if self.prompt.kill_ring.is_empty() {
            return;
        }
        let s = self.prompt.kill_ring.clone();
        self.prompt.text.insert_str(self.prompt.cursor, &s);
        self.prompt.cursor += s.len();
        self.prompt.history_idx = None;
    }

    pub(super) fn cursor_up(&mut self) {
        let (row, col) = self.cursor_row_col();
        if row > 0 {
            self.move_up();
        } else if col > 0 {
            self.move_line_start();
        } else {
            self.recall_prev();
        }
    }

    pub(super) fn cursor_down(&mut self) {
        let (row, _col) = self.cursor_row_col();
        let last_row = self.prompt.text.matches('\n').count();
        if row < last_row {
            self.move_down();
        } else if self.prompt.cursor < self.prompt.text.len() {
            self.move_line_end();
        } else {
            self.recall_next();
        }
    }

    pub(super) fn current_line_text(&self) -> Option<String> {
        let rel = self
            .viewport
            .nav_cursor
            .saturating_sub(self.viewport.log_off);
        // Prefer the raw markdown source (markers intact) when the rendered
        // line carries a position map; slice the source over the row's full
        // content range so a soft-wrapped row yields its raw fragment.
        if let Some(rl) = self.viewport.log_vis.get(rel).and_then(|v| v.raw.as_ref()) {
            if rl.map.len() >= 2 {
                let end = rl.map.last()?;
                if rl.hard_break && end > 0 {
                    return Some(rl.source[..end].to_string());
                }
                let start = rl.map.first()?;
                if start < end {
                    return Some(rl.source[start..end].to_string());
                }
            }
            // Degenerate (whole-line) raw — no per-char map. The display
            // (e.g. a table grid) doesn't map 1:1 to the source, so yank
            // returns the full source line. A table border row carries an
            // empty source — it has no markdown to yank, so return nothing
            // rather than the rendered box-drawing characters.
            if !rl.source.is_empty() {
                return Some(rl.source.to_string());
            }
            return None;
        }
        let vl = self.viewport.log_vis.get(rel)?;
        let s = &vl.rendered;
        let n = s.chars().count();
        let (cstart, cend) = vl.content;
        let cstart = cstart.min(n);
        let cend = cend.min(n);
        if cstart >= cend {
            return None;
        }
        let b0 = s.char_indices().nth(cstart).map_or(s.len(), |(b, _)| b);
        let b1 = s.char_indices().nth(cend).map_or(s.len(), |(b, _)| b);
        Some(s[b0..b1].to_string())
    }

    pub(super) fn recall_prev(&mut self) {
        if self.prompt.history.is_empty() {
            return;
        }
        match self.prompt.history_idx {
            None => {
                self.prompt.stash.clone_from(&self.prompt.text);
                let idx = self.prompt.history.len() - 1;
                self.prompt.history_idx = Some(idx);
                self.set_input(self.prompt.history[idx].clone());
                self.prompt.cursor = 0;
            }
            Some(0) => {}
            Some(i) => {
                let idx = i - 1;
                self.prompt.history_idx = Some(idx);
                self.set_input(self.prompt.history[idx].clone());
                self.prompt.cursor = 0;
            }
        }
    }

    pub(super) fn recall_next(&mut self) {
        match self.prompt.history_idx {
            None => {}
            Some(i) => {
                if i + 1 >= self.prompt.history.len() {
                    self.prompt.history_idx = None;
                    let stash = std::mem::take(&mut self.prompt.stash);
                    self.set_input(stash);
                } else {
                    let idx = i + 1;
                    self.prompt.history_idx = Some(idx);
                    self.set_input(self.prompt.history[idx].clone());
                }
            }
        }
    }

    pub(super) fn set_input(&mut self, s: String) {
        self.prompt.text = s;
        self.prompt.cursor = self.prompt.text.len();
    }

    pub(super) fn clear_input(&mut self) {
        self.prompt.text.clear();
        self.prompt.cursor = 0;
        self.prompt.history_idx = None;
        self.prompt.slash_complete = None;
    }
}
