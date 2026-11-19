use std::sync::Arc;

use ratatui::style::Style;
use ratatui::text::Span;

use crate::tui::{DetailKey, DETAIL_VIEW_ROWS};

use crate::tui::view::component::Cx;
use crate::tui::view::prim::{self, RawLine, RenderLine};

struct DetailVisualRow {
    text: String,
    raw: RawLine,
}

fn detail_visual_rows(lines: &[String], inner: usize) -> Vec<DetailVisualRow> {
    let mut rows = Vec::new();
    for line in lines {
        let source: Arc<str> = Arc::from(line.as_str());
        let indent_len = line
            .bytes()
            .take_while(|&byte| byte == b' ' || byte == b'\t')
            .count();
        let indent_chars = line[..indent_len].chars().count();
        let body = &line[indent_len..];
        let ascii_body = body.is_ascii();
        let body_offsets = if ascii_body {
            Vec::new()
        } else {
            std::iter::once(0)
                .chain(body.char_indices().map(|(byte, c)| byte + c.len_utf8()))
                .collect::<Vec<_>>()
        };
        let segments = prim::wrap_pre(line, inner.max(1));
        let mut consumed = 0usize;
        for (index, segment) in segments.into_iter().enumerate() {
            let body_chars = segment.chars().count().saturating_sub(indent_chars);
            let raw = if ascii_body && consumed + body_chars <= body.len() {
                RawLine::linear(
                    source.clone(),
                    indent_len + consumed,
                    body_chars,
                    index == 0,
                )
            } else {
                let map = (0..=body_chars)
                    .map(|offset| {
                        indent_len
                            + if ascii_body {
                                (consumed + offset).min(body.len())
                            } else {
                                body_offsets
                                    .get(consumed + offset)
                                    .copied()
                                    .unwrap_or(body.len())
                            }
                    })
                    .collect();
                RawLine::new(source.clone(), map, index == 0)
            };
            rows.push(DetailVisualRow { text: segment, raw });
            consumed += body_chars;
        }
    }
    rows
}

pub(super) fn detail_row_count(lines: &[String], width: usize, deco: &[Span<'static>]) -> usize {
    if lines.is_empty() {
        return 0;
    }
    let inner = width.saturating_sub(prim::span_width(deco) + 3).max(1);
    lines
        .iter()
        .map(|line| prim::wrap_pre_count(line, inner))
        .sum()
}

pub(super) fn detail_str_row_count(text: &str, width: usize, deco: &[Span<'static>]) -> usize {
    let trimmed = text.trim_end_matches('\n');
    if trimmed.is_empty() {
        return 0;
    }
    let inner = width.saturating_sub(prim::span_width(deco) + 3).max(1);
    trimmed
        .split('\n')
        .map(|line| prim::wrap_pre_count(line, inner))
        .sum()
}

pub(super) fn detail_box(
    lines: &[String],
    key: &DetailKey,
    tail: bool,
    cx: &Cx,
    deco: &[Span<'static>],
    style: Style,
) -> Vec<RenderLine> {
    let expanded = cx.app.expanded_details.get(key);
    let Some(expanded) = expanded else {
        return Vec::new();
    };
    // Reserve one cell after the mini-scrollbar as outer right margin.
    let inner = cx.width.saturating_sub(prim::span_width(deco) + 3);
    let rows = detail_visual_rows(lines, inner);
    let total = rows.len();
    let target =
        |line: RenderLine, row: usize| line.with_detail_row((*key).clone(), total, tail, row);
    let max = total.saturating_sub(DETAIL_VIEW_ROWS);
    let start = expanded
        .scroll
        .unwrap_or(if tail { max } else { 0 })
        .min(max);
    let visible_rows = total.min(DETAIL_VIEW_ROWS);
    let visible = &rows[start..total.min(start + visible_rows)];
    let raised = style.bg(cx.theme.surface);
    let focused = cx
        .app
        .detail_focus
        .as_ref()
        .is_some_and(|focus| focus.key == *key);
    let track = Style::new().fg(cx.theme.subtle).bg(cx.theme.surface);
    let thumb = Style::new()
        .fg(if focused {
            cx.theme.user
        } else {
            cx.theme.muted
        })
        .bg(cx.theme.surface);
    let thumb_row = (start * visible_rows.saturating_sub(1)).checked_div(max);
    let mut out = Vec::with_capacity(visible_rows);
    for (row, visual) in visible.iter().enumerate() {
        let used = prim::width(&visual.text);
        let mut line_deco = deco.to_vec();
        line_deco.push(Span::styled(" ", raised));
        let detail_span = prim::render(
            line_deco,
            vec![Span::styled(visual.text.clone(), raised)],
            vec![
                Span::styled(" ".repeat(inner.saturating_sub(used)), raised),
                if max == 0 {
                    Span::styled(" ", raised)
                } else {
                    Span::styled(
                        if thumb_row == Some(row) { "┃" } else { "│" },
                        if thumb_row == Some(row) { thumb } else { track },
                    )
                },
                Span::raw(" "),
            ],
        )
        .with_raw(visual.raw.clone());
        out.push(target(detail_span, start + row));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::{App, DetailState, ServiceTier, ThinkingLevel};

    #[test]
    fn detail_row_count_matches_rendered_detail_box_at_wrap_boundary() {
        let mut app = App::new(
            "openai/gpt-4o".to_string(),
            ThinkingLevel::Off,
            ServiceTier::Auto,
            0,
            lofi_types::CompactionConfig::default(),
            String::new(),
        );
        let key = DetailKey::Exec(0);
        app.expanded_details.insert(
            key.clone(),
            DetailState {
                turn: 0,
                scroll: None,
            },
        );
        let deco = vec![Span::raw("  ")];
        // inner = width - (deco(2) + 3) = 10, so an 11-column line wraps onto 2
        // rows; subtracting 2 instead of 3 would mispredict 1 row here.
        let width = 15;
        let line = "abcdefghijk".to_string();
        let lines = vec![line.clone()];
        let cx = Cx {
            app: &app,
            theme: app.theme,
            width,
            active_turn: false,
        };
        let rendered = detail_box(&lines, &key, false, &cx, &deco, Style::default());
        assert_eq!(rendered.len(), 2);
        assert_eq!(detail_row_count(&lines, width, &deco), rendered.len());
        assert_eq!(detail_str_row_count(&line, width, &deco), rendered.len());
    }
}
