#[allow(clippy::wildcard_imports)]
use super::*;
use crate::tui::InfoLine;

fn render_info_lines(lines: &[InfoLine], width: usize, t: Theme) -> Vec<Line<'static>> {
    let section_style = t.heading_style();
    let rule_style = t.subtle_style();
    let mut out = Vec::new();
    for line in lines {
        match line {
            InfoLine::Section(label) => {
                let label_width = prim::width(label);
                let rule_width = width.saturating_sub(label_width.saturating_add(1));
                let mut spans = vec![Span::styled(label.clone(), section_style)];
                if rule_width > 0 {
                    spans.push(Span::raw(" "));
                    spans.push(Span::styled("─".repeat(rule_width), rule_style));
                }
                out.push(Line::from(spans));
            }
            InfoLine::Text(line) => out.extend(prim::wrap_line_styled(line, width)),
        }
    }
    out
}

pub(in crate::tui::view) fn render_info_modal(f: &mut Frame, area: Rect, app: &mut App) {
    let t = app.theme;
    let Some(info) = app.info_mut() else {
        return;
    };
    let title = format!(" {} ", info.title);
    let max_w = area.width.saturating_sub(4) as usize;
    let line_w = |line: &InfoLine| match line {
        InfoLine::Section(label) => prim::width(label),
        InfoLine::Text(line) => line
            .spans
            .iter()
            .map(|span| prim::width(span.content.as_ref()))
            .sum::<usize>(),
    };
    let max_body = info.lines.iter().map(line_w).max().unwrap_or(0);
    let inner_w = max_body.min(max_w).max(prim::width(&title));
    let max_body_h = (area.height as usize).saturating_sub(4);
    let body_w = inner_w.max(1);
    let wrapped = render_info_lines(&info.lines, body_w, t);
    let total = wrapped.len();
    let need_sb = total > max_body_h;
    let view_h = total.min(max_body_h);
    let desired_frame_h = u16::try_from(view_h + 4).unwrap_or(12);

    let key_style = t.key_hint_style();
    let dim = t.muted_style();
    let mut hint_spans: Vec<Span<'static>> = Vec::new();
    if need_sb {
        hint_spans.push(Span::styled("j/k", key_style));
        hint_spans.push(Span::styled(" to scroll · ", dim));
    }
    hint_spans.push(Span::styled("y", key_style));
    hint_spans.push(Span::styled(" to copy · ", dim));
    hint_spans.push(Span::styled("q", key_style));
    hint_spans.push(Span::styled(" to dismiss", dim));

    let w = u16::try_from(inner_w + 4).unwrap_or(40).min(area.width);
    let popup = centered_modal(area, w, desired_frame_h);
    f.render_widget(Clear, popup);
    let rows = render_modal_frame(f, popup, t, modal_title(t, title), Line::from(hint_spans));
    let actual_view_h = rows.content.height as usize;
    info.total = total;
    info.view_h = actual_view_h;
    if info.scroll > info.max_scroll() {
        info.scroll = info.max_scroll();
    }
    let scroll = info.scroll;
    let body_rect = Rect {
        width: u16::try_from(body_w)
            .unwrap_or(rows.content.width)
            .min(rows.content.width),
        ..rows.content
    };
    let body = Paragraph::new(wrapped).scroll((u16::try_from(scroll).unwrap_or(u16::MAX), 0));
    f.render_widget(body, body_rect);

    if need_sb {
        let scroll_area = modal_scroll_area(rows.content);
        prim::render_scrollbar(
            f,
            scroll_area.gutter,
            scroll,
            actual_view_h,
            total,
            t.subtle,
            t.user,
        );
    }
}
