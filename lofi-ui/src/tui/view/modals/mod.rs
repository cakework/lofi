#[allow(clippy::wildcard_imports)]
use super::*;
use crate::tui::Theme;
use ratatui::widgets::{Block as WidgetBlock, BorderType, Padding};

mod background;
mod confirm;
mod info;
mod pickers;

pub(super) use background::render_background_modal;
pub(super) use confirm::render_confirm_modal;
pub(super) use info::render_info_modal;
pub(super) use pickers::{
    render_model_picker, render_paths_picker, render_picker, render_policy_picker,
    render_service_picker, render_slash_complete, render_theme_picker, render_thinking_picker,
    render_tree_picker,
};

/// Shared modal chrome: the terminal's default background, one-cell
/// horizontal breathing room, and a primary border so the active overlay is
/// immediately obvious.
fn rich_modal_block(t: Theme) -> WidgetBlock<'static> {
    WidgetBlock::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(t.primary))
        .padding(Padding::horizontal(1))
        .style(t.text_style())
}

fn modal_title(t: Theme, title: impl Into<String>) -> Line<'static> {
    let title = title.into();
    Line::from(Span::styled(
        title.trim().to_string(),
        t.accent_heading_style(),
    ))
}

fn modal_help(t: Theme, text: impl Into<String>) -> Line<'static> {
    let text = text.into();
    let leading = text.starts_with(' ');
    let trailing = text.ends_with(' ');
    let groups: Vec<&str> = text.trim().split("  ").filter(|s| !s.is_empty()).collect();
    let key_style = t.key_hint_style();
    let description_style = t.subtle_style();
    let mut spans = Vec::new();
    if leading {
        spans.push(Span::styled(" ", description_style));
    }
    for (index, group) in groups.iter().enumerate() {
        if index > 0 {
            spans.push(Span::styled("  ", description_style));
        }
        let (key, description) = group.split_once(' ').unwrap_or((group, ""));
        spans.push(Span::styled(key.to_string(), key_style));
        if !description.is_empty() {
            spans.push(Span::styled(format!(" {description}"), description_style));
        }
    }
    if trailing {
        spans.push(Span::styled(" ", description_style));
    }
    Line::from(spans)
}

fn centered_modal(area: Rect, width: u16, desired_h: u16) -> Rect {
    let width = width.min(area.width);
    let height = desired_h.min(area.height);
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(height) / 2;
    Rect::new(x, y, width, height)
}

#[derive(Clone, Copy)]
struct ModalRows {
    title: Rect,
    content: Rect,
    help: Rect,
}

fn modal_rows(inner: Rect) -> ModalRows {
    let title_h = inner.height.min(1);
    let help_h = inner.height.saturating_sub(title_h).min(1);
    let content_y = inner.y.saturating_add(title_h);
    let content_h = inner.height.saturating_sub(title_h + help_h);
    ModalRows {
        title: Rect::new(inner.x, inner.y, inner.width, title_h),
        content: Rect::new(inner.x, content_y, inner.width, content_h),
        help: Rect::new(
            inner.x,
            inner.bottom().saturating_sub(help_h),
            inner.width,
            help_h,
        ),
    }
}

fn render_modal_help_line(f: &mut Frame, area: Rect, help: Line<'static>) {
    if area.height > 0 {
        f.render_widget(
            Paragraph::new(help).alignment(ratatui::layout::Alignment::Center),
            area,
        );
    }
}

fn render_modal_frame(
    f: &mut Frame,
    popup: Rect,
    t: Theme,
    title: Line<'static>,
    help: Line<'static>,
) -> ModalRows {
    let block = rich_modal_block(t);
    let rows = modal_rows(block.inner(popup));
    f.render_widget(block, popup);
    if rows.title.height > 0 {
        f.render_widget(Paragraph::new(title), rows.title);
    }
    render_modal_help_line(f, rows.help, help);
    rows
}

fn modal_scroll_area(content: Rect) -> prim::ScrollArea {
    prim::scroll_area(Rect::new(
        content.x,
        content.y,
        content.width.saturating_add(1),
        content.height,
    ))
}
