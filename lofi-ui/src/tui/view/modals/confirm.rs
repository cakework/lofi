#[allow(clippy::wildcard_imports)]
use super::*;
use ratatui::widgets::{Block as WidgetBlock, Padding};

pub(in crate::tui::view) fn render_confirm_modal(f: &mut Frame, area: Rect, app: &mut App) {
    const MAX_COMMAND_ROWS: usize = 12;

    use ratatui::layout::Alignment;
    let t = app.theme;
    let Some(req) = app.confirms.pending.first() else {
        return;
    };

    f.buffer_mut()
        .set_style(area, t.panel_style().patch(t.subtle_style()));

    let desired_w = area.width.saturating_mul(3).saturating_div(5).max(52);
    let w = desired_w.min(100).min(area.width);
    let content_w = w.saturating_sub(6).max(1) as usize;
    let reason = req
        .reason
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let explanation = match &reason {
        lofi_core::ConfirmReason::Policy { reason } => {
            let reason = reason.trim();
            if reason.is_empty() {
                "Shell wants to run this command".to_string()
            } else {
                format!("Shell wants to run this command: {reason}")
            }
        }
        lofi_core::ConfirmReason::OutsideRoot {
            dir,
            narrow,
            marker,
            requested,
            current,
        } => {
            let shown = confirm_grant_unit(dir, narrow.as_deref(), app.confirms.narrow);
            // Only a grant that holds the shown unit is described as already
            // covering it: a read-only grant for the requested directory does
            // not make the project root read-only.
            let widens = current.as_ref().is_some_and(|grant| {
                grant.access == lofi_types::Access::Read
                    && std::path::Path::new(shown).starts_with(&grant.dir)
            });
            let unit = crate::tui::text::abbreviate_path(std::path::Path::new(shown), content_w);
            let access = match requested {
                lofi_types::Access::Read => "read",
                lofi_types::Access::Write => "read and write",
            };
            if widens {
                format!("{unit} and its subdirectories are read-only; grant write access too?")
            } else if app.confirms.narrow && narrow.is_some() {
                let root = crate::tui::text::abbreviate_path(std::path::Path::new(dir), content_w);
                format!("lofi wants {access} access to {unit} and its subdirectories, not {root}")
            } else if let Some(marker) = marker {
                format!(
                    "lofi wants {access} access to {unit} and its subdirectories (project root: {marker})"
                )
            } else {
                format!("lofi wants {access} access to {unit} and its subdirectories")
            }
        }
        lofi_core::ConfirmReason::AutoEvaluating { started_at } => {
            format!("Auto evaluation for {}s...", started_at.elapsed().as_secs())
        }
        lofi_core::ConfirmReason::AutoAsk { reason } => {
            if reason.trim().is_empty() {
                "Auto evaluation asks for your approval.".to_string()
            } else {
                format!("Auto evaluation asks: {reason}")
            }
        }
        lofi_core::ConfirmReason::AutoRisk { reason } => {
            if reason.trim().is_empty() {
                "Auto evaluation sees clear harm.".to_string()
            } else {
                format!("Auto evaluation sees clear harm: {reason}")
            }
        }
        lofi_core::ConfirmReason::AutoFailed { reason } => {
            format!("Auto evaluation could not decide: {reason}")
        }
    };
    let explanation_lines = prim::wrap(&explanation, content_w);
    // A directory-grant request shows a path, not a command: abbreviate it
    // so it fits the panel instead of wrapping. Shell commands stay whole.
    let command = match &reason {
        lofi_core::ConfirmReason::OutsideRoot { dir, narrow, .. } => {
            let unit = confirm_grant_unit(dir, narrow.as_deref(), app.confirms.narrow);
            crate::tui::text::abbreviate_path(std::path::Path::new(unit), content_w)
        }
        _ => req.command.clone(),
    };
    // Commands are preformatted source: preserve leading blank lines and
    // indentation instead of treating them as flow text.
    let cmd_lines = prim::wrap_pre(&command, content_w);
    // Keep the dialog compact even on a tall terminal. The command panel is
    // a viewport rather than a truncation point, so every wrapped row remains
    // reachable with the scrolling keys.
    let explanation_rows = explanation_lines.len().max(1);
    let non_command_rows = explanation_rows + 9;
    let available_command_rows = area
        .height
        .saturating_sub(u16::try_from(non_command_rows).unwrap_or(u16::MAX))
        .max(1) as usize;
    let command_rows = cmd_lines
        .len()
        .clamp(1, MAX_COMMAND_ROWS)
        .min(available_command_rows);
    let desired_frame_h = u16::try_from(command_rows + non_command_rows).unwrap_or(u16::MAX);
    let popup = centered_modal(area, w, desired_frame_h);
    f.render_widget(Clear, popup);

    let key = t.key_hint_style();
    let mut hints = vec![
        Span::styled("↑/↓", key),
        Span::styled(" scroll  ", t.subtle_style()),
        Span::styled("←/→", key),
        Span::styled(" choose  ", t.subtle_style()),
    ];
    // Only a grant for a project root offers the request's own directory as a
    // narrower unit, so only that prompt advertises the toggle.
    if matches!(
        &reason,
        lofi_core::ConfirmReason::OutsideRoot {
            narrow: Some(_),
            ..
        }
    ) {
        hints.push(Span::styled("i", key));
        hints.push(Span::styled(" this dir  ", t.subtle_style()));
    }
    hints.push(Span::styled("enter", key));
    hints.push(Span::styled(" confirm  ", t.subtle_style()));
    hints.push(Span::styled("esc", key));
    hints.push(Span::styled(" deny", t.subtle_style()));
    let help = Line::from(hints);

    let queue_count = app.confirms.pending.len();
    let counter = if queue_count > 1 {
        format!("  1/{queue_count}")
    } else {
        String::new()
    };
    let title = format!("Permission Required{counter}");
    let chrome = render_modal_frame(
        f,
        popup,
        t,
        Line::from(Span::styled(
            title,
            Style::new().fg(t.warn).add_modifier(Modifier::BOLD),
        )),
        help,
    );
    if chrome.content.height == 0 {
        return;
    }

    let rows = Layout::vertical([
        Constraint::Length(u16::try_from(explanation_rows).unwrap_or(u16::MAX)),
        Constraint::Length(1), // gap
        Constraint::Length(1), // content label
        Constraint::Min(1),    // command panel
        Constraint::Length(1), // gap
        Constraint::Length(1), // actions
        Constraint::Length(1), // gap
    ])
    .split(chrome.content);
    f.render_widget(
        Paragraph::new(explanation_lines.join("\n")).style(t.muted_style()),
        rows[0],
    );
    f.render_widget(
        Paragraph::new(Span::styled("Command", t.accent_heading_style())),
        rows[2],
    );

    let command_panel = WidgetBlock::default()
        .style(t.panel_style().patch(t.text_style()))
        .padding(Padding::horizontal(1));
    let command_inner = command_panel.inner(rows[3]);
    let command_scroll_area = modal_scroll_area(command_inner);
    let command_content = command_scroll_area.content;
    f.render_widget(command_panel, rows[3]);
    app.confirms.total = cmd_lines.len();
    app.confirms.view_h = command_content.height as usize;
    let max_scroll = app.confirms.total.saturating_sub(app.confirms.view_h);
    app.confirms.scroll = app.confirms.scroll.min(max_scroll);
    let body_lines: Vec<Line> = cmd_lines
        .iter()
        .skip(app.confirms.scroll)
        .take(app.confirms.view_h)
        .map(|line| {
            Line::from(Span::styled(
                line.clone(),
                t.panel_style().patch(t.text_style()),
            ))
        })
        .collect();
    f.render_widget(Paragraph::new(body_lines), command_content);
    if app.confirms.total > app.confirms.view_h {
        prim::render_scrollbar(
            f,
            command_scroll_area.gutter,
            app.confirms.scroll,
            app.confirms.view_h,
            app.confirms.total,
            t.subtle,
            t.user,
        );
    }

    let button = |label: &str, selected: bool| {
        let style = if selected {
            t.selected_item_style()
        } else {
            t.panel_style().patch(t.muted_style())
        };
        Span::styled(format!(" {label} "), style)
    };
    let labels = app.confirm_choice_labels();
    let mut spans: Vec<Span> = Vec::new();
    for (i, label) in labels.iter().enumerate() {
        if i > 0 {
            spans.push(Span::raw("  "));
        }
        spans.push(button(label, app.confirms.selected == i));
    }
    let actions = Line::from(spans).alignment(Alignment::Right);
    f.render_widget(Paragraph::new(actions), rows[5]);
}

/// The directory a directory-grant prompt names: the project root it offers
/// by default, or the requested directory once the scope is narrowed.
fn confirm_grant_unit<'a>(dir: &'a str, narrow: Option<&'a str>, narrowed: bool) -> &'a str {
    match (narrowed, narrow) {
        (true, Some(narrow)) => narrow,
        _ => dir,
    }
}
