#[allow(clippy::wildcard_imports)]
use super::*;
use crate::tui::tree::tree_entry_loading_label;
use crate::tui::{PickerList, PickerState, TreeEntry, TreePickerState};

fn relative_age(then: std::time::SystemTime) -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(then)
        .unwrap_or_default()
        .as_secs();
    match seconds {
        0..60 => "just now".to_string(),
        60..3600 => format!("{}m ago", seconds / 60),
        3600..172_800 => format!("{}h ago", seconds / 3600),
        172_800..31_536_000 => format!("{}d ago", seconds / 86_400),
        _ => format!("{}y ago", seconds / 31_536_000),
    }
}

fn resume_metadata(message_count: usize, last_active: std::time::SystemTime) -> String {
    let messages = if message_count == 1 { "msg" } else { "msgs" };
    format!("{message_count} {messages} · {}", relative_age(last_active))
}

/// Color only the semantic role label; keep the colon and transcript preview
/// in the normal foreground so the row remains easy to scan.
fn resume_preview_spans(t: Theme, preview: String) -> Vec<Span<'static>> {
    let role = if preview.starts_with("user:") {
        Some(("user", t.user))
    } else if preview.starts_with("agent:") {
        Some(("agent", t.agent))
    } else {
        None
    };
    let Some((role, color)) = role else {
        return vec![Span::styled(preview, t.text_style())];
    };
    let rest = preview[role.len()..].to_string();
    vec![
        Span::styled(
            role.to_string(),
            Style::new().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(rest, t.text_style()),
    ]
}

/// Shared list-modal frame: centers the popup, renders bordered chrome, runs
/// `ListState` with persistent `scroll_offset`, and draws the scrollbar from
/// `state.offset()`.
fn render_list_frame<T>(
    f: &mut Frame,
    area: Rect,
    t: Theme,
    title: &str,
    help: &str,
    picker: &PickerList<T>,
    desired_content_w: usize,
    max_rows: usize,
    build_items: impl FnOnce(usize) -> Vec<ListItem<'static>>,
) {
    use ratatui::widgets::ListState;
    let total = picker.len();
    let chrome_w = prim::width(title).max(prim::width(help));
    let w = u16::try_from(desired_content_w.max(chrome_w) + 4)
        .unwrap_or(40)
        .min(area.width);
    let visible_rows = total.min(max_rows);
    let desired_frame_h = u16::try_from(visible_rows + 4).unwrap_or(24);
    let popup = centered_modal(area, w, desired_frame_h);
    f.render_widget(Clear, popup);
    let frame = render_modal_frame(f, popup, t, modal_title(t, title), modal_help(t, help));
    let scroll_area = modal_scroll_area(frame.content);
    let content = scroll_area.content;
    let items = build_items(content.width as usize);
    let list = List::new(items)
        .style(t.text_style())
        .highlight_style(t.selected_item_style());
    let mut state = ListState::default()
        .with_offset(picker.scroll_offset.get())
        .with_selected((!picker.is_empty()).then_some(picker.selected));
    f.render_stateful_widget(list, content, &mut state);
    picker.scroll_offset.set(state.offset());
    if total > content.height as usize {
        prim::render_scrollbar(
            f,
            scroll_area.gutter,
            state.offset(),
            content.height as usize,
            total,
            t.subtle,
            t.user,
        );
    }
}

/// Shared single-column picker renderer (model, theme, thinking, policy, service).
fn render_pick_modal<T>(
    f: &mut Frame,
    area: Rect,
    t: Theme,
    title: &str,
    help: &str,
    picker: &PickerList<T>,
    rows: &[String],
    active: Option<usize>,
) {
    let content_w = rows.iter().map(|r| prim::width(r)).max().unwrap_or(0);
    let active_style = t.accent_heading_style();
    let inactive_style = t.text_style();
    render_list_frame(f, area, t, title, help, picker, content_w, 20, |_| {
        rows.iter()
            .enumerate()
            .map(|(i, row)| {
                ListItem::new(Span::styled(
                    row.clone(),
                    if Some(i) == active {
                        active_style
                    } else {
                        inactive_style
                    },
                ))
            })
            .collect()
    });
}

pub(in crate::tui::view) fn render_picker(
    f: &mut Frame,
    area: Rect,
    app: &App,
    picker: &PickerState,
) {
    use ratatui::widgets::ListState;
    let t = app.theme;
    let total = picker.list.len();
    let title = " Resume a session ";
    let help = " ↑/↓ navigate  enter resume  esc close ";
    let desired_rows = total.clamp(1, 12);
    let desired_frame_h = u16::try_from(desired_rows + 4).unwrap_or(16);

    // Size from only the selected viewport. Progressive enrichment of an old
    // off-screen row must not make every frame walk and allocate the full list.
    let preliminary_start = picker
        .list
        .selected
        .saturating_sub(desired_rows.saturating_sub(1))
        .min(total.saturating_sub(desired_rows));
    let row_data = |row: &crate::tui::PickerEntry| match &row.details {
        Some(entry) => (
            entry.last_message.clone(),
            resume_metadata(entry.message_count, entry.last_active),
        ),
        None => (
            row.preview
                .clone()
                .unwrap_or_else(|| "loading…".to_string()),
            relative_age(row.file.last_active()),
        ),
    };
    let content_w = picker.list.items
        [preliminary_start..(preliminary_start + desired_rows.min(total))]
        .iter()
        .map(|row| {
            let (preview, meta) = row_data(row);
            prim::width(&preview) + 2 + prim::width(&meta)
        })
        .max()
        .unwrap_or(0);
    let chrome_w = prim::width(title).max(prim::width(help));
    let w = u16::try_from(content_w.max(chrome_w) + 4)
        .unwrap_or(40)
        .min(area.width);
    let popup = centered_modal(area, w, desired_frame_h);
    f.render_widget(Clear, popup);
    let rows = render_modal_frame(f, popup, t, modal_title(t, title), modal_help(t, help));
    let scroll_area = modal_scroll_area(rows.content);
    let content = scroll_area.content;
    let view_h = content.height as usize;
    let start = picker
        .list
        .selected
        .saturating_sub(view_h.saturating_sub(1))
        .min(total.saturating_sub(view_h));
    let end = (start + view_h).min(total);
    let row_width = content.width as usize;
    let items: Vec<ListItem> = picker.list.items[start..end]
        .iter()
        .map(|row| {
            let (preview, meta) = row_data(row);
            let meta_w = prim::width(&meta);
            let preview_w = row_width.saturating_sub(meta_w.saturating_add(2));
            let preview = prim::truncate(&preview, preview_w);
            ListItem::new(
                HStack::new(row_width)
                    .left(resume_preview_spans(t, preview))
                    .right([Span::styled(meta, t.muted_style())])
                    .build(),
            )
        })
        .collect();
    let list = List::new(items)
        .style(t.text_style())
        .highlight_style(t.selected_item_style());
    let selected =
        (picker.list.selected < total).then_some(picker.list.selected.saturating_sub(start));
    let mut state = ListState::default().with_selected(selected);
    f.render_stateful_widget(list, content, &mut state);
    if total > view_h {
        prim::render_scrollbar(
            f,
            scroll_area.gutter,
            start,
            view_h,
            total,
            t.subtle,
            t.user,
        );
    }
}

pub(in crate::tui::view) fn render_slash_complete(f: &mut Frame, area: Rect, app: &App) {
    use ratatui::widgets::ListState;
    let Some(sc) = &app.prompt.slash_complete else {
        return;
    };
    let t = app.theme;
    let n_max = sc.len().min(8);
    let row_w = SLASH_COMMANDS
        .iter()
        .map(|(cmd, desc)| prim::width(cmd) + 2 + prim::width(desc))
        .max()
        .unwrap_or(20);
    let help_w = prim::width(" ↑/↓ navigate  enter complete  esc close ");
    let w = u16::try_from(row_w.max(help_w) + 4)
        .unwrap_or(40)
        .min(area.width);
    // Anchor horizontally at the cursor's column within the prompt area,
    // so the popover tracks the cursor as the user types. `prompt.rect` is
    // already inset past the gutter, and the cursor x is relative to its
    // left edge, so add directly. Clamp so the popover stays on screen.
    let content_width = app.prompt.rect.width as usize;
    let (cursor_row, cursor_x) = app.input_cursor_pos(content_width);
    let cursor_screen_x = app
        .prompt
        .rect
        .x
        .saturating_add(u16::try_from(cursor_x).unwrap_or(u16::MAX));
    let cursor_screen_y = app
        .prompt
        .rect
        .y
        .saturating_add(u16::try_from(cursor_row).unwrap_or(u16::MAX));
    // Border and bottom hint consume three rows; autocomplete deliberately
    // has no title so the first interior row is immediately useful.
    let avail_above = cursor_screen_y.saturating_sub(area.y) as usize;
    let n = n_max.min(avail_above.saturating_sub(3));
    if n == 0 {
        return;
    }
    let frame_h = u16::try_from(n + 3).unwrap_or(11);
    let popup_y = cursor_screen_y.saturating_sub(frame_h);
    let popup_x = cursor_screen_x.min(area.right().saturating_sub(w));
    let popup = Rect::new(popup_x, popup_y, w, frame_h);
    f.render_widget(Clear, popup);
    let block = rich_modal_block(t);
    let inner = block.inner(popup);
    let help_h = inner.height.min(1);
    let content_rows = Rect::new(
        inner.x,
        inner.y,
        inner.width,
        inner.height.saturating_sub(help_h),
    );
    let help = Rect::new(
        inner.x,
        inner.bottom().saturating_sub(help_h),
        inner.width,
        help_h,
    );
    f.render_widget(block, popup);
    render_modal_help_line(
        f,
        help,
        modal_help(t, " ↑/↓ navigate  enter complete  esc close "),
    );
    let total = sc.len();
    let need_sb = total > content_rows.height as usize;
    let scroll_area = modal_scroll_area(content_rows);
    let content = scroll_area.content;
    let cmd_style = t.text_style();
    let desc_style = t.subtle_style();
    let items: Vec<ListItem> = sc
        .items
        .iter()
        .map(|&idx| {
            let (cmd, desc) = SLASH_COMMANDS[idx];
            ListItem::new(Line::from(vec![
                Span::styled(cmd.to_string(), cmd_style),
                Span::styled(format!("  {desc}"), desc_style),
            ]))
        })
        .collect();
    let list = List::new(items)
        .style(t.text_style())
        .highlight_style(t.selected_item_style());
    let mut state = ListState::default()
        .with_offset(sc.scroll_offset.get())
        .with_selected((!sc.is_empty()).then_some(sc.selected));
    f.render_stateful_widget(list, content, &mut state);
    sc.scroll_offset.set(state.offset());
    if need_sb {
        prim::render_scrollbar(
            f,
            scroll_area.gutter,
            state.offset(),
            content_rows.height as usize,
            total,
            t.subtle,
            t.user,
        );
    }
}

pub(in crate::tui::view) fn render_model_picker(
    f: &mut Frame,
    area: Rect,
    app: &App,
    picker: &PickerList<lofi_types::ModelChoice>,
) {
    let rows: Vec<String> = picker
        .items
        .iter()
        .map(|c| {
            let mut s = format!("{}/{}", c.provider, c.id);
            if !c.name.is_empty() && c.name != c.id {
                s.push_str("  ");
                s.push_str(&c.name);
            }
            if !c.thinking_levels.is_empty() {
                s.push_str("  ·thinks");
            }
            if c.supports_image {
                s.push_str("  ·img");
            }
            s
        })
        .collect();
    let current = app.model.switch_label();
    let active = picker
        .items
        .iter()
        .position(|c| format!("{}/{}", c.provider, c.id) == current);
    render_pick_modal(
        f,
        area,
        app.theme,
        " Switch model ",
        " ↑/↓ navigate  enter switch  esc close ",
        picker,
        &rows,
        active,
    );
}

pub(in crate::tui::view) fn render_theme_picker(
    f: &mut Frame,
    area: Rect,
    app: &App,
    picker: &PickerList<lofi_types::ThemeMode>,
) {
    let rows: Vec<String> = picker
        .items
        .iter()
        .map(|m| m.as_str().to_string())
        .collect();
    let active = picker
        .items
        .iter()
        .position(|m| *m == app.theme_manager.mode());
    render_pick_modal(
        f,
        area,
        app.theme,
        " Color scheme ",
        " ↑/↓ navigate  enter apply  esc close ",
        picker,
        &rows,
        active,
    );
}

pub(in crate::tui::view) fn render_thinking_picker(
    f: &mut Frame,
    area: Rect,
    app: &App,
    picker: &PickerList<lofi_types::ThinkingLevel>,
) {
    let rows: Vec<String> = picker
        .items
        .iter()
        .map(|l| l.as_str().to_string())
        .collect();
    let active = picker.items.iter().position(|l| *l == app.model.thinking);
    render_pick_modal(
        f,
        area,
        app.theme,
        " Thinking level ",
        " ↑/↓ navigate  enter apply  esc close ",
        picker,
        &rows,
        active,
    );
}

pub(in crate::tui::view) fn render_paths_picker(
    f: &mut Frame,
    area: Rect,
    app: &App,
    picker: &PickerList<lofi_core::paths::PathRow>,
) {
    let t = app.theme;
    let title = " Allowed paths ";
    let help = " ↑/↓ navigate  space toggle  d forget  state rows only  esc close ";
    let content_w = picker
        .items
        .iter()
        .map(|row| {
            format!(
                "{}  {}  {}",
                crate::tui::text::tilde_path(&row.path),
                row.access.as_str(),
                row.source.label()
            )
        })
        .map(|r| prim::width(&r))
        .max()
        .unwrap_or(0);
    render_list_frame(
        f,
        area,
        t,
        title,
        help,
        picker,
        content_w,
        20,
        |content_width| {
            // Right columns stay fixed-width so access and source stay readable when
            // a path is longer than the dialog is wide; the path column absorbs the
            // shortfall by abbreviating middle components, then truncating.
            let acc_w = prim::width("write");
            let src_w = prim::width("workspace");
            let path_w = content_width.saturating_sub(acc_w + src_w + 4).max(1);
            picker
                .items
                .iter()
                .map(|row| {
                    let mut path = prim::truncate(
                        &crate::tui::text::abbreviate_path(&row.path, path_w),
                        path_w,
                    );
                    path.push_str(&" ".repeat(path_w.saturating_sub(prim::width(&path))));
                    let label = format!(
                        "{}  {:>acc_w$}  {:>src_w$}",
                        path,
                        row.access.as_str(),
                        row.source.label()
                    );
                    let style = if row.source == lofi_core::paths::PathSource::State {
                        t.text_style()
                    } else {
                        t.muted_style()
                    };
                    ListItem::new(Span::styled(label, style))
                })
                .collect()
        },
    );
}

pub(in crate::tui::view) fn render_policy_picker(
    f: &mut Frame,
    area: Rect,
    app: &App,
    picker: &PickerList<lofi_types::BashApprovalMode>,
) {
    let rows: Vec<String> = picker
        .items
        .iter()
        .map(|m| crate::tui::policy_mode_label(*m).to_string())
        .collect();
    let auto_mode = app.auto_mode_configured();
    let current = app.policy_override().map_or_else(
        || lofi_core::default_approval_mode(auto_mode),
        |o| o.effective(auto_mode),
    );
    let active = picker.items.iter().position(|m| *m == current);
    render_pick_modal(
        f,
        area,
        app.theme,
        " Bash policy ",
        " ↑/↓ navigate  enter apply  esc close ",
        picker,
        &rows,
        active,
    );
}

pub(in crate::tui::view) fn render_service_picker(
    f: &mut Frame,
    area: Rect,
    app: &App,
    picker: &PickerList<lofi_types::ServiceTier>,
) {
    let rows: Vec<String> = picker
        .items
        .iter()
        .map(|t| t.as_str().to_string())
        .collect();
    let active = picker
        .items
        .iter()
        .position(|t| *t == app.model.service_tier);
    render_pick_modal(
        f,
        area,
        app.theme,
        " Service tier ",
        " ↑/↓ navigate  enter apply  esc close ",
        picker,
        &rows,
        active,
    );
}

/// A hydrated row carries its own label; a skeleton row renders the
/// placeholder for its kind, so the row list holds no text of its own.
fn tree_entry_label(entry: &TreeEntry) -> &str {
    if entry.label.is_empty() {
        tree_entry_loading_label(entry.source_kind)
    } else {
        &entry.label
    }
}

pub(in crate::tui::view) fn render_tree_picker(
    f: &mut Frame,
    area: Rect,
    app: &App,
    picker: &TreePickerState,
) {
    use ratatui::widgets::ListState;
    let t = app.theme;
    let total = picker.list.len();
    let title = if picker.loading {
        " Loading session tree… "
    } else {
        " Roll back to a turn "
    };
    let help = " ↑/↓ navigate  enter restore  esc close ";
    let desired_rows = total.clamp(1, 20);
    let preliminary_start = picker
        .list
        .selected
        .saturating_sub(desired_rows.saturating_sub(1))
        .min(total.saturating_sub(desired_rows));
    let content_w = picker.list.items
        [preliminary_start..(preliminary_start + desired_rows.min(total))]
        .iter()
        .map(|entry| prim::width(&entry.prefix) + prim::width(tree_entry_label(entry)))
        .max()
        .unwrap_or(0);
    let chrome_w = prim::width(title).max(prim::width(help));
    let w = u16::try_from(content_w.max(chrome_w) + 4)
        .unwrap_or(40)
        .min(area.width);
    let desired_frame_h = u16::try_from(desired_rows + 4).unwrap_or(24);
    let popup = centered_modal(area, w, desired_frame_h);
    f.render_widget(Clear, popup);
    let rows = render_modal_frame(f, popup, t, modal_title(t, title), modal_help(t, help));
    let scroll_area = modal_scroll_area(rows.content);
    let content = scroll_area.content;
    let view_h = content.height as usize;
    let start = picker
        .list
        .selected
        .saturating_sub(view_h.saturating_sub(1))
        .min(total.saturating_sub(view_h));
    let end = (start + view_h).min(total);
    let tree_art = t.subtle_style();
    let items: Vec<ListItem> = picker.list.items[start..end]
        .iter()
        .map(|entry| {
            let label = tree_entry_label(entry);
            let kind_color = if label.starts_with("user:") {
                t.user
            } else if label.starts_with("agent:") {
                t.agent
            } else if label.starts_with("tool:") || label.starts_with("exec:") {
                t.success
            } else {
                t.muted
            };
            let mut label_style = Style::new().fg(kind_color);
            if entry.is_active {
                label_style = label_style.add_modifier(Modifier::BOLD);
            }
            ListItem::new(Line::from(vec![
                Span::styled(entry.prefix.to_string(), tree_art),
                Span::styled(label.to_string(), label_style),
            ]))
        })
        .collect();
    let list = List::new(items)
        .style(t.text_style())
        .highlight_style(t.selected_item_style());
    let selected =
        (picker.list.selected < total).then_some(picker.list.selected.saturating_sub(start));
    let mut state = ListState::default().with_selected(selected);
    f.render_stateful_widget(list, content, &mut state);
    if total > view_h {
        prim::render_scrollbar(
            f,
            scroll_area.gutter,
            start,
            view_h,
            total,
            t.subtle,
            t.user,
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn resume_preview_highlights_role_labels() {
        let t = Theme::dark();
        let user = resume_preview_spans(t, "user: hello".to_string());
        assert_eq!(user[0].content, "user");
        assert_eq!(user[0].style.fg, Some(t.user));
        assert_eq!(user[1].content, ": hello");
        assert_eq!(user[1].style.fg, Some(t.fg));

        let agent = resume_preview_spans(t, "agent: hi".to_string());
        assert_eq!(agent[0].content, "agent");
        assert_eq!(agent[0].style.fg, Some(t.agent));
        assert_eq!(agent[1].content, ": hi");
    }

    #[test]
    fn paths_picker_scrollbar_uses_viewport_offset_not_selected_index() {
        let mut app = App::new(
            "openai/gpt-4o".to_string(),
            lofi_types::ThinkingLevel::Off,
            lofi_types::ServiceTier::Auto,
            0,
            lofi_types::CompactionConfig::default(),
            String::new(),
        );
        let rows: Vec<lofi_core::paths::PathRow> = (0..30)
            .map(|i| lofi_core::paths::PathRow {
                path: std::path::PathBuf::from(format!("/granted/dir_{i:02}")),
                access: lofi_types::Access::Read,
                source: lofi_core::paths::PathSource::State,
            })
            .collect();
        // Selected row 6 fits within the 12-row viewport with offset 0, so the
        // scrollbar thumb must stay anchored at the top row of the gutter.
        app.open_modal(crate::tui::ActiveModal::Paths(PickerList::new(rows, 6)));
        let mut term = Terminal::new(TestBackend::new(80, 16)).unwrap();
        term.draw(|f| crate::tui::view::render(f, &mut app))
            .unwrap();
        let buf = term.backend().buffer();
        let gx = (0..80)
            .find(|&x| (0..16).any(|y| buf[(x, y)].symbol() == "┃"))
            .unwrap();
        let scrollbar_cells: Vec<(u16, &str)> = (0..16)
            .filter_map(|y| {
                let sym = buf[(gx, y)].symbol();
                matches!(sym, "┃" | "│").then_some((y, sym))
            })
            .collect();
        assert!(!scrollbar_cells.is_empty(), "expected paths scrollbar");
        assert_eq!(
            scrollbar_cells[0].1, "┃",
            "thumb must start at the top when viewport offset is 0, got {scrollbar_cells:?}"
        );
    }
}
