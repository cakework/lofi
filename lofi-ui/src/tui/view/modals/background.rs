#[allow(clippy::wildcard_imports)]
use super::*;
use crate::tui::{
    ActiveModal, BackgroundKill, BackgroundModalState, BackgroundTab, BackgroundView,
    JobOutputView, JobRow, JobViewContent, SubagentRow, SubagentTranscriptView,
};
use lofi_types::short_id;
use ratatui::style::Color;

/// Compact `mm:ss` for a job's running/elapsed wall time.
fn fmt_job_duration(ms: u64) -> String {
    let s = ms / 1000;
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    }
}

fn job_color(color: lofi_core::JobColor) -> Option<Color> {
    match color {
        lofi_core::JobColor::Default => None,
        lofi_core::JobColor::Indexed(index) => Some(Color::Indexed(index)),
        lofi_core::JobColor::Rgb(red, green, blue) => Some(Color::Rgb(red, green, blue)),
    }
}

fn job_span_style(attrs: lofi_core::JobStyle) -> Style {
    let mut style = Style::new();
    if let Some(color) = job_color(attrs.foreground) {
        style = style.fg(color);
    }
    if let Some(color) = job_color(attrs.background) {
        style = style.bg(color);
    }
    if attrs.attributes.contains(lofi_core::JobAttributes::BOLD) {
        style = style.add_modifier(Modifier::BOLD);
    }
    if attrs.attributes.contains(lofi_core::JobAttributes::DIM) {
        style = style.add_modifier(Modifier::DIM);
    }
    if attrs.attributes.contains(lofi_core::JobAttributes::ITALIC) {
        style = style.add_modifier(Modifier::ITALIC);
    }
    if attrs
        .attributes
        .contains(lofi_core::JobAttributes::UNDERLINE)
    {
        style = style.add_modifier(Modifier::UNDERLINED);
    }
    if attrs.attributes.contains(lofi_core::JobAttributes::INVERSE) {
        style = style.add_modifier(Modifier::REVERSED);
    }
    style
}

/// The one dialog for running work. It keeps its own state on `App`, so
/// it is taken out for the body: the drill-in renders turn components,
/// whose render context borrows `&App`.
pub(in crate::tui::view) fn render_background_modal(f: &mut Frame, area: Rect, app: &mut App) {
    app.refresh_background_modal();
    let Some(mut modal) = app.take_background_modal() else {
        return;
    };
    if let Some(view) = &mut modal.view {
        render_background_view(f, area, app, view);
    } else {
        render_background_list(f, area, app, &mut modal);
    }
    app.modal = Some(ActiveModal::Background(modal));
}

/// The drill-in of one row: a job's output or a subagent's transcript.
fn render_background_view(f: &mut Frame, area: Rect, app: &App, view: &mut BackgroundView) {
    match view {
        BackgroundView::Job(job) => render_job_output(f, area, app, job),
        BackgroundView::Subagent(subagent) => {
            render_subagent_transcript(f, area, app, subagent);
        }
    }
}

/// The list of the active tab, under the tab bar.
fn render_background_list(f: &mut Frame, area: Rect, app: &App, modal: &mut BackgroundModalState) {
    use ratatui::widgets::ListState;
    let t = app.theme;
    let jobs = app.jobs_rows();
    let (rows, empty) = match modal.tab {
        BackgroundTab::Subagents => (
            modal
                .subagents
                .rows
                .iter()
                .map(|row| (subagent_row(row), row.running))
                .collect::<Vec<_>>(),
            "no subagents",
        ),
        BackgroundTab::Jobs => (
            jobs.iter()
                .map(|job| (job_row(job), job.running()))
                .collect::<Vec<_>>(),
            "no jobs",
        ),
        BackgroundTab::Timers => (
            modal
                .timers
                .rows
                .iter()
                .map(|timer| (timer_row(timer), true))
                .collect::<Vec<_>>(),
            "no timers",
        ),
    };
    let labels = background_tab_labels(modal, jobs.len());
    let tabs_w = labels.iter().map(|label| prim::width(label)).sum::<usize>()
        + 2 * labels.len().saturating_sub(1);
    let title = " background ";
    let help = if modal.confirm.is_some() {
        format!(
            " stop {}? y confirms  any other key cancels ",
            kill_label(modal, &jobs)
        )
    } else {
        " tab or h/l switch  j/k select  enter open  x stop  X stop all  r reload  esc close "
            .to_string()
    };
    let content_w = rows
        .iter()
        .map(|(text, _)| prim::width(text))
        .max()
        .unwrap_or(0)
        .max(tabs_w)
        .max(prim::width(empty));
    let w = u16::try_from(content_w.max(prim::width(title).max(prim::width(&help))) + 4)
        .unwrap_or(40)
        .min(area.width);
    let visible_rows = rows.len().clamp(1, 18);
    let desired_frame_h = u16::try_from(visible_rows + 5).unwrap_or(9);
    let popup = centered_modal(area, w, desired_frame_h);
    f.render_widget(Clear, popup);
    let frame = render_modal_frame(f, popup, t, modal_title(t, title), modal_help(t, &help));
    let chunks = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(frame.content);
    f.render_widget(
        Paragraph::new(background_tab_bar(t, modal, &labels)),
        chunks[0],
    );
    let body = chunks[1];
    if rows.is_empty() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(empty, t.muted_style()))),
            body,
        );
        return;
    }
    let items: Vec<ListItem> = rows
        .iter()
        .map(|(text, running)| {
            let style = if *running {
                t.text_style()
            } else {
                t.muted_style()
            };
            ListItem::new(Span::styled(text.clone(), style))
        })
        .collect();
    let list = List::new(items)
        .style(t.text_style())
        .highlight_style(t.selected_item_style());
    let selected = modal.selected().min(rows.len() - 1);
    let mut state = ListState::default().with_selected(Some(selected));
    f.render_stateful_widget(list, body, &mut state);
    if rows.len() > body.height as usize {
        let scroll_area = modal_scroll_area(body);
        prim::render_scrollbar(
            f,
            scroll_area.gutter,
            state.offset(),
            body.height as usize,
            rows.len(),
            t.subtle,
            t.user,
        );
    }
}

/// Tab labels with the live count of every section, so a user on one tab
/// still sees that work is running elsewhere.
fn background_tab_labels(modal: &BackgroundModalState, jobs: usize) -> Vec<String> {
    BackgroundTab::ALL
        .iter()
        .map(|tab| {
            let count = match tab {
                BackgroundTab::Subagents => modal.subagents.rows.len(),
                BackgroundTab::Jobs => jobs,
                BackgroundTab::Timers => modal.timers.rows.len(),
            };
            format!("{} ({count})", tab.title())
        })
        .collect()
}

fn background_tab_bar(t: Theme, modal: &BackgroundModalState, labels: &[String]) -> Line<'static> {
    let mut spans = Vec::new();
    for (i, (tab, label)) in BackgroundTab::ALL.iter().zip(labels).enumerate() {
        if i > 0 {
            spans.push(Span::raw("  "));
        }
        let style = if *tab == modal.tab {
            t.selected_item_style()
        } else {
            t.muted_style()
        };
        spans.push(Span::styled(label.clone(), style));
    }
    Line::from(spans)
}

/// What the armed kill would stop, for the help line.
fn kill_label(modal: &BackgroundModalState, jobs: &[JobRow]) -> String {
    let selected = modal.selected();
    match &modal.confirm {
        Some(BackgroundKill::All(tab)) => format!("all {}", tab.label()),
        Some(BackgroundKill::Job(_)) => jobs.get(selected).map_or_else(
            || "this job".to_string(),
            |job| {
                let cmd = match job {
                    JobRow::Live(job) => one_line(&job.cmd, 28),
                    JobRow::Stale(job) => job
                        .cmd
                        .as_deref()
                        .map_or_else(String::new, |cmd| one_line(cmd, 28)),
                };
                if cmd.is_empty() {
                    format!("job {}", short_id(job.id()))
                } else {
                    format!("job {} ({cmd})", short_id(job.id()))
                }
            },
        ),
        Some(BackgroundKill::Subagent(id)) => format!("subagent {id}"),
        Some(BackgroundKill::Timer(_)) => "this timer".to_string(),
        None => "this row".to_string(),
    }
}

/// A subagent row states how the run stands, then what the registry knows
/// about it -- model, spend, elapsed time -- and finally what it is doing.
/// Every column keeps its width and a field the row lacks shows a dash, so
/// the labels start on one column, as the commands of a job row do.
fn subagent_row(row: &SubagentRow) -> String {
    let state = if row.running { "running" } else { "done" };
    let model = row
        .model
        .as_deref()
        .map_or_else(|| "-".to_string(), |model| prim::truncate(model, 24));
    let cost = row
        .cost
        .map_or_else(|| "-".to_string(), |cost| format!("${cost:.4}"));
    let elapsed = if row.running {
        prim::fmt_duration(std::time::Duration::from_millis(row.elapsed_ms))
    } else {
        "-".to_string()
    };
    format!(
        "{state:<7} {model:<24} {cost:>7} {elapsed:>6}  {}",
        run_label(&row.preview)
    )
}

/// A run's one-line label. A store preview carries the role a session picker
/// shows -- `agent: …`, `user: …` -- because it lists turns. This list holds
/// runs, so the role only takes width the label needs.
fn run_label(preview: &str) -> String {
    let text = preview
        .strip_prefix("agent: ")
        .or_else(|| preview.strip_prefix("user: "))
        .unwrap_or(preview);
    one_line(text, 48)
}

fn job_row(row: &JobRow) -> String {
    let (state, exit, elapsed, cmd) = match row {
        JobRow::Live(job) => (
            job.state.clone(),
            job.exit_code
                .map_or_else(|| "-".to_string(), |code| code.to_string()),
            fmt_job_duration(job.duration_ms),
            one_line(&job.cmd, 48),
        ),
        // The process, the log, and the duration went with the run that
        // owned them. What it ran is all the transcript could keep.
        JobRow::Stale(job) => (
            "stale".to_string(),
            "-".to_string(),
            "-".to_string(),
            job.cmd.as_deref().map_or_else(
                || "(no command recorded)".to_string(),
                |cmd| one_line(cmd, 48),
            ),
        ),
    };
    format!(
        "{state:<9} {:>5} {exit:>3} {elapsed:>7}  {cmd}",
        short_id(row.id())
    )
}

fn timer_row(timer: &lofi_core::TimerInfo) -> String {
    let due = timer
        .due
        .duration_since(std::time::SystemTime::now())
        .map_or_else(
            |_| "due".to_string(),
            |left| format!("in {}", fmt_job_duration(left.as_millis() as u64)),
        );
    let kind = timer.every.map_or_else(
        || "once".to_string(),
        |every| format!("every {}", fmt_job_duration(every.as_millis() as u64)),
    );
    let message = timer
        .message
        .as_deref()
        .map_or_else(String::new, |message| one_line(message, 40));
    format!("{due:<10} {kind:<14} {message}")
}

/// One row's worth of a command or a message, clipped to a column width.
fn one_line(text: &str, max: usize) -> String {
    prim::truncate(&prim::one_line(text), max)
}

/// Subagent transcript drill-in: the same turn components as the main
/// transcript, rendered per frame so only the visible window is
/// materialized. Scroll is bottom-anchored like the main transcript.
fn render_subagent_transcript(
    f: &mut Frame,
    area: Rect,
    app: &App,
    view: &mut SubagentTranscriptView,
) {
    let t = app.theme;
    let title = format!(" subagent {} transcript ", view.id);
    let help = " ↑/↓ scroll  g top  G bottom  esc back ";
    let w = area.width.saturating_sub(4).clamp(40, 100);
    let h = area.height.saturating_sub(4).clamp(8, 28);
    let popup = centered_modal(area, w, h);
    f.render_widget(Clear, popup);
    let rows = render_modal_frame(f, popup, t, modal_title(t, title), modal_help(t, help));
    let body_h = rows.content.height as usize;
    view.view_h = body_h;

    if let Some(error) = &view.load_error {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                error.clone(),
                Style::new().fg(t.error),
            ))),
            rows.content,
        );
        return;
    }

    // A width change re-measures every turn; a reload re-measures only
    // the turns whose signature changed, which the refresh kept unmeasured.
    let width = rows.content.width as usize;
    if width == 0 || body_h == 0 {
        return;
    }
    let cx = component::Cx {
        app,
        theme: t,
        width,
        active_turn: false,
    };
    if view.width != width {
        // A reflow invalidates every measurement, and the reload
        // compensation has no prior layout to compare against.
        view.heights = vec![None; view.turns.len()];
        view.prior_heights.clear();
        view.width = width;
    }
    // Only the turns the refresh invalidated need a measure pass.
    let changed: Vec<bool> = view.heights.iter().map(Option::is_none).collect();
    for (i, height) in view.heights.iter_mut().enumerate() {
        if height.is_none() {
            *height = Some(blocks::render_turn_height(&cx, &view.turns[i]));
        }
    }
    let heights: Vec<usize> = view.heights.iter().flatten().copied().collect();
    let total = heights.iter().sum::<usize>() + heights.len().saturating_sub(1);
    if view.reloaded && view.scroll > 0 && !view.prior_heights.is_empty() {
        // Scroll is a distance from the bottom. A change strictly below
        // the window top shifts the visible content unless scroll follows
        // it; a change at or above the window top needs no compensation. A
        // bottom-anchored view (scroll == 0) stays pinned instead. The
        // rule is turn-granular: a changed turn straddling the top counts
        // its whole delta even when the divergence sits above the top.
        let end_old = view.total.saturating_sub(view.scroll);
        let start_old = end_old.saturating_sub(body_h);
        let mut drift: isize = 0;
        let mut pos = 0usize;
        for (i, &old_h) in view.prior_heights.iter().enumerate() {
            if i > 0 {
                pos += 1;
            }
            if i < heights.len() && changed[i] && pos + old_h > start_old {
                drift += heights[i].cast_signed() - old_h.cast_signed();
            }
            pos += old_h;
        }
        // Turns past the prior layout were appended at the bottom, always
        // below the window top.
        for (i, &new_h) in heights.iter().enumerate().skip(view.prior_heights.len()) {
            drift += new_h.cast_signed() + isize::from(i > 0);
        }
        if drift > 0 {
            view.scroll = view.scroll.saturating_add(drift.unsigned_abs());
        } else {
            view.scroll = view.scroll.saturating_sub(drift.unsigned_abs());
        }
    }
    view.reloaded = false;
    view.prior_heights.clear();
    view.total = total;
    view.scroll = view.scroll.min(view.total.saturating_sub(body_h));

    let end = view.total.saturating_sub(view.scroll);
    let start = end.saturating_sub(body_h);
    let mut lines: Vec<Line> = Vec::with_capacity(body_h);
    let mut pos = 0usize;
    for (i, turn) in view.turns.iter().enumerate() {
        // The blank separator between turns matches the main transcript.
        if i > 0 {
            if (start..end).contains(&pos) {
                lines.push(Line::default());
            }
            pos += 1;
        }
        let height = heights[i];
        if height > 0 && pos + height > start && pos < end {
            let from = start.saturating_sub(pos);
            let to = end.min(pos + height) - pos;
            lines.extend(
                blocks::render_turn_window(&cx, turn, from..to)
                    .into_iter()
                    .map(|rl| rl.line),
            );
        }
        pos += height;
        if pos >= end {
            break;
        }
    }
    f.render_widget(Paragraph::new(lines), rows.content);
    if view.total > body_h {
        let scroll_area = modal_scroll_area(rows.content);
        prim::render_scrollbar(
            f,
            scroll_area.gutter,
            start,
            body_h,
            view.total,
            t.subtle,
            t.user,
        );
    }
}

/// One job's output: its parsed terminal screen, or its log tail.
fn render_job_output(f: &mut Frame, area: Rect, app: &App, view: &JobOutputView) {
    let t = app.theme;
    match &view.content {
        JobViewContent::Terminal(screen) => {
            let title = format!(" job {} terminal ", short_id(view.id));
            let popup = centered_modal(
                area,
                screen.cols.saturating_add(4),
                screen.rows.saturating_add(4),
            );
            f.render_widget(Clear, popup);
            let rows = render_modal_frame(
                f,
                popup,
                t,
                modal_title(t, title),
                modal_help(t, " esc back "),
            );
            let visible = screen.lines.iter().map(|line| {
                Line::from(
                    line.spans
                        .iter()
                        .map(|span| Span::styled(span.text.clone(), job_span_style(span.style)))
                        .collect::<Vec<_>>(),
                )
            });
            f.render_widget(Paragraph::new(visible.collect::<Vec<_>>()), rows.content);
        }
        JobViewContent::Log { lines, scroll, .. } => {
            let title = format!(" job {} log ", short_id(view.id));
            let help = " up/down scroll  g top  G follow  esc back ";
            let w = area.width.saturating_sub(4).clamp(40, 100);
            let h = area.height.saturating_sub(4).clamp(8, 28);
            let popup = centered_modal(area, w, h);
            f.render_widget(Clear, popup);
            let rows = render_modal_frame(f, popup, t, modal_title(t, title), modal_help(t, help));
            let body_h = rows.content.height as usize;
            let total = lines.len();
            let end = total.saturating_sub(*scroll);
            let start = end.saturating_sub(body_h);
            let visible: Vec<Line> = lines
                .iter()
                .skip(start)
                .take(body_h)
                .map(|line| Line::from(Span::styled(line.clone(), t.text_style())))
                .collect();
            f.render_widget(Paragraph::new(visible), rows.content);
            if total > body_h {
                let scroll_area = modal_scroll_area(rows.content);
                prim::render_scrollbar(
                    f,
                    scroll_area.gutter,
                    start,
                    body_h,
                    total,
                    t.subtle,
                    t.user,
                );
            }
        }
    }
}
