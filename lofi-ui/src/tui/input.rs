#![allow(clippy::wildcard_imports)]

use super::*;

#[allow(clippy::too_many_lines, clippy::match_same_arms)]
pub(super) fn handle_event(
    ev: &Event,
    app: &mut App,
    agent: Option<&BackendRef>,
    current_run: &mut Option<RunHandle>,
) {
    if matches!(ev, Event::FocusGained) {
        app.update_theme(theme::ThemeEvent::Refresh);
        return;
    }
    if let Event::Mouse(m) = ev {
        if !app.modal_open() {
            handle_mouse(*m, app);
        }
        return;
    }
    if let Event::Paste(s) = ev {
        if app.viewport.mode == Mode::Input && !app.modal_open() {
            app.viewport.sel = None;
            app.insert_str(s);
        }
        return;
    }
    let Event::Key(k) = ev else {
        return;
    };
    if !matches!(k.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
        return;
    }
    if app.handle_modal_stack_key(k) {
        return;
    }

    if k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL) {
        handle_ctrl_c(app, current_run);
        return;
    }

    // Tab and Escape peel a focused detail before their outer action: Tab
    // does not drop to the input area and Escape does not abort the run
    // while an expansion is open.
    if matches!(k.code, KeyCode::Tab | KeyCode::Esc) && app.detail_focus.is_some() {
        app.collapse_detail_focus();
        return;
    }
    // Escape aborts the run only from the input area. In NAV and SELECT it
    // peels a mode layer instead (SELECT -> NAV -> input).
    if k.code == KeyCode::Esc
        && current_run.is_some()
        && app.viewport.mode == Mode::Input
        && app.prompt.slash_complete.is_none()
    {
        interrupt_run(app, current_run);
        return;
    }

    if app.viewport.mode != Mode::Input {
        app.prompt.last_kill_was_kill = false;
        match app.viewport.mode {
            Mode::Navigate => handle_nav_key(k, app),
            Mode::Select => handle_select_key(k, app),
            Mode::Input => {}
        }
        return;
    }

    app.viewport.sel = None;
    let append_kill = app.prompt.last_kill_was_kill;
    app.prompt.last_kill_was_kill = false;

    if k.code == KeyCode::Enter && k.modifiers.contains(KeyModifiers::ALT) {
        app.insert_newline();
        app.refresh_slash_complete();
        return;
    }
    if k.code == KeyCode::Char('j') && k.modifiers.contains(KeyModifiers::CONTROL) {
        app.insert_newline();
        app.refresh_slash_complete();
        return;
    }
    if app.handle_popover_key(k) {
        return;
    }
    match k.code {
        KeyCode::Enter
            if (current_run.is_some() || app.lifecycle_busy) && !app.prompt.text.is_empty() =>
        {
            app.debug_mark_submit();
            let prompt = std::mem::take(&mut app.prompt.text);
            app.prompt.cursor = 0;
            app.prompt.history_idx = None;
            app.prompt.slash_complete = None;
            if app.slash_command(&prompt) {
                return;
            }
            app.prompt.history.push(prompt.clone());
            app.prompt_queue.push(QueuedPrompt {
                text: prompt,
                kind: lofi_types::PromptKind::User,
            });
            // Signal the live run to yield after its current round. Setting the
            // flag here (not on the next event) closes the race against a
            // blocking tool call during which no events flow.
            if let Some(r) = current_run {
                r.preempt.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            // The queued prompt renders in the transcript tail; a scrolled-up
            // view would leave it below the viewport.
            app.pin_to_latest();
            return;
        }
        KeyCode::Enter if current_run.is_none() && !app.prompt.text.is_empty() => {
            app.debug_mark_submit();
            let prompt = std::mem::take(&mut app.prompt.text);
            app.prompt.cursor = 0;
            app.prompt.history_idx = None;
            app.prompt.slash_complete = None;
            if app.slash_command(&prompt) {
                return;
            }
            app.prompt.history.push(prompt.clone());
            if let Some((command, exclude_from_context)) = parse_user_shell(&prompt) {
                spawn_user_shell(app, current_run, command, exclude_from_context);
                return;
            }
            let Some(agent) = agent else {
                app.transcript.push_turn(Turn {
                    prompt,
                    kind: lofi_types::PromptKind::User,
                    blocks: Vec::new(),
                });
                if let Some(hint) = &app.model.no_models_hint {
                    if let Some(turn) = app.transcript.turns.last_mut() {
                        turn.blocks.push(Block::Error(hint.clone()));
                    }
                }
                app.pin_to_latest();
                return;
            };
            spawn_agent_run(
                app,
                current_run,
                agent,
                Some(prompt),
                lofi_types::PromptKind::User,
            );
        }
        KeyCode::Backspace if k.modifiers.contains(KeyModifiers::ALT) => app.kill_word_back(),
        KeyCode::Backspace => app.backspace(),
        KeyCode::Delete => app.delete_forward_char(),
        KeyCode::Left => app.move_left(),
        KeyCode::Right => app.move_right(),
        KeyCode::Up if k.modifiers.contains(KeyModifiers::ALT) => {
            // Pop the most recent queued prompt for editing. Skip Notices:
            // system-injected text has no business round-tripping through
            // the editor where the user could resubmit it as if typed.
            let mut user_idx = None;
            for (i, q) in app.prompt_queue.iter().enumerate().rev() {
                if q.kind == lofi_types::PromptKind::User {
                    user_idx = Some(i);
                    break;
                }
            }
            if let Some(i) = user_idx {
                let queued = app.prompt_queue.remove(i);
                app.prompt.text = queued.text;
                app.prompt.cursor = app.prompt.text.len();
                app.prompt.history_idx = None;
                app.refresh_slash_complete();
            }
        }
        KeyCode::Up => app.cursor_up(),
        KeyCode::Down => app.cursor_down(),
        KeyCode::Tab => app.enter_nav(),
        KeyCode::Esc => app.clear_input(),
        KeyCode::PageUp => app.page_up(),
        KeyCode::PageDown => app.page_down(),
        KeyCode::Char('a') if k.modifiers.contains(KeyModifiers::CONTROL) => app.move_line_start(),
        KeyCode::Char('e') if k.modifiers.contains(KeyModifiers::CONTROL) => app.move_line_end(),
        // ctrl+b opens the background dialog on an empty prompt, so the
        // readline binding still works while the user is typing.
        KeyCode::Char('b')
            if k.modifiers.contains(KeyModifiers::CONTROL) && app.prompt.text.is_empty() =>
        {
            let tab = app.newest_background_tab();
            app.open_background_modal(tab);
        }
        KeyCode::Char('b') if k.modifiers.contains(KeyModifiers::CONTROL) => app.move_left(),
        KeyCode::Char('f') if k.modifiers.contains(KeyModifiers::CONTROL) => app.move_right(),
        KeyCode::Char('n') if k.modifiers.contains(KeyModifiers::CONTROL) => app.cursor_down(),
        KeyCode::Char('p') if k.modifiers.contains(KeyModifiers::CONTROL) => app.cursor_up(),
        KeyCode::Char('h') if k.modifiers.contains(KeyModifiers::CONTROL) => app.backspace(),
        KeyCode::Char('k') if k.modifiers.contains(KeyModifiers::CONTROL) => {
            app.kill_line_end(append_kill);
        }
        KeyCode::Char('u') if k.modifiers.contains(KeyModifiers::CONTROL) => app.kill_line_start(),
        KeyCode::Char('w') if k.modifiers.contains(KeyModifiers::CONTROL) => app.kill_word_back(),
        KeyCode::Char('y') if k.modifiers.contains(KeyModifiers::CONTROL) => app.yank(),
        KeyCode::Char('b') if k.modifiers.contains(KeyModifiers::ALT) => app.move_word_back(),
        KeyCode::Char('f') if k.modifiers.contains(KeyModifiers::ALT) => app.move_word_fwd(),
        KeyCode::Char('d') if k.modifiers.contains(KeyModifiers::ALT) => app.kill_word_fwd(),
        KeyCode::Char('<') if k.modifiers.contains(KeyModifiers::ALT) => app.prompt.cursor = 0,
        KeyCode::Char('>') if k.modifiers.contains(KeyModifiers::ALT) => {
            app.prompt.cursor = app.prompt.text.len();
        }
        KeyCode::Char('d') if k.modifiers.contains(KeyModifiers::CONTROL) => {
            if app.prompt.text.is_empty() {
                request_quit(app, current_run);
            } else {
                app.delete_forward_char();
            }
        }

        KeyCode::Char(c)
            if !k
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            app.insert_char(c);
        }
        _ => {}
    }
    app.refresh_slash_complete();
}

pub(super) fn handle_ctrl_c(app: &mut App, current_run: &mut Option<RunHandle>) {
    // Navigate/Select: Ctrl-C is a "give me the prompt" key, not a cancel. It
    // drops the user onto the newest transcript line and focuses the editor
    // without ever interrupting the current turn.
    if app.viewport.mode != Mode::Input {
        app.enter_input();
        app.pin_to_latest();
        app.ctrl_c_at = None;
        return;
    }
    // A non-empty prompt is the more immediate thing to dismiss; clearing it
    // also takes priority over cancelling the turn so a single Ctrl-C never
    // both wipes text and kills a run.
    if !app.prompt.text.is_empty() {
        app.clear_input();
        app.ctrl_c_at = None;
        return;
    }
    if current_run.is_some() {
        // A cancel the turn never answers leaves the key with nothing to do:
        // it only raises the flag, and nothing is polling it. Escalate to
        // quit, which gives the turn one short flush budget and then detaches
        // its task.
        if app.cancel_unacknowledged() {
            request_quit(app, current_run);
        } else {
            interrupt_run(app, current_run);
        }
        app.ctrl_c_at = None;
        return;
    }
    let now = Instant::now();
    if app
        .ctrl_c_at
        .is_some_and(|t| now.duration_since(t) < QUIT_DOUBLE_PRESS)
    {
        app.should_quit = true;
    } else {
        app.ctrl_c_at = Some(now);
    }
}

#[derive(Clone, Copy)]
pub(super) enum WordMotion {
    NextStart { big: bool },
    PrevStart { big: bool },
    NextEnd { big: bool },
}

/// Dispatches the vim line-movement keys (`$`, `^`, `0`, `w`/`b`/`e`/`W`/`B`/`E`).
/// Returns `true` when the key was a motion so the caller can skip its own
/// match. Shared by Navigate and Select.
pub(super) fn apply_motion(k: &KeyEvent, app: &mut App) -> bool {
    match k.code {
        KeyCode::Char('$') => {
            let (_, cend) = app.cursor_content_range();
            app.nav_set_col(cend.saturating_sub(1));
            true
        }
        KeyCode::Char('0') => {
            let (cstart, _) = app.cursor_content_range();
            app.nav_set_col(cstart);
            true
        }
        KeyCode::Char('^') => {
            app.nav_set_col(app.first_nonblank_col());
            true
        }
        KeyCode::Char('w') => {
            app.nav_word_motion(WordMotion::NextStart { big: false });
            true
        }
        KeyCode::Char('W') => {
            app.nav_word_motion(WordMotion::NextStart { big: true });
            true
        }
        KeyCode::Char('b') => {
            app.nav_word_motion(WordMotion::PrevStart { big: false });
            true
        }
        KeyCode::Char('B') => {
            app.nav_word_motion(WordMotion::PrevStart { big: true });
            true
        }
        KeyCode::Char('e') => {
            app.nav_word_motion(WordMotion::NextEnd { big: false });
            true
        }
        KeyCode::Char('E') => {
            app.nav_word_motion(WordMotion::NextEnd { big: true });
            true
        }
        _ => false,
    }
}

pub(super) fn handle_nav_key(k: &KeyEvent, app: &mut App) {
    if k.modifiers.contains(KeyModifiers::CONTROL) {
        return;
    }
    if k.modifiers.contains(KeyModifiers::SHIFT) && matches!(k.code, KeyCode::Up | KeyCode::Down) {
        return;
    }
    if apply_motion(k, app) {
        return;
    }
    match k.code {
        KeyCode::Char('j') | KeyCode::Down => {
            app.viewport.sel = None;
            app.nav_move(1);
        }
        KeyCode::Char('k') | KeyCode::Up => {
            app.viewport.sel = None;
            app.nav_move(-1);
        }
        KeyCode::Char('h') | KeyCode::Left => {
            app.viewport.sel = None;
            app.nav_col_delta(-1);
        }
        KeyCode::Char('l') | KeyCode::Right => {
            app.viewport.sel = None;
            app.nav_col_delta(1);
        }
        KeyCode::Enter | KeyCode::Char(' ') => {
            app.viewport.sel = None;
            if !app.toggle_cursor_detail() && k.code == KeyCode::Enter {
                app.yank_line();
                app.enter_input();
            }
        }
        KeyCode::Char('g') => {
            app.viewport.sel = None;
            app.nav_top();
        }
        KeyCode::Char('G') => {
            app.viewport.sel = None;
            app.nav_bottom();
        }
        KeyCode::Char('v') => app.enter_select(),
        KeyCode::Char('y') => {
            app.yank_line();
            app.enter_input();
        }
        KeyCode::Char('i') | KeyCode::Tab | KeyCode::Esc => app.enter_input(),
        KeyCode::Char('[') if app.detail_focus.is_none() => app.nav_jump_turn(-1),
        KeyCode::Char(']') if app.detail_focus.is_none() => app.nav_jump_turn(1),
        KeyCode::PageUp => app.page_up(),
        KeyCode::PageDown => app.page_down(),
        _ => {}
    }
}

pub(super) fn handle_select_key(k: &KeyEvent, app: &mut App) {
    if k.modifiers.contains(KeyModifiers::CONTROL) {
        return;
    }
    if k.modifiers.contains(KeyModifiers::SHIFT) && matches!(k.code, KeyCode::Up | KeyCode::Down) {
        return;
    }
    if apply_motion(k, app) {
        return;
    }
    match k.code {
        KeyCode::Enter | KeyCode::Char(' ') if app.detail_focus.is_some() => {
            app.toggle_cursor_detail();
        }
        KeyCode::Char('j') | KeyCode::Down => app.nav_move(1),
        KeyCode::Char('k') | KeyCode::Up => app.nav_move(-1),
        KeyCode::Char('h') | KeyCode::Left => app.nav_col_delta(-1),
        KeyCode::Char('l') | KeyCode::Right => app.nav_col_delta(1),
        KeyCode::Char('g') => app.nav_top(),
        KeyCode::Char('G') => app.nav_bottom(),
        KeyCode::Char('y') | KeyCode::Enter => {
            app.yank_selection();
            app.enter_input();
        }
        KeyCode::Tab | KeyCode::Esc => {
            app.viewport.mode = Mode::Navigate;
            app.viewport.sel = None;
        }
        KeyCode::Char('[') if app.detail_focus.is_none() => app.nav_jump_turn(-1),
        KeyCode::Char(']') if app.detail_focus.is_none() => app.nav_jump_turn(1),
        KeyCode::PageUp => app.page_up(),
        KeyCode::PageDown => app.page_down(),
        _ => {}
    }
}

pub(super) fn handle_mouse(m: MouseEvent, app: &mut App) {
    let in_log = m.row >= app.viewport.log_rect.y
        && m.row < app.viewport.log_rect.y + app.viewport.log_rect.height
        && m.column >= app.viewport.log_rect.x
        && m.column < app.viewport.log_rect.x + app.viewport.log_rect.width;
    let can_select = app.viewport.mode == Mode::Input || app.viewport.mode == Mode::Navigate;
    let detail_at = |app: &App, row: u16| {
        let rel = row.saturating_sub(app.viewport.log_rect.y) as usize;
        app.viewport.log_details.get(rel)?.clone()
    };
    match m.kind {
        MouseEventKind::ScrollUp | MouseEventKind::ScrollDown if in_log => {
            let delta = if matches!(m.kind, MouseEventKind::ScrollUp) {
                -1
            } else {
                1
            };
            let cell = log_cell(app, m.row, m.column);
            app.viewport.nav_cursor = cell.0;
            if let Some(detail) = detail_at(app, m.row) {
                if app.viewport.mode == Mode::Navigate {
                    app.detail_focus = Some(DetailFocus::on(&detail, cell.1));
                }
            } else {
                app.collapse_detail_focus();
            }
            if !app.scroll_cursor_detail(delta) {
                app.scroll_nav(if delta < 0 { -3 } else { 3 });
            }
        }
        MouseEventKind::Down(MouseButton::Left) if can_select => {
            app.viewport.sel = None;
            if in_log {
                let cell = log_cell(app, m.row, m.column);
                if app.viewport.mode == Mode::Navigate {
                    if let Some(detail) = detail_at(app, m.row) {
                        if detail.row.is_some() {
                            app.detail_focus = Some(DetailFocus::on(&detail, cell.1));
                            app.viewport.sel = Some(Selection {
                                start: cell,
                                end: cell,
                            });
                            return;
                        }
                    }
                }
                app.viewport.sel = Some(Selection {
                    start: cell,
                    end: cell,
                });
            }
        }
        MouseEventKind::Drag(MouseButton::Left) if in_log && app.detail_focus.is_some() => {
            let cell = log_cell(app, m.row, m.column);
            if let Some(detail) = detail_at(app, m.row) {
                if let Some(focus) = app.detail_focus.as_mut() {
                    if detail.key == focus.key {
                        if let Some(row) = detail.row {
                            focus.cursor = row;
                            focus.col = cell.1;
                            app.viewport.nav_cursor = cell.0;
                            app.viewport.nav_col = cell.1;
                            if let Some(sel) = app.viewport.sel.as_mut() {
                                sel.end = cell;
                            }
                        }
                    }
                }
            }
        }
        MouseEventKind::Drag(MouseButton::Left) if in_log && can_select => {
            let cell = log_cell(app, m.row, m.column);
            if let Some(sel) = app.viewport.sel.as_mut() {
                sel.end = cell;
            }
        }
        MouseEventKind::Up(MouseButton::Left) if can_select => {
            app.yank_selection();
            // Return to Input mode after a drag-yank so the user can
            // immediately type (mouse drag is a quick-peek action).
            app.enter_input();
        }
        _ => {}
    }
}

/// Map a screen cell inside the log viewport to (select-line index, char
/// index) in `log_vis`, using display width so wide chars land correctly.
/// The char index is clamped to the line's content range so a press/drag in
/// the gutter or padding snaps to the content edge — the selection never
/// starts or ends in the decorative whitespace.
pub(super) fn log_cell(app: &App, row: u16, column: u16) -> (usize, usize) {
    let rel_y = row.saturating_sub(app.viewport.log_rect.y) as usize;
    let rel_x = column.saturating_sub(app.viewport.log_rect.x) as usize;
    let line_idx = app.viewport.log_off.saturating_add(rel_y);
    let line = app.viewport.log_vis.get(rel_y);
    let col = line.map_or(rel_x, |vl| {
        let s = &vl.rendered;
        let (cstart, cend) = vl.content;
        col_to_char_idx(s, rel_x).clamp(cstart, cend)
    });
    (line_idx, col)
}
