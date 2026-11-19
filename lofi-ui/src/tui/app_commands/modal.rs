use super::*;

fn choice_label(choice: ConfirmResponse) -> &'static str {
    match choice {
        ConfirmResponse::Allow => "Allow",
        ConfirmResponse::Deny => "Deny",
        ConfirmResponse::Grant {
            access: lofi_types::Access::Read,
            ..
        } => "Read only",
        ConfirmResponse::Grant {
            access: lofi_types::Access::Write,
            ..
        } => "Read & write",
    }
}

impl App {
    pub(in crate::tui) fn open_modal(&mut self, next: ActiveModal) {
        self.close_modal();
        self.modal = Some(next);
    }

    pub(in crate::tui) fn close_modal(&mut self) {
        match self.modal.take() {
            Some(ActiveModal::Resume(_)) => {
                self.picker_generation.fetch_add(1, Ordering::Relaxed);
            }
            Some(ActiveModal::Tree {
                picker, snapshot, ..
            }) => {
                self.picker_generation.fetch_add(1, Ordering::Relaxed);
                self.release_tree_resources(picker.list.items, snapshot);
            }
            _ => {}
        }
    }

    pub(in crate::tui) fn background_modal(&self) -> Option<&BackgroundModalState> {
        match &self.modal {
            Some(ActiveModal::Background(modal)) => Some(modal),
            _ => None,
        }
    }

    pub(in crate::tui) fn background_modal_mut(&mut self) -> Option<&mut BackgroundModalState> {
        match &mut self.modal {
            Some(ActiveModal::Background(modal)) => Some(modal),
            _ => None,
        }
    }

    pub(in crate::tui) fn take_background_modal(&mut self) -> Option<Box<BackgroundModalState>> {
        if matches!(self.modal, Some(ActiveModal::Background(_))) {
            let Some(ActiveModal::Background(modal)) = self.modal.take() else {
                unreachable!()
            };
            Some(modal)
        } else {
            None
        }
    }

    pub(in crate::tui) fn info(&self) -> Option<&InfoModal> {
        match &self.modal {
            Some(ActiveModal::Info(info)) => Some(info),
            _ => None,
        }
    }

    pub(in crate::tui) fn info_mut(&mut self) -> Option<&mut InfoModal> {
        match &mut self.modal {
            Some(ActiveModal::Info(info)) => Some(info),
            _ => None,
        }
    }

    pub(in crate::tui) fn queue_confirmation(&mut self, req: lofi_core::ConfirmRequest) {
        if Self::confirmation_ready(&req) {
            self.confirms.pending.push(req);
        } else {
            self.confirms.deferred.push(req);
        }
    }

    /// Retire completed requests and promote auto-mode evaluations once they
    /// need user attention. The grace period is a presentation policy: core
    /// and tool layers emit the evaluating request immediately.
    pub(in crate::tui) fn refresh_confirmations(&mut self) -> bool {
        let old_front = self.confirms.pending.first().map(|req| req.id);
        let old_pending = self.confirms.pending.len();
        let old_deferred = self.confirms.deferred.len();

        self.confirms
            .pending
            .retain(|req| req.active.load(std::sync::atomic::Ordering::Relaxed));

        let mut waiting = Vec::with_capacity(self.confirms.deferred.len());
        for req in self.confirms.deferred.drain(..) {
            if !req.active.load(std::sync::atomic::Ordering::Relaxed) {
                continue;
            }
            if Self::confirmation_ready(&req) {
                self.confirms.pending.push(req);
            } else {
                waiting.push(req);
            }
        }
        self.confirms.deferred = waiting;

        let new_front = self.confirms.pending.first().map(|req| req.id);
        if old_front != new_front {
            self.confirms.selected = 0;
            self.confirms.narrow = false;
            self.confirms.scroll = 0;
            self.confirms.total = 0;
            self.confirms.view_h = 0;
        }
        // The terminal report counts its debounce from here. Compare ids
        // against the stored front rather than against `old_front`: a request
        // answered between ticks leaves the list before this runs, and the
        // next one has to start its own clock.
        if new_front != self.confirms.waiting.map(|(id, _)| id) {
            self.confirms.waiting = new_front.map(|id| (id, Instant::now()));
        }

        old_pending != self.confirms.pending.len()
            || old_deferred != self.confirms.deferred.len()
            || old_front != new_front
    }

    fn confirmation_ready(req: &lofi_core::ConfirmRequest) -> bool {
        match *req
            .reason
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            lofi_core::ConfirmReason::AutoEvaluating { started_at } => {
                started_at.elapsed() >= AUTO_MODE_UI_GRACE
            }
            _ => true,
        }
    }

    /// Whether a centered modal is open. The inline slash-complete popover is
    /// excluded because input remains active beneath it.
    pub(in crate::tui) fn modal_open(&self) -> bool {
        self.modal.is_some() || !self.confirms.pending.is_empty()
    }

    /// Dispatch a key to the topmost visible popup. This order is the reverse
    /// of the popup render order in `view::render`.
    pub(in crate::tui) fn handle_modal_stack_key(&mut self, k: &KeyEvent) -> bool {
        if !self.confirms.pending.is_empty() {
            return self.handle_confirm_key(k);
        }
        if self.modal.is_some() {
            return self.handle_modal_key(k);
        }
        self.handle_popover_key(k)
    }

    pub(in crate::tui) fn handle_confirm_key(&mut self, k: &KeyEvent) -> bool {
        if self.confirms.pending.is_empty() {
            return false;
        }
        // Auto-mode may have approved between the last draw and this key.
        // Consume the key while dismissing that stale modal rather than
        // letting it leak through to the prompt or a queued confirmation.
        if !self.confirms.pending[0]
            .active
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            self.confirms.pending.remove(0);
            self.confirms.selected = 0;
            self.confirms.narrow = false;
            self.confirms.scroll = 0;
            self.confirms.total = 0;
            self.confirms.view_h = 0;
            return true;
        }
        let choices = self.confirm_choices();
        let len = choices.len();
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let max_scroll = self.confirms.total.saturating_sub(self.confirms.view_h);
        let response = match k.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.confirms.scroll = self.confirms.scroll.saturating_sub(1);
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.confirms.scroll = self.confirms.scroll.saturating_add(1).min(max_scroll);
                None
            }
            KeyCode::PageUp => {
                self.confirms.scroll = self
                    .confirms
                    .scroll
                    .saturating_sub(self.confirms.view_h.max(1));
                None
            }
            KeyCode::PageDown => {
                self.confirms.scroll = self
                    .confirms
                    .scroll
                    .saturating_add(self.confirms.view_h.max(1))
                    .min(max_scroll);
                None
            }
            KeyCode::Home => {
                self.confirms.scroll = 0;
                None
            }
            KeyCode::End => {
                self.confirms.scroll = max_scroll;
                None
            }
            KeyCode::Left | KeyCode::Char('h') => {
                self.confirms.selected = self.confirms.selected.saturating_sub(1);
                None
            }
            KeyCode::Right | KeyCode::Char('l') => {
                self.confirms.selected = (self.confirms.selected + 1).min(len - 1);
                None
            }
            KeyCode::Tab | KeyCode::BackTab => {
                self.confirms.selected = (self.confirms.selected + 1) % len;
                None
            }
            KeyCode::Enter => choices.get(self.confirms.selected.min(len - 1)).copied(),
            KeyCode::Char('a' | 'A' | 'y' | 'Y') => choices.first().copied(),
            KeyCode::Char('w' | 'W') => choices.iter().find(|c| c.is_grant_write()).copied(),
            KeyCode::Char('r' | 'R') => choices.iter().find(|c| c.is_grant_read()).copied(),
            KeyCode::Char('i' | 'I') => {
                self.confirms.narrow = self.confirm_offers_narrow() && !self.confirms.narrow;
                None
            }
            KeyCode::Char('d' | 'D' | 'n' | 'N') | KeyCode::Esc => Some(ConfirmResponse::Deny),
            KeyCode::Char('c') if ctrl => Some(ConfirmResponse::Deny),
            _ => None,
        };
        if let Some(choice) = response {
            let req = self.confirms.pending.remove(0);
            req.active
                .store(false, std::sync::atomic::Ordering::Relaxed);
            let _ = req.respond.send(choice);
            self.confirms.selected = 0;
            self.confirms.narrow = false;
            self.confirms.scroll = 0;
            self.confirms.total = 0;
            self.confirms.view_h = 0;
        }
        true
    }

    /// The button labels for the front confirmation's choices. A Deny that
    /// auto mode will answer on its own carries the seconds left, so the
    /// user can see the prompt will not wait forever.
    pub(in crate::tui) fn confirm_choice_labels(&self) -> Vec<String> {
        let auto_deny_in = self
            .confirms
            .pending
            .first()
            .and_then(|req| {
                *req.deny_at
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
            })
            .map(|deadline| deadline.saturating_duration_since(std::time::Instant::now()))
            .map(|remaining| remaining.as_secs() + u64::from(remaining.subsec_millis() > 0));
        self.confirm_choices()
            .into_iter()
            .map(|choice| match (choice, auto_deny_in) {
                (ConfirmResponse::Deny, Some(secs)) => {
                    format!("Deny ({}s)", secs.max(1))
                }
                _ => choice_label(choice).to_string(),
            })
            .collect()
    }

    /// Whether the front prompt offers the requested directory as a second,
    /// narrower unit beside the project root.
    fn confirm_offers_narrow(&self) -> bool {
        let Some(req) = self.confirms.pending.first() else {
            return false;
        };
        let reason = req
            .reason
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        matches!(
            *reason,
            lofi_core::ConfirmReason::OutsideRoot {
                narrow: Some(_),
                ..
            }
        )
    }

    /// The response choices the front confirmation offers, in button order.
    /// A directory grant for a read offers read-only first, because that is
    /// the least access the operation needs; a write offers only read and
    /// write, because read-only cannot satisfy it.
    fn confirm_choices(&self) -> Vec<ConfirmResponse> {
        use lofi_types::Access;
        let Some(req) = self.confirms.pending.first() else {
            return Vec::new();
        };
        let reason = req
            .reason
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match *reason {
            lofi_core::ConfirmReason::OutsideRoot { requested, .. } => {
                // The buttons choose the access; the scope the answer grants
                // is the toggle the modal carries beside them.
                let narrow = self.confirms.narrow;
                if requested == Access::Read {
                    vec![
                        ConfirmResponse::Grant {
                            access: Access::Read,
                            narrow,
                        },
                        ConfirmResponse::Grant {
                            access: Access::Write,
                            narrow,
                        },
                        ConfirmResponse::Deny,
                    ]
                } else {
                    vec![
                        ConfirmResponse::Grant {
                            access: Access::Write,
                            narrow,
                        },
                        ConfirmResponse::Deny,
                    ]
                }
            }
            _ => vec![ConfirmResponse::Allow, ConfirmResponse::Deny],
        }
    }

    /// Handle keys for the read-only information modal. Copy leaves the modal
    /// open; dismiss and navigation keys are consumed instead of reaching the
    /// prompt.
    pub(in crate::tui) fn handle_info_key(&mut self, k: &KeyEvent) -> bool {
        if self.info().is_none() {
            return false;
        }
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        match k.code {
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter => {
                self.close_modal();
            }
            KeyCode::Char('y') => {
                if let Some(info) = self.info() {
                    let text = info
                        .lines
                        .iter()
                        .map(|line| match line {
                            InfoLine::Section(label) => label.clone(),
                            InfoLine::Text(line) => line
                                .spans
                                .iter()
                                .map(|span| span.content.as_ref())
                                .collect::<String>(),
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    self.yank_text(&text);
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if let Some(i) = self.info_mut() {
                    i.scroll_down();
                }
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if let Some(i) = self.info_mut() {
                    i.scroll_up();
                }
            }
            KeyCode::Char('n') if ctrl => {
                if let Some(i) = self.info_mut() {
                    i.scroll_down();
                }
            }
            KeyCode::Char('p') if ctrl => {
                if let Some(i) = self.info_mut() {
                    i.scroll_up();
                }
            }
            KeyCode::PageDown => {
                if let Some(i) = self.info_mut() {
                    i.scroll_page_down();
                }
            }
            KeyCode::PageUp => {
                if let Some(i) = self.info_mut() {
                    i.scroll_page_up();
                }
            }
            _ => {}
        }
        true
    }

    fn confirm_active_list_modal(&mut self) {
        match self.modal.take() {
            Some(ActiveModal::Resume(picker)) => self.picker_confirm_inner(picker),
            Some(ActiveModal::Tree {
                picker, snapshot, ..
            }) => self.tree_picker_confirm_inner(picker, snapshot),
            Some(ActiveModal::Model(picker)) => self.model_picker_confirm_inner(&picker),
            Some(ActiveModal::Thinking(picker)) => self.thinking_picker_confirm_inner(&picker),
            Some(ActiveModal::Policy(picker)) => self.policy_picker_confirm_inner(&picker),
            Some(ActiveModal::Service(picker)) => self.service_picker_confirm_inner(&picker),
            Some(ActiveModal::Theme(picker)) => self.theme_picker_confirm_inner(&picker),
            other => self.modal = other,
        }
    }

    /// Unified key dispatch for modal overlays.
    pub(in crate::tui) fn handle_modal_key(&mut self, k: &KeyEvent) -> bool {
        match self.modal.as_ref() {
            None => return false,
            Some(ActiveModal::Info(_)) => return self.handle_info_key(k),
            Some(ActiveModal::Background(_)) => return self.handle_background_key(k),
            Some(ActiveModal::Paths(_)) => return self.handle_paths_key(k),
            Some(
                ActiveModal::Resume(_)
                | ActiveModal::Tree { .. }
                | ActiveModal::Model(_)
                | ActiveModal::Thinking(_)
                | ActiveModal::Policy(_)
                | ActiveModal::Service(_)
                | ActiveModal::Theme(_),
            ) => {}
        }
        let len = self.active_list_nav_mut().map_or(0, |m| m.len());
        // A progressive picker draws before it has rows. Escape still closes
        // it, but Enter cannot confirm an absent selection.
        if len == 0 && matches!(k.code, KeyCode::Enter | KeyCode::Tab | KeyCode::BackTab) {
            return true;
        }
        match k.code {
            KeyCode::Enter => {
                self.confirm_active_list_modal();
                return true;
            }
            // With a single entry, Tab/Shift+Tab confirm outright instead of
            // cycling (a no-op) — same as pressing Enter.
            KeyCode::Tab | KeyCode::BackTab if len == 1 => {
                self.confirm_active_list_modal();
                return true;
            }
            KeyCode::Esc | KeyCode::Char('q') => {
                self.close_modal();
                return true;
            }
            _ => {}
        }
        if let Some(nav) = self.active_list_nav_mut() {
            nav.handle_nav_key(k, true);
        }
        if matches!(self.modal, Some(ActiveModal::Tree { .. })) {
            self.request_tree_viewport();
        }
        true
    }

    fn active_list_nav_mut(&mut self) -> Option<&mut dyn ListNav> {
        match self.modal.as_mut()? {
            ActiveModal::Resume(p) => Some(&mut p.list),
            ActiveModal::Tree { picker, .. } => Some(&mut picker.list),
            ActiveModal::Model(m) => Some(m),
            ActiveModal::Thinking(t) => Some(t),
            ActiveModal::Policy(p) => Some(p),
            ActiveModal::Paths(p) => Some(p),
            ActiveModal::Service(s) => Some(s),
            ActiveModal::Theme(t) => Some(t),
            ActiveModal::Background(_) | ActiveModal::Info(_) => None,
        }
    }

    pub(in crate::tui) fn handle_popover_key(&mut self, k: &KeyEvent) -> bool {
        let Some(len) = self.prompt.slash_complete.as_ref().map(PickerList::len) else {
            return false;
        };
        match k.code {
            KeyCode::Enter => {
                self.slash_complete_accept();
                return true;
            }
            // With a single candidate, Tab/Shift+Tab accept it outright
            // instead of cycling (a no-op) — same as pressing Enter.
            KeyCode::Tab | KeyCode::BackTab if len == 1 => {
                self.slash_complete_accept();
                return true;
            }
            KeyCode::Esc => {
                self.prompt.slash_complete = None;
                return true;
            }
            _ => {}
        }
        self.prompt
            .slash_complete
            .as_mut()
            .is_some_and(|popover| popover.handle_nav_key(k, false))
    }
}
