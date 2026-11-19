use super::*;

impl App {
    /// `/model`: open the model-picker overlay populated from the retained
    /// `model.choices`. The current model is pre-selected so the user sees
    /// where they are. No-op (with a notice) when no model is available.
    pub(in crate::tui) fn open_model_picker(&mut self) {
        if self.model.choices.is_empty() {
            self.notify(
                NotifyKind::Warn,
                "no models available (set a provider API key or configure backend models)",
            );
            return;
        }
        let current = self.model.switch_label();
        let selected = self
            .model
            .choices
            .iter()
            .position(|c| format!("{}/{}", c.provider, c.id) == current)
            .unwrap_or(0);
        self.open_modal(ActiveModal::Model(PickerList::new(
            self.model.choices.clone(),
            selected,
        )));
    }

    pub(in crate::tui) fn model_picker_confirm_inner(
        &mut self,
        picker: &PickerList<lofi_types::ModelChoice>,
    ) {
        if let Some(choice) = picker.current() {
            self.model.pending_switch = Some(format!("{}/{}", choice.provider, choice.id));
        }
    }

    #[cfg(test)]
    pub(in crate::tui) fn model_picker_confirm(&mut self) {
        if let Some(ActiveModal::Model(picker)) = self.modal.take() {
            self.model_picker_confirm_inner(&picker);
        }
    }

    /// `/thinking`: open the thinking-level picker for the current model.
    /// Offers `off` plus the model's declared `thinking_levels` (deduped),
    /// pre-selected at the current level. Shows a notice instead of opening
    /// when the current model supports no thinking levels (only `off`).
    pub(in crate::tui) fn open_thinking_picker(&mut self) {
        let levels = self.current_thinking_choices();
        if levels.len() <= 1 {
            self.notify(
                NotifyKind::Info,
                "current model does not support thinking levels",
            );
            return;
        }
        let selected = levels
            .iter()
            .position(|l| l == &self.model.thinking)
            .unwrap_or(0);
        self.open_modal(ActiveModal::Thinking(PickerList::new(levels, selected)));
    }

    fn current_thinking_choices(&self) -> Vec<ThinkingLevel> {
        let mut out = vec![ThinkingLevel::Off];
        let label = self.model.switch_label();
        if let Some(c) = self
            .model
            .choices
            .iter()
            .find(|c| format!("{}/{}", c.provider, c.id) == label)
        {
            for level in &c.thinking_levels {
                if level != &ThinkingLevel::Off && !out.contains(level) {
                    out.push(level.clone());
                }
            }
        }
        out
    }

    /// `/service`: open the service-tier picker for the current model.
    /// Offers `auto` plus the model's declared `service_tiers` (deduped),
    /// pre-selected at the current tier. Shows a notice instead of opening
    /// when the current model declares no service tiers beyond `auto`.
    pub(in crate::tui) fn open_service_picker(&mut self) {
        let tiers = self.current_service_choices();
        if tiers.len() <= 1 {
            self.notify(
                NotifyKind::Info,
                "current model does not support service tiers",
            );
            return;
        }
        let selected = tiers
            .iter()
            .position(|t| t == &self.model.service_tier)
            .unwrap_or(0);
        self.open_modal(ActiveModal::Service(PickerList::new(tiers, selected)));
    }

    fn current_service_choices(&self) -> Vec<ServiceTier> {
        let mut out = vec![ServiceTier::Auto];
        let label = self.model.switch_label();
        if let Some(c) = self
            .model
            .choices
            .iter()
            .find(|c| format!("{}/{}", c.provider, c.id) == label)
        {
            for tier in &c.service_tiers {
                if tier != &ServiceTier::Auto && !out.contains(tier) {
                    out.push(tier.clone());
                }
            }
        }
        out
    }

    pub(in crate::tui) fn service_picker_confirm_inner(
        &mut self,
        picker: &PickerList<ServiceTier>,
    ) {
        if let Some(tier) = picker.current() {
            let query = self.model.format_switch_query(
                self.model.switch_label(),
                &self.model.thinking,
                tier,
            );
            self.model.pending_switch = Some(query);
        }
    }

    #[cfg(test)]
    pub(in crate::tui) fn service_picker_confirm(&mut self) {
        if let Some(ActiveModal::Service(picker)) = self.modal.take() {
            self.service_picker_confirm_inner(&picker);
        }
    }

    /// `/theme`: open the color-scheme picker (Auto / Light / Dark),
    /// pre-selected on the currently active mode.
    pub(in crate::tui) fn open_theme_picker(&mut self) {
        let modes = THEME_MODES.to_vec();
        let selected = modes
            .iter()
            .position(|m| *m == self.theme_manager.mode())
            .unwrap_or(0);
        self.open_modal(ActiveModal::Theme(PickerList::new(modes, selected)));
    }

    pub(in crate::tui) fn theme_picker_confirm_inner(
        &mut self,
        picker: &PickerList<lofi_types::ThemeMode>,
    ) {
        if let Some(&mode) = picker.current() {
            self.update_theme(theme::ThemeEvent::SetMode(mode));
        }
    }

    #[cfg(test)]
    pub(in crate::tui) fn theme_picker_confirm(&mut self) {
        if let Some(ActiveModal::Theme(picker)) = self.modal.take() {
            self.theme_picker_confirm_inner(&picker);
        }
    }

    /// `/paths`: open the allowed-paths manager. Shows the workspace root,
    /// the builtin read roots, the config grants, and the remembered state
    /// grants. Only state rows are editable.
    pub(in crate::tui) fn open_paths_picker(&mut self) {
        let Some(paths) = self.paths() else {
            self.notify(NotifyKind::Error, "no agent: /paths is unavailable");
            return;
        };
        self.open_modal(ActiveModal::Paths(PickerList::new(
            paths.snapshot().rows,
            0,
        )));
    }

    /// Applies a `/paths` row edit through the paths handle and refreshes
    /// the modal's rows from the new snapshot.
    fn paths_apply(
        &mut self,
        edit: impl FnOnce(&lofi_core::paths::PathsHandle) -> lofi_error::Result<()>,
    ) {
        let Some(paths) = self.paths() else {
            return;
        };
        if let Err(error) = edit(&paths) {
            self.notify(NotifyKind::Error, format!("paths change failed: {error}"));
        }
        let rows = paths.snapshot().rows;
        if let Some(ActiveModal::Paths(picker)) = &mut self.modal {
            picker.items = rows;
            picker.clamp();
        }
    }

    /// Key handling for the `/paths` modal. `space`/`enter` toggles a state
    /// row's access, `d` forgets it; other rows only navigate.
    pub(in crate::tui) fn handle_paths_key(&mut self, k: &KeyEvent) -> bool {
        let Some(ActiveModal::Paths(picker)) = &mut self.modal else {
            return false;
        };
        if picker.is_empty() {
            if matches!(k.code, KeyCode::Esc | KeyCode::Char('q')) {
                self.close_modal();
            }
            return true;
        }
        if matches!(k.code, KeyCode::Esc | KeyCode::Char('q')) {
            self.close_modal();
            return true;
        }
        if picker.handle_nav_key(k, true) {
            return true;
        }
        let Some(row) = picker.current().cloned() else {
            return true;
        };
        match k.code {
            KeyCode::Char(' ' | 't') | KeyCode::Enter
                if row.source == lofi_core::paths::PathSource::State =>
            {
                let next = if row.access == lofi_types::Access::Read {
                    lofi_types::Access::Write
                } else {
                    lofi_types::Access::Read
                };
                let dir = row.path.clone();
                self.paths_apply(move |paths| paths.set_grant(&dir, next));
            }
            KeyCode::Char('d' | 'D' | 'x' | 'X')
                if row.source == lofi_core::paths::PathSource::State =>
            {
                let dir = row.path.clone();
                self.paths_apply(move |paths| paths.forget_grant(&dir));
            }
            _ => {}
        }
        true
    }

    /// `/policy`: open the bash approval mode dialog. The choice is
    /// session-scoped and never written to `policy.toml`.
    pub(in crate::tui) fn open_policy_picker(&mut self) {
        let Some(override_handle) = self.policy_override() else {
            self.notify(NotifyKind::Error, "no agent: /policy is unavailable");
            return;
        };
        let auto_mode = self.auto_mode_configured();
        let mut modes = vec![
            lofi_types::BashApprovalMode::AllowAll,
            lofi_types::BashApprovalMode::AskManual,
        ];
        if auto_mode {
            modes.push(lofi_types::BashApprovalMode::AskAuto);
        }
        modes.push(lofi_types::BashApprovalMode::DenyAll);
        let current = override_handle.effective(auto_mode);
        let selected = modes.iter().position(|m| *m == current).unwrap_or(1);
        self.open_modal(ActiveModal::Policy(PickerList::new(modes, selected)));
    }

    pub(in crate::tui) fn policy_picker_confirm_inner(
        &mut self,
        picker: &PickerList<lofi_types::BashApprovalMode>,
    ) {
        if let Some(mode) = picker.current() {
            if let Some(override_handle) = self.policy_override() {
                override_handle.set(Some(*mode));
            }
        }
    }

    #[cfg(test)]
    pub(in crate::tui) fn policy_picker_confirm(&mut self) {
        if let Some(ActiveModal::Policy(picker)) = self.modal.take() {
            self.policy_picker_confirm_inner(&picker);
        }
    }

    pub(in crate::tui) fn thinking_picker_confirm_inner(
        &mut self,
        picker: &PickerList<ThinkingLevel>,
    ) {
        if let Some(level) = picker.current() {
            let query = self.model.format_switch_query(
                self.model.switch_label(),
                level,
                &self.model.service_tier,
            );
            self.model.pending_switch = Some(query);
        }
    }

    #[cfg(test)]
    pub(in crate::tui) fn thinking_picker_confirm(&mut self) {
        if let Some(ActiveModal::Thinking(picker)) = self.modal.take() {
            self.thinking_picker_confirm_inner(&picker);
        }
    }

    pub(in crate::tui) fn apply_model_switch(
        &mut self,
        label: &str,
        level: ThinkingLevel,
        tier: ServiceTier,
        context_window: Option<u64>,
    ) {
        self.model.label = label.to_string();
        self.model.thinking_label =
            (level != ThinkingLevel::Off).then(|| format!(":{}", level.as_str()));
        self.model.thinking = level;
        self.model.service_label =
            (tier != ServiceTier::Auto).then(|| format!("@{}", tier.as_str()));
        self.model.service_tier = tier;
        self.model.ctx_limit = context_window
            .filter(|&l| l > 0)
            .unwrap_or(DEFAULT_CTX_LIMIT);
        self.lifecycle.set_context_window(self.model.ctx_limit);
        self.notify(
            NotifyKind::Info,
            format!(
                "switched to {} for the next turn",
                self.model.session_label()
            ),
        );
        self.bump_render_epoch();
    }
}
