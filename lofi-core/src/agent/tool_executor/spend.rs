/// Child spend of one exec call: the runs it spawned, which the session
/// hook folds at the end, and the spend of the foreground runs it awaited,
/// which their own results carried.
#[derive(Clone, Default)]
pub(super) struct ChildSpend {
    pub(super) spawned: Vec<String>,
    /// Every id this call started, including the ones a resume priced from
    /// their own body: the call releases all of them when it ends.
    pub(super) claimed: Vec<String>,
    pub(super) awaited: crate::agent::subagent::Rollup,
}

impl ChildSpend {
    /// Take one run body's usage. A resume installs the new run's totals over
    /// the entry's, so a run's body is where its own spend is read from.
    pub(super) fn add_run(&mut self, body: &serde_json::Value) {
        if let Some(usage) = body.get("usage") {
            self.awaited.add_usage(usage);
        }
    }

    /// Drop a spawned run from the fold: its spend is priced here instead.
    pub(super) fn forget(&mut self, id: &str) {
        self.spawned.retain(|kept| kept != id);
    }

    pub(super) fn is_empty(&self) -> bool {
        self.spawned.is_empty() && self.awaited.is_empty()
    }

    /// The result's `subagents` record: the session's fold of the spawned
    /// runs plus the runs this call awaited itself.
    pub(super) fn record(&self, folded: Option<&serde_json::Value>) -> serde_json::Value {
        let mut total = self.awaited.clone();
        if let Some(folded) = folded {
            total.merge(folded);
        }
        total.json()
    }
}

/// A bound ending `left` from now, as a call's own `timeout_ms` states it.
#[cfg(test)]
pub(super) fn clock(left: std::time::Duration) -> lofi_types::Deadline {
    lofi_types::Deadline::at(std::time::Instant::now() + left)
}

pub(super) fn clamp_child(
    op: lofi_code::SubagentOp,
    mut args: serde_json::Value,
    clock: lofi_types::Deadline,
) -> (serde_json::Value, Option<u64>) {
    // A wait or a mailbox receive must not block past the call that awaits
    // them, and a foreground child must not outlive it either, so each takes
    // what is left. A spawned run outlives the call that started it, so its
    // own timeout stands as given and one that sets none stays unbounded: it
    // is left alone below. Each kind names its bound with its own key.
    let (key, child, fill) = match op {
        lofi_code::SubagentOp::Run => ("timeout_ms", true, true),
        lofi_code::SubagentOp::Wait => ("timeout_ms", false, true),
        // A wait that names nothing waits for the run; a receive that names
        // nothing returns at once, so its absent bound stays absent.
        lofi_code::SubagentOp::MailboxReceive => ("wait_ms", false, false),
        _ => return (args, None),
    };
    let Some(left) = clock.remaining(std::time::Instant::now()) else {
        return (args, None);
    };
    let asked = match args.get(key) {
        None | Some(serde_json::Value::Null) => None,
        Some(asked) => match asked.as_u64() {
            Some(asked) => Some(asked),
            // A value that is not a positive integer is the hook's to report.
            None => return (args, None),
        },
    };
    if asked.is_none() && !fill {
        return (args, None);
    }
    // A spent clock still asks the hook: a wait resolves at once on a run
    // that has ended, and a child starts and expires rather than hanging.
    // Zero would be read as an invalid argument.
    // Rounded up, like every bound taken from the clock: a wait that ends a
    // millisecond before the wall reads as its own outcome instead of the
    // budget that cut it.
    let left = u64::try_from(left.as_nanos().div_ceil(1_000_000))
        .unwrap_or(u64::MAX)
        .max(1);
    let used = asked.map_or(left, |asked| asked.min(left));
    // A wait returns the run's own shape, so the cap it was cut to is not the
    // wait's to report: the run it names already said what it was given.
    let cut = if child {
        asked.filter(|asked| *asked > left)
    } else {
        None
    };
    if let Some(map) = args.as_object_mut() {
        map.insert(key.to_string(), serde_json::json!(used));
        // What the caller asked for, so the run reports the allowance it
        // was cut from and not only what it was left with.
        if let Some(asked) = cut {
            map.insert("requested_timeout_ms".to_string(), serde_json::json!(asked));
        }
    }
    (args, cut)
}
