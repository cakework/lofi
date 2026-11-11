use std::time::{Duration, Instant};

/// When the work an owner started may still run.
///
/// One instant, carried from whatever owns work into everything it starts, so a
/// tool never keeps a bound of its own that has to be held in step by hand.
/// `None` is unbounded: nothing here cuts that work.
///
/// Only a timeout the caller set narrows the instant — an `exec` call's
/// `timeoutMs`, a tool's `timeout_ms` — and it can shorten the bound it
/// inherits and never extend it. Nothing else invents a bound: a run has none,
/// so work it starts runs as long as it is useful.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Deadline(Option<Instant>);

impl Deadline {
    /// No bound at all.
    pub const NONE: Self = Self(None);

    /// A bound that ends at `end`.
    #[must_use]
    pub fn at(end: Instant) -> Self {
        Self(Some(end))
    }

    /// This bound narrowed to `timeout` from `now`.
    ///
    /// A timeout too large to represent is no timeout, so a nonsense duration
    /// does not become a panic or an instant in the past.
    #[must_use]
    pub fn bind(self, timeout: Option<Duration>, now: Instant) -> Self {
        let Some(timeout) = timeout else {
            return self;
        };
        let Some(wanted) = now.checked_add(timeout).map(Self::at) else {
            return self;
        };
        match (self.0, wanted.0) {
            (Some(end), Some(wanted)) => Self::at(end.min(wanted)),
            (None, _) => wanted,
            (_, None) => self,
        }
    }

    /// What is left before the bound, saturating at zero.
    #[must_use]
    pub fn remaining(self, now: Instant) -> Option<Duration> {
        self.0.map(|end| end.saturating_duration_since(now))
    }

    /// The instant the bound ends.
    #[must_use]
    pub fn instant(self) -> Option<Instant> {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    #[test]
    fn a_timeout_shortens_the_bound_it_inherits() {
        let now = Instant::now();
        let bound = Deadline::at(now + secs(10));

        assert_eq!(
            bound.bind(Some(secs(3)), now).instant(),
            Some(now + secs(3)),
            "a shorter timeout wins"
        );
        assert_eq!(
            bound.bind(Some(secs(30)), now).instant(),
            Some(now + secs(10)),
            "a timeout never extends the bound it inherits"
        );
        assert_eq!(bound.bind(None, now), bound, "no timeout leaves the bound");
    }

    #[test]
    fn unbounded_work_takes_only_a_timeout() {
        let now = Instant::now();

        assert_eq!(Deadline::NONE.bind(None, now), Deadline::NONE);
        assert_eq!(
            Deadline::NONE.bind(Some(secs(5)), now).instant(),
            Some(now + secs(5)),
            "a timeout is the only thing that bounds unbounded work"
        );
        assert_eq!(
            Deadline::NONE.bind(Some(Duration::MAX), now).instant(),
            None,
            "a timeout too large to represent is no bound"
        );
    }

    #[test]
    fn what_is_left_saturates_at_zero() {
        let now = Instant::now();
        let bound = Deadline::at(now + secs(30));

        assert_eq!(bound.remaining(now), Some(secs(30)));
        assert_eq!(bound.remaining(now + secs(30)), Some(Duration::ZERO));
        assert_eq!(
            bound.remaining(now + secs(31)),
            Some(Duration::ZERO),
            "a passed bound reads zero rather than underflowing"
        );
        assert_eq!(Deadline::NONE.remaining(now + secs(3600)), None);
    }
}
