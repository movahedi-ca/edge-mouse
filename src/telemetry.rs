//! Stage-level telemetry for the edge engine (mouse-future Phase 0).
//!
//! Stage timers record integer nanoseconds only: no floats on the hot
//! record path. Counters cover retries, cache hits, and escalations.
//! Pure synchronous Rust; no async runtime anywhere near these types.

use std::time::Instant;

/// The instrumented stages of one resolve/act/verify step, in pipeline order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Stage {
    Perceive,
    Normalize,
    Cache,
    Exact,
    Bm25,
    Ranker,
    Convert,
    Move,
    Verify,
    Ocr,
    Escalate,
}

impl Stage {
    /// Number of stages; also the timer slot count.
    pub const COUNT: usize = 11;

    const fn index(self) -> usize {
        self as usize
    }
}

/// A started wall-clock measurement. `elapsed_nanos` returns whole
/// nanoseconds as a u64, saturated on overflow.
#[derive(Debug, Clone)]
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Start measuring now.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Whole nanoseconds elapsed since `start`, integer only.
    pub fn elapsed_nanos(&self) -> u64 {
        self.start.elapsed().as_nanos().min(u64::MAX as u128) as u64
    }
}

/// Per-run accumulation of stage timings and event counters.
#[derive(Debug, Default)]
pub struct RunTelemetry {
    stage_nanos: [u64; Stage::COUNT],
    retries: u64,
    cache_hits: u64,
    escalations: u64,
}

impl RunTelemetry {
    /// Empty telemetry for one run.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `nanos` to the accumulated time for `stage` (saturating).
    pub fn record(&mut self, stage: Stage, nanos: u64) {
        self.stage_nanos[stage.index()] = self.stage_nanos[stage.index()].saturating_add(nanos);
    }

    /// Accumulated nanoseconds for one stage.
    pub fn stage_nanos(&self, stage: Stage) -> u64 {
        self.stage_nanos[stage.index()]
    }

    /// Sum of all stage timers (saturating).
    pub fn total_nanos(&self) -> u64 {
        self.stage_nanos
            .iter()
            .fold(0u64, |acc, &n| acc.saturating_add(n))
    }

    /// Count one retry.
    pub fn inc_retries(&mut self) {
        self.retries = self.retries.saturating_add(1);
    }

    /// Count one identity-cache hit.
    pub fn inc_cache_hits(&mut self) {
        self.cache_hits = self.cache_hits.saturating_add(1);
    }

    /// Count one Tier-1 escalation.
    pub fn inc_escalations(&mut self) {
        self.escalations = self.escalations.saturating_add(1);
    }

    /// Total retries recorded.
    pub fn retries(&self) -> u64 {
        self.retries
    }

    /// Total cache hits recorded.
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits
    }

    /// Total escalations recorded.
    pub fn escalations(&self) -> u64 {
        self.escalations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_accumulates_per_stage() {
        let mut t = RunTelemetry::new();
        t.record(Stage::Bm25, 100);
        t.record(Stage::Bm25, 50);
        t.record(Stage::Cache, 7);
        assert_eq!(t.stage_nanos(Stage::Bm25), 150);
        assert_eq!(t.stage_nanos(Stage::Cache), 7);
        assert_eq!(t.stage_nanos(Stage::Move), 0);
        assert_eq!(t.total_nanos(), 157);
    }

    #[test]
    fn counters_increment() {
        let mut t = RunTelemetry::new();
        t.inc_retries();
        t.inc_retries();
        t.inc_cache_hits();
        t.inc_escalations();
        assert_eq!(t.retries(), 2);
        assert_eq!(t.cache_hits(), 1);
        assert_eq!(t.escalations(), 1);
    }

    #[test]
    fn counters_saturate_at_u64_max() {
        let mut t = RunTelemetry::new();
        t.record(Stage::Move, u64::MAX);
        t.record(Stage::Move, 1);
        assert_eq!(t.stage_nanos(Stage::Move), u64::MAX);
        t.retries = u64::MAX;
        t.inc_retries();
        assert_eq!(t.retries(), u64::MAX);
    }

    #[test]
    fn timer_elapsed_is_integer_nanos() {
        let timer = Timer::start();
        let a = timer.elapsed_nanos();
        let b = timer.elapsed_nanos();
        assert!(b >= a);
    }
}
