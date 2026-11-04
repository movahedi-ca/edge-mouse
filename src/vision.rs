//! VM-local vision for the edge engine (mouse-future Phase 2, Layer 5).
//!
//! Validated captures (failures are errors, never silent zeros),
//! box-filter 64-bit dHash with correct BGRX luma, predicate-based
//! tri-state verification, damage-driven settle with a measured
//! fixed-sleep fallback, and password-field zeroing before any screenshot
//! bytes leave the VM.
//!
//! Pure synchronous Rust: no async runtime on this path. The XDamage
//! probe needs a live X connection and is only constructed on the VM via
//! [`auto_probe`]; unit tests drive [`ScriptedProbe`] instead.
//!
//! Verification is predicate-based (B05): each step declares what should
//! be true. Change signals (subtree diff, pixel hash) are supporting
//! evidence only and never independently imply `Confirmed`.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::{ensure, Result};

use super::predicates::{EvalCtx, Predicate, PredicateValue};
use super::telemetry::{RunTelemetry, Stage, Timer};

// ---------------------------------------------------------------------------
// Captures: validated, never silent zeros.
// ---------------------------------------------------------------------------

/// One raw frame in XImage layout: 4 bytes per pixel, BGRX order.
#[derive(Debug, Clone)]
pub struct Capture {
    pub buf: Vec<u8>,
    /// XImage bytes_per_line; never assumed to equal `w * 4`.
    pub stride: usize,
    pub w: u32,
    pub h: u32,
}

impl Capture {
    /// Zero the password-field boxes from `ctx` in place. Call before any
    /// bytes of this capture leave the VM (escalation payload, artifact).
    pub fn redact_passwords(&mut self, ctx: &EvalCtx) {
        redact_password_fields(&mut self.buf, self.stride, &ctx.password_boxes());
    }
}

/// H18: validate BEFORE filtering. A malformed capture is an error, not a
/// misleading hash.
pub fn validate_capture(c: &Capture) -> Result<()> {
    ensure!(
        c.w > 0 && c.h > 0,
        "vision: empty capture ({}x{})",
        c.w,
        c.h
    );
    ensure!(
        c.stride >= c.w as usize * 4,
        "vision: stride {} < row width {}",
        c.stride,
        c.w as usize * 4
    );
    let need = c.stride * (c.h as usize - 1) + c.w as usize * 4;
    ensure!(
        c.buf.len() >= need,
        "vision: buffer {} too short for {}x{} stride {} (need {})",
        c.buf.len(),
        c.w,
        c.h,
        c.stride,
        need
    );
    Ok(())
}

/// Box-filter the capture down to a 9x8 luma grid. Each cell is the mean
/// BT.601 luma of its pixel block; blocks are sized by integer division
/// so every source pixel belongs to exactly one cell.
pub fn box_filter_9x8(c: &Capture) -> Result<[[u32; 9]; 8]> {
    validate_capture(c)?;
    let (w, h) = (c.w as usize, c.h as usize);
    let mut cells = [[0u32; 9]; 8];
    for (row, cells_row) in cells.iter_mut().enumerate() {
        for (col, cell) in cells_row.iter_mut().enumerate() {
            let x0 = (col * w) / 9;
            let x1 = ((col + 1) * w) / 9;
            let y0 = (row * h) / 8;
            let y1 = ((row + 1) * h) / 8;
            let x1e = x1.max(x0 + 1);
            let y1e = y1.max(y0 + 1);
            // v1.2, fix 5: the per-pixel bounds check was provably
            // redundant (validate_capture guarantees the buffer covers
            // stride*(h-1) + w*4 bytes, and every (x, y) here is inside
            // [0, w) x [0, h)). One check per cell; the inner loop runs
            // over slices so LLVM can vectorize the luma accumulation.
            ensure!(
                (y1e - 1) * c.stride + (x1e - 1) * 4 + 2 < c.buf.len(),
                "vision: cell ({row},{col}) out of bounds"
            );
            let (mut acc, mut n) = (0u64, 0u64);
            for y in y0..y1e {
                let row_start = y * c.stride;
                let row_px = &c.buf[row_start + x0 * 4..row_start + x1e * 4];
                // X11 pixels are BGRX: 29*B + 150*G + 77*R.
                acc += row_px
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| 29 * p[0] as u64 + 150 * p[1] as u64 + 77 * p[2] as u64)
                    .sum::<u64>();
                n += (x1e - x0) as u64;
            }
            *cell = (acc / n.max(1)) as u32 >> 8;
        }
    }
    Ok(cells)
}

/// 64-bit dHash over the 9x8 grid. The bit counter increments ONLY when a
/// comparison is stored: 9 columns hold 8 comparisons, 8 rows, 64 bits
/// total. Shifting a u64 by >= 64 panics, so the counter must never reach
/// 64; the debug_assert pins the 64-comparison contract.
pub fn dhash_9x8(cells: &[[u32; 9]; 8]) -> u64 {
    let mut hash = 0u64;
    let mut bit = 0u32;
    for row in cells.iter() {
        for col in 0..8 {
            if row[col + 1] > row[col] {
                hash |= 1u64 << bit;
            }
            bit += 1;
        }
    }
    debug_assert_eq!(bit, 64, "dHash must store exactly 64 comparisons");
    hash
}

/// Hamming distance between two hashes.
#[inline]
pub fn hash_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Mean luma of the 9x8 grid (v1.2, fix 16). dHash is blind to uniform
/// color changes (a white->black flip has Hamming distance 0); the mean
/// luma is level-sensitive and catches them. The two hashes are
/// complementary: dHash for structure, luma for level.
pub fn mean_luma_9x8(cells: &[[u32; 9]; 8]) -> f32 {
    let mut sum = 0u64;
    for row in cells.iter() {
        for &c in row.iter() {
            sum += c as u64;
        }
    }
    sum as f32 / 72.0
}

/// Luma difference threshold: a mean-luma shift above this counts as a
/// level change even when the dHash distance is 0 (uniform region).
pub const LUMA_CUTOFF: f32 = 12.0;

// ---------------------------------------------------------------------------
// Verdicts, evidence, stats.
// ---------------------------------------------------------------------------

/// B05: the verifier's verdict. Only `Confirmed` may cache or commit;
/// only `Unchanged` may retry (never on `commit: true` or non-idempotent
/// actions); `Inconclusive` escalates with evidence and never clicks;
/// `AlreadySatisfied` never clicks. The runner enforces these rules; this
/// module only reports honestly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The declared predicate is true after the action.
    Confirmed,
    /// State demonstrably stable and the predicate false.
    Unchanged,
    /// Predicate unknown, pixels moving, or settle timed out.
    Inconclusive,
    /// Predicate true before the action: do not click.
    AlreadySatisfied,
}

/// Which settle path a verification took. Recorded per step in
/// [`VerifyEvidence::settle`] and aggregated in [`VerifierStats`]: the
/// fixed-sleep fallback must stay visible, never silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettlePath {
    XDamage,
    FixedSleep,
}

/// Per-step verification evidence for the runner.
#[derive(Debug, Clone)]
pub struct VerifyEvidence {
    pub verdict: Verdict,
    pub settle: Option<SettleOutcome>,
    /// Hamming distance baseline -> frame1, when frames were compared.
    pub frame_distance: Option<u32>,
}

/// Aggregate verifier telemetry, flushed into the run journal evidence by
/// the runner. `RunTelemetry` (telemetry.rs) stays stage-timer shaped;
/// settle-path visibility lives here.
#[derive(Debug, Default, Clone)]
pub struct VerifierStats {
    pub xdamage_settles: u64,
    pub fixed_sleep_fallbacks: u64,
    pub confirmed: u64,
    pub unchanged: u64,
    pub inconclusive: u64,
    pub already_satisfied: u64,
}

impl VerifierStats {
    fn note(&mut self, ev: &VerifyEvidence) {
        if let Some(s) = &ev.settle {
            match s.path {
                SettlePath::XDamage => self.xdamage_settles += 1,
                SettlePath::FixedSleep => self.fixed_sleep_fallbacks += 1,
            }
        }
        match ev.verdict {
            Verdict::Confirmed => self.confirmed += 1,
            Verdict::Unchanged => self.unchanged += 1,
            Verdict::Inconclusive => self.inconclusive += 1,
            Verdict::AlreadySatisfied => self.already_satisfied += 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Damage-driven settle.
// ---------------------------------------------------------------------------

/// Damage feed probe. `poll_fraction` returns the fraction of the screen
/// that changed since the last call, or `None` when the feed is
/// unavailable or broke mid-wait.
pub trait DamageProbe {
    fn poll_fraction(&mut self) -> Option<f64>;
    fn path(&self) -> SettlePath;
}

#[derive(Debug, Clone, Copy)]
pub struct SettleOutcome {
    pub stable: bool,
    pub path: SettlePath,
    pub elapsed_ms: u64,
}

/// Fixed-sleep fallback: used ONLY where XDamage is unavailable. 150ms is
/// the midpoint of the 50-300ms damage bound.
pub const FIXED_FALLBACK_SLEEP_MS: u64 = 150;
/// Damage poll cadence.
pub const DAMAGE_POLL_MS: u64 = 4;
/// Changed-pixel fraction that counts as "still moving".
pub const DAMAGE_FRACTION: f64 = 0.001;
/// Consecutive quiet polls required before proceeding.
pub const DAMAGE_QUIET_POLLS: u32 = 2;
/// Settle wait bounds.
pub const SETTLE_FLOOR_MS: u64 = 50;
pub const SETTLE_CEILING_MS: u64 = 300;

/// Damage-driven settle. Proceeds when changed pixels stay below `frac`
/// for `quiet` consecutive polls, no earlier than `floor_ms` and no later
/// than `ceiling_ms`, polling every `poll_ms`.
///
/// A probe on the fixed-sleep path sleeps [`FIXED_FALLBACK_SLEEP_MS`]
/// once and reports stable: the fallback is honest about what it is, and
/// the path is recorded in the outcome. A probe whose feed breaks
/// mid-wait reports unstable so the caller goes `Inconclusive` instead of
/// guessing.
pub fn wait_stable(
    probe: &mut dyn DamageProbe,
    frac: f64,
    quiet: u32,
    floor_ms: u64,
    ceiling_ms: u64,
    poll_ms: u64,
) -> SettleOutcome {
    let start = Instant::now();
    let elapsed_ms = || start.elapsed().as_millis().min(u64::MAX as u128) as u64;
    if probe.path() == SettlePath::FixedSleep {
        std::thread::sleep(Duration::from_millis(FIXED_FALLBACK_SLEEP_MS));
        return SettleOutcome {
            stable: true,
            path: SettlePath::FixedSleep,
            elapsed_ms: elapsed_ms(),
        };
    }
    let quiet = quiet.max(1);
    let mut good = 0u32;
    loop {
        std::thread::sleep(Duration::from_millis(poll_ms));
        let elapsed = elapsed_ms();
        match probe.poll_fraction() {
            None => {
                return SettleOutcome {
                    stable: false,
                    path: probe.path(),
                    elapsed_ms: elapsed,
                }
            }
            Some(f) => {
                if f < frac {
                    good += 1;
                } else {
                    good = 0;
                }
                if good >= quiet && elapsed >= floor_ms {
                    return SettleOutcome {
                        stable: true,
                        path: probe.path(),
                        elapsed_ms: elapsed,
                    };
                }
                if elapsed >= ceiling_ms {
                    return SettleOutcome {
                        stable: false,
                        path: probe.path(),
                        elapsed_ms: elapsed,
                    };
                }
            }
        }
    }
}

/// Live XDamage probe. Accumulates `DamageNotify` bounding-box areas per
/// poll window as an upper bound on the changed-pixel fraction (areas may
/// overlap, so this overestimates rather than misses motion). Created
/// only where the server offers the DAMAGE extension; otherwise
/// [`XDamageProbe::connect`] returns `None` and the caller takes the
/// fixed-sleep fallback.
pub struct XDamageProbe {
    conn: x11rb::rust_connection::RustConnection,
    damage: x11rb::protocol::damage::Damage,
    screen_area: u64,
}

impl XDamageProbe {
    /// Connect and register for damage on the root window. Returns `None`
    /// when there is no X server or the DAMAGE extension is missing, in
    /// which case the caller must use the fixed-sleep fallback.
    pub fn connect() -> Option<Self> {
        use x11rb::connection::Connection;
        use x11rb::connection::RequestConnection;
        use x11rb::protocol::damage;
        let (conn, _) = x11rb::connect(None).ok()?;
        // Extension missing (or unreadable) on this server: the caller
        // takes the fixed-sleep fallback.
        let _ = conn
            .extension_information(damage::X11_EXTENSION_NAME)
            .ok()??;
        let root = conn.setup().roots.first()?;
        let screen_area = (root.width_in_pixels as u64)
            .checked_mul(root.height_in_pixels as u64)
            .filter(|&a| a > 0)?;
        let damage = conn.generate_id().ok()?;
        damage::create(&conn, damage, root.root, damage::ReportLevel::NON_EMPTY)
            .ok()?
            .check()
            .ok()?;
        Some(Self {
            conn,
            damage,
            screen_area,
        })
    }
}

impl DamageProbe for XDamageProbe {
    fn path(&self) -> SettlePath {
        SettlePath::XDamage
    }

    fn poll_fraction(&mut self) -> Option<f64> {
        use x11rb::connection::Connection;
        use x11rb::protocol::damage;
        use x11rb::protocol::Event;
        let mut area = 0u64;
        loop {
            match self.conn.poll_for_event() {
                Ok(Some(Event::DamageNotify(n))) if n.damage == self.damage => {
                    area = area.saturating_add(n.area.width as u64 * n.area.height as u64);
                }
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(_) => return None,
            }
        }
        // Acknowledge and clear the accumulated damage so the next poll
        // window starts clean.
        if let Ok(cookie) = damage::subtract(&self.conn, self.damage, 0u32, 0u32) {
            let _ = cookie.check();
        }
        Some(area as f64 / self.screen_area as f64)
    }
}

impl Drop for XDamageProbe {
    fn drop(&mut self) {
        let _ = x11rb::protocol::damage::destroy(&self.conn, self.damage);
    }
}

/// Fallback probe: the damage feed is unavailable, so every wait takes
/// the fixed sleep. `poll_fraction` is never consulted on this path.
pub struct FixedSleepProbe;

impl DamageProbe for FixedSleepProbe {
    fn poll_fraction(&mut self) -> Option<f64> {
        None
    }
    fn path(&self) -> SettlePath {
        SettlePath::FixedSleep
    }
}

/// Scripted probe for unit tests: yields the scripted fractions in order,
/// then repeats the last value indefinitely.
pub struct ScriptedProbe {
    pub fractions: VecDeque<f64>,
    pub path: SettlePath,
}

impl ScriptedProbe {
    pub fn new(path: SettlePath, fractions: Vec<f64>) -> Self {
        Self {
            fractions: fractions.into(),
            path,
        }
    }
}

impl DamageProbe for ScriptedProbe {
    fn poll_fraction(&mut self) -> Option<f64> {
        if self.fractions.len() > 1 {
            self.fractions.pop_front()
        } else {
            self.fractions.front().copied()
        }
    }
    fn path(&self) -> SettlePath {
        self.path
    }
}

/// The best available settle probe: XDamage when the server offers it,
/// the fixed-sleep fallback otherwise. The path taken is recorded in
/// every [`SettleOutcome`] and aggregated in [`VerifierStats`].
pub fn auto_probe() -> Box<dyn DamageProbe> {
    match XDamageProbe::connect() {
        Some(p) => Box::new(p),
        None => Box::new(FixedSleepProbe),
    }
}

// ---------------------------------------------------------------------------
// Predicate-based verification.
// ---------------------------------------------------------------------------

/// Hamming cutoffs. Budgets until calibrated on VM clips of
/// animation-only vs real change.
pub const STABLE_CUTOFF: u32 = 6;
pub const SETTLED_CUTOFF: u32 = 4;

/// Predicate value plus region pixels captured during the hover dwell,
/// BEFORE the press. If the predicate is already true here, the runner
/// must NOT click again.
///
/// v1.2, fix 16 (also resolves M6): the region hash AND mean luma are
/// the PRE-ACTION anchor, captured in dispatch before the click. The old
/// code hashed a post-action frame and diffed two post-action frames.
/// `region_luma` is the level-sensitive complement to the dHash (which
/// is blind to uniform color changes).
pub struct Baseline {
    pub predicate_before: PredicateValue,
    pub region: (u32, u32, u32, u32),
    pub region_hash: u64,
    pub region_luma: f32,
}

/// Capture the baseline during the hover dwell. When the predicate is
/// already true, no pixels are captured (region_hash is 0) and the
/// verdict will be `AlreadySatisfied`.
pub fn capture_baseline(
    pred: &Predicate,
    eval: &dyn Fn(&Predicate) -> PredicateValue,
    region: (u32, u32, u32, u32),
    capture: &dyn Fn((u32, u32, u32, u32)) -> Result<Capture>,
) -> Result<Baseline> {
    let predicate_before = eval(pred);
    let (region_hash, region_luma) = if predicate_before.is_true() {
        (0, 0.0)
    } else {
        let c = capture(region)?;
        let cells = box_filter_9x8(&c)?;
        (dhash_9x8(&cells), mean_luma_9x8(&cells))
    };
    Ok(Baseline {
        predicate_before,
        region,
        region_hash,
        region_luma,
    })
}

pub struct VerifyHooks<'a> {
    /// Predicate evaluator; may re-snapshot on each call. Called up to
    /// twice per verification (right after the action, after settle).
    pub eval: &'a dyn Fn(&Predicate) -> PredicateValue,
    pub capture: &'a dyn Fn((u32, u32, u32, u32)) -> Result<Capture>,
    pub settle: &'a mut dyn DamageProbe,
    /// Last-resort text read for text predicates. The structured stub
    /// reports `Busy` until the Phase 5 worker lands.
    pub ocr: &'a dyn Fn((u32, u32, u32, u32)) -> OcrOutcome,
    pub telemetry: &'a mut RunTelemetry,
    pub stats: &'a mut VerifierStats,
}

/// B05: change is evidence, never proof. `Confirmed` requires the declared
/// predicate true; `Unchanged` requires demonstrably stable AND false;
/// everything else is `Inconclusive`. Only `Confirmed` may cache or
/// commit; that rule is enforced by the runner, the sole consumer of
/// this verdict.
pub fn check_against(
    baseline: &Baseline,
    pred: &Predicate,
    hooks: VerifyHooks<'_>,
) -> VerifyEvidence {
    let timer = Timer::start();
    let VerifyHooks {
        eval,
        capture,
        settle,
        ocr,
        telemetry,
        stats,
    } = hooks;

    macro_rules! finish {
        ($verdict:expr, $settle:expr, $dist:expr) => {{
            let ev = VerifyEvidence {
                verdict: $verdict,
                settle: $settle,
                frame_distance: $dist,
            };
            telemetry.record(Stage::Verify, timer.elapsed_nanos());
            stats.note(&ev);
            return ev;
        }};
    }

    if baseline.predicate_before.is_true() {
        finish!(Verdict::AlreadySatisfied, None, None);
    }
    if eval(pred).is_true() {
        finish!(Verdict::Confirmed, None, None);
    }
    // Predicate false or unknown: consult region pixels as supporting signal.
    // v1.2, fix 16: compare against the PRE-ACTION baseline (dHash for
    // structure, mean luma for level). A uniform white->black flip has
    // dHash distance 0 but a large luma shift. The luma check ONLY applies
    // when the baseline dHash is 0 (uniform region); for structured
    // regions, dHash alone decides (a focus ring changes luma but is
    // still "unchanged" if the dHash is stable).
    let (frame1_hash, frame1_luma) = match capture(baseline.region)
        .and_then(|c| box_filter_9x8(&c))
        .map(|cells| (dhash_9x8(&cells), mean_luma_9x8(&cells)))
    {
        Ok(h) => h,
        Err(_) => finish!(Verdict::Inconclusive, None, None),
    };
    let dist = hash_distance(baseline.region_hash, frame1_hash);
    // Uniform baseline (dHash=0): luma is the level signal.
    // Structured baseline: dHash distance decides.
    let level_changed =
        baseline.region_hash == 0 && (baseline.region_luma - frame1_luma).abs() >= LUMA_CUTOFF;
    if dist < STABLE_CUTOFF && !level_changed {
        // Stable and matching the baseline, predicate false -> Unchanged.
        finish!(Verdict::Unchanged, None, Some(dist));
    }
    // Pixels moved: damage-driven settle, not a fixed sleep.
    let s = wait_stable(
        settle,
        DAMAGE_FRACTION,
        DAMAGE_QUIET_POLLS,
        SETTLE_FLOOR_MS,
        SETTLE_CEILING_MS,
        DAMAGE_POLL_MS,
    );
    if !s.stable {
        finish!(Verdict::Inconclusive, Some(s), Some(dist));
    }
    let frame2 = match capture(baseline.region)
        .and_then(|c| box_filter_9x8(&c))
        .map(|cells| dhash_9x8(&cells))
    {
        Ok(h) => h,
        Err(_) => finish!(Verdict::Inconclusive, Some(s), Some(dist)),
    };
    if hash_distance(frame1_hash, frame2) > SETTLED_CUTOFF {
        // Still moving after the settle window: animation, not the action.
        finish!(Verdict::Inconclusive, Some(s), Some(dist));
    }
    // One more predicate read after the settle, then OCR as a last resort
    // for text predicates.
    //
    // v1.2, fix 13: a false HasText on a non-empty tree routes through
    // the OCR path instead of verdicting Unchanged directly. The old
    // code sent Bool(false) straight to Unchanged, so OCR never ran for
    // HasText (eval never returns Text, and the OCR arm only ran on
    // Unknown/Text).
    let verdict = match eval(pred) {
        PredicateValue::Bool(true) => Verdict::Confirmed,
        PredicateValue::Bool(false) => match pred {
            Predicate::HasText(want) => match ocr(baseline.region) {
                OcrOutcome::Text(seen) if seen.contains(want.as_str()) => Verdict::Confirmed,
                OcrOutcome::Text(_) => Verdict::Unchanged,
                OcrOutcome::Timeout | OcrOutcome::Busy | OcrOutcome::Error(_) => {
                    Verdict::Inconclusive
                }
            },
            _ => Verdict::Unchanged,
        },
        // Unknown, or a text value the verifier cannot interpret as
        // satisfied: fall through to the OCR last-resort below.
        PredicateValue::Unknown | PredicateValue::Text(_) => match pred {
            Predicate::HasText(want) => match ocr(baseline.region) {
                OcrOutcome::Text(seen) if seen.contains(want.as_str()) => Verdict::Confirmed,
                OcrOutcome::Text(_) => Verdict::Unchanged,
                OcrOutcome::Timeout | OcrOutcome::Busy | OcrOutcome::Error(_) => {
                    Verdict::Inconclusive
                }
            },
            _ => Verdict::Inconclusive,
        },
    };
    finish!(verdict, Some(s), Some(dist));
}

// ---------------------------------------------------------------------------
// OCR: structured stub. The real bounded worker (capacity 2, per-job
// deadlines, generation ids, Tesseract) lands in Phase 5 (B5). Until then
// the stub reports Busy so callers treat OCR as unavailable, never as
// empty text.
// ---------------------------------------------------------------------------

/// Outcome of one OCR read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OcrOutcome {
    Text(String),
    Timeout,
    Busy,
    Error(String),
}

/// One OCR job: region in device px, an absolute deadline, and the tree
/// generation the region belongs to. The real worker drops results from
/// a stale generation; the stub never produces results.
#[derive(Debug, Clone)]
pub struct OcrRequest {
    pub region: (u32, u32, u32, u32),
    pub deadline_ms: u64,
    pub generation: u64,
}

pub struct OcrStub;

impl OcrStub {
    /// Always `Busy`: no worker is wired yet. The signature already
    /// carries deadlines and generation ids so B5 drops in behind it.
    pub fn submit(&self, _req: &OcrRequest) -> OcrOutcome {
        OcrOutcome::Busy
    }
}

// ---------------------------------------------------------------------------
// Credential hygiene: password-field zeroing before any screenshot bytes
// leave the VM.
// ---------------------------------------------------------------------------

/// Zero every `input[type=password]` box in the frame buffer in place.
/// Boxes are (x, y, w, h) in device px matching the capture. Only the B,
/// G, R bytes are cleared; X is left untouched. Overflow-safe: boxes
/// outside the buffer are skipped, never wrapped.
pub fn redact_password_fields(buf: &mut [u8], stride: usize, boxes: &[(u32, u32, u32, u32)]) {
    let len = buf.len() as u64;
    for &(x, y, w, h) in boxes {
        for row in y..y.saturating_add(h) {
            for col in x..x.saturating_add(w) {
                let idx = row as u64 * stride as u64 + col as u64 * 4;
                if idx + 2 < len {
                    let i = idx as usize;
                    buf[i] = 0;
                    buf[i + 1] = 0;
                    buf[i + 2] = 0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicates::AxNode;
    use std::cell::Cell;

    // --- synthetic frames: 90x80, 10px per dHash cell column/row ---

    fn solid(w: u32, h: u32, v: u8) -> Capture {
        let stride = w as usize * 4;
        let mut buf = vec![0u8; stride * h as usize];
        #[allow(clippy::chunks_exact_to_as_chunks)]
        // verified against known answers; restyle risks drift
        for px in buf.chunks_exact_mut(4) {
            px[0] = v;
            px[1] = v;
            px[2] = v;
            px[3] = 255;
        }
        Capture { buf, stride, w, h }
    }

    /// Frame with a bright block covering grid columns `c0..c1`.
    fn block_frame(c0: u32, c1: u32) -> Capture {
        let (w, h) = (90u32, 80u32);
        let mut cap = solid(w, h, 10);
        for y in 0..h as usize {
            for x in (c0 as usize * 10)..(c1 as usize * 10).min(w as usize) {
                let i = y * cap.stride + x * 4;
                cap.buf[i] = 200;
                cap.buf[i + 1] = 200;
                cap.buf[i + 2] = 200;
            }
        }
        cap
    }

    fn hash_of(cap: &Capture) -> u64 {
        dhash_9x8(&box_filter_9x8(cap).unwrap())
    }

    fn baseline_with_hash(hash: u64) -> Baseline {
        // Test helper: the baseline luma must match the solid(90,80,10)
        // used at call sites (luma 10.0), not an arbitrary 128.0, or the
        // v1.2 fix-16 luma check sees a phantom level change.
        Baseline {
            predicate_before: PredicateValue::Bool(false),
            region: (0, 0, 90, 80),
            region_hash: hash,
            region_luma: 10.0,
        }
    }

    /// Drive `check_against` with scripted eval values, frames, damage
    /// fractions, and OCR outcome.
    fn run_verify(
        pred: &Predicate,
        baseline: &Baseline,
        eval_seq: Vec<PredicateValue>,
        frames: Vec<Capture>,
        probe_fractions: Vec<f64>,
        ocr: OcrOutcome,
    ) -> (VerifyEvidence, VerifierStats, RunTelemetry) {
        let eval_idx = Cell::new(0);
        let eval = |_p: &Predicate| {
            let i = eval_idx.get();
            eval_idx.set(i + 1);
            eval_seq.get(i).cloned().unwrap_or(PredicateValue::Unknown)
        };
        let frame_idx = Cell::new(0);
        let capture = |_r: (u32, u32, u32, u32)| {
            let i = frame_idx.get();
            frame_idx.set(i + 1);
            frames
                .get(i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("no more frames"))
        };
        let mut probe = ScriptedProbe::new(SettlePath::XDamage, probe_fractions);
        let ocr_fn = |_r: (u32, u32, u32, u32)| ocr.clone();
        let mut telemetry = RunTelemetry::new();
        let mut stats = VerifierStats::default();
        let ev = check_against(
            baseline,
            pred,
            VerifyHooks {
                eval: &eval,
                capture: &capture,
                settle: &mut probe,
                ocr: &ocr_fn,
                telemetry: &mut telemetry,
                stats: &mut stats,
            },
        );
        (ev, stats, telemetry)
    }

    // --- capture validation ---

    #[test]
    fn validate_accepts_well_formed_capture() {
        assert!(validate_capture(&solid(90, 80, 10)).is_ok());
    }

    #[test]
    fn validate_rejects_malformed_captures() {
        let mut c = solid(90, 80, 10);
        c.w = 0;
        assert!(validate_capture(&c).is_err());
        let mut c = solid(90, 80, 10);
        c.stride = 90 * 4 - 1;
        assert!(validate_capture(&c).is_err());
        let mut c = solid(90, 80, 10);
        c.buf.truncate(c.buf.len() - 1);
        assert!(validate_capture(&c).is_err());
        // Stride larger than w*4 (padded XImage rows) is fine.
        let mut c = solid(90, 80, 10);
        c.stride += 32;
        c.buf = vec![0u8; c.stride * 79 + 90 * 4];
        assert!(validate_capture(&c).is_ok());
    }

    // --- dHash ---

    #[test]
    fn dhash_bit_counter_fix_all_64_bits() {
        // Every comparison true: the fixed counter stores exactly 64 bits.
        // The old per-cell counter shifted by >= 64 and panicked.
        let cells: [[u32; 9]; 8] = std::array::from_fn(|_| std::array::from_fn(|c| c as u32));
        assert_eq!(dhash_9x8(&cells), u64::MAX);
        // No comparison true: zero hash.
        let zeros = [[0u32; 9]; 8];
        assert_eq!(dhash_9x8(&zeros), 0);
    }

    #[test]
    fn box_filter_uses_bgrx_luma_order() {
        // 9x8 capture, one pixel per cell. BGRX order: idx=B, idx+1=G, idx+2=R.
        let mut cap = solid(9, 8, 0);
        // Cell (0,0): pure blue (BGRX: buf[0] is B) -> (29*255)>>8 = 28.
        cap.buf[0] = 255;
        // Cell (0,1): pure green -> (150*255)>>8 = 149.
        cap.buf[4 + 1] = 255;
        // Cell (0,2): pure red -> (77*255)>>8 = 76.
        cap.buf[2 * 4 + 2] = 255;
        let cells = box_filter_9x8(&cap).unwrap();
        assert_eq!(cells[0][0], 28);
        assert_eq!(cells[0][1], 149);
        assert_eq!(cells[0][2], 76);
    }

    #[test]
    fn block_frame_distances_match_cutoff_design() {
        let h0 = hash_of(&solid(90, 80, 10));
        let h1 = hash_of(&block_frame(3, 9));
        let h2 = hash_of(&block_frame(4, 9));
        assert_eq!(h0, 0);
        // 8 row transitions dark->bright: over the STABLE_CUTOFF of 6.
        assert!(hash_distance(h0, h1) >= STABLE_CUTOFF);
        // Shifted block: 16 differing bits, over the SETTLED_CUTOFF of 4.
        assert!(hash_distance(h1, h2) > SETTLED_CUTOFF);
        assert_eq!(hash_distance(h1, h1), 0);
    }

    // --- settle ---

    #[test]
    fn wait_stable_xdamage_quiet_after_floor() {
        let mut probe = ScriptedProbe::new(SettlePath::XDamage, vec![0.0; 50]);
        let s = wait_stable(&mut probe, 0.001, 2, 50, 300, 4);
        assert!(s.stable);
        assert_eq!(s.path, SettlePath::XDamage);
        assert!(s.elapsed_ms >= 50, "elapsed {} < floor 50", s.elapsed_ms);
    }

    #[test]
    fn wait_stable_xdamage_never_quiet_hits_ceiling() {
        let mut probe = ScriptedProbe::new(SettlePath::XDamage, vec![0.5; 200]);
        let s = wait_stable(&mut probe, 0.001, 2, 50, 300, 4);
        assert!(!s.stable);
        assert!(
            s.elapsed_ms >= 300,
            "elapsed {} < ceiling 300",
            s.elapsed_ms
        );
    }

    #[test]
    fn wait_stable_fixed_sleep_path_is_recorded() {
        let mut probe = ScriptedProbe::new(SettlePath::FixedSleep, vec![]);
        let s = wait_stable(&mut probe, 0.001, 2, 50, 300, 4);
        assert!(s.stable);
        assert_eq!(s.path, SettlePath::FixedSleep);
        assert!(
            s.elapsed_ms >= FIXED_FALLBACK_SLEEP_MS,
            "elapsed {} < fallback {}",
            s.elapsed_ms,
            FIXED_FALLBACK_SLEEP_MS
        );
    }

    #[test]
    fn wait_stable_broken_feed_is_unstable() {
        struct Dead;
        impl DamageProbe for Dead {
            fn poll_fraction(&mut self) -> Option<f64> {
                None
            }
            fn path(&self) -> SettlePath {
                SettlePath::XDamage
            }
        }
        let mut probe = Dead;
        let s = wait_stable(&mut probe, 0.001, 2, 50, 300, 4);
        assert!(!s.stable);
        assert_eq!(s.path, SettlePath::XDamage);
    }

    // --- labelled verifier scenarios (false Confirmed is a release blocker) ---

    fn done_pred() -> Predicate {
        Predicate::HasText("Done".to_string())
    }

    #[test]
    fn scenario_spinner_only_never_confirms() {
        // Spinner animates: pixels change, predicate stays false, damage
        // never settles. Must be Inconclusive, never Confirmed.
        let base = baseline_with_hash(hash_of(&solid(90, 80, 10)));
        let (ev, stats, telemetry) = run_verify(
            &done_pred(),
            &base,
            vec![PredicateValue::Bool(false)],
            vec![block_frame(3, 9)],
            vec![0.5; 200],
            OcrOutcome::Busy,
        );
        assert_eq!(ev.verdict, Verdict::Inconclusive);
        assert_ne!(ev.verdict, Verdict::Confirmed);
        assert_eq!(stats.inconclusive, 1);
        assert_eq!(stats.xdamage_settles, 1);
        assert!(telemetry.stage_nanos(Stage::Verify) > 0);
    }

    #[test]
    fn scenario_focus_ring_change_is_unchanged() {
        // Focus ring appeared: pixels changed once, then stable, predicate
        // false. Unchanged, never Confirmed.
        // v1.2, fix 13: uses a non-text predicate (Focused) because HasText
        // now routes through OCR; with Busy OCR a false HasText is
        // Inconclusive, not Unchanged.
        let base = baseline_with_hash(hash_of(&solid(90, 80, 10)));
        let (ev, stats, _) = run_verify(
            &Predicate::Focused,
            &base,
            vec![PredicateValue::Bool(false), PredicateValue::Bool(false)],
            vec![block_frame(3, 9), block_frame(3, 9)],
            vec![0.0; 50],
            OcrOutcome::Busy,
        );
        assert_eq!(ev.verdict, Verdict::Unchanged);
        assert_eq!(stats.unchanged, 1);
    }

    #[test]
    fn scenario_clock_tick_is_inconclusive() {
        // Clock keeps ticking: damage never quiets for 2 consecutive polls.
        let base = baseline_with_hash(hash_of(&solid(90, 80, 10)));
        let fractions: Vec<f64> = (0..200)
            .map(|i| if i % 2 == 0 { 0.05 } else { 0.0 })
            .collect();
        let (ev, stats, _) = run_verify(
            &done_pred(),
            &base,
            vec![PredicateValue::Bool(false)],
            vec![block_frame(3, 9)],
            fractions,
            OcrOutcome::Busy,
        );
        assert_eq!(ev.verdict, Verdict::Inconclusive);
        assert_eq!(stats.inconclusive, 1);
    }

    #[test]
    fn scenario_pre_existing_text_is_already_satisfied() {
        // Predicate true during the hover dwell: no capture, no click.
        let ctx = EvalCtx {
            nodes: vec![AxNode {
                node_id: 1,
                role: "status".to_string(),
                name: "Done".to_string(),
                text: "Done".to_string(),
                ..Default::default()
            }],
            url: String::new(),
            target_node_id: None,
        };
        let eval = |p: &Predicate| super::super::predicates::eval(p, &ctx);
        let capture = |_r: (u32, u32, u32, u32)| -> Result<Capture> {
            panic!("capture must not run for an already-satisfied predicate")
        };
        let base = capture_baseline(&done_pred(), &eval, (0, 0, 90, 80), &capture).unwrap();
        assert!(base.predicate_before.is_true());
        assert_eq!(base.region_hash, 0);
        assert_eq!(base.region_luma, 0.0);

        let mut probe = ScriptedProbe::new(SettlePath::XDamage, vec![0.0]);
        let mut telemetry = RunTelemetry::new();
        let mut stats = VerifierStats::default();
        let ev = check_against(
            &base,
            &done_pred(),
            VerifyHooks {
                eval: &eval,
                capture: &capture,
                settle: &mut probe,
                ocr: &|_r| OcrOutcome::Busy,
                telemetry: &mut telemetry,
                stats: &mut stats,
            },
        );
        assert_eq!(ev.verdict, Verdict::AlreadySatisfied);
        assert_eq!(stats.already_satisfied, 1);
        assert_eq!(stats.xdamage_settles, 0, "no settle for AlreadySatisfied");
    }

    #[test]
    fn scenario_delayed_spa_update_confirms_after_settle() {
        // SPA applies the update ~300ms in: first read false, pixels move,
        // settle quiets, second read true. Confirmed.
        let base = baseline_with_hash(hash_of(&solid(90, 80, 10)));
        let (ev, stats, _) = run_verify(
            &done_pred(),
            &base,
            vec![PredicateValue::Bool(false), PredicateValue::Bool(true)],
            vec![block_frame(3, 9), block_frame(3, 9)],
            vec![0.0; 50],
            OcrOutcome::Busy,
        );
        assert_eq!(ev.verdict, Verdict::Confirmed);
        assert_eq!(stats.confirmed, 1);
        assert_eq!(stats.xdamage_settles, 1);
    }

    #[test]
    fn scenario_animation_loop_is_inconclusive() {
        // Settles briefly but frame1 vs frame2 still differ: animation,
        // not the action. Inconclusive.
        let base = baseline_with_hash(hash_of(&solid(90, 80, 10)));
        let (ev, stats, _) = run_verify(
            &done_pred(),
            &base,
            vec![PredicateValue::Bool(false), PredicateValue::Bool(false)],
            vec![block_frame(3, 9), block_frame(4, 9)],
            vec![0.0; 50],
            OcrOutcome::Busy,
        );
        assert_eq!(ev.verdict, Verdict::Inconclusive);
        assert_eq!(stats.inconclusive, 1);
    }

    #[test]
    fn scenario_genuine_transition_confirms_immediately() {
        // Predicate true right after the action: Confirmed with no settle
        // and no second frame.
        let base = baseline_with_hash(hash_of(&solid(90, 80, 10)));
        let (ev, stats, _) = run_verify(
            &done_pred(),
            &base,
            vec![PredicateValue::Bool(true)],
            vec![],
            vec![0.0],
            OcrOutcome::Busy,
        );
        assert_eq!(ev.verdict, Verdict::Confirmed);
        assert_eq!(stats.confirmed, 1);
        assert_eq!(stats.xdamage_settles, 0, "no settle needed");
        assert!(ev.frame_distance.is_none());
    }

    #[test]
    fn scenario_ocr_last_resort_for_text_predicate() {
        // Predicate Unknown after settle; OCR reads the wanted text.
        let base = baseline_with_hash(hash_of(&solid(90, 80, 10)));
        let (ev, _, _) = run_verify(
            &done_pred(),
            &base,
            vec![PredicateValue::Unknown, PredicateValue::Unknown],
            vec![block_frame(3, 9), block_frame(3, 9)],
            vec![0.0; 50],
            OcrOutcome::Text("Done".to_string()),
        );
        assert_eq!(ev.verdict, Verdict::Confirmed);
    }

    #[test]
    fn scenario_ocr_busy_after_unknown_is_inconclusive() {
        // Same, but the OCR stub is Busy: honest Inconclusive, never a
        // guess, never a confirm.
        let base = baseline_with_hash(hash_of(&solid(90, 80, 10)));
        let (ev, _, _) = run_verify(
            &done_pred(),
            &base,
            vec![PredicateValue::Unknown, PredicateValue::Unknown],
            vec![block_frame(3, 9), block_frame(3, 9)],
            vec![0.0; 50],
            OcrOutcome::Busy,
        );
        assert_eq!(ev.verdict, Verdict::Inconclusive);
    }

    // --- credential hygiene ---

    #[test]
    fn redact_password_fields_zeroes_only_the_box() {
        let mut cap = solid(90, 80, 200);
        // Box at (10,10) size 20x10 in device px.
        redact_password_fields(&mut cap.buf, cap.stride, &[(10, 10, 20, 10)]);
        let at = |x: usize, y: usize| &cap.buf[y * cap.stride + x * 4..y * cap.stride + x * 4 + 4];
        // Inside: BGR zeroed, X untouched.
        assert_eq!(at(15, 15), &[0, 0, 0, 255]);
        // Outside: intact.
        assert_eq!(at(5, 5), &[200, 200, 200, 255]);
        assert_eq!(at(35, 15), &[200, 200, 200, 255]);
    }

    #[test]
    fn redact_password_fields_tolerates_out_of_bounds_boxes() {
        let mut cap = solid(16, 16, 200);
        // Absurd box: skipped, never wraps.
        redact_password_fields(
            &mut cap.buf,
            cap.stride,
            &[(u32::MAX - 4, u32::MAX - 4, 100, 100)],
        );
        assert!(cap.buf.iter().step_by(4).all(|&b| b == 200));
    }

    #[test]
    fn capture_redact_passwords_uses_ctx_boxes() {
        let mut cap = solid(90, 80, 200);
        let ctx = EvalCtx {
            nodes: vec![AxNode {
                node_id: 1,
                role: "textbox".to_string(),
                is_password: true,
                box_px: (0, 0, 90, 80),
                ..Default::default()
            }],
            ..Default::default()
        };
        cap.redact_passwords(&ctx);
        assert!(cap.buf.iter().step_by(4).take(3).all(|&b| b == 0));
        // X channel untouched.
        assert!(cap.buf.iter().skip(3).step_by(4).all(|&b| b == 255));
    }

    // --- OCR stub ---

    #[test]
    fn ocr_stub_reports_busy_with_deadline_and_generation() {
        let stub = OcrStub;
        let out = stub.submit(&OcrRequest {
            region: (0, 0, 90, 80),
            deadline_ms: 500,
            generation: 42,
        });
        assert_eq!(out, OcrOutcome::Busy);
    }
}
