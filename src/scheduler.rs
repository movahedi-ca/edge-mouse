//! Movement scheduler: honest timing for every gesture (mouse-future Phase 3).
//!
//! One scheduler for move, click, drag, and scroll: button state is a
//! parameter ([`ButtonPlan]`), and every gesture consumes movement time,
//! hover dwell, press dwell, and inter-step gap from its [`SpeedProfile`].
//!
//! Profiles:
//! - [`SpeedProfile::Legacy`] reproduces the current driver's durations
//!   exactly (same formula, same uniform noise) and is the DEFAULT and the
//!   rollback. The parity gate is measured, not hoped: p50 within +/-15%
//!   per distance bucket against `docs/baseline-2026-11.json`, mean
//!   ln(MT/ID) within +/-0.3 of -2.32.
//! - [`SpeedProfile::Human`] and [`SpeedProfile::Fast`] are PROVISIONAL
//!   until the KS harness (`rust/examples/ks_harness.rs`) fits their Fitts
//!   a/b from recorded reaches and they pass KS(ln MT/ID) + velocity-shape.
//!   Until then they are unreachable from the default path
//!   ([`SpeedProfile::is_available`]).
//!
//! Hard rules honored here:
//! - No tokio anywhere on this path. The [`InputThread`] is plain
//!   `std::thread`; the bridge's HTTP plane keeps its own runtime.
//! - Gesture dispatch crosses to the input thread on an `rtrb` SPSC ring;
//!   side effects (geometry/pointer requests) cross on a second ring and
//!   are drained between flushed frames. No mutex is held across sleeps:
//!   the input thread OWNS the `RustConnection`.
//! - Pointer/damage state broadcasts back on a hand-rolled [`SeqLock`]
//!   (latest-value telemetry, no queue, no new dependency).
//! - Absolute `CLOCK_MONOTONIC` deadlines (`libc::clock_nanosleep`);
//!   overdue samples are skipped, never queued.
//! - A sub-pixel [`ResidualAccum`](crate::legacy::ResidualAccum)
//!   carries fractional deltas across ticks: the integer sum converges
//!   with zero drift to the exact target.
//! - [`PredictiveDrift`] keeps the cursor alive during Tier-1 waits. The
//!   scheduler never calls Tier-1 itself: it only accepts the optional
//!   top-3 targets and the confirmation index.
//!
//! Trajectory geometry (easing, arc-length cursor, Bezier builders) lives
//! in [`crate::legacy`] (vendored from the bridge's `x11_mouse.rs`) and is
//! shared, not duplicated: this module owns timing policy, profiles,
//! threading, and drift.

use std::cell::RefCell;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result};
use rtrb::{Consumer, Producer, RingBuffer};
use x11rb::connection::Connection;
use x11rb::protocol::xproto::ConnectionExt as _;
use x11rb::protocol::xproto::Window;
use x11rb::protocol::xtest::ConnectionExt as _;
use x11rb::rust_connection::RustConnection;
use x11rb::CURRENT_TIME;

use crate::legacy::{
    asymmetric_min_jerk, build_trajectory, build_trajectory_fitts, check_in_window,
    deadline_add_ms, monotonic_now, sleep_until_deadline, timespec_after, ArcLengthCursor,
    ResidualAccum, XorShift64, MAX_CHORD_PX, OVERDUE_LATCH_MS,
};
use crate::legacy::{legacy_duration_ms, WindowRect};

// ---------------------------------------------------------------------------
// Speed profiles
// ---------------------------------------------------------------------------

/// Movement speed profile. `Human`/`Fast` are provisional until the KS
/// harness fits and accepts them; `Legacy` is the default and rollback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeedProfile {
    Legacy,
    Human,
    Fast,
}

/// Lognormal as median + sigma: sample = median * exp(sigma * z).
/// (The plan's honesty rule: never an ambiguous min/max pair.)
#[derive(Debug, Clone, Copy)]
pub struct Lognormal {
    pub median_ms: f64,
    pub sigma: f64,
}

impl Lognormal {
    /// Draw one sample from a standard-normal `z`.
    pub fn sample(&self, z: f64) -> f64 {
        self.median_ms * (self.sigma * z).exp()
    }
}

/// Per-profile timing parameters.
pub struct ProfileParams {
    /// True only for Legacy (Fitts path unused).
    pub legacy: bool,
    /// Fitts MT = a + b * log2(D / W + 1), seconds. Provisional for Human/Fast.
    pub a: f64,
    pub b: f64,
    /// Lognormal sigma applied around the Fitts median (provisional).
    pub sigma: f64,
    pub hover: Lognormal,
    pub dwell: Lognormal,
    pub gap: Lognormal,
    /// KS-gate acceptance. Legacy is the measured current behavior, so it
    /// ships as accepted; Human/Fast stay provisional until fitted.
    pub validated: bool,
}

impl SpeedProfile {
    pub fn params(&self) -> ProfileParams {
        match self {
            SpeedProfile::Legacy => ProfileParams {
                legacy: true,
                a: 0.0,
                b: 0.0,
                sigma: 0.0,
                hover: Lognormal {
                    median_ms: 85.0,
                    sigma: 0.35,
                }, // ~50-140ms
                dwell: Lognormal {
                    median_ms: 50.0,
                    sigma: 0.35,
                }, // ~30-80ms
                gap: Lognormal {
                    median_ms: 140.0,
                    sigma: 0.4,
                }, // ~80-220ms
                validated: true, // it IS the current behavior
            },
            // Provisional until fitted from recorded reaches + KS-accepted.
            SpeedProfile::Human => ProfileParams {
                legacy: false,
                a: 0.050,
                b: 0.150,
                sigma: 0.12,
                hover: Lognormal {
                    median_ms: 85.0,
                    sigma: 0.35,
                },
                dwell: Lognormal {
                    median_ms: 50.0,
                    sigma: 0.35,
                },
                gap: Lognormal {
                    median_ms: 220.0,
                    sigma: 0.4,
                },
                validated: false,
            },
            // Provisional until fitted from recorded reaches + KS-accepted.
            SpeedProfile::Fast => ProfileParams {
                legacy: false,
                a: 0.022,
                b: 0.068,
                sigma: 0.12,
                hover: Lognormal {
                    median_ms: 21.0,
                    sigma: 0.35,
                }, // ~12-35ms
                dwell: Lognormal {
                    median_ms: 26.0,
                    sigma: 0.35,
                }, // ~16-40ms
                gap: Lognormal {
                    median_ms: 70.0,
                    sigma: 0.4,
                }, // ~40-120ms
                validated: false,
            },
        }
    }

    /// Human/Fast are unreachable from the default path until the KS gate
    /// passes. Callers that need a provisional profile must opt in by
    /// checking this and naming the provisional status in their report.
    pub fn is_available(&self) -> bool {
        self.params().validated
    }
}

// ---------------------------------------------------------------------------
// Fitts timing + the ONE W formula
// ---------------------------------------------------------------------------

/// Clamp bounds for Fitts movement time (plan Layer 4).
pub const MT_MIN_MS: f64 = 16.0;
pub const MT_MAX_MS: f64 = 3000.0;

/// ONE W formula: projected target extent along the approach direction.
///
/// Tested on horizontal, vertical, and diagonal approaches. `min(w,h)` is
/// used only for degenerate approaches (zero-length); W is clamped to at
/// least 1px so D/W can never blow up.
pub fn target_width(w: f64, h: f64, approach: (f64, f64)) -> f64 {
    let len = approach.0.hypot(approach.1);
    if len < 1e-6 {
        return w.min(h).max(1.0);
    }
    let (ux, uy) = (approach.0 / len, approach.1 / len);
    (w * ux.abs() + h * uy.abs()).max(1.0)
}

/// Fitts MT in seconds (unclamped): a + b * log2(D / W + 1).
#[inline]
pub fn fitts_mt_s(a: f64, b: f64, dist_px: f64, w_eff: f64) -> f64 {
    let w = w_eff.max(1.0);
    a + b * (dist_px / w + 1.0).log2()
}

/// Fitts movement-time median in ms, clamped to [16ms, 3000ms].
pub fn fitts_mt_median_ms(a: f64, b: f64, dist_px: f64, w_eff: f64) -> f64 {
    (fitts_mt_s(a, b, dist_px, w_eff) * 1000.0).clamp(MT_MIN_MS, MT_MAX_MS)
}

/// Full Fitts sample: median * exp(sigma * z), clamped to [16ms, 3000ms].
/// Lognormal as median + sigma, per the plan's honesty rule.
pub fn fitts_mt_sample_ms(a: f64, b: f64, sigma: f64, dist_px: f64, w_eff: f64, z: f64) -> f64 {
    (fitts_mt_median_ms(a, b, dist_px, w_eff) * (sigma * z).exp()).clamp(MT_MIN_MS, MT_MAX_MS)
}

/// Fitts index of difficulty in bits: log2(D / W + 1), W >= 1px.
pub fn fitts_id(dist_px: f64, w_eff: f64) -> f64 {
    (dist_px / w_eff.max(1.0) + 1.0).log2()
}

/// Overshoot probability rises with Fitts ID and falls toward zero on
/// large targets (large W means small ID). Shape is provisional like the
/// rest of Human/Fast; the KS harness will calibrate it.
pub fn overshoot_probability(id_bits: f64) -> f64 {
    (0.10 + 0.12 * id_bits).clamp(0.0, 0.85)
}

// ---------------------------------------------------------------------------
// OU tremor
// ---------------------------------------------------------------------------

/// Ornstein-Uhlenbeck tremor (plan H3): mean-reverting jitter with time
/// constant tau ~= 40-80ms, replacing the driver's white gaussian jitter
/// on the Human/Fast path.
///
///   a = exp(-dt / tau); j = j * a + sigma * sqrt(1 - a*a) * z
pub struct OuTremor {
    tau_ms: f64,
    sigma_px: f64,
    jx: f64,
    jy: f64,
}

impl OuTremor {
    pub fn new(tau_ms: f64, sigma_px: f64) -> Self {
        Self {
            tau_ms: tau_ms.max(1.0),
            sigma_px: sigma_px.max(0.0),
            jx: 0.0,
            jy: 0.0,
        }
    }

    /// Advance by `dt_ms` with standard-normal draws; returns (jx, jy) px.
    pub fn step(&mut self, dt_ms: f64, zx: f64, zy: f64) -> (f64, f64) {
        let a = (-dt_ms / self.tau_ms).exp();
        let b = self.sigma_px * (1.0 - a * a).max(0.0).sqrt();
        self.jx = self.jx * a + b * zx;
        self.jy = self.jy * a + b * zy;
        (self.jx, self.jy)
    }

    pub fn state(&self) -> (f64, f64) {
        (self.jx, self.jy)
    }
}

// ---------------------------------------------------------------------------
// Gesture planning
// ---------------------------------------------------------------------------

/// Gesture kind. One scheduler, button state as a parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GestureKind {
    Move,
    Click,
    DoubleClick,
    Drag,
    Scroll,
}

/// Target box for the W computation (page space, px).
#[derive(Debug, Clone, Copy)]
pub struct TargetBox {
    pub w: f64,
    pub h: f64,
}

/// Button behavior for a planned gesture.
#[derive(Debug, Clone, Copy)]
pub enum ButtonPlan {
    /// Pure move: no button changes.
    None,
    /// Drag: press before the move, release after.
    DownThroughMove,
    /// Click: move, hover, press, dwell, release.
    Click { dwell_ms: f64 },
    /// Double-click: move, hover, press/release twice.
    DoubleClick { dwell_ms: f64, inter_ms: f64 },
    /// Scroll: move, then emit `ticks` vertical wheel events (XTest
    /// button 4 = wheel up, button 5 = wheel down). Positive ticks
    /// scroll down, negative scroll up. v1.2, fix 11: a Scroll gesture
    /// used to plan ButtonPlan::None and send no wheel events at all.
    Scroll { ticks: i32 },
}

/// A fully planned gesture: path geometry plus the nominal timing the
/// emitter realizes against absolute deadlines.
pub struct PlannedEmit {
    pub profile: SpeedProfile,
    pub kind: GestureKind,
    /// Path geometry, arc-length resampled (4px). The cursor maps sample
    /// index to arc position, so the sample count is independent of the
    /// point count.
    pub points: Vec<(f64, f64)>,
    /// Emission count. Legacy: points.len() (per-point cadence, parity).
    /// Human/Fast: one per flushed frame at the measured display cadence.
    pub samples: usize,
    /// Nominal ms between samples.
    pub spacing_ms: f64,
    /// Half-width of the uniform spacing noise (Legacy 0.3, Human/Fast 0).
    pub spacing_noise: f64,
    pub button: ButtonPlan,
    pub hover_ms: f64,
    /// Movement-time deadline span in ms (informational + assertions).
    pub mt_ms: f64,
    pub seed: u64,
    pub exact_end: (i32, i32),
    /// Measured display cadence in ms (Human/Fast frame pacing).
    pub frame_ms: f64,
}

/// Check every planned point against the SEC-1 rect (reject, never clamp).
fn check_planned_path(rect: &WindowRect, pts: &[(f64, f64)]) -> Result<()> {
    for &(px, py) in pts {
        check_in_window(Some(rect), px.round() as i32, py.round() as i32)?;
    }
    Ok(())
}

/// Plan a Legacy gesture: the exact legacy trajectory builder, SEC-1
/// checked with retries and a chord-capped linear fallback (same policy
/// as the driver's `plan_trajectory`), Legacy timing.
fn plan_legacy_path(
    from: (f64, f64),
    to: (f64, f64),
    rect: &WindowRect,
    rng: &mut XorShift64,
) -> Result<Vec<(f64, f64)>> {
    check_in_window(Some(rect), from.0.round() as i32, from.1.round() as i32)?;
    check_in_window(Some(rect), to.0.round() as i32, to.1.round() as i32)?;
    for _ in 0..4 {
        let pts = build_trajectory(from.0, from.1, to.0, to.1, rng);
        if check_planned_path(rect, &pts).is_ok() {
            return Ok(pts);
        }
    }
    // Direct leg, then chord-capped linear (convex rect: endpoints inside
    // implies the whole leg is inside).
    let single = build_trajectory(from.0, from.1, to.0, to.1, rng);
    if check_planned_path(rect, &single).is_ok() {
        return Ok(single);
    }
    let dx = to.0 - from.0;
    let dy = to.1 - from.1;
    let d = dx.hypot(dy);
    let n = ((d / MAX_CHORD_PX).ceil() as usize).max(1);
    let pts: Vec<(f64, f64)> = (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (from.0 + dx * t, from.1 + dy * t)
        })
        .collect();
    check_planned_path(rect, &pts)?;
    Ok(pts)
}

/// Plan a Human/Fast gesture: matched control points, ID-driven overshoot
/// tested against the rect with a direct-leg fallback.
fn plan_fitts_path(
    from: (f64, f64),
    to: (f64, f64),
    overshoot_p: f64,
    rect: &WindowRect,
    rng: &mut XorShift64,
) -> Result<Vec<(f64, f64)>> {
    check_in_window(Some(rect), from.0.round() as i32, from.1.round() as i32)?;
    check_in_window(Some(rect), to.0.round() as i32, to.1.round() as i32)?;
    for _ in 0..4 {
        let pts = build_trajectory_fitts(from.0, from.1, to.0, to.1, overshoot_p, rng);
        if check_planned_path(rect, &pts).is_ok() {
            return Ok(pts);
        }
    }
    // Direct matched leg without overshoot, then chord-capped linear.
    let direct = build_trajectory_fitts(from.0, from.1, to.0, to.1, 0.0, rng);
    if check_planned_path(rect, &direct).is_ok() {
        return Ok(direct);
    }
    let dx = to.0 - from.0;
    let dy = to.1 - from.1;
    let d = dx.hypot(dy);
    let n = ((d / MAX_CHORD_PX).ceil() as usize).max(1);
    let pts: Vec<(f64, f64)> = (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (from.0 + dx * t, from.1 + dy * t)
        })
        .collect();
    check_planned_path(rect, &pts)?;
    Ok(pts)
}

/// Plan one gesture against the profile. Human/Fast draw their Fitts MT
/// from the provisional a/b until the KS harness fits them; the plan
/// records which profile produced it so provisional output is labeled.
#[allow(clippy::too_many_arguments)]
pub fn plan_gesture(
    profile: SpeedProfile,
    kind: GestureKind,
    from: (f64, f64),
    to: (f64, f64),
    target: TargetBox,
    rect: &WindowRect,
    seed: u64,
    frame_ms: f64,
) -> Result<PlannedEmit> {
    let params = profile.params();
    // v1.2, fix 14: from_seed, not with_salt. The seed must fully
    // determine the trajectory; with_salt mixed wall-clock time.
    let mut rng = XorShift64::from_seed(seed);

    let (points, mt_ms, samples, spacing_ms, spacing_noise, hover_ms, button) = if params.legacy {
        let (hz, dz) = rng.gauss_pair();
        let pts = plan_legacy_path(from, to, rect, &mut rng)?;
        let total = ArcLengthCursor::new(&pts).total();
        let mt = legacy_duration_ms(total);
        let n = pts.len().max(2);
        let base_ms = (1.2 / total.max(1.0)).clamp(0.003, 0.008) * 1000.0;
        let button = match kind {
            GestureKind::Move => ButtonPlan::None,
            // v1.2, fix 11: Scroll used to plan ButtonPlan::None and send
            // no wheel events at all. Direction follows the gesture vector
            // (down when to.y >= from.y); 3 ticks is the default scroll.
            GestureKind::Scroll => ButtonPlan::Scroll {
                ticks: if to.1 >= from.1 { 3 } else { -3 },
            },
            GestureKind::Drag => ButtonPlan::DownThroughMove,
            GestureKind::Click => ButtonPlan::Click {
                dwell_ms: params.dwell.sample(dz),
            },
            GestureKind::DoubleClick => ButtonPlan::DoubleClick {
                dwell_ms: params.dwell.sample(dz),
                inter_ms: 60.0,
            },
        };
        (pts, mt, n, base_ms, 0.3, params.hover.sample(hz), button)
    } else {
        let gz = rng.gauss_pair().0;
        let (hz, dz) = rng.gauss_pair();
        let dx = to.0 - from.0;
        let dy = to.1 - from.1;
        let dist = dx.hypot(dy);
        let w_eff = target_width(target.w, target.h, (dx, dy));
        let id = fitts_id(dist, w_eff);
        let mt = fitts_mt_sample_ms(params.a, params.b, params.sigma, dist, w_eff, gz);
        let op = overshoot_probability(id);
        let pts = plan_fitts_path(from, to, op, rect, &mut rng)?;
        let frame = frame_ms.max(1.0);
        let n = ((mt / frame).round() as usize).max(2);
        let button = match kind {
            GestureKind::Move => ButtonPlan::None,
            // v1.2, fix 11: Scroll used to plan ButtonPlan::None and send
            // no wheel events at all. Direction follows the gesture vector
            // (down when to.y >= from.y); 3 ticks is the default scroll.
            GestureKind::Scroll => ButtonPlan::Scroll {
                ticks: if to.1 >= from.1 { 3 } else { -3 },
            },
            GestureKind::Drag => ButtonPlan::DownThroughMove,
            GestureKind::Click => ButtonPlan::Click {
                dwell_ms: params.dwell.sample(dz),
            },
            GestureKind::DoubleClick => ButtonPlan::DoubleClick {
                dwell_ms: params.dwell.sample(dz),
                inter_ms: 60.0,
            },
        };
        (
            pts,
            mt,
            n,
            mt / (n - 1) as f64,
            0.0,
            params.hover.sample(hz),
            button,
        )
    };

    Ok(PlannedEmit {
        profile,
        kind,
        points,
        samples,
        spacing_ms,
        spacing_noise,
        button,
        hover_ms,
        mt_ms,
        seed,
        exact_end: (to.0.round() as i32, to.1.round() as i32),
        frame_ms,
    })
}

// ---------------------------------------------------------------------------
// Emission against absolute deadlines
// ---------------------------------------------------------------------------

/// Report from one emission: counts + measured wall time.
#[derive(Debug, Clone, Copy)]
pub struct EmitReport {
    pub emitted: usize,
    pub skipped: usize,
    pub duration_ms: f64,
}

fn xtest_motion(conn: &RustConnection, root: Window, x: i32, y: i32) -> Result<()> {
    // The SEC-1 check (via `check`) already verified i16 range; the cast
    // below is infallible on that path.
    conn.xtest_fake_input(6, 0, CURRENT_TIME, root, x as i16, y as i16, 0)
        .context("XTest motion")?;
    Ok(())
}

fn xtest_button(conn: &RustConnection, root: Window, press: bool) -> Result<()> {
    let ev = if press { 4 } else { 5 };
    conn.xtest_fake_input(ev, 1, CURRENT_TIME, root, 0, 0, 0)
        .context("XTest button")?;
    conn.flush().context("XTest flush")?;
    Ok(())
}

/// Emit one vertical wheel tick via XTest (button 4 = up, button 5 =
/// down). v1.2, fix 11.
fn xtest_wheel(conn: &RustConnection, root: Window, down: bool) -> Result<()> {
    let ev = if down { 5 } else { 4 };
    conn.xtest_fake_input(ev, 1, CURRENT_TIME, root, 0, 0, 0)
        .context("XTest wheel press")?;
    conn.xtest_fake_input(ev, 0, CURRENT_TIME, root, 0, 0, 0)
        .context("XTest wheel release")?;
    conn.flush().context("XTest flush")?;
    Ok(())
}

/// Emit one planned gesture against absolute CLOCK_MONOTONIC deadlines.
///
/// `check` is the SEC-1 gate (reject, never clamp); `on_frame` runs after
/// every flush , the input thread drains its side-effect queue there, the
/// direct path passes a no-op.
fn emit_inner(
    conn: &RustConnection,
    root: Window,
    check: &dyn Fn(i32, i32) -> Result<()>,
    plan: &PlannedEmit,
    on_frame: &mut dyn FnMut(),
) -> Result<EmitReport> {
    let t0 = monotonic_now();

    if matches!(plan.button, ButtonPlan::DownThroughMove) {
        xtest_button(conn, root, true)?;
    }

    let mut cursor = ArcLengthCursor::new(&plan.points);
    let total = cursor.total();
    let n = plan.samples.max(2);
    let inv_last = 1.0 / ((n - 1) as f64);
    // v1.2, fix 14: from_seed, not with_salt (see plan_gesture).
    let mut rng = XorShift64::from_seed(plan.seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut accum = ResidualAccum::new();
    let mut tremor = OuTremor::new(60.0, 0.5);
    let mut deadline = monotonic_now();
    let mut emitted = 0usize;
    let mut skipped = 0usize;
    let mut last_sent: Option<(i32, i32)> = None;

    // Degenerate path: single motion to the exact end.
    if total < 1e-6 || plan.points.len() < 2 {
        let (ex, ey) = plan.exact_end;
        check(ex, ey)?;
        xtest_motion(conn, root, ex, ey)?;
        // v1.2, fix 30: no flush here; the single post-gesture flush
        // below covers it.
        on_frame();
        emitted += 1;
    } else {
        for i in 0..n {
            if i > 0 {
                let noise = if plan.spacing_noise > 0.0 {
                    rng.uniform(1.0 - plan.spacing_noise, 1.0 + plan.spacing_noise)
                } else {
                    1.0
                };
                deadline = deadline_add_ms(&deadline, plan.spacing_ms * noise);
                let latch = deadline_add_ms(&deadline, OVERDUE_LATCH_MS);
                if timespec_after(&monotonic_now(), &latch) {
                    // Overdue past the latch: skip the stale sample, never
                    // queue it. The deadline schedule is unchanged.
                    skipped += 1;
                    continue;
                }
                sleep_until_deadline(&deadline);
            }

            let target_d = asymmetric_min_jerk((i as f64) * inv_last) * total;
            let base_pt = cursor.advance_to(target_d);

            let (jx, jy) = if i > 0 && i + 1 < n {
                let (ox, oy) = match plan.profile {
                    SpeedProfile::Legacy => {
                        let (gx, gy) = rng.gauss_pair();
                        (base_pt.0 + gx * 0.6, base_pt.1 + gy * 0.6)
                    }
                    SpeedProfile::Human | SpeedProfile::Fast => {
                        let (zx, zy) = rng.gauss_pair();
                        let (tx, ty) = tremor.step(plan.spacing_ms, zx, zy);
                        (base_pt.0 + tx, base_pt.1 + ty)
                    }
                };
                // Jitter stays inside the checked space: drop it when the
                // rounded pixel would leave the window rect.
                if check(ox.round() as i32, oy.round() as i32).is_ok() {
                    (ox, oy)
                } else {
                    base_pt
                }
            } else {
                base_pt
            };

            let (rx, ry) = if i + 1 == n {
                // Exact arrival: zero cumulative drift.
                accum.snap(plan.exact_end.0, plan.exact_end.1)
            } else {
                accum.push(jx, jy)
            };
            // Fail closed: the accumulator carry can shift the rounded
            // pixel by 1px at the rect border; SEC-1 rejects, never clamps.
            check(rx, ry)?;

            if last_sent != Some((rx, ry)) || i + 1 == n {
                xtest_motion(conn, root, rx, ry)?;
                // v1.2, fix 30: no per-point flush. X11 is buffered by
                // design; a flush per pixel is a synchronous round trip
                // per point. on_frame() still runs per point for the
                // drift detector; the socket flush happens once below.
                on_frame();
                last_sent = Some((rx, ry));
                emitted += 1;
            }
        }
    }

    // Single flush for the whole motion phase (fix 30). The button phase
    // below flushes its own events via xtest_button/xtest_wheel.
    conn.flush().context("XTest flush")?;

    // Button phase continues the same absolute-deadline schedule
    // (hover -> press -> dwell -> release).
    match plan.button {
        ButtonPlan::None => {}
        ButtonPlan::DownThroughMove => {
            xtest_button(conn, root, false)?;
        }
        // v1.2, fix 11: Scroll gestures emit real wheel events.
        ButtonPlan::Scroll { ticks } => {
            let down = ticks >= 0;
            for _ in 0..ticks.unsigned_abs() {
                xtest_wheel(conn, root, down)?;
            }
        }
        ButtonPlan::Click { dwell_ms } => {
            deadline = deadline_add_ms(&deadline, plan.hover_ms);
            sleep_until_deadline(&deadline);
            xtest_button(conn, root, true)?;
            deadline = deadline_add_ms(&deadline, dwell_ms);
            sleep_until_deadline(&deadline);
            xtest_button(conn, root, false)?;
        }
        ButtonPlan::DoubleClick { dwell_ms, inter_ms } => {
            deadline = deadline_add_ms(&deadline, plan.hover_ms);
            sleep_until_deadline(&deadline);
            xtest_button(conn, root, true)?;
            deadline = deadline_add_ms(&deadline, dwell_ms);
            sleep_until_deadline(&deadline);
            xtest_button(conn, root, false)?;
            deadline = deadline_add_ms(&deadline, inter_ms);
            sleep_until_deadline(&deadline);
            xtest_button(conn, root, true)?;
            deadline = deadline_add_ms(&deadline, dwell_ms);
            sleep_until_deadline(&deadline);
            xtest_button(conn, root, false)?;
        }
    }

    let t1 = monotonic_now();
    let duration_ms =
        (t1.tv_sec - t0.tv_sec) as f64 * 1000.0 + (t1.tv_nsec - t0.tv_nsec) as f64 / 1e6;
    Ok(EmitReport {
        emitted,
        skipped,
        duration_ms,
    })
}

/// Direct (non-thread) emission of a planned gesture. Used by the KS
/// harness and tests; production actuation goes through [`InputThread`].
pub fn emit_direct(
    conn: &RustConnection,
    root: Window,
    rect: &WindowRect,
    plan: &PlannedEmit,
) -> Result<EmitReport> {
    let check = |x: i32, y: i32| check_in_window(Some(rect), x, y);
    emit_inner(conn, root, &check, plan, &mut || {})
}

// ---------------------------------------------------------------------------
// SeqLock: latest-value broadcast from the input thread
// ---------------------------------------------------------------------------

/// Hand-rolled sequence lock: latest-value pointer/damage telemetry
/// broadcast from the input thread back to the HTTP plane. One writer
/// (the input thread), many readers. No mutex, no new dependency.
///
/// Protocol: the writer bumps the sequence to odd, writes the payload,
/// bumps to even. A reader retries when it observes an odd sequence or a
/// sequence change across the read, so it never sees a torn write.
/// `T: Copy` keeps both sides allocation-free.
pub struct SeqLock<T: Copy> {
    seq: AtomicU64,
    data: UnsafeCell<T>,
}

// SAFETY: the seqlock protocol gives the single writer exclusive access
// during the odd-sequence window; readers only read during even windows
// and retry on any race. `T: Copy + Send` makes the bitwise copy safe to
// move across threads.
unsafe impl<T: Copy + Send> Send for SeqLock<T> {}
unsafe impl<T: Copy + Send> Sync for SeqLock<T> {}

impl<T: Copy> SeqLock<T> {
    pub const fn new(value: T) -> Self {
        Self {
            seq: AtomicU64::new(0),
            data: UnsafeCell::new(value),
        }
    }

    /// Writer side: the input thread ONLY.
    pub fn store(&self, value: T) {
        let s = self.seq.load(Ordering::Relaxed);
        self.seq.store(s + 1, Ordering::Release); // odd: write in progress
                                                  // SAFETY: we hold the odd-sequence window; no reader samples now.
        unsafe {
            *self.data.get() = value;
        }
        self.seq.store(s + 2, Ordering::Release); // even: consistent
    }

    /// Reader side: any thread. Spins briefly on a torn read.
    pub fn load(&self) -> T {
        loop {
            let s1 = self.seq.load(Ordering::Acquire);
            if s1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            // SAFETY: s1 even means no writer is inside the critical
            // section; the trailing sequence check catches a race.
            let v = unsafe { *self.data.get() };
            std::sync::atomic::fence(Ordering::Acquire);
            let s2 = self.seq.load(Ordering::Acquire);
            if s1 == s2 {
                return v;
            }
        }
    }
}

/// Latest-value pointer/damage state broadcast by the input thread.
#[derive(Debug, Clone, Copy, Default)]
pub struct InputState {
    pub x: i32,
    pub y: i32,
    /// Bitmask of held buttons (bit 0 = button 1).
    pub buttons: u8,
    /// Flushed frames since thread start.
    pub frame_seq: u64,
    /// Monotonic microseconds of the last emission.
    pub last_emit_us: u64,
}

// ---------------------------------------------------------------------------
// Input thread: owns the RustConnection, driven over rtrb rings
// ---------------------------------------------------------------------------

/// Commands crossing the gesture ring (SPSC, wait-free).
pub enum GestureCmd {
    Emit(PlannedEmit),
    Shutdown,
}

/// Side effects drained between flushed frames: geometry/pointer requests
/// that must never block mid-trajectory. Separate SPSC ring so a drain
/// can never swallow a queued gesture.
pub enum SideEffect {
    QueryPointer,
    SetRect(WindowRect),
    Ping,
}

const GESTURE_RING_CAP: usize = 256;
const SIDE_RING_CAP: usize = 64;
/// Max side effects drained per between-frame drain call.
const DRAIN_BUDGET: usize = 16;

/// The actuation thread. Owns the `RustConnection` for its whole life:
/// no mutex, no tokio, no shared borrows across the thread boundary.
/// Gesture dispatch arrives on the rtrb SPSC ring; pointer/damage state
/// broadcasts back on the [`SeqLock`].
pub struct InputThread {
    // The SPSC contract lives on the ring: one logical producer. The
    // mutex serializes concurrent dispatchers (tokio workers share this
    // handle) and is held only for the push itself , never across a
    // sleep or a frame.
    gesture_tx: std::sync::Mutex<Producer<GestureCmd>>,
    side_tx: std::sync::Mutex<Producer<SideEffect>>,
    state: Arc<SeqLock<InputState>>,
    unpark: thread::Thread,
    thread: Option<JoinHandle<()>>,
}

impl InputThread {
    /// Spawn the input thread, moving the X11 connection onto it.
    pub fn spawn(conn: RustConnection, root: Window, rect: WindowRect) -> Self {
        let (gesture_tx, gesture_rx) = RingBuffer::<GestureCmd>::new(GESTURE_RING_CAP);
        let (side_tx, side_rx) = RingBuffer::<SideEffect>::new(SIDE_RING_CAP);
        let state = Arc::new(SeqLock::new(InputState::default()));
        let thread = thread::Builder::new()
            .name("edge-input".into())
            .spawn({
                let state = Arc::clone(&state);
                move || {
                    InputRunner {
                        gesture_rx,
                        side_rx: RefCell::new(side_rx),
                        state,
                        conn,
                        root,
                        rect,
                        frame_seq: 0,
                        shutdown: false,
                        pending_rect: None,
                        pending_query: false,
                    }
                    .run()
                }
            })
            .expect("spawn edge input thread");
        let unpark = thread.thread().clone();
        Self {
            gesture_tx: std::sync::Mutex::new(gesture_tx),
            side_tx: std::sync::Mutex::new(side_tx),
            state,
            unpark,
            thread: Some(thread),
        }
    }

    /// Dispatch a gesture across the SPSC ring; wakes the thread.
    /// Fails closed when the ring is full (the caller retries or sheds;
    /// actuation is never silently dropped into a queue).
    pub fn dispatch_gesture(&self, cmd: GestureCmd) -> Result<(), rtrb::PushError<GestureCmd>> {
        let r = self
            .gesture_tx
            .lock()
            .expect("gesture ring producer mutex")
            .push(cmd);
        self.unpark.unpark();
        r
    }

    /// Queue a side effect (geometry/pointer request). Drained between
    /// flushed frames; never blocks mid-trajectory.
    pub fn dispatch_side_effect(&self, se: SideEffect) -> Result<(), rtrb::PushError<SideEffect>> {
        let r = self
            .side_tx
            .lock()
            .expect("side ring producer mutex")
            .push(se);
        self.unpark.unpark();
        r
    }

    /// Latest broadcast pointer/damage state.
    pub fn state(&self) -> InputState {
        self.state.load()
    }

    /// Ask the thread to stop and join it.
    pub fn shutdown(mut self) {
        let _ = self.dispatch_gesture(GestureCmd::Shutdown);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

struct InputRunner {
    gesture_rx: Consumer<GestureCmd>,
    side_rx: RefCell<Consumer<SideEffect>>,
    state: Arc<SeqLock<InputState>>,
    conn: RustConnection,
    root: Window,
    rect: WindowRect,
    frame_seq: u64,
    shutdown: bool,
    /// v1.2, fix 7: a SetRect/QueryPointer that arrives mid-gesture is
    /// STASHED here, not dropped. The old on_frame popped the command
    /// and `break`ed, deleting it despite the "deferred" comment; SEC-1
    /// then kept the stale rect for the whole gesture. The stash is
    /// applied right after the gesture completes.
    pending_rect: Option<WindowRect>,
    pending_query: bool,
}

impl InputRunner {
    fn run(mut self) {
        while !self.shutdown {
            self.drain_side_effects();
            match self.gesture_rx.pop() {
                Ok(GestureCmd::Emit(plan)) => {
                    if let Err(e) = self.emit(&plan) {
                        // Actuation errors are terminal for the gesture,
                        // not the thread: keep serving.
                        eprintln!("edge-input: gesture failed: {e:#}");
                    }
                }
                Ok(GestureCmd::Shutdown) => self.shutdown = true,
                Err(_) => thread::park(), // empty: sleep until dispatch unparks
            }
        }
    }

    /// Drain side-effect commands between flushed frames. Only the side
    /// ring is touched here, so queued gestures are never disturbed.
    /// Returns true when a shutdown was requested.
    fn drain_side_effects(&mut self) {
        let mut rx = self.side_rx.borrow_mut();
        for _ in 0..DRAIN_BUDGET {
            match rx.pop() {
                Ok(SideEffect::QueryPointer) => {
                    if let Ok(cookie) = self.conn.query_pointer(self.root) {
                        if let Ok(r) = cookie.reply() {
                            let mut s = self.state.load();
                            s.x = i32::from(r.root_x);
                            s.y = i32::from(r.root_y);
                            self.state.store(s);
                        }
                    }
                }
                Ok(SideEffect::SetRect(rect)) => {
                    self.rect = rect;
                }
                Ok(SideEffect::Ping) => {}
                Err(_) => break,
            }
        }
    }

    fn broadcast_frame(&mut self, x: i32, y: i32) {
        let now = monotonic_now();
        self.frame_seq += 1;
        let mut s = self.state.load();
        s.x = x;
        s.y = y;
        s.frame_seq = self.frame_seq;
        s.last_emit_us = (now.tv_sec as u64) * 1_000_000 + (now.tv_nsec as u64) / 1_000;
        self.state.store(s);
    }

    fn emit(&mut self, plan: &PlannedEmit) -> Result<EmitReport> {
        // Split borrows: the drain closure touches only the side ring and
        // the broadcast state, never the connection.
        let side_rx = &self.side_rx;
        let runner_state = &self.state;
        let frame_seq = &mut self.frame_seq;
        let pending_rect = &mut self.pending_rect;
        let pending_query = &mut self.pending_query;
        let check_rect = self.rect;
        let check = |x: i32, y: i32| check_in_window(Some(&check_rect), x, y);
        let mut on_frame = || {
            // Drain side effects between flushed frames; no mutex held
            // across sleeps (the runner owns everything here). Pointer
            // queries and rect changes that arrive mid-gesture are
            // STASHED (fix 7), not dropped: the side ring is a consumer,
            // so re-queueing is impossible, and the old `break` deleted
            // the command despite the "deferred" comment.
            let mut rx = side_rx.borrow_mut();
            for _ in 0..DRAIN_BUDGET {
                match rx.pop() {
                    Ok(SideEffect::QueryPointer) => {
                        *pending_query = true;
                        break;
                    }
                    Ok(SideEffect::SetRect(rect)) => {
                        *pending_rect = Some(rect);
                        break;
                    }
                    Ok(SideEffect::Ping) => {}
                    Err(_) => break,
                }
            }
            drop(rx);
            *frame_seq += 1;
            let mut s = runner_state.load();
            s.frame_seq = *frame_seq;
            runner_state.store(s);
        };
        let report = emit_inner(&self.conn, self.root, &check, plan, &mut on_frame);
        // Apply the deferred side effects (fix 7): the stashed rect takes
        // effect before the next gesture's SEC-1 check; the stashed query
        // runs now.
        if let Some(rect) = self.pending_rect.take() {
            self.rect = rect;
        }
        if std::mem::replace(&mut self.pending_query, false) {
            if let Ok(cookie) = self.conn.query_pointer(self.root) {
                if let Ok(r) = cookie.reply() {
                    let mut s = self.state.load();
                    s.x = i32::from(r.root_x);
                    s.y = i32::from(r.root_y);
                    self.state.store(s);
                }
            }
        }
        if report.is_ok() {
            // Latest-value broadcast: the gesture ended exactly on target.
            self.broadcast_frame(plan.exact_end.0, plan.exact_end.1);
        }
        report
    }
}

// ---------------------------------------------------------------------------
// Predictive drift during Tier-1 waits
// ---------------------------------------------------------------------------

/// A candidate target for predictive drift.
#[derive(Debug, Clone, Copy)]
pub struct DriftTarget {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// Predictive drift while the runner waits on Tier-1 (plan Layer 4).
///
/// A frozen cursor during a 2s model call is itself a temporal tell. The
/// engine pre-computes trajectories to the top-3 candidates; this struct
/// drifts at low velocity toward their geometric barycenter. On
/// confirmation, [`PredictiveDrift::splice`] splices the confirmed spline
/// onto the in-flight state, hiding 80-150ms of actuation inside planning
/// time.
///
/// The scheduler never calls Tier-1: it only accepts the optional top-3
/// precomputed paths and the confirmation index.
pub struct PredictiveDrift {
    pos: (f64, f64),
    barycenter: (f64, f64),
    paths: [Vec<(f64, f64)>; 3],
    speed_px_per_s: f64,
    arrived: bool,
}

impl PredictiveDrift {
    /// Start drifting from `from` toward the barycenter of the three
    /// precomputed candidate paths. `speed_px_per_s` is the low drift
    /// velocity (tuning stays below the perceptual threshold on the VM
    /// display; the plan leaves the exact value open).
    pub fn new(from: (f64, f64), paths: [Vec<(f64, f64)>; 3], speed_px_per_s: f64) -> Self {
        let mut bx = 0.0;
        let mut by = 0.0;
        for p in &paths {
            let e = p.last().copied().unwrap_or(from);
            bx += e.0;
            by += e.1;
        }
        Self {
            pos: from,
            barycenter: (bx / 3.0, by / 3.0),
            paths,
            speed_px_per_s: speed_px_per_s.max(0.0),
            arrived: false,
        }
    }

    /// Advance the drift by `dt_s`; returns the position to emit.
    pub fn step(&mut self, dt_s: f64) -> (f64, f64) {
        if self.arrived || self.speed_px_per_s <= 0.0 {
            return self.pos;
        }
        let dx = self.barycenter.0 - self.pos.0;
        let dy = self.barycenter.1 - self.pos.1;
        let d = dx.hypot(dy);
        let step = self.speed_px_per_s * dt_s;
        if d <= step || d < 1e-9 {
            self.pos = self.barycenter;
            self.arrived = true;
        } else {
            self.pos.0 += dx / d * step;
            self.pos.1 += dy / d * step;
        }
        self.pos
    }

    pub fn position(&self) -> (f64, f64) {
        self.pos
    }

    pub fn barycenter(&self) -> (f64, f64) {
        self.barycenter
    }

    pub fn arrived(&self) -> bool {
        self.arrived
    }

    /// Splice the confirmed target's precomputed spline onto the in-flight
    /// state: drop the path prefix behind the point nearest the current
    /// drift position, then lead in from the current position with a
    /// chord-capped leg so there is no teleport.
    pub fn splice(self, confirmed: usize) -> Vec<(f64, f64)> {
        let path = &self.paths[confirmed.min(2)];
        let mut best = 0;
        let mut best_d = f64::INFINITY;
        for (i, &p) in path.iter().enumerate() {
            let d = (p.0 - self.pos.0).hypot(p.1 - self.pos.1);
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        let mut out = Vec::with_capacity(path.len() - best + 2);
        // Chord-capped lead-in from the drift position to the splice point.
        let target = path.get(best).copied().unwrap_or(self.pos);
        let lead_d = (target.0 - self.pos.0).hypot(target.1 - self.pos.1);
        let n = ((lead_d / MAX_CHORD_PX).ceil() as usize).max(1);
        for i in 0..n {
            let t = i as f64 / n as f64;
            out.push((
                self.pos.0 + (target.0 - self.pos.0) * t,
                self.pos.1 + (target.1 - self.pos.1) * t,
            ));
        }
        out.extend_from_slice(&path[best..]);
        out
    }
}

// ---------------------------------------------------------------------------
// Unit tests (headless: no X11)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Baseline p50s, computed from docs/baseline-2026-11.json
    // (median of the 20 durations_us per distance bucket).
    const BASELINE_P50_MS: [(f64, f64); 5] = [
        (100.0, 212.5),
        (200.0, 314.4),
        (400.0, 321.6),
        (600.0, 494.0),
        (800.0, 650.7),
    ];
    /// Mean ln(MT/ID) over all baseline samples, MT in seconds,
    /// ID = log2(D/24+1) bits. Computed from the same JSON: -2.320.
    const BASELINE_MEAN_LN_MT_ID: f64 = -2.32;

    #[test]
    fn w_formula_three_axes() {
        // Horizontal approach: W = w.
        assert!((target_width(100.0, 20.0, (1.0, 0.0)) - 100.0).abs() < 1e-9);
        // Vertical approach: W = h.
        assert!((target_width(100.0, 20.0, (0.0, 1.0)) - 20.0).abs() < 1e-9);
        // Diagonal 45 degrees on a square: w*|ux| + h*|uy|.
        let w = target_width(100.0, 100.0, (1.0, 1.0));
        assert!((w - 141.4213562373095).abs() < 1e-6, "w={w}");
        // Approach direction sign does not matter.
        assert!(
            (target_width(100.0, 20.0, (-3.0, -4.0)) - target_width(100.0, 20.0, (3.0, 4.0))).abs()
                < 1e-9
        );
        // Degenerate approach: min(w,h).
        assert!((target_width(100.0, 20.0, (0.0, 0.0)) - 20.0).abs() < 1e-9);
        // Clamp: never below 1px.
        assert!((target_width(0.0, 0.0, (1.0, 0.0)) - 1.0).abs() < 1e-9);
        assert!((target_width(100.0, 20.0, (1e-9, 0.0)) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn mt_clamps() {
        // Tiny distance with small a/b: clamped to the 16ms floor.
        assert!((fitts_mt_median_ms(0.005, 0.010, 1.0, 24.0) - 16.0).abs() < 1e-9);
        // Huge distance: clamped to the 3000ms ceiling.
        assert!((fitts_mt_median_ms(0.05, 0.15, 1e9, 1.0) - 3000.0).abs() < 1e-9);
        // Sane middle: a + b*log2(D/W+1) in seconds -> ms.
        let mt = fitts_mt_median_ms(0.05, 0.15, 400.0, 24.0);
        let expect = (0.05 + 0.15 * (400.0f64 / 24.0 + 1.0).log2()) * 1000.0;
        assert!((mt - expect).abs() < 1e-6, "mt={mt} expect={expect}");
    }

    #[test]
    fn legacy_duration_reproduces_baseline() {
        // PARITY GATE (unit level): the Legacy formula against the measured
        // baseline p50s, +/-15% per bucket. Constants are FROM
        // docs/baseline-2026-11.json, not invented.
        for (d, p50) in BASELINE_P50_MS {
            let mt = legacy_duration_ms(d);
            let rel = (mt - p50).abs() / p50;
            assert!(
                rel <= 0.15,
                "Legacy MT {mt:.1}ms at {d}px deviates {:.1}% from baseline p50 {p50}ms",
                rel * 100.0,
            );
        }
        // Mean ln(MT/ID), MT in seconds, within +/-0.3 of -2.32.
        let mean = BASELINE_P50_MS
            .iter()
            .map(|(d, _)| {
                let id = (d / 24.0 + 1.0).log2();
                (legacy_duration_ms(*d) / 1000.0 / id).ln()
            })
            .sum::<f64>()
            / BASELINE_P50_MS.len() as f64;
        assert!(
            (mean - BASELINE_MEAN_LN_MT_ID).abs() <= 0.3,
            "mean ln(MT/ID) = {mean:.3}, baseline -2.32"
        );
    }

    #[test]
    fn overshoot_probability_shape() {
        // Rises with ID, bounded, zero-ish floor on huge targets.
        let p_small = overshoot_probability(fitts_id(100.0, 200.0));
        let p_big = overshoot_probability(fitts_id(800.0, 8.0));
        assert!(p_big > p_small, "{p_big} vs {p_small}");
        assert!((0.0..=0.85).contains(&p_small));
        assert!((0.0..=0.85).contains(&p_big));
    }

    #[test]
    fn ou_tremor_mean_reverts() {
        let mut t = OuTremor::new(60.0, 0.5);
        // With zero draws the state decays as exp(-dt/tau).
        let (jx, _) = t.step(30.0, 2.0, 0.0);
        let a = (-30.0f64 / 60.0).exp();
        let b = 0.5 * (1.0 - a * a).sqrt();
        assert!((jx - b * 2.0).abs() < 1e-12);
        let (jx2, _) = t.step(600.0, 0.0, 0.0);
        assert!(jx2.abs() < jx.abs() * 0.01, "must decay toward zero");
        // Stationary scale stays bounded under sustained noise.
        let mut t2 = OuTremor::new(60.0, 0.5);
        let mut rng = XorShift64::with_salt(42);
        let mut max_j = 0.0f64;
        for _ in 0..10000 {
            let (zx, zy) = rng.gauss_pair();
            let (jx, jy) = t2.step(8.0, zx, zy);
            max_j = max_j.max(jx.abs()).max(jy.abs());
        }
        assert!(max_j < 5.0, "OU state blew up: {max_j}");
    }

    #[test]
    fn seqlock_latest_value() {
        let lock = Arc::new(SeqLock::new(0u64));
        // Single-threaded sanity.
        lock.store(7);
        assert_eq!(lock.load(), 7);
        // Concurrent writer/reader with a MONOTONIC store sequence (reset
        // to 0 before spawning): the reader always sees a fully written
        // value and never goes backwards.
        lock.store(0);
        let w = Arc::clone(&lock);
        let writer = thread::spawn(move || {
            for i in 1..=5000u64 {
                w.store(i);
            }
        });
        let mut last = 0u64;
        while last < 5000 {
            let v = lock.load();
            assert!(v >= last, "seqlock went backwards: {v} < {last}");
            last = v;
        }
        writer.join().unwrap();
        assert_eq!(lock.load(), 5000);
    }

    #[test]
    fn rtrb_rings_dispatch() {
        // Gesture ring carries commands without loss or reorder.
        let (mut tx, mut rx) = RingBuffer::<GestureCmd>::new(8);
        tx.push(GestureCmd::Shutdown).unwrap();
        assert!(matches!(rx.pop().unwrap(), GestureCmd::Shutdown));
        // Full ring fails closed (never silently queues).
        let (mut tx2, _rx2) = RingBuffer::<SideEffect>::new(2);
        tx2.push(SideEffect::Ping).unwrap();
        tx2.push(SideEffect::Ping).unwrap();
        assert!(tx2.push(SideEffect::Ping).is_err());
        // Side ring drains independently of the gesture ring.
        let (mut stx, mut srx) = RingBuffer::<SideEffect>::new(4);
        stx.push(SideEffect::Ping).unwrap();
        assert!(matches!(srx.pop().unwrap(), SideEffect::Ping));
    }

    #[test]
    fn predictive_drift_reaches_barycenter_and_splices() {
        let from = (0.0, 0.0);
        let mk = |ex: f64, ey: f64| vec![from, (ex, ey)];
        let mut drift = PredictiveDrift::new(
            from,
            [mk(300.0, 0.0), mk(0.0, 300.0), mk(300.0, 300.0)],
            60.0,
        );
        // Barycenter of (300,0), (0,300), (300,300) = (200,200).
        let (bx, by) = drift.barycenter();
        assert!((bx - 200.0).abs() < 1e-9 && (by - 200.0).abs() < 1e-9);
        // Drift advances toward it and stops there.
        let mut pos = from;
        for _ in 0..1000 {
            pos = drift.step(0.1);
            if drift.arrived() {
                break;
            }
        }
        assert!(drift.arrived());
        assert!((pos.0 - 200.0).abs() < 1e-9 && (pos.1 - 200.0).abs() < 1e-9);
        // Splice starts at the drift position and continues the confirmed path.
        let spliced = drift.splice(0);
        assert!(!spliced.is_empty());
        assert!((spliced[0].0 - 200.0).abs() < 1e-9);
        assert!((spliced[0].1 - 200.0).abs() < 1e-9);
        let last = spliced[spliced.len() - 1];
        assert!((last.0 - 300.0).abs() < 1e-9 && (last.1 - 0.0).abs() < 1e-9);
        // Chord cap holds on the lead-in.
        for w in spliced.windows(2) {
            let c = (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1);
            assert!(c <= MAX_CHORD_PX + 1e-9, "lead-in chord {c}");
        }
    }

    #[test]
    fn plan_gesture_legacy_shapes() {
        let rect = WindowRect {
            x: 0,
            y: 0,
            w: 1920,
            h: 1080,
        };
        let plan = plan_gesture(
            SpeedProfile::Legacy,
            GestureKind::Click,
            (100.0, 100.0),
            (500.0, 100.0),
            TargetBox { w: 24.0, h: 24.0 },
            &rect,
            12345,
            16.7,
        )
        .unwrap();
        assert_eq!(plan.profile, SpeedProfile::Legacy);
        assert!(plan.points.len() >= 2);
        assert_eq!(plan.samples, plan.points.len());
        assert!(plan.spacing_noise > 0.0);
        assert!(plan.mt_ms > 0.0);
        assert!(matches!(plan.button, ButtonPlan::Click { .. }));
        assert!(plan.hover_ms > 0.0);
        assert_eq!(plan.exact_end, (500, 100));
        // Every planned point passes SEC-1.
        for &(px, py) in &plan.points {
            assert!(check_in_window(Some(&rect), px.round() as i32, py.round() as i32).is_ok());
        }
    }

    #[test]
    fn plan_gesture_fitts_provisional_shapes() {
        // Provisional profiles plan but are NOT available by default.
        assert!(!SpeedProfile::Human.is_available());
        assert!(!SpeedProfile::Fast.is_available());
        assert!(SpeedProfile::Legacy.is_available());
        let rect = WindowRect {
            x: 0,
            y: 0,
            w: 1920,
            h: 1080,
        };
        for profile in [SpeedProfile::Human, SpeedProfile::Fast] {
            let plan = plan_gesture(
                profile,
                GestureKind::Move,
                (100.0, 100.0),
                (500.0, 300.0),
                TargetBox { w: 24.0, h: 24.0 },
                &rect,
                999,
                16.7,
            )
            .unwrap();
            assert!((MT_MIN_MS..=MT_MAX_MS).contains(&plan.mt_ms));
            assert!(plan.samples >= 2);
            assert_eq!(plan.spacing_noise, 0.0);
            for &(px, py) in &plan.points {
                assert!(check_in_window(Some(&rect), px.round() as i32, py.round() as i32).is_ok());
            }
        }
    }

    #[test]
    fn lognormal_median_sigma() {
        let l = Lognormal {
            median_ms: 100.0,
            sigma: 0.35,
        };
        assert!((l.sample(0.0) - 100.0).abs() < 1e-9);
        assert!(l.sample(1.0) > 100.0);
        assert!(l.sample(-1.0) < 100.0);
    }
}
