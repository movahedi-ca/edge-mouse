//! Vendored legacy input primitives.
//!
//! Adapted verbatim from `rust/src/input/x11_mouse.rs` in
//! chromebook-vm-bridge (MIT). These are the pure, X11-free pieces the
//! edge engine depends on: the verified window rect and its SEC-1 gate,
//! the legacy timing model, the trajectory builders, the easing function,
//! and the small geometry/RNG/time utilities behind them. `pub(crate)`
//! visibility from the original is `pub` here; behavior is unchanged.

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};

/// Minimum travel distance that still gets a Bezier leg; shorter moves
/// use capped linear steps.
const MIN_BEZIER_DIST_PX: f64 = 15.0;
/// No two consecutive trajectory points may be farther apart than this.
/// Chord cap for resampled paths; used by the scheduler's splice lead-in.
pub const MAX_CHORD_PX: f64 = 12.0;
const MAX_CHORD_SQ: f64 = MAX_CHORD_PX * MAX_CHORD_PX;
/// de Casteljau flatness tolerance (Cairo/Skia-style).
const FLATNESS_TOL_PX: f64 = 0.5;
const FLATNESS_TOL_SQ: f64 = FLATNESS_TOL_PX * FLATNESS_TOL_PX;
const MAX_SUBDIV_DEPTH: u32 = 10;
/// Uniform spacing after arc-length resampling.
const RESAMPLE_SPACING_PX: f64 = 4.0;

/// Legacy movement-time model: the pre-scheduler driver's emit formula.
///
/// N = total/4 + 1 points, base = clamp(1.2/total, 0.003, 0.008) s per
/// point, uniform(0.7, 1.3) noise with mean 1. Returns the nominal (mean)
/// duration in milliseconds for a trajectory of `total_arc_px` arc length.
pub fn legacy_duration_ms(total_arc_px: f64) -> f64 {
    let n = total_arc_px / 4.0 + 1.0;
    let base = (1.2 / total_arc_px.max(1.0)).clamp(0.003, 0.008);
    // v1.2: sample 0 emits with no sleep (the emit loop only sleeps for
    // i > 0), so wall time is (n-1)*base, not n*base. The old formula
    // reported ~2x the actual duration for short moves (n = 2).
    (n - 1.0).max(0.0) * base * 1000.0
}

/// Overdue latch for absolute-deadline emission: a sample more than this
/// far past its deadline is skipped (never queued). Covers routine
/// clock_nanosleep wakeup jitter (sub-ms) without masking real stalls.
pub const OVERDUE_LATCH_MS: f64 = 8.0;

/// A 2D point in device pixels.
pub type Pt = (f64, f64);
/// A cubic Bezier leg: four control points.
type Cubic = (Pt, Pt, Pt, Pt);

/// Verified browser window rectangle on the root screen, in screen pixels.
/// Populated from fresh X11 geometry queries before each visible action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowRect {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

/// SEC-1 gate: reject any coordinate outside the verified window rect.
///
/// Free function (no X11 needed) so it is unit-testable and so callers can
/// pre-validate before acquiring the mouse. Uses `i64` bounds math to prevent
/// signed `i32` overflow and verifies `i16` range for `XTestFakeInput`.
#[inline]
pub fn check_in_window(rect: Option<&WindowRect>, x: i32, y: i32) -> Result<()> {
    let r = rect
        .ok_or_else(|| anyhow!("SEC-1: window rect not set; refusing to drive the raw desktop"))?;
    let xi = i64::from(x);
    let yi = i64::from(y);
    let rx0 = i64::from(r.x);
    let ry0 = i64::from(r.y);
    let rx1 = rx0 + i64::from(r.w);
    let ry1 = ry0 + i64::from(r.h);
    let fits_i16 = i16::try_from(x).is_ok() && i16::try_from(y).is_ok();
    let inside = fits_i16 && xi >= rx0 && yi >= ry0 && xi < rx1 && yi < ry1;
    if inside {
        Ok(())
    } else {
        Err(anyhow!(
            "SEC-1: coordinate ({x},{y}) outside verified window rect \
             ({},{},{},{}) - rejected, not clamped",
            r.x,
            r.y,
            r.w,
            r.h
        ))
    }
}

// ---------------------------------------------------------------------------
// Deterministic RNG
// ---------------------------------------------------------------------------

/// Deterministic xorshift64* RNG.
///
/// Seeded from the system clock XOR a per-action salt, so every trajectory
/// is unique but reproducible within a single action's plan+emit pair.
#[derive(Debug, Clone)]
pub struct XorShift64(u64);

impl XorShift64 {
    pub fn seeded() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0x9E37_79B9_7F4A_7C15, |d| d.as_nanos() as u64);
        Self(nanos | 1)
    }

    pub fn with_salt(salt: u64) -> Self {
        let mut s = Self::seeded();
        s.0 ^= salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        // Guarantee non-zero state after XORing salt (0 is an absorbing state
        // for xorshift64*) and warm up the state once to diffuse low timestamp bits.
        s.0 |= 1;
        let _ = s.next_u64();
        s
    }

    /// Deterministic seed: NO wall-clock mixing. `plan_gesture` and
    /// `emit_inner` must use this so `PlannedEmit.seed` fully determines
    /// the trajectory and timing noise (v1.2, fix 14). `with_salt` mixes
    /// `SystemTime::now()`, which broke the determinism contract.
    /// `seed | 1` avoids the zero absorbing state.
    pub fn from_seed(seed: u64) -> Self {
        let mut s = Self(seed | 1);
        let _ = s.next_u64(); // warm up once, same as with_salt
        s
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        const INV_2POW53: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * INV_2POW53
    }

    #[inline]
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }

    /// Two independent standard normal variates via a single Box-Muller transform.
    #[inline]
    pub fn gauss_pair(&mut self) -> (f64, f64) {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
        (r * c, r * s)
    }
}

// ---------------------------------------------------------------------------
// Pure geometry helpers (all unit-testable, no X11).
// ---------------------------------------------------------------------------

#[inline]
fn dist_sq(a: Pt, b: Pt) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

#[inline]
fn dist(a: Pt, b: Pt) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

#[inline]
fn lerp(a: Pt, b: Pt, t: f64) -> Pt {
    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
}

#[inline]
fn midpoint(a: Pt, b: Pt) -> Pt {
    ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5)
}

/// Asymmetric minimum-jerk easing.
///
///   s(t) = (1 - A) * mj(t) + A * warp(t),   t in [0, 1]
///   mj(t)   = 10t^3 - 15t^4 + 6t^5     (symmetric minimum-jerk: zero
///                                       endpoint velocity/acceleration,
///                                       velocity peak at t = 0.50)
///   warp(t) = 1 - (1 - t)^P            (decel-heavy power warp: zero end
///                                       velocity, velocity peak at t = 0)
///
///   A = 0.45, P = 2.35  ->  velocity peak near t ~= 0.38: the fast
///   ballistic onset of a human reach followed by a long corrective tail.
///   s(0) = 0, s(1) = 1, s'(1) = 0, s monotone non-decreasing on [0, 1]
///   (both terms have non-negative derivative there). Note s'(0) = A*P > 0:
///   the onset is intentionally fast; the sub-pixel residual accumulator
///   keeps the fractional early velocity instead of truncating it away.
pub const EASE_BLEND_A: f64 = 0.45;
pub const EASE_WARP_P: f64 = 2.35;

#[inline]
pub fn asymmetric_min_jerk(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    let mj = t * t * t * (10.0 + t * (-15.0 + 6.0 * t));
    let warp = 1.0 - (1.0 - t).powf(EASE_WARP_P);
    (1.0 - EASE_BLEND_A) * mj + EASE_BLEND_A * warp
}

/// Squared distance from point `p` to segment `a -> b`. Avoids `sqrt` inside
/// inner subdivision and smoothness loops.
#[inline]
fn dist_sq_point_to_segment(p: Pt, a: Pt, b: Pt) -> f64 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let l2 = (dx * dx + dy * dy).max(1e-9);
    let t = (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2).clamp(0.0, 1.0);
    let qx = (a.0 + t * dx) - p.0;
    let qy = (a.1 + t * dy) - p.1;
    qx * qx + qy * qy
}

#[inline]
fn flat_enough_sq(p0: Pt, p1: Pt, p2: Pt, p3: Pt, tol_sq: f64) -> bool {
    dist_sq_point_to_segment(p1, p0, p3) < tol_sq && dist_sq_point_to_segment(p2, p0, p3) < tol_sq
}

#[inline]
fn subdivide(p0: Pt, p1: Pt, p2: Pt, p3: Pt) -> (Cubic, Cubic) {
    let m01 = midpoint(p0, p1);
    let m12 = midpoint(p1, p2);
    let m23 = midpoint(p2, p3);
    let m012 = midpoint(m01, m12);
    let m123 = midpoint(m12, m23);
    let mid = midpoint(m012, m123);
    ((p0, m01, m012, mid), (mid, m123, m23, p3))
}

/// Adaptive subdivision. Terminates when the segment is BOTH no longer than
/// `MAX_CHORD_PX` AND flat within tolerance, or at max depth. Checks
/// `chord_sq <= MAX_CHORD_SQ` first to short-circuit flatness projections on
/// upper tree levels, and uses squared distances throughout to avoid square
/// roots. Appends points after `p0` into `out`.
fn adaptive_points(p0: Pt, p1: Pt, p2: Pt, p3: Pt, depth: u32, out: &mut Vec<Pt>) {
    let chord_sq = dist_sq(p0, p3);
    if depth >= MAX_SUBDIV_DEPTH
        || (chord_sq <= MAX_CHORD_SQ && flat_enough_sq(p0, p1, p2, p3, FLATNESS_TOL_SQ))
    {
        out.push(p3);
        return;
    }
    let (l, r) = subdivide(p0, p1, p2, p3);
    adaptive_points(l.0, l.1, l.2, l.3, depth + 1, out);
    adaptive_points(r.0, r.1, r.2, r.3, depth + 1, out);
}

/// Uniform arc-length resampling: walk the polyline emitting a point every
/// `spacing` px. Guarantees the chord cap survives to the emitted path.
///
/// Computes `dist(prev, cur)` once per input segment (rather than per emitted
/// sample) and interpolates directly from `(prev, cur)` to eliminate cumulative
/// floating-point drift.
fn resample_arc_length(points: &[Pt], spacing: f64) -> Vec<Pt> {
    if points.len() < 2 {
        return points.to_vec();
    }
    let mut out = Vec::with_capacity(points.len() * 2);
    out.push(points[0]);
    let mut carry = 0.0; // distance already covered since last emitted point
    let mut prev = points[0];
    for &cur in &points[1..] {
        let seg_len = dist(prev, cur);
        if seg_len < 1e-12 {
            continue;
        }
        let mut rem = seg_len;
        let mut consumed = 0.0;
        while carry + rem >= spacing {
            let need = spacing - carry;
            consumed += need;
            rem = seg_len - consumed;
            out.push(lerp(prev, cur, consumed / seg_len));
            carry = 0.0;
        }
        carry += rem;
        prev = cur;
    }
    if let (Some(&last), Some(&o)) = (points.last(), out.last()) {
        if dist_sq(o, last) > 1e-18 {
            out.push(last);
        }
    }
    out
}

/// Monotonic O(N) arc-length cursor over precomputed cumulative lengths.
///
/// Segment lengths are computed once at construction, and the cursor only
/// ever advances forward (the eased arc distance is monotone by
/// construction), so each `advance_to` is amortized O(1): it resumes from
/// the previous segment instead of rescanning from the start.
#[derive(Debug, Clone)]
pub struct ArcLengthCursor {
    pts: Vec<Pt>,
    cum: Vec<f64>,
    seg: usize,
    total: f64,
}

impl ArcLengthCursor {
    pub fn new(pts: &[Pt]) -> Self {
        let mut cum = Vec::with_capacity(pts.len());
        let mut acc = 0.0;
        cum.push(0.0);
        for w in pts.windows(2) {
            acc += dist(w[0], w[1]);
            cum.push(acc);
        }
        Self {
            pts: pts.to_vec(),
            cum,
            seg: 0,
            total: acc,
        }
    }

    /// Total arc length of the path.
    pub fn total(&self) -> f64 {
        self.total
    }

    /// Position at arc distance `d`. `d` must be non-decreasing across
    /// calls (the easing guarantees this); the cursor never rescans.
    pub fn advance_to(&mut self, d: f64) -> Pt {
        let n = self.pts.len();
        if n == 0 {
            return (0.0, 0.0);
        }
        if n == 1 || d <= 0.0 {
            return self.pts[0];
        }
        if d >= self.total {
            return self.pts[n - 1];
        }
        while self.seg + 1 < n - 1 && self.cum[self.seg + 1] < d {
            self.seg += 1;
        }
        let seg_len = self.cum[self.seg + 1] - self.cum[self.seg];
        let t = if seg_len < 1e-9 {
            0.0
        } else {
            ((d - self.cum[self.seg]) / seg_len).clamp(0.0, 1.0)
        };
        lerp(self.pts[self.seg], self.pts[self.seg + 1], t)
    }
}

/// Sub-pixel residual accumulator (rx, ry).
///
/// Each fractional position is added to the carry before rounding and the
/// rounding residual is retained, so the discrete integer sum converges
/// with zero cumulative drift to the exact target: slow initiation and
/// terminal fine-correction keep their fractional velocity instead of
/// truncating it away.
#[derive(Debug, Clone, Default)]
pub struct ResidualAccum {
    rx: f64,
    ry: f64,
}

impl ResidualAccum {
    pub fn new() -> Self {
        Self { rx: 0.0, ry: 0.0 }
    }

    /// Fold a fractional position into the accumulator; returns the
    /// integer pixel to emit this tick.
    pub fn push(&mut self, x: f64, y: f64) -> (i32, i32) {
        let ex = x + self.rx;
        let ey = y + self.ry;
        let ix = ex.round();
        let iy = ey.round();
        self.rx = ex - ix;
        self.ry = ey - iy;
        (ix as i32, iy as i32)
    }

    /// Snap exactly onto the integer target and clear the carry. The final
    /// emission of every gesture uses this: arrival is exact, zero drift.
    pub fn snap(&mut self, x: i32, y: i32) -> (i32, i32) {
        self.rx = 0.0;
        self.ry = 0.0;
        (x, y)
    }
}

// ---------------------------------------------------------------------------
// Monotonic-clock deadline helpers
// ---------------------------------------------------------------------------

/// CLOCK_MONOTONIC timestamp for absolute deadline scheduling.
pub fn monotonic_now() -> libc::timespec {
    let mut ts = std::mem::MaybeUninit::<libc::timespec>::uninit();
    // SAFETY: clock_gettime with CLOCK_MONOTONIC always succeeds on Linux;
    // the timespec is fully initialized before assume_init.
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, ts.as_mut_ptr());
        ts.assume_init()
    }
}

/// Absolute sleep until `deadline` on CLOCK_MONOTONIC. Returns immediately
/// when the deadline already passed: overdue samples are skipped by the
/// caller, never queued.
pub fn sleep_until_deadline(deadline: &libc::timespec) {
    // SAFETY: absolute nanosleep on a valid timespec; no signal handling,
    // so the remaining-time out-param is irrelevant (null).
    unsafe {
        libc::clock_nanosleep(
            libc::CLOCK_MONOTONIC,
            libc::TIMER_ABSTIME,
            deadline,
            std::ptr::null_mut(),
        );
    }
}

/// `base + ms` as an absolute timespec (overflow-safe via i128 nanos).
pub fn deadline_add_ms(base: &libc::timespec, ms: f64) -> libc::timespec {
    let nanos =
        base.tv_sec as i128 * 1_000_000_000 + base.tv_nsec as i128 + (ms * 1_000_000.0) as i128;
    libc::timespec {
        tv_sec: (nanos / 1_000_000_000) as libc::time_t,
        tv_nsec: (nanos % 1_000_000_000) as libc::c_long,
    }
}

/// True when `a` is strictly after `b` on the monotonic clock.
pub fn timespec_after(a: &libc::timespec, b: &libc::timespec) -> bool {
    (a.tv_sec, a.tv_nsec) > (b.tv_sec, b.tv_nsec)
}

// ---------------------------------------------------------------------------
// Trajectory builders
// ---------------------------------------------------------------------------

/// Capped linear steps from `p0` to `p3` when skipping Bezier curves or
/// falling back near window boundaries. Guarantees every segment <= `MAX_CHORD_PX`.
fn capped_linear_leg(p0: Pt, p3: Pt) -> Vec<Pt> {
    let d = dist(p0, p3);
    if d <= MAX_CHORD_PX {
        return vec![p0, p3];
    }
    let n = (d / MAX_CHORD_PX).ceil() as usize;
    let inv_n = 1.0 / (n as f64);
    (0..=n).map(|i| lerp(p0, p3, (i as f64) * inv_n)).collect()
}

/// Build one cubic Bezier leg p0 -> p3 with human-like control points.
/// Returns a uniformly resampled polyline (or capped linear steps for
/// short legs). The chord cap holds on every path.
fn cubic_leg(p0: Pt, p3: Pt, rng: &mut XorShift64) -> Vec<Pt> {
    let dx = p3.0 - p0.0;
    let dy = p3.1 - p0.1;
    let d = dx.hypot(dy);
    if d < MIN_BEZIER_DIST_PX {
        return capped_linear_leg(p0, p3);
    }
    // Unit vector along p0 -> p3 and perpendicular normal (-dy/d, dx/d),
    // avoiding `atan2` + 4 trig evaluations per leg.
    let dxu = dx / d;
    let dyu = dy / d;
    let ux = -dyu;
    let uy = dxu;

    let d1 = d * 0.01 * rng.uniform(4.0, 8.0);
    let d2 = d * 0.01 * rng.uniform(4.0, 8.0);
    let s1: f64 = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
    let s2: f64 = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
    let p1 = (
        p0.0 + 0.25 * dx + s1 * d1 * ux,
        p0.1 + 0.25 * dy + s1 * d1 * uy,
    );
    let p2 = (
        p0.0 + 0.75 * dx + s2 * d2 * ux,
        p0.1 + 0.75 * dy + s2 * d2 * uy,
    );
    let est_pts = ((d / RESAMPLE_SPACING_PX) as usize).saturating_add(4);
    let mut raw = Vec::with_capacity(est_pts);
    raw.push(p0);
    adaptive_points(p0, p1, p2, p3, 0, &mut raw);
    resample_arc_length(&raw, RESAMPLE_SPACING_PX)
}

/// Full trajectory from (x0,y0) to (x1,y1), including the 60%-probability
/// overshoot-and-correct for long moves. Uniformly resampled.
///
/// This is the LEGACY trajectory builder: control-point signs and the
/// overshoot rule are exactly the pre-scheduler driver's, so the Legacy
/// branch reproduces the measured baseline.
pub fn build_trajectory(x0: f64, y0: f64, x1: f64, y1: f64, rng: &mut XorShift64) -> Vec<Pt> {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let d = dx.hypot(dy);
    if d < MIN_BEZIER_DIST_PX {
        return capped_linear_leg((x0, y0), (x1, y1));
    }
    // Leg endpoints: optionally overshoot first along the travel axis, then correct.
    let mut pts = Vec::with_capacity(((d / RESAMPLE_SPACING_PX) as usize).saturating_add(16));
    pts.push((x0, y0));

    let mut sx = x0;
    let mut sy = y0;
    if d > 400.0 && rng.next_f64() < 0.6 {
        let overshoot_dist = rng.uniform(8.0, 22.0);
        let ox = x1 + overshoot_dist * (dx / d);
        let oy = y1 + overshoot_dist * (dy / d);
        let leg1 = cubic_leg((sx, sy), (ox, oy), rng);
        pts.extend(leg1.into_iter().skip(1));
        sx = ox;
        sy = oy;
    }
    let leg_final = cubic_leg((sx, sy), (x1, y1), rng);
    pts.extend(leg_final.into_iter().skip(1)); // skip duplicated joint
    pts
}

/// One cubic Bezier leg with the matched control-point rule (Human/Fast):
/// both perpendicular offsets share one sign (a single consistent bow),
/// except ~10% of long moves which take opposite signs (S-curve).
/// Uniformly resampled like [`cubic_leg`].
pub fn cubic_leg_matched(p0: Pt, p3: Pt, rng: &mut XorShift64) -> Vec<Pt> {
    let dx = p3.0 - p0.0;
    let dy = p3.1 - p0.1;
    let d = dx.hypot(dy);
    if d < MIN_BEZIER_DIST_PX {
        return capped_linear_leg(p0, p3);
    }
    let dxu = dx / d;
    let dyu = dy / d;
    let ux = -dyu;
    let uy = dxu;

    let d1 = d * 0.01 * rng.uniform(4.0, 8.0);
    let d2 = d * 0.01 * rng.uniform(4.0, 8.0);
    let s1: f64 = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
    // s2 = s1 except ~10% of long moves.
    let s2: f64 = if d > 400.0 && rng.next_f64() < 0.10 {
        -s1
    } else {
        s1
    };
    let p1 = (
        p0.0 + 0.25 * dx + s1 * d1 * ux,
        p0.1 + 0.25 * dy + s1 * d1 * uy,
    );
    let p2 = (
        p0.0 + 0.75 * dx + s2 * d2 * ux,
        p0.1 + 0.75 * dy + s2 * d2 * uy,
    );
    let est_pts = ((d / RESAMPLE_SPACING_PX) as usize).saturating_add(4);
    let mut raw = Vec::with_capacity(est_pts);
    raw.push(p0);
    adaptive_points(p0, p1, p2, p3, 0, &mut raw);
    resample_arc_length(&raw, RESAMPLE_SPACING_PX)
}

/// Human/Fast trajectory builder: matched control points
/// ([`cubic_leg_matched`]) and an overshoot probability driven by the
/// Fitts ID (rises with ID; the caller passes it in). The overshoot leg is
/// still tested against the window rect by the caller with a direct-leg
/// fallback; this builder only shapes the path.
pub fn build_trajectory_fitts(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    overshoot_p: f64,
    rng: &mut XorShift64,
) -> Vec<Pt> {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let d = dx.hypot(dy);
    if d < MIN_BEZIER_DIST_PX {
        return capped_linear_leg((x0, y0), (x1, y1));
    }
    let mut pts = Vec::with_capacity(((d / RESAMPLE_SPACING_PX) as usize).saturating_add(16));
    pts.push((x0, y0));

    let mut sx = x0;
    let mut sy = y0;
    if d > 150.0 && rng.next_f64() < overshoot_p.clamp(0.0, 1.0) {
        let overshoot_dist = rng.uniform(8.0, 22.0);
        let ox = x1 + overshoot_dist * (dx / d);
        let oy = y1 + overshoot_dist * (dy / d);
        let leg1 = cubic_leg_matched((sx, sy), (ox, oy), rng);
        pts.extend(leg1.into_iter().skip(1));
        sx = ox;
        sy = oy;
    }
    let leg_final = cubic_leg_matched((sx, sy), (x1, y1), rng);
    pts.extend(leg_final.into_iter().skip(1)); // skip duplicated joint
    pts
}
