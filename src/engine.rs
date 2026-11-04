//! Engine: the integrated edge pipeline (mouse-future B7).
//!
//! This module wires the phase builds into one working pipeline behind
//! the runbook's [`StepDriver`] contract, for a single
//! (page, natural-language target, declared predicate) task:
//!
//! ```text
//! perceive  CDP/DOM snapshot -> versioned tree (perceive.rs)
//! resolve   resolver cascade -> target box or typed escalation (resolver.rs)
//! actuate   typed coords -> scheduler Legacy profile -> X11 click (scheduler.rs)
//! verify    vision.rs predicate verdict + damage-driven settle timing
//! journal   runbook.rs records the run (sqlite, no double-execute)
//! ```
//!
//! Composition notes (mechanical, no redesign of any module):
//!
//! - Perception, actuation, and capture are traits. The bridge's live
//!   wiring and the VM benchmark example provide them; unit tests use
//!   fakes. The engine itself never touches CDP or the network.
//! - The resolver feed mirrors [`perceive::feed_resolver`] exactly, with
//!   one addition: backend node ids recorded via [`StepDriver::note_failed`]
//!   are excluded on the retry feed, so a retry never re-resolves to a
//!   known-bad target.
//! - Actuation is the scheduler's [`SpeedProfile::Legacy`] branch
//!   (`plan_gesture` + `emit_direct`). Human/Fast stay provisional:
//!   [`check_profile_available`] refuses them unless the KS gate has
//!   marked them available.
//! - Verification sequencing: the runbook calls `predicate_now` before
//!   `dispatch` (the AlreadySatisfied gate) and `verify` after. The
//!   pre-action predicate value is captured in `dispatch` from the
//!   pre-action snapshot, so `verify` can build vision's [`Baseline`]
//!   faithfully: `predicate_before` is what the pre-action context said,
//!   and the region hash is the post-action anchor. The pixel-change
//!   signal is supporting evidence only; the declared predicate decides.
//! - Only [`Verdict::Confirmed`] commits to the identity cache.
//! - No tokio anywhere on this path. The actuation thread owns its X11
//!   connection; perception/capture implementations that need async
//!   (CDP) block on their own runtime outside this module.

use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

use x11rb::protocol::xproto::ConnectionExt as _;
use x11rb::protocol::xproto::Window;
use x11rb::rust_connection::RustConnection;

use crate::coords::{to_screen, FrameTransform, MainViewportCss, RootDevicePx};
use crate::legacy::WindowRect;
use crate::perceive::GenerationSnapshot;
use crate::predicates::{eval as eval_predicate, EvalCtx, Predicate, PredicateValue};
use crate::resolver::{
    normalize_desc, EngineConfig, Gates, Hit, MissReason, Resolution, ResolveOpts, ResolverEngine,
    Rng01,
};
use crate::runbook::{self, RunConfig, RunOutcome, SqliteJournalStore, Step, StepDriver};
use crate::scheduler::{
    emit_direct, plan_gesture, EmitReport, GestureKind, SpeedProfile, TargetBox,
};
use crate::telemetry::{RunTelemetry, Stage, Timer};
use crate::vision::{
    box_filter_9x8, check_against, dhash_9x8, Baseline, Capture, DamageProbe, OcrOutcome, Verdict,
    VerifierStats, VerifyEvidence, VerifyHooks,
};

// ---------------------------------------------------------------------------
// EngineRng: deterministic xorshift64 for the resolver cascade.
// ---------------------------------------------------------------------------

/// The resolver needs a `&mut dyn Rng01` for interior-point sampling.
/// The driver's XorShift64 lives in the input module and only implements
/// `Rng01` under `#[cfg(test)]`; this tiny local xorshift keeps the
/// engine self-contained with identical determinism under a fixed seed.
struct EngineRng(u64);

impl EngineRng {
    fn next(&mut self) -> u64 {
        let mut x = self.0 | 1;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

impl Rng01 for EngineRng {
    fn uniform01(&mut self) -> f64 {
        ((self.next() >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------
// Geometry: binds the typed coordinate spaces to one live window.
// ---------------------------------------------------------------------------

/// The typed conversion inputs for one benchmark/app window. Carried as a
/// struct so no bare tuple crosses the engine boundary.
#[derive(Debug, Clone, Copy)]
pub struct Geometry {
    /// Device pixel ratio of the page.
    pub dpr: f64,
    /// Page zoom factor (1.0 unless the page zooms).
    pub zoom: f64,
    /// X11 client-window content origin in root device px (typed).
    pub window_origin: RootDevicePx,
}

// ---------------------------------------------------------------------------
// Pluggable pipeline stages (traits).
// ---------------------------------------------------------------------------

/// One perception refresh: the immutable tree snapshot plus the
/// predicate-evaluation context built from it, plus the live window and
/// domain for the runbook's allowlist gate.
pub struct PerceptionData {
    pub snapshot: Arc<GenerationSnapshot>,
    pub eval_ctx: EvalCtx,
    pub window_title: String,
    pub domain: String,
    /// Full page text (document.body.innerText, lowercased by the caller)
    /// for HasText predicates against non-interactive content (headings,
    /// etc.). The DOM snapshot only carries interactive elements.
    pub page_text: String,
}

/// Synchronous perception source. The VM benchmark implements this over
/// CDP (blocking on its own runtime); unit tests use fixtures.
pub trait Perception {
    fn snapshot(&mut self) -> Result<PerceptionData, String>;
}

/// Report from one actuation.
pub struct ActuationReport {
    pub emit: EmitReport,
    pub from_px: (f64, f64),
    pub to_px: (f64, f64),
    pub profile: SpeedProfile,
}

/// Synchronous actuation backend: typed viewport point in, click done.
pub trait Actuator {
    fn click(
        &mut self,
        at: MainViewportCss,
        wh: (f64, f64),
        seed: u64,
    ) -> Result<ActuationReport, String>;
}

/// Synchronous capture backend: region BGRX captures for the verifier,
/// plus one redacted full-viewport shot for the escalation context.
pub trait Capturer {
    fn capture_region(&mut self, region: (u32, u32, u32, u32)) -> Result<Capture, String>;
    fn full_shot(&mut self) -> Option<Vec<u8>>;
}

// ---------------------------------------------------------------------------
// Profiles: Legacy is the default; Human/Fast stay provisional.
// ---------------------------------------------------------------------------

/// Refuse provisional profiles until the KS gate marks them available.
/// The engine default is [`SpeedProfile::Legacy`].
pub fn check_profile_available(p: SpeedProfile) -> Result<(), String> {
    if p.is_available() {
        Ok(())
    } else {
        Err(format!(
            "profile {p:?} is provisional (KS gate not passed); refusing"
        ))
    }
}

// ---------------------------------------------------------------------------
// X11Actuator: scheduler Legacy profile -> XTest click.
// ---------------------------------------------------------------------------

/// Real actuation: `plan_gesture(Legacy, Click)` + `emit_direct` against
/// the caller's X11 connection. Owns no threads and touches no async
/// runtime; the caller keeps the connection alive.
pub struct X11Actuator {
    conn: RustConnection,
    root: Window,
    rect: WindowRect,
    geom: Geometry,
    profile: SpeedProfile,
    seed: u64,
}

impl X11Actuator {
    pub fn new(conn: RustConnection, root: Window, rect: WindowRect, geom: Geometry) -> Self {
        Self {
            conn,
            root,
            rect,
            geom,
            profile: SpeedProfile::Legacy,
            seed: 1,
        }
    }

    /// Switch profile. Provisional profiles are refused (see
    /// [`check_profile_available`]).
    pub fn set_profile(&mut self, p: SpeedProfile) -> Result<(), String> {
        check_profile_available(p)?;
        self.profile = p;
        Ok(())
    }

    pub fn profile(&self) -> SpeedProfile {
        self.profile
    }

    fn pointer_now(&self) -> Result<(f64, f64), String> {
        let r = self
            .conn
            .query_pointer(self.root)
            .map_err(|e| format!("query_pointer: {e}"))?
            .reply()
            .map_err(|e| format!("query_pointer reply: {e}"))?;
        Ok((f64::from(r.root_x), f64::from(r.root_y)))
    }
}

impl Actuator for X11Actuator {
    fn click(
        &mut self,
        at: MainViewportCss,
        wh: (f64, f64),
        _seed: u64,
    ) -> Result<ActuationReport, String> {
        self.seed = self.seed.wrapping_add(1);
        let seed = self.seed;
        // Typed conversion: the ONE viewport-to-device path.
        let to = to_screen(at, self.geom.zoom, self.geom.dpr, self.geom.window_origin);
        let from = self.pointer_now()?;
        let plan = plan_gesture(
            self.profile,
            GestureKind::Click,
            from,
            (to.0, to.1),
            TargetBox { w: wh.0, h: wh.1 },
            &self.rect,
            seed,
            16.7,
        )
        .map_err(|e| format!("plan_gesture: {e}"))?;
        // Direct emission: the KS-harness path. Production actuation goes
        // through the scheduler's InputThread; the benchmark example
        // drives emit_direct on its own thread, never on a tokio worker.
        let emit = emit_direct(&self.conn, self.root, &self.rect, &plan)
            .map_err(|e| format!("emit_direct: {e}"))?;
        Ok(ActuationReport {
            emit,
            from_px: from,
            to_px: (to.0, to.1),
            profile: self.profile,
        })
    }
}

// ---------------------------------------------------------------------------
// EngineDriver: StepDriver over perceive/resolve/actuate/verify.
// ---------------------------------------------------------------------------

/// What the last dispatch left behind: the winning hit, the pre-action
/// perception, the pre-action predicate value, and the text shortlist.
struct LastStep {
    hit: Hit,
    data: PerceptionData,
    pred_before: PredicateValue,
    shortlist: Vec<String>,
    /// Device-px click point and target box, for the benchmark's
    /// screenshot overlay (honest miss review).
    click_px: (f64, f64),
    target_box_px: (f64, f64, f64, f64),
}

/// Per-step wall-clock timings the benchmark report needs.
#[derive(Debug, Clone, Default)]
pub struct StepTimings {
    /// perceive + resolve + actuate wall time (describe-to-click).
    pub acquire_ms: f64,
    /// verify() wall time.
    pub verify_ms: f64,
    /// Damage-quiet detection window, when the settle probe ran.
    pub settle_ms: Option<f64>,
}

/// The integrated driver. Generic over the perception, actuation, and
/// capture backends so the same pipeline runs on the VM and in tests.
pub struct EngineDriver<P, A, C> {
    perception: P,
    actuator: A,
    capturer: C,
    settle: Box<dyn DamageProbe>,
    resolver: ResolverEngine,
    rng: EngineRng,
    geom: Geometry,
    origin: String,
    profile_id: String,
    route: String,
    viewport: (f64, f64),
    last: Option<LastStep>,
    /// Live window title + domain, refreshed on every perception read
    /// (used by the runbook's allowlist gate even before any dispatch).
    context: Option<(String, String)>,
    /// Set on the first dispatch; distinguishes "completed with a click"
    /// from "completed as already-satisfied".
    dispatched: bool,
    failed_ids: Vec<u64>,
    timings: StepTimings,
    telemetry: RunTelemetry,
    stats: VerifierStats,
}

impl<P: Perception, A: Actuator, C: Capturer> EngineDriver<P, A, C> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        perception: P,
        actuator: A,
        capturer: C,
        settle: Box<dyn DamageProbe>,
        geom: Geometry,
        origin: String,
        profile_id: String,
        route: String,
        viewport: (f64, f64),
        gates: Gates,
    ) -> Self {
        let resolver = ResolverEngine::new(
            EngineConfig {
                profile_id: profile_id.clone(),
                origin: origin.clone(),
                viewport,
                cache_cap: 512,
            },
            0,
            gates,
        );
        Self {
            perception,
            actuator,
            capturer,
            settle,
            resolver,
            rng: EngineRng(0xB7E),
            geom,
            origin,
            profile_id,
            route,
            viewport,
            last: None,
            context: None,
            dispatched: false,
            failed_ids: Vec::new(),
            timings: StepTimings::default(),
            telemetry: RunTelemetry::new(),
            stats: VerifierStats::default(),
        }
    }

    /// Feed one snapshot into the resolver, excluding failed ids.
    /// Mirrors [`crate::perceive::feed_resolver`]; the exclusion is
    /// the retry contract from [`StepDriver::note_failed`].
    fn feed(&mut self, snap: &GenerationSnapshot) {
        let t = Timer::start();
        self.resolver.begin_generation(snap.generation);
        // Identity chain: coordinates are already main-viewport CSS px,
        // normalized once at tree-insertion time (same contract as
        // feed_resolver).
        let chain = [FrameTransform {
            scroll: (0.0, 0.0),
            content_origin: (0.0, 0.0),
            nested_offset: (0.0, 0.0),
            css_scale: 1.0,
            dpr: 1.0,
        }];
        for n in &snap.nodes {
            if self.failed_ids.contains(&n.backend_node_id) {
                continue;
            }
            self.resolver.insert(
                n.backend_node_id,
                &n.role,
                &n.name,
                &n.text,
                n.at,
                n.wh,
                n.enabled && n.visible,
                &chain,
            );
        }
        self.resolver.finalize();
        self.telemetry.record(Stage::Normalize, t.elapsed_nanos());
    }

    /// Evaluate the predicate against a perception snapshot, target-aware.
    fn eval_in(
        &self,
        data: &PerceptionData,
        pred: &Predicate,
        target: Option<u64>,
    ) -> PredicateValue {
        let mut ctx = data.eval_ctx.clone();
        ctx.target_node_id = target;
        // HasText also matches against the full page text (the DOM snapshot
        // only carries interactive elements; headings and other static text
        // live in page_text).
        if let Predicate::HasText(want) = pred {
            if data.page_text.contains(want) {
                return PredicateValue::Bool(true);
            }
        }
        eval_predicate(pred, &ctx)
    }

    /// Region for the verifier: the hit's node box in device px.
    fn verify_region(&self, hit: &Hit) -> (u32, u32, u32, u32) {
        let scale = self.geom.zoom * self.geom.dpr;
        let scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
        // Node box top-left from the pre-action snapshot when available;
        // otherwise the interior point minus half extents.
        let (bx, by) = match &self.last {
            Some(l) => l
                .data
                .snapshot
                .by_backend
                .get(&hit.backend_node_id)
                .map(|&i| l.data.snapshot.nodes[i].at)
                .unwrap_or((hit.at.0 - hit.w / 2.0, hit.at.1 - hit.h / 2.0)),
            None => (hit.at.0 - hit.w / 2.0, hit.at.1 - hit.h / 2.0),
        };
        let tl = to_screen(
            MainViewportCss(bx, by),
            self.geom.zoom,
            self.geom.dpr,
            self.geom.window_origin,
        );
        let w = ((hit.w * scale).round() as u32).max(1);
        let h = ((hit.h * scale).round() as u32).max(1);
        (tl.0.max(0.0) as u32, tl.1.max(0.0) as u32, w, h)
    }

    fn shortlist_text(&mut self, desc: &str) -> Vec<String> {
        match self.resolver.natural_top_k(desc, 8) {
            Ok(top) => top
                .iter()
                .map(|(i, score, _)| {
                    let c = &self.resolver.candidates()[*i];
                    format!("{} {:?} score={:.2}", c.backend_node_id, c.role, score)
                })
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Run one (describe, predicate) task through the full pipeline with
    /// journaling. One step per call; the journal path persists the run.
    pub fn run_single(&mut self, spec: &TaskSpec, journal_path: &str) -> TaskReport {
        let book = [Step {
            describe: spec.describe.clone(),
            predicate: spec.predicate.clone(),
            commit: false,
            idempotent: spec.idempotent,
            window: spec.window.clone(),
            domain: spec.domain.clone(),
        }];
        let cfg = RunConfig {
            profile_id: self.profile_id.clone(),
            window_allowlist: vec![spec.window.clone()],
            domain_allowlist: vec![spec.domain.clone()],
            max_steps: 4,
        };
        let mut store = SqliteJournalStore::open(journal_path).expect("journal store must open");
        let (_state, outcome) = runbook::run(&book, &cfg, self, &mut store);
        let already_satisfied = matches!(outcome, RunOutcome::Complete) && !self.dispatched;
        TaskReport {
            outcome,
            stage_nanos: stage_vec(&self.telemetry),
            retries: self.telemetry.retries(),
            cache_hits: self.telemetry.cache_hits(),
            escalations: self.telemetry.escalations(),
            stats: stats_summary(&self.stats),
            timings: self.timings.clone(),
            resolve_stage: self.last.as_ref().map(|l| l.hit.stage.to_string()),
            confidence: self.last.as_ref().map(|l| l.hit.confidence),
            already_satisfied,
            click_px: self.last.as_ref().map(|l| l.click_px),
            target_box_px: self.last.as_ref().map(|l| l.target_box_px),
            journal_path: journal_path.to_string(),
        }
    }

    /// Last dispatch's wall-clock breakdown (benchmark use).
    pub fn timings(&self) -> &StepTimings {
        &self.timings
    }

    /// Post-action snapshot with navigation-aware stabilization.
    ///
    /// After a click that triggers a navigation, the DOM is mid-swap: an
    /// immediate snapshot captures the old document (or a blank/partial new
    /// one), and the predicate is then evaluated against the wrong page,
    /// yielding false Unchanged/Inconclusive verdicts. This polls the
    /// perception source until two consecutive snapshots agree (same node
    /// count and same text hash), up to ~4s, so the verify step reads the
    /// settled destination document.
    fn stable_snapshot(&mut self) -> Result<PerceptionData, String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::thread;
        use std::time::Duration;

        fn fingerprint(data: &PerceptionData) -> (usize, u64) {
            let mut h = DefaultHasher::new();
            let n = data.snapshot.nodes.len();
            for node in &data.snapshot.nodes {
                node.name.hash(&mut h);
                node.text.hash(&mut h);
            }
            (n, h.finish())
        }

        let mut prev = self.perception.snapshot()?;
        let mut prev_fp = fingerprint(&prev);
        for _ in 0..8 {
            thread::sleep(Duration::from_millis(500));
            let cur = self.perception.snapshot()?;
            let cur_fp = fingerprint(&cur);
            if cur_fp == prev_fp {
                return Ok(cur);
            }
            prev = cur;
            prev_fp = cur_fp;
        }
        Ok(prev)
    }
}

fn stage_vec(t: &RunTelemetry) -> Vec<(String, u64)> {
    use Stage::*;
    let names = [
        (Perceive, "perceive"),
        (Normalize, "normalize"),
        (Cache, "cache"),
        (Bm25, "bm25"),
        (Ranker, "ranker"),
        (Convert, "convert"),
        (Move, "move"),
        (Verify, "verify"),
        (Ocr, "ocr"),
        (Escalate, "escalate"),
    ];
    names
        .iter()
        .map(|(s, n)| (n.to_string(), t.stage_nanos(*s)))
        .collect()
}

/// One (describe, predicate) task.
#[derive(Debug, Clone)]
pub struct TaskSpec {
    pub describe: String,
    pub predicate: Predicate,
    pub window: String,
    pub domain: String,
    pub idempotent: bool,
}

/// Owned report for one task run (JSON-serializable by the caller).
/// `RunOutcome` is Debug-only, so this report is Debug-only too.
#[derive(Debug)]
pub struct TaskReport {
    pub outcome: RunOutcome,
    pub stage_nanos: Vec<(String, u64)>,
    pub retries: u64,
    pub cache_hits: u64,
    pub escalations: u64,
    pub stats: StatsSummary,
    pub timings: StepTimings,
    pub resolve_stage: Option<String>,
    pub confidence: Option<f64>,
    /// True when the run completed without any dispatch (the predicate
    /// was already satisfied on the pre-read; no click happened).
    pub already_satisfied: bool,
    /// Device-px click point and resolved target box (screenshot overlay).
    pub click_px: Option<(f64, f64)>,
    pub target_box_px: Option<(f64, f64, f64, f64)>,
    pub journal_path: String,
}

#[derive(Debug, Clone, Default)]
pub struct StatsSummary {
    pub xdamage_settles: u64,
    pub fixed_sleep_fallbacks: u64,
    pub confirmed: u64,
    pub unchanged: u64,
    pub inconclusive: u64,
    pub already_satisfied: u64,
}

fn stats_summary(s: &VerifierStats) -> StatsSummary {
    StatsSummary {
        xdamage_settles: s.xdamage_settles,
        fixed_sleep_fallbacks: s.fixed_sleep_fallbacks,
        confirmed: s.confirmed,
        unchanged: s.unchanged,
        inconclusive: s.inconclusive,
        already_satisfied: s.already_satisfied,
    }
}

// ---------------------------------------------------------------------------
// StepDriver implementation: the pipeline behind the runbook.
// ---------------------------------------------------------------------------

impl<P: Perception, A: Actuator, C: Capturer> StepDriver for EngineDriver<P, A, C> {
    fn current_context(&self) -> (String, String) {
        match &self.context {
            Some((w, d)) => (w.clone(), d.clone()),
            None => (String::new(), String::new()),
        }
    }

    fn predicate_now(&mut self, pred: &Predicate) -> PredicateValue {
        let data = match self.perception.snapshot() {
            Ok(d) => d,
            Err(_) => return PredicateValue::Unknown,
        };
        let target = self.last.as_ref().map(|l| l.hit.backend_node_id);
        let v = self.eval_in(&data, pred, target);
        // Refresh the live context even on a pure read, without touching
        // the last dispatch record.
        self.context = Some((data.window_title.clone(), data.domain.clone()));
        v
    }

    fn dispatch(&mut self, step: &Step) -> Result<u64, String> {
        let t_all = Instant::now();

        // 1. Perceive.
        let t = Timer::start();
        let data = self
            .perception
            .snapshot()
            .map_err(|e| format!("perceive: {e}"))?;
        self.telemetry.record(Stage::Perceive, t.elapsed_nanos());
        self.feed(&data.snapshot);

        // Pre-action predicate value, for the verify baseline.
        let pred_before = self.eval_in(&data, &step.predicate, None);

        // 2. Resolve.
        let t = Timer::start();
        let opts = ResolveOpts {
            route: self.route.clone(),
            bypass_cache: false,
        };
        let resolution = self.resolver.resolve(&step.describe, &opts, &mut self.rng);
        let resolve_nanos = t.elapsed_nanos();
        let shortlist = self.shortlist_text(&step.describe);
        let hit = match resolution {
            Resolution::Hit(h) => {
                let stage = match h.stage {
                    crate::resolver::Stage::Cache => Stage::Cache,
                    crate::resolver::Stage::Exact => Stage::Bm25,
                    crate::resolver::Stage::Bm25 => Stage::Bm25,
                    crate::resolver::Stage::Ranker => Stage::Ranker,
                };
                self.telemetry.record(stage, resolve_nanos);
                if matches!(h.stage, crate::resolver::Stage::Cache) {
                    self.telemetry.inc_cache_hits();
                }
                h
            }
            Resolution::Miss { reason, .. } => {
                self.telemetry.record(Stage::Escalate, resolve_nanos);
                self.telemetry.inc_escalations();
                return Err(match reason {
                    MissReason::PerceptionEmpty => {
                        "resolution miss: PerceptionEmpty (no candidates)".to_string()
                    }
                    MissReason::NoSearchableTerms => {
                        format!("resolution miss: NoSearchableTerms ({:?})", step.describe)
                    }
                    MissReason::RankingUncertain => {
                        format!("resolution miss: RankingUncertain; top-k: {shortlist:?}")
                    }
                });
            }
        };
        if hit.needs_refinement {
            return Err(format!(
                "target {} under 12x12 physical px: crop-and-zoom refinement not wired in B7",
                hit.backend_node_id
            ));
        }

        // 3. Actuate.
        let t = Timer::start();
        let rep = self
            .actuator
            .click(hit.at, (hit.w, hit.h), 0)
            .map_err(|e| format!("actuate: {e}"))?;
        self.telemetry.record(Stage::Move, t.elapsed_nanos());

        // Device-px target box for the benchmark overlay, from the
        // pre-action snapshot (same snapshot the hit came from).
        let target_box_px = data
            .snapshot
            .by_backend
            .get(&hit.backend_node_id)
            .map(|&i| {
                let n = &data.snapshot.nodes[i];
                let tl = to_screen(
                    MainViewportCss(n.at.0, n.at.1),
                    self.geom.zoom,
                    self.geom.dpr,
                    self.geom.window_origin,
                );
                let br = to_screen(
                    MainViewportCss(n.at.0 + n.wh.0, n.at.1 + n.wh.1),
                    self.geom.zoom,
                    self.geom.dpr,
                    self.geom.window_origin,
                );
                (tl.0, tl.1, (br.0 - tl.0).max(1.0), (br.1 - tl.1).max(1.0))
            })
            .unwrap_or((rep.to_px.0, rep.to_px.1, 1.0, 1.0));

        self.timings.acquire_ms = t_all.elapsed().as_secs_f64() * 1000.0;
        self.timings.settle_ms = None;
        self.dispatched = true;
        self.last = Some(LastStep {
            hit: hit.clone(),
            data,
            pred_before,
            shortlist,
            click_px: rep.to_px,
            target_box_px,
        });
        Ok(hit.backend_node_id)
    }

    fn verify(&mut self, step: &Step) -> Verdict {
        let t_all = Instant::now();
        let (hit_id, pred_before, region) = match &self.last {
            Some(l) => (
                l.hit.backend_node_id,
                l.pred_before.clone(),
                self.verify_region(&l.hit),
            ),
            None => return Verdict::Inconclusive,
        };

        // Fresh post-action perception for the predicate re-read, stabilized
        // across a potential navigation (see stable_snapshot).
        let data = match self.stable_snapshot() {
            Ok(d) => d,
            Err(_) => return Verdict::Inconclusive,
        };
        let mut ctx = data.eval_ctx.clone();
        ctx.target_node_id = Some(hit_id);

        // Post-action region anchor (the pre-action predicate state is
        // already established by the runner's predicate_now gate).
        let region_hash = match self.capturer.capture_region(region).and_then(|c| {
            box_filter_9x8(&c)
                .map(|cells| dhash_9x8(&cells))
                .map_err(|e| format!("dhash: {e}"))
        }) {
            Ok(h) => h,
            Err(_) => return Verdict::Inconclusive,
        };
        let baseline = Baseline {
            predicate_before: pred_before,
            region,
            region_hash,
        };

        let ev: VerifyEvidence = {
            // RefCell: the hooks want `&dyn Fn`, but capture mutates the
            // backend. Interior mutability keeps the closure `Fn`.
            let capturer = RefCell::new(&mut self.capturer);
            let capture = |r: (u32, u32, u32, u32)| {
                capturer
                    .borrow_mut()
                    .capture_region(r)
                    .map_err(|e| anyhow::anyhow!(e))
            };
            let settle: &mut dyn DamageProbe = &mut *self.settle;
            let telemetry = &mut self.telemetry;
            let stats = &mut self.stats;
            let hooks = VerifyHooks {
                eval: &|p: &Predicate| {
                    // HasText also matches the full page text (see eval_in).
                    if let Predicate::HasText(want) = p {
                        if data.page_text.contains(want) {
                            return PredicateValue::Bool(true);
                        }
                    }
                    eval_predicate(p, &ctx)
                },
                capture: &capture,
                settle,
                ocr: &|_| OcrOutcome::Busy,
                telemetry,
                stats,
            };
            check_against(&baseline, &step.predicate, hooks)
        };

        // Only Confirmed may commit to the identity cache.
        if matches!(ev.verdict, Verdict::Confirmed) {
            let desc_norm = normalize_desc(&step.describe);
            let th = self
                .resolver
                .candidates()
                .iter()
                .find(|c| c.backend_node_id == hit_id)
                .map(|c| c.prep.text_hash);
            if let Some(th) = th {
                let profile_id = self.profile_id.clone();
                let origin = self.origin.clone();
                let route = self.route.clone();
                self.resolver
                    .commit_cache(&profile_id, &origin, &route, &desc_norm, hit_id, th);
            }
        }

        self.timings.verify_ms = t_all.elapsed().as_secs_f64() * 1000.0;
        // The settle window is timed by the caller's probe wrapper when
        // one is installed; without it the verify wall time stands in.
        ev.verdict
    }

    fn resnapshot(&mut self) {
        // Dispatch always re-snapshots; nothing cached to invalidate here.
        // Failed ids persist so the retry feed excludes them.
    }

    fn note_failed(&mut self, id: u64) {
        if !self.failed_ids.contains(&id) {
            self.failed_ids.push(id);
        }
    }

    fn escalation_shot(&mut self) -> Option<Vec<u8>> {
        self.capturer.full_shot()
    }

    fn shortlist(&self) -> Vec<String> {
        self.last
            .as_ref()
            .map(|l| l.shortlist.clone())
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Unit tests: the full pipeline against fakes (no X11, no CDP).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perceive::{TreeNode, VersionedTree};
    use crate::predicates::AxNode;
    use crate::runbook::EscalateReason;
    use crate::vision::{ScriptedProbe, SettlePath};
    use std::collections::VecDeque;

    const VIEWPORT: (f64, f64) = (1280.0, 800.0);

    fn tnode(id: u64, role: &str, name: &str, x: f64, focused: bool) -> (TreeNode, AxNode) {
        let t = TreeNode::new(
            id,
            1,
            role,
            name,
            "",
            "main",
            (x, 20.0),
            (80.0, 30.0),
            1.0,
            true,
            true,
            false,
        );
        let a = AxNode {
            node_id: id,
            role: role.to_string(),
            name: name.to_string(),
            text: String::new(),
            value: String::new(),
            enabled: true,
            visible: true,
            focused,
            is_password: false,
            box_px: (x as u32, 20, 80, 30),
            attributes: Vec::new(),
        };
        (t, a)
    }

    /// Scripted perception: each snapshot() pops the next focused flag.
    struct FakePerception {
        script: VecDeque<bool>,
        nodes: Vec<(u64, &'static str, &'static str, f64)>,
    }

    impl FakePerception {
        fn new(script: Vec<bool>) -> Self {
            Self {
                script: script.into(),
                nodes: vec![
                    (1000, "button", "Search jobs", 10.0),
                    (1001, "button", "Search tenders", 100.0),
                    (1002, "link", "Home", 190.0),
                ],
            }
        }
    }

    impl Perception for FakePerception {
        fn snapshot(&mut self) -> Result<PerceptionData, String> {
            let focused = self.script.pop_front().unwrap_or(false);
            let mut tree = VersionedTree::new(VIEWPORT);
            let tnodes: Vec<TreeNode> = self
                .nodes
                .iter()
                .map(|(id, role, name, x)| tnode(*id, role, name, *x, focused).0)
                .collect();
            tree.rebuild_from_snapshot(tnodes, vec!["main".to_string()]);
            let anxodes: Vec<AxNode> = self
                .nodes
                .iter()
                .map(|(id, role, name, x)| tnode(*id, role, name, *x, focused).1)
                .collect();
            Ok(PerceptionData {
                snapshot: tree.snapshot(),
                eval_ctx: EvalCtx {
                    nodes: anxodes,
                    url: "https://example.com/".to_string(),
                    target_node_id: None,
                },
                window_title: "Google Chrome".to_string(),
                domain: "example.com".to_string(),
                page_text: String::new(),
            })
        }
    }

    struct FakeActuator {
        pub clicks: Vec<(MainViewportCss, (f64, f64))>,
    }

    impl Actuator for FakeActuator {
        fn click(
            &mut self,
            at: MainViewportCss,
            wh: (f64, f64),
            _seed: u64,
        ) -> Result<ActuationReport, String> {
            self.clicks.push((at, wh));
            Ok(ActuationReport {
                emit: EmitReport {
                    emitted: 12,
                    skipped: 0,
                    duration_ms: 210.0,
                },
                from_px: (0.0, 0.0),
                to_px: (at.0, at.1),
                profile: SpeedProfile::Legacy,
            })
        }
    }

    struct FakeCapturer;

    impl Capturer for FakeCapturer {
        fn capture_region(&mut self, _region: (u32, u32, u32, u32)) -> Result<Capture, String> {
            // Solid gray 16x16: dhash is 0 and stable across calls.
            Ok(Capture {
                buf: vec![128u8; 16 * 16 * 4],
                stride: 64,
                w: 16,
                h: 16,
            })
        }
        fn full_shot(&mut self) -> Option<Vec<u8>> {
            None
        }
    }

    fn geom() -> Geometry {
        Geometry {
            dpr: 1.0,
            zoom: 1.0,
            window_origin: RootDevicePx(0.0, 0.0),
        }
    }

    fn driver(script: Vec<bool>) -> EngineDriver<FakePerception, FakeActuator, FakeCapturer> {
        EngineDriver::new(
            FakePerception::new(script),
            FakeActuator { clicks: Vec::new() },
            FakeCapturer,
            Box::new(ScriptedProbe::new(SettlePath::FixedSleep, vec![0.0])),
            geom(),
            "example.com".to_string(),
            "test".to_string(),
            "/".to_string(),
            VIEWPORT,
            Gates::default(),
        )
    }

    fn spec(describe: &str, predicate: Predicate) -> TaskSpec {
        TaskSpec {
            describe: describe.to_string(),
            predicate,
            window: "Google Chrome".to_string(),
            domain: "example.com".to_string(),
            idempotent: true,
        }
    }

    #[test]
    fn full_pipeline_click_confirms() {
        // Pre-action: nothing focused. Post-action: the target is focused.
        let mut d = driver(vec![false, false, true, true]);
        let rep = d.run_single(
            &spec("Search jobs", Predicate::Focused),
            &format!("/tmp/b7-engine-test-1-{}.sqlite", std::process::id()),
        );
        assert!(
            matches!(rep.outcome, RunOutcome::Complete),
            "outcome: {:?}",
            rep.outcome
        );
        assert_eq!(rep.stats.confirmed, 1);
        // "Search jobs" is a unique exact name: stage 0 fires, correctly.
        assert_eq!(rep.resolve_stage.as_deref(), Some("exact"));
        assert!(!rep.already_satisfied);
        assert!(rep.timings.acquire_ms > 0.0);
        assert!(rep.confidence.unwrap_or(0.0) > 0.0);
        std::fs::remove_file(format!(
            "/tmp/b7-engine-test-1-{}.sqlite",
            std::process::id()
        ))
        .ok();
    }

    #[test]
    fn already_satisfied_never_clicks() {
        // Predicate true on the very first read: no dispatch, no click.
        let mut d = driver(vec![true]);
        let rep = d.run_single(
            &spec("Search jobs", Predicate::HasText("Search jobs".to_string())),
            &format!("/tmp/b7-engine-test-2-{}.sqlite", std::process::id()),
        );
        assert!(
            matches!(rep.outcome, RunOutcome::Complete),
            "outcome: {:?}",
            rep.outcome
        );
        // The AlreadySatisfied path returns Done from the pre-read without
        // verification: no click, no check_against. The report flag is the
        // honest signal, not the verifier stats.
        assert!(rep.already_satisfied);
        assert_eq!(rep.stats.already_satisfied, 0);
        std::fs::remove_file(format!(
            "/tmp/b7-engine-test-2-{}.sqlite",
            std::process::id()
        ))
        .ok();
    }

    #[test]
    fn resolve_miss_escalates_typed() {
        let mut d = driver(vec![false, false]);
        let rep = d.run_single(
            &spec("blargh nonexistent widget zzz", Predicate::Focused),
            &format!("/tmp/b7-engine-test-3-{}.sqlite", std::process::id()),
        );
        match &rep.outcome {
            RunOutcome::Escalated(e) => {
                assert_eq!(e.reason, EscalateReason::DispatchRejected);
                assert!(
                    e.context.evidence.contains("RankingUncertain")
                        || e.context.evidence.contains("NoSearchableTerms"),
                    "evidence: {}",
                    e.context.evidence
                );
            }
            other => panic!("expected Escalated, got {other:?}"),
        }
        std::fs::remove_file(format!(
            "/tmp/b7-engine-test-3-{}.sqlite",
            std::process::id()
        ))
        .ok();
    }

    #[test]
    fn note_failed_excludes_on_retry_feed() {
        let mut d = driver(vec![false, false, false, false]);
        // Prime the resolver through one dispatch cycle manually: feed
        // happens inside dispatch, so drive two dispatches via run_single
        // on two one-step books. Simpler: exercise feed+resolve directly.
        let data = d.perception.snapshot().expect("snapshot");
        d.feed(&data.snapshot);
        let opts = ResolveOpts {
            route: "/".to_string(),
            bypass_cache: true,
        };
        let first = match d.resolver.resolve("Search jobs", &opts, &mut d.rng) {
            Resolution::Hit(h) => h.backend_node_id,
            other => panic!("expected hit, got {other:?}"),
        };
        assert_eq!(first, 1000);
        d.note_failed(1000);
        let data2 = d.perception.snapshot().expect("snapshot");
        d.feed(&data2.snapshot);
        let second = match d.resolver.resolve("Search jobs", &opts, &mut d.rng) {
            Resolution::Hit(h) => h.backend_node_id,
            other => panic!("expected hit, got {other:?}"),
        };
        assert_eq!(second, 1001, "retry must exclude the failed id");
    }

    #[test]
    fn provisional_profiles_refused() {
        assert!(check_profile_available(SpeedProfile::Legacy).is_ok());
        assert!(check_profile_available(SpeedProfile::Human).is_err());
        assert!(check_profile_available(SpeedProfile::Fast).is_err());
    }
}
