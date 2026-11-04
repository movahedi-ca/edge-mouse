//! Durable runbook runner (mouse-future Phase 2, Layer 6).
//!
//! The escalation budget is computed ONCE and carried in [`ExecCtx`];
//! the journal is persisted BEFORE and AFTER every state-changing action;
//! restart reconciles "dispatched but unconfirmed" actions by re-reading
//! the predicate, never by blindly repeating. One retry with a fresh
//! snapshot, failed ids inserted before any escalate, never an auto-retry
//! on `commit: true`. Hard window/domain allowlists plus a per-run step
//! ceiling independent of the escalation budget. Bounded escalation
//! context: the last 3 screenshots (WebP, password-redacted by the
//! driver before compression), the text shortlist, failed ids, and
//! evidence; the full trace (this journal) persists locally only.
//!
//! Pure synchronous Rust. Journal writes are blocking by design: a
//! journaled state must be durable before the action it describes. A
//! failed journal write parks the run rather than proceeding on a lie.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use super::predicates::{Predicate, PredicateValue};
use super::vision::Verdict;

// ---------------------------------------------------------------------------
// Steps, config, budget.
// ---------------------------------------------------------------------------

/// One runbook step. `predicate` is the declared post-condition (B05):
/// the verifier confirms it, never "any change".
#[derive(Debug, Clone)]
pub struct Step {
    pub describe: String,
    pub predicate: Predicate,
    /// Checkpoint step: the runner pauses for review before dispatching.
    /// Never auto-retried.
    pub commit: bool,
    /// Safe to attempt twice (no submit/pay/send/delete side effects).
    pub idempotent: bool,
    /// Expected window title and domain: the lane this step may act in.
    /// Empty means undeclared; the live-context allowlist still applies.
    pub window: String,
    pub domain: String,
}

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub profile_id: String,
    /// Hard allowlists: the runner refuses to act outside them. A step
    /// naming any other window title or domain escalates immediately.
    pub window_allowlist: Vec<String>,
    pub domain_allowlist: Vec<String>,
    /// Hard iteration ceiling per run, independent of the escalation
    /// budget: a run that cannot finish in N steps is parked, not looped.
    pub max_steps: usize,
}

/// B06: execution context. The budget is computed once and threaded
/// through; no path recomputes it, and no path passes `usize::MAX`.
pub struct ExecCtx<'a> {
    pub budget: usize,
    pub cfg: &'a RunConfig,
    pub book_len: usize,
}

/// Overflow-safe ceiling division, then clamp to [2, 5].
///
/// The division comes first, so no intermediate can overflow: `n / 5`
/// never exceeds `n`. (The naive `(n + 4) / 5` overflows near
/// `usize::MAX` in debug and wraps in release; this form is identical in
/// both profiles.) The `% 5 != 0` form is the task's literal formula;
/// keep it verbatim for auditability.
/// Tested for n = 0, 1, 5, 6, 25, 10_000 and
/// `usize::MAX`, in debug and release.
#[allow(clippy::manual_is_multiple_of)]
pub fn max_escalations(n_steps: usize) -> usize {
    let ceil = n_steps / 5 + usize::from(n_steps % 5 != 0);
    ceil.clamp(2, 5)
}

// ---------------------------------------------------------------------------
// Book hashing: pure-Rust SHA-256, no new dependency.
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    #[allow(clippy::chunks_exact_to_as_chunks)]
    // verified against known answers; restyle risks drift
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, slot) in w.iter_mut().take(16).enumerate() {
            *slot = u32::from_be_bytes(chunk[4 * i..4 * i + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = String::with_capacity(64);
    for word in h {
        out.push_str(&format!("{word:08x}"));
    }
    out
}

/// Canonical book hash. `resume` refuses a journal whose book hash does
/// not match the book it is asked to resume.
pub fn book_hash(book: &[Step]) -> String {
    let mut buf = Vec::new();
    for s in book {
        buf.extend_from_slice(s.describe.as_bytes());
        buf.push(0x1f);
        buf.extend_from_slice(format!("{:?}", s.predicate).as_bytes());
        buf.push(0x1f);
        buf.push(s.commit as u8);
        buf.push(s.idempotent as u8);
        buf.extend_from_slice(s.window.as_bytes());
        buf.push(0x1f);
        buf.extend_from_slice(s.domain.as_bytes());
        buf.push(0x1e);
    }
    sha256_hex(&buf)
}

// ---------------------------------------------------------------------------
// Durable journal.
// ---------------------------------------------------------------------------

/// H17: durable journal. Written before AND after each state-changing
/// action. On restart, an action journaled "dispatched" but not
/// "confirmed" is reconciled (re-read the predicate), never repeated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunJournal {
    pub run_id: String,
    pub book_hash: String,
    pub cursor: usize,
    /// "dispatched:step{i}:attempt{k}" before the action;
    /// "emit_confirmed:step{i}:attempt{k}" immediately after the X11 emit
    /// returns (before verify);
    /// "confirmed:step{i}" / "unconfirmed:step{i}:{verdict}" after;
    /// "escalated:..." / "parked:..." / "checkpoint:..." at stops.
    pub last_outcome: Option<String>,
    pub profile_id: String,
    pub origin: String,
    pub tree_generation: u64,
    /// Steps whose actions are confirmed done: never re-executed.
    pub committed_steps: Vec<usize>,
    pub checkpoints: Vec<usize>,
}

pub trait JournalStore {
    fn load(&self, book_hash: &str) -> Option<RunJournal>;
    /// Must be durable on success. The runner parks the run on failure
    /// rather than proceeding on a lie.
    fn save(&mut self, j: &RunJournal) -> Result<(), String>;
}

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS run_journal (
    book_hash TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    cursor INTEGER NOT NULL,
    last_outcome TEXT,
    profile_id TEXT NOT NULL,
    origin TEXT NOT NULL,
    tree_generation INTEGER NOT NULL,
    committed_steps TEXT NOT NULL,
    checkpoints TEXT NOT NULL,
    updated_at INTEGER NOT NULL
)";

/// SQLite-backed journal on the existing `rusqlite` dependency.
pub struct SqliteJournalStore {
    conn: Connection,
}

impl SqliteJournalStore {
    pub fn open(path: &str) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    pub fn open_in_memory() -> rusqlite::Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }
}

impl JournalStore for SqliteJournalStore {
    fn load(&self, book_hash: &str) -> Option<RunJournal> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT run_id, cursor, last_outcome, profile_id, origin,
                        tree_generation, committed_steps, checkpoints
                 FROM run_journal WHERE book_hash = ?1",
            )
            .ok()?;
        stmt.query_row([book_hash], |row| {
            let committed: String = row.get(6)?;
            let checkpoints: String = row.get(7)?;
            Ok(RunJournal {
                book_hash: book_hash.to_string(),
                run_id: row.get(0)?,
                cursor: row.get::<_, i64>(1)? as usize,
                last_outcome: row.get(2)?,
                profile_id: row.get(3)?,
                origin: row.get(4)?,
                tree_generation: row.get::<_, i64>(5)? as u64,
                committed_steps: serde_json::from_str(&committed).unwrap_or_default(),
                checkpoints: serde_json::from_str(&checkpoints).unwrap_or_default(),
            })
        })
        .ok()
    }

    fn save(&mut self, j: &RunJournal) -> Result<(), String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        self.conn
            .execute(
                "INSERT INTO run_journal
                     (book_hash, run_id, cursor, last_outcome, profile_id, origin,
                      tree_generation, committed_steps, checkpoints, updated_at)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)
                 ON CONFLICT(book_hash) DO UPDATE SET
                     run_id=excluded.run_id, cursor=excluded.cursor,
                     last_outcome=excluded.last_outcome, profile_id=excluded.profile_id,
                     origin=excluded.origin, tree_generation=excluded.tree_generation,
                     committed_steps=excluded.committed_steps,
                     checkpoints=excluded.checkpoints, updated_at=excluded.updated_at",
                params![
                    j.book_hash,
                    j.run_id,
                    j.cursor as i64,
                    j.last_outcome,
                    j.profile_id,
                    j.origin,
                    j.tree_generation as i64,
                    serde_json::to_string(&j.committed_steps).unwrap_or_else(|_| "[]".to_string()),
                    serde_json::to_string(&j.checkpoints).unwrap_or_else(|_| "[]".to_string()),
                    now,
                ],
            )
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Driver seam, outcomes, state.
// ---------------------------------------------------------------------------

/// The runner's seam to the act/perceive layers. The real implementation
/// is owned by later phases; every method is synchronous and the runner
/// never touches tokio.
pub trait StepDriver {
    /// Current window title and domain, for the allowlist gate. Empty
    /// strings skip the live-context check (perception not ready yet).
    fn current_context(&self) -> (String, String);
    /// Re-read the predicate without acting. Used for the
    /// AlreadySatisfied check and for restart reconciliation.
    fn predicate_now(&mut self, pred: &Predicate) -> PredicateValue;
    /// Perform the state-changing action. `Ok` carries a target id used
    /// for retry bookkeeping (0 when the step has no single target).
    fn dispatch(&mut self, step: &Step) -> Result<u64, String>;
    /// Verify the outcome after dispatch (predicate-based; see vision).
    /// The runner treats ONLY `Verdict::Confirmed` as a landed action
    /// eligible for cache commit; `AlreadySatisfied` marks the step done
    /// without any action; `Unchanged` may trigger the single retry;
    /// `Inconclusive` escalates and never clicks.
    fn verify(&mut self, step: &Step) -> Verdict;
    /// Fresh perception snapshot before a retry.
    fn resnapshot(&mut self);
    /// Record a failed target id; the retry must exclude it. Inserted
    /// BEFORE any escalate carrying failed ids.
    fn note_failed(&mut self, id: u64);
    /// One redacted WebP screenshot for the escalation context ring. The
    /// driver MUST zero password fields before compression.
    fn escalation_shot(&mut self) -> Option<Vec<u8>>;
    /// Text shortlist for the escalation context.
    fn shortlist(&self) -> Vec<String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalateReason {
    LowConfidence,
    VerifyFailed,
    PerceptionEmpty,
    Inconclusive,
    AlreadySatisfied,
    AllowlistViolation,
    DispatchRejected,
}

/// Bounded escalation context. Tier-1 sees the last 3 screenshots (WebP,
/// password-redacted by the driver before compression), the text
/// shortlist, failed ids, and evidence; the full trace (the journal)
/// persists locally only.
#[derive(Debug, Clone)]
pub struct EscalationContext {
    pub shots: Vec<Vec<u8>>,
    pub shortlist: Vec<String>,
    pub failed_ids: Vec<u64>,
    pub evidence: String,
}

#[derive(Debug, Clone)]
pub struct Escalation {
    pub step: usize,
    pub reason: EscalateReason,
    pub describe: String,
    pub context: EscalationContext,
}

#[derive(Debug)]
pub enum RunOutcome {
    Complete,
    /// Paused at a `commit: true` step for review. `resume()` executes it.
    Checkpoint {
        step: usize,
    },
    Escalated(Escalation),
    Parked {
        reason: String,
    },
}

pub struct RunState {
    pub journal: RunJournal,
    pub tier1_calls: usize,
    pub strikes_on: Option<usize>,
    /// Sliding window of the last 3 escalation screenshots (WebP,
    /// password-redacted by the driver), most recent last. Carried in
    /// memory across `resume()` calls: escalations stop the run, so a
    /// per-`drive()` ring would never hold more than one shot. Starts
    /// empty after a process restart; the journal (the durable trace)
    /// stays lean and never carries image bytes.
    pub shot_ring: VecDeque<Vec<u8>>,
}

// ---------------------------------------------------------------------------
// Runner.
// ---------------------------------------------------------------------------

pub fn run(
    book: &[Step],
    cfg: &RunConfig,
    driver: &mut dyn StepDriver,
    store: &mut dyn JournalStore,
) -> (RunState, RunOutcome) {
    let hash = book_hash(book);
    // H17: resume a persisted journal for the same book and profile, if any.
    let mut state = match store.load(&hash) {
        Some(j) if j.profile_id == cfg.profile_id => RunState {
            journal: j,
            tier1_calls: 0,
            strikes_on: None,
            shot_ring: VecDeque::new(),
        },
        _ => RunState {
            journal: RunJournal {
                run_id: uuid::Uuid::new_v4().to_string(),
                book_hash: hash,
                cursor: 0,
                last_outcome: None,
                profile_id: cfg.profile_id.clone(),
                origin: String::new(),
                tree_generation: 0,
                committed_steps: Vec::new(),
                checkpoints: Vec::new(),
            },
            tier1_calls: 0,
            strikes_on: None,
            shot_ring: VecDeque::new(),
        },
    };
    let outcome = drive(book, cfg, &mut state, driver, store);
    (state, outcome)
}

pub fn resume(
    book: &[Step],
    cfg: &RunConfig,
    state: &mut RunState,
    driver: &mut dyn StepDriver,
    store: &mut dyn JournalStore,
) -> RunOutcome {
    // Defensive: refuse to resume a different book against this journal.
    if state.journal.book_hash != book_hash(book) {
        return RunOutcome::Parked {
            reason: "book hash mismatch: refusing to resume a different runbook".to_string(),
        };
    }
    drive(book, cfg, state, driver, store)
}

/// Cap the escalation screenshot ring at 3, most recent last.
fn push_shot(ring: &mut VecDeque<Vec<u8>>, shot: Vec<u8>) {
    if ring.len() >= 3 {
        ring.pop_front();
    }
    ring.push_back(shot);
}

fn persist(state: &RunState, store: &mut dyn JournalStore) -> Result<(), String> {
    store
        .save(&state.journal)
        .map_err(|e| format!("journal write failed: {e}"))
}

fn allowlisted(value: &str, list: &[String]) -> bool {
    list.iter().any(|entry| value.contains(entry))
}

/// Hard allowlist gate. Returns the violation reason, if any. A step
/// naming a window title or domain outside the allowlists escalates
/// immediately instead of acting; so does a live context outside them.
fn allowlist_violation(
    ctx: &ExecCtx,
    step: &Step,
    live_window: &str,
    live_domain: &str,
) -> Option<String> {
    if !step.window.is_empty() && !allowlisted(&step.window, &ctx.cfg.window_allowlist) {
        return Some(format!("step window {:?} outside allowlist", step.window));
    }
    if !step.domain.is_empty() && !allowlisted(&step.domain, &ctx.cfg.domain_allowlist) {
        return Some(format!("step domain {:?} outside allowlist", step.domain));
    }
    if !live_window.is_empty() && !allowlisted(live_window, &ctx.cfg.window_allowlist) {
        return Some(format!("live window {live_window:?} outside allowlist"));
    }
    if !live_domain.is_empty() && !allowlisted(live_domain, &ctx.cfg.domain_allowlist) {
        return Some(format!("live domain {live_domain:?} outside allowlist"));
    }
    None
}

/// Journal marker written immediately after `driver.dispatch` returns Ok:
/// the X11 emit completed, verify has not run yet. Distinct from the
/// pre-emit "dispatched" marker, which cannot tell crash-before-emit from
/// crash-after-emit.
fn emit_confirmed_marker(last_outcome: &Option<String>, i: usize) -> bool {
    match last_outcome.as_deref() {
        Some(s) => s.starts_with(&format!("emit_confirmed:step{i}:")),
        None => false,
    }
}

/// Post-verify "unconfirmed" markers: the emit happened, the outcome was
/// never recorded.
fn unconfirmed_marker(last_outcome: &Option<String>, i: usize) -> bool {
    match last_outcome.as_deref() {
        Some(s) => s.starts_with(&format!("unconfirmed:step{i}:")),
        None => false,
    }
}

/// Pre-emit "dispatched" marker with NO following "emit_confirmed": the
/// journal went out before the X11 call, so the action may never have
/// happened. Safe to re-plan (execute_step re-checks AlreadySatisfied
/// before dispatching).
fn dispatched_unemitted_marker(last_outcome: &Option<String>, i: usize) -> bool {
    match last_outcome.as_deref() {
        // "emit_confirmed:..." does not start with "dispatched:", so the
        // prefix test alone distinguishes the two crash points.
        Some(s) => s.starts_with(&format!("dispatched:step{i}:")),
        None => false,
    }
}

enum StepResult {
    Done,
    Escalate {
        reason: EscalateReason,
        evidence: String,
        failed_ids: Vec<u64>,
    },
    Park {
        reason: String,
    },
}

fn mark_done(
    state: &mut RunState,
    store: &mut dyn JournalStore,
    i: usize,
    outcome: String,
) -> Result<(), String> {
    if !state.journal.committed_steps.contains(&i) {
        state.journal.committed_steps.push(i);
    }
    state.journal.cursor += 1;
    state.strikes_on = None;
    state.journal.last_outcome = Some(outcome);
    persist(state, store)
}

fn drive(
    book: &[Step],
    cfg: &RunConfig,
    state: &mut RunState,
    driver: &mut dyn StepDriver,
    store: &mut dyn JournalStore,
) -> RunOutcome {
    // B06: the budget is computed once here and carried in the ctx.
    let ctx = ExecCtx {
        budget: max_escalations(book.len()),
        cfg,
        book_len: book.len(),
    };
    let mut steps_this_run = 0usize;

    while state.journal.cursor < book.len() {
        // Hard iteration ceiling, independent of the escalation budget.
        if steps_this_run >= ctx.cfg.max_steps {
            let reason = format!("per-run step ceiling {} reached", ctx.cfg.max_steps);
            state.journal.last_outcome = Some(format!("parked:{reason}"));
            if let Err(e) = persist(state, store) {
                return RunOutcome::Parked { reason: e };
            }
            return RunOutcome::Parked { reason };
        }
        steps_this_run += 1;
        let i = state.journal.cursor;
        let step = &book[i];

        // H17: never re-execute confirmed work.
        if state.journal.committed_steps.contains(&i) {
            state.journal.cursor += 1;
            continue;
        }

        let (live_window, live_domain) = driver.current_context();
        if !live_domain.is_empty() {
            state.journal.origin = live_domain.clone();
        }

        if let Some(reason) = allowlist_violation(&ctx, step, &live_window, &live_domain) {
            return escalate(
                state,
                &ctx,
                store,
                driver,
                i,
                step,
                EscalateReason::AllowlistViolation,
                &reason,
                &[],
            );
        }

        // commit:true checkpoint: persist the pause, return for review.
        // resume() executes the step.
        if step.commit && !state.journal.checkpoints.contains(&i) {
            state.journal.checkpoints.push(i);
            state.journal.last_outcome = Some(format!("checkpoint:step{i}"));
            if let Err(e) = persist(state, store) {
                return RunOutcome::Parked { reason: e };
            }
            return RunOutcome::Checkpoint { step: i };
        }

        // Restart reconciliation.
        // - "dispatched" WITHOUT "emit_confirmed": the journal went out
        //   before the X11 emit, so the action may never have happened.
        //   Safe to re-plan: fall through to execute_step, which re-checks
        //   AlreadySatisfied before dispatching. No predicate re-read is
        //   needed (there is nothing to reconcile: no confirmed emit).
        // - "emit_confirmed" (or a post-verify "unconfirmed"): the emit
        //   happened. Re-read the predicate; never blindly repeat.
        if dispatched_unemitted_marker(&state.journal.last_outcome, i) {
            // Fall through to execute_step: fresh plan.
        } else if emit_confirmed_marker(&state.journal.last_outcome, i)
            || unconfirmed_marker(&state.journal.last_outcome, i)
        {
            match driver.predicate_now(&step.predicate) {
                PredicateValue::Bool(true) => {
                    // The action landed before the kill. Never repeat it.
                    if let Err(e) =
                        mark_done(state, store, i, format!("confirmed:step{i}:reconciled"))
                    {
                        return RunOutcome::Parked { reason: e };
                    }
                    continue;
                }
                PredicateValue::Bool(false) => {
                    // Re-read says the action did not take effect: fall
                    // through to a fresh attempt, which re-checks
                    // AlreadySatisfied before dispatching.
                }
                // Unknown, or a text value the runner cannot interpret as
                // satisfied: escalate rather than repeat blindly.
                PredicateValue::Unknown | PredicateValue::Text(_) => {
                    return escalate(
                        state,
                        &ctx,
                        store,
                        driver,
                        i,
                        step,
                        EscalateReason::Inconclusive,
                        "restart reconcile: predicate unknown after dispatched-but-unconfirmed; refusing to repeat blindly",
                        &[],
                    );
                }
            }
        }

        match execute_step(i, step, &ctx, state, driver, store) {
            StepResult::Done => {
                if let Err(e) = mark_done(state, store, i, format!("confirmed:step{i}")) {
                    return RunOutcome::Parked { reason: e };
                }
            }
            StepResult::Escalate {
                reason,
                evidence,
                failed_ids,
            } => {
                return escalate(
                    state,
                    &ctx,
                    store,
                    driver,
                    i,
                    step,
                    reason,
                    &evidence,
                    &failed_ids,
                );
            }
            StepResult::Park { reason } => {
                state.journal.last_outcome = Some(format!("parked:step{i}:{reason}"));
                let _ = persist(state, store);
                return RunOutcome::Parked { reason };
            }
        }
    }
    RunOutcome::Complete
}

fn execute_step(
    i: usize,
    step: &Step,
    _ctx: &ExecCtx,
    state: &mut RunState,
    driver: &mut dyn StepDriver,
    store: &mut dyn JournalStore,
) -> StepResult {
    // AlreadySatisfied before any dispatch: a satisfied step is never clicked.
    if driver.predicate_now(&step.predicate).is_true() {
        return StepResult::Done;
    }
    let mut failed_ids: Vec<u64> = Vec::new();

    // Journal BEFORE the state-changing action.
    state.journal.last_outcome = Some(format!("dispatched:step{i}:attempt0"));
    if let Err(e) = persist(state, store) {
        return StepResult::Park { reason: e };
    }

    // Attempt 0.
    let target = match driver.dispatch(step) {
        Ok(id) => {
            // The X11 emit returned: journal it BEFORE verify. On restart,
            // "dispatched" without this marker means the emit may never
            // have happened (safe to re-plan); with it, reconcile re-reads
            // the predicate as today.
            state.journal.last_outcome = Some(format!("emit_confirmed:step{i}:attempt0"));
            if let Err(e) = persist(state, store) {
                return StepResult::Park { reason: e };
            }
            id
        }
        Err(e) => {
            return StepResult::Escalate {
                reason: EscalateReason::DispatchRejected,
                evidence: e,
                failed_ids,
            }
        }
    };
    let verdict = driver.verify(step);
    // Journal AFTER the action, before interpreting the outcome.
    match verdict {
        Verdict::Confirmed | Verdict::AlreadySatisfied => {
            state.journal.last_outcome = Some(format!("confirmed:step{i}:attempt0"));
            if let Err(e) = persist(state, store) {
                return StepResult::Park { reason: e };
            }
            StepResult::Done
        }
        Verdict::Unchanged | Verdict::Inconclusive => {
            state.journal.last_outcome = Some(format!("unconfirmed:step{i}:attempt0:{verdict:?}"));
            if let Err(e) = persist(state, store) {
                return StepResult::Park { reason: e };
            }
            if verdict == Verdict::Inconclusive {
                return StepResult::Escalate {
                    reason: EscalateReason::Inconclusive,
                    evidence: "inconclusive evidence; never clicks blind".to_string(),
                    failed_ids,
                };
            }
            // Unchanged: one retry, only when idempotent and not a commit step.
            if !(step.idempotent && !step.commit) {
                return StepResult::Escalate {
                    reason: EscalateReason::VerifyFailed,
                    evidence:
                        "unchanged after action; not retryable (non-idempotent or commit step)"
                            .to_string(),
                    failed_ids,
                };
            }
            // Retry: re-snapshot FIRST, insert the failed id BEFORE any
            // escalate that could follow.
            failed_ids.push(target);
            driver.note_failed(target);
            driver.resnapshot();
            if driver.predicate_now(&step.predicate).is_true() {
                return StepResult::Done;
            }
            state.journal.last_outcome = Some(format!("dispatched:step{i}:attempt1"));
            if let Err(e) = persist(state, store) {
                return StepResult::Park { reason: e };
            }
            let target2 = match driver.dispatch(step) {
                Ok(id) => {
                    state.journal.last_outcome = Some(format!("emit_confirmed:step{i}:attempt1"));
                    if let Err(e) = persist(state, store) {
                        return StepResult::Park { reason: e };
                    }
                    id
                }
                Err(e) => {
                    return StepResult::Escalate {
                        reason: EscalateReason::DispatchRejected,
                        evidence: e,
                        failed_ids,
                    }
                }
            };
            let verdict2 = driver.verify(step);
            match verdict2 {
                Verdict::Confirmed | Verdict::AlreadySatisfied => {
                    state.journal.last_outcome = Some(format!("confirmed:step{i}:attempt1"));
                    if let Err(e) = persist(state, store) {
                        return StepResult::Park { reason: e };
                    }
                    StepResult::Done
                }
                v => {
                    failed_ids.push(target2);
                    driver.note_failed(target2);
                    state.journal.last_outcome =
                        Some(format!("unconfirmed:step{i}:attempt1:{v:?}"));
                    if let Err(e) = persist(state, store) {
                        return StepResult::Park { reason: e };
                    }
                    StepResult::Escalate {
                        reason: if v == Verdict::Inconclusive {
                            EscalateReason::Inconclusive
                        } else {
                            EscalateReason::VerifyFailed
                        },
                        evidence: format!("retry verdict {v:?}, not confirmed"),
                        failed_ids,
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn escalate(
    state: &mut RunState,
    ctx: &ExecCtx,
    store: &mut dyn JournalStore,
    driver: &mut dyn StepDriver,
    i: usize,
    step: &Step,
    reason: EscalateReason,
    evidence: &str,
    failed_ids: &[u64],
) -> RunOutcome {
    // 2-strike breaker: the same step escalating twice in a row parks the
    // run instead of looping on Tier-1.
    if state.strikes_on == Some(i) {
        let reason_text = format!("2-strike breaker at step {i}");
        state.journal.last_outcome = Some(format!("parked:step{i}:strike"));
        if let Err(e) = persist(state, store) {
            return RunOutcome::Parked {
                reason: format!("{reason_text} (journal: {e})"),
            };
        }
        return RunOutcome::Parked {
            reason: reason_text,
        };
    }
    if state.tier1_calls >= ctx.budget {
        let reason_text = format!("escalation budget {} exhausted at step {i}", ctx.budget);
        state.journal.last_outcome = Some(format!("parked:step{i}:budget"));
        let _ = persist(state, store);
        return RunOutcome::Parked {
            reason: reason_text,
        };
    }
    state.tier1_calls += 1;
    state.strikes_on = Some(i);
    state.journal.last_outcome = Some(format!("escalated:step{i}:{reason:?}"));
    let _ = persist(state, store);

    // Bounded escalation context: the driver's fresh shot plus the ring,
    // capped at 3, most recent last.
    if let Some(shot) = driver.escalation_shot() {
        push_shot(&mut state.shot_ring, shot);
    }
    let context = EscalationContext {
        shots: state.shot_ring.iter().cloned().collect(),
        shortlist: driver.shortlist(),
        failed_ids: failed_ids.to_vec(),
        evidence: evidence.to_string(),
    };
    RunOutcome::Escalated(Escalation {
        step: i,
        reason,
        describe: step.describe.clone(),
        context,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, VecDeque};

    // --- fakes ---

    struct ScriptDriver {
        dispatches: Vec<String>,
        dispatch_results: VecDeque<Result<u64, String>>,
        predicate_values: VecDeque<PredicateValue>,
        verify_values: VecDeque<Verdict>,
        context: (String, String),
        failed: Vec<u64>,
        resnapshots: usize,
        shots: VecDeque<Vec<u8>>,
        shortlist: Vec<String>,
    }

    impl ScriptDriver {
        fn new() -> Self {
            Self {
                dispatches: Vec::new(),
                dispatch_results: VecDeque::new(),
                predicate_values: VecDeque::new(),
                verify_values: VecDeque::new(),
                context: ("Chrome - Portal".to_string(), "portal.example".to_string()),
                failed: Vec::new(),
                resnapshots: 0,
                shots: VecDeque::new(),
                shortlist: vec!["shortlist item".to_string()],
            }
        }
    }

    impl StepDriver for ScriptDriver {
        fn current_context(&self) -> (String, String) {
            self.context.clone()
        }
        fn predicate_now(&mut self, _pred: &Predicate) -> PredicateValue {
            self.predicate_values
                .pop_front()
                .unwrap_or(PredicateValue::Bool(false))
        }
        fn dispatch(&mut self, step: &Step) -> Result<u64, String> {
            self.dispatches.push(step.describe.clone());
            self.dispatch_results.pop_front().unwrap_or(Ok(7))
        }
        fn verify(&mut self, _step: &Step) -> Verdict {
            self.verify_values.pop_front().unwrap_or(Verdict::Confirmed)
        }
        fn resnapshot(&mut self) {
            self.resnapshots += 1;
        }
        fn note_failed(&mut self, id: u64) {
            self.failed.push(id);
        }
        fn escalation_shot(&mut self) -> Option<Vec<u8>> {
            self.shots.pop_front()
        }
        fn shortlist(&self) -> Vec<String> {
            self.shortlist.clone()
        }
    }

    struct MemStore {
        map: HashMap<String, RunJournal>,
        fail_save: bool,
    }

    impl MemStore {
        fn new() -> Self {
            Self {
                map: HashMap::new(),
                fail_save: false,
            }
        }
    }

    impl JournalStore for MemStore {
        fn load(&self, book_hash: &str) -> Option<RunJournal> {
            self.map.get(book_hash).cloned()
        }
        fn save(&mut self, j: &RunJournal) -> Result<(), String> {
            if self.fail_save {
                return Err("disk is on fire".to_string());
            }
            self.map.insert(j.book_hash.clone(), j.clone());
            Ok(())
        }
    }

    fn step(describe: &str) -> Step {
        Step {
            describe: describe.to_string(),
            predicate: Predicate::HasText("Done".to_string()),
            commit: false,
            idempotent: true,
            window: "Chrome".to_string(),
            domain: "portal.example".to_string(),
        }
    }

    fn cfg() -> RunConfig {
        RunConfig {
            profile_id: "p".to_string(),
            window_allowlist: vec!["Chrome".to_string()],
            domain_allowlist: vec!["portal.example".to_string()],
            max_steps: 100,
        }
    }

    fn seeded_journal(book: &[Step], f: impl FnOnce(&mut RunJournal)) -> MemStore {
        let mut store = MemStore::new();
        let mut j = RunJournal {
            run_id: "seeded".to_string(),
            book_hash: book_hash(book),
            cursor: 0,
            last_outcome: None,
            profile_id: "p".to_string(),
            origin: String::new(),
            tree_generation: 0,
            committed_steps: Vec::new(),
            checkpoints: Vec::new(),
        };
        f(&mut j);
        store.save(&j).unwrap();
        store
    }

    // --- budget ---

    #[test]
    fn budget_table() {
        assert_eq!(max_escalations(0), 2);
        assert_eq!(max_escalations(1), 2);
        assert_eq!(max_escalations(5), 2);
        assert_eq!(max_escalations(6), 2);
        assert_eq!(max_escalations(25), 5);
        assert_eq!(max_escalations(10_000), 5);
    }

    #[test]
    fn budget_huge_n_is_overflow_safe() {
        // The naive (n + 4) / 5 overflows here in debug and wraps in
        // release; the division-first form is identical in both.
        assert_eq!(max_escalations(usize::MAX), 5);
        assert_eq!(max_escalations(usize::MAX - 1), 5);
    }

    // --- hashing ---

    #[test]
    fn sha256_matches_nist_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn book_hash_stable_and_sensitive() {
        let a = vec![step("one"), step("two")];
        let b = vec![step("one"), step("two")];
        let c = vec![step("one"), step("three")];
        assert_eq!(book_hash(&a), book_hash(&b));
        assert_ne!(book_hash(&a), book_hash(&c));
        assert_eq!(book_hash(&a).len(), 64);
    }

    // --- journal store ---

    #[test]
    fn sqlite_journal_round_trip() {
        let mut store = SqliteJournalStore::open_in_memory().unwrap();
        let j = RunJournal {
            run_id: "r1".to_string(),
            book_hash: "h1".to_string(),
            cursor: 3,
            last_outcome: Some("dispatched:step3:attempt0".to_string()),
            profile_id: "p".to_string(),
            origin: "portal.example".to_string(),
            tree_generation: 9,
            committed_steps: vec![0, 1, 2],
            checkpoints: vec![2],
        };
        store.save(&j).unwrap();
        let back = store.load("h1").unwrap();
        assert_eq!(back.run_id, "r1");
        assert_eq!(back.cursor, 3);
        assert_eq!(back.last_outcome, j.last_outcome);
        assert_eq!(back.origin, "portal.example");
        assert_eq!(back.tree_generation, 9);
        assert_eq!(back.committed_steps, vec![0, 1, 2]);
        assert_eq!(back.checkpoints, vec![2]);
        assert!(store.load("nope").is_none());
    }

    // --- happy path + no double-execute ---

    #[test]
    fn happy_path_commits_and_never_repeats_on_rerun() {
        let book = vec![step("s0"), step("s1")];
        let cfg = cfg();
        let mut store = MemStore::new();
        let mut driver = ScriptDriver::new();
        let (state, outcome) = run(&book, &cfg, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Complete));
        assert_eq!(driver.dispatches, vec!["s0", "s1"]);
        assert_eq!(state.journal.committed_steps, vec![0, 1]);
        assert_eq!(state.journal.cursor, 2);

        // A fresh process loading the same journal must not repeat anything.
        let mut driver2 = ScriptDriver::new();
        let (state2, outcome2) = run(&book, &cfg, &mut driver2, &mut store);
        assert!(matches!(outcome2, RunOutcome::Complete));
        assert!(driver2.dispatches.is_empty(), "confirmed action repeated!");
        assert_eq!(state2.journal.cursor, 2);
    }

    #[test]
    fn sqlite_backed_run_survives_process_restart() {
        let dir = std::env::temp_dir().join(format!("mf-journal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("journal.db");
        let path_s = path.to_str().unwrap().to_string();
        let book = vec![step("s0"), step("s1")];
        let cfg = cfg();
        {
            let mut store = SqliteJournalStore::open(&path_s).unwrap();
            let mut driver = ScriptDriver::new();
            let (_, outcome) = run(&book, &cfg, &mut driver, &mut store);
            assert!(matches!(outcome, RunOutcome::Complete));
            assert_eq!(driver.dispatches.len(), 2);
        }
        // "Process restart": new store handle on the same file.
        {
            let mut store = SqliteJournalStore::open(&path_s).unwrap();
            let mut driver = ScriptDriver::new();
            let (_, outcome) = run(&book, &cfg, &mut driver, &mut store);
            assert!(matches!(outcome, RunOutcome::Complete));
            assert!(
                driver.dispatches.is_empty(),
                "confirmed action repeated after restart!"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    // --- restart reconciliation ---

    #[test]
    fn restart_before_dispatch_runs_normally() {
        let book = vec![step("s0")];
        let store = seeded_journal(&book, |_| {});
        let mut driver = ScriptDriver::new();
        let cfg = cfg();
        let mut store = store;
        let mut state = RunState {
            journal: store.load(&book_hash(&book)).unwrap(),
            tier1_calls: 0,
            strikes_on: None,
            shot_ring: VecDeque::new(),
        };
        let outcome = resume(&book, &cfg, &mut state, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Complete));
        assert_eq!(driver.dispatches, vec!["s0"]);
    }

    #[test]
    fn restart_after_dispatch_predicate_true_never_repeats() {
        let book = vec![step("s0")];
        let store = seeded_journal(&book, |j| {
            j.last_outcome = Some("emit_confirmed:step0:attempt0".to_string());
        });
        let mut driver = ScriptDriver::new();
        // Re-read finds the action landed.
        driver
            .predicate_values
            .push_back(PredicateValue::Bool(true));
        let cfg = cfg();
        let mut store = store;
        let mut state = RunState {
            journal: store.load(&book_hash(&book)).unwrap(),
            tier1_calls: 0,
            strikes_on: None,
            shot_ring: VecDeque::new(),
        };
        let outcome = resume(&book, &cfg, &mut state, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Complete));
        assert!(driver.dispatches.is_empty(), "dispatched action repeated!");
        assert_eq!(state.journal.committed_steps, vec![0]);
    }

    #[test]
    fn restart_after_dispatch_predicate_false_attempts_once() {
        let book = vec![step("s0")];
        let store = seeded_journal(&book, |j| {
            j.last_outcome = Some("emit_confirmed:step0:attempt0".to_string());
        });
        let mut driver = ScriptDriver::new();
        // Re-read: not landed. Pre-dispatch check: not satisfied. Verify: confirmed.
        driver
            .predicate_values
            .push_back(PredicateValue::Bool(false));
        driver
            .predicate_values
            .push_back(PredicateValue::Bool(false));
        driver.verify_values.push_back(Verdict::Confirmed);
        let cfg = cfg();
        let mut store = store;
        let mut state = RunState {
            journal: store.load(&book_hash(&book)).unwrap(),
            tier1_calls: 0,
            strikes_on: None,
            shot_ring: VecDeque::new(),
        };
        let outcome = resume(&book, &cfg, &mut state, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Complete));
        assert_eq!(driver.dispatches, vec!["s0"]);
    }

    #[test]
    fn restart_after_dispatch_predicate_unknown_escalates_never_repeats() {
        let book = vec![step("s0")];
        let store = seeded_journal(&book, |j| {
            j.last_outcome = Some("emit_confirmed:step0:attempt0".to_string());
        });
        let mut driver = ScriptDriver::new();
        driver.predicate_values.push_back(PredicateValue::Unknown);
        let cfg = cfg();
        let mut store = store;
        let mut state = RunState {
            journal: store.load(&book_hash(&book)).unwrap(),
            tier1_calls: 0,
            strikes_on: None,
            shot_ring: VecDeque::new(),
        };
        let outcome = resume(&book, &cfg, &mut state, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Escalated(_)));
        assert!(driver.dispatches.is_empty(), "blind repeat after unknown!");
    }

    #[test]
    fn restart_after_dispatched_without_emit_replans_fresh() {
        // Crash point 1: the journal went out BEFORE the X11 emit, so the
        // action may never have happened. Restart must re-plan (a fresh
        // dispatch), not predicate-re-read and not escalate.
        let book = vec![step("s0")];
        let store = seeded_journal(&book, |j| {
            j.last_outcome = Some("dispatched:step0:attempt0".to_string());
        });
        let mut driver = ScriptDriver::new();
        // execute_step's AlreadySatisfied check: not satisfied.
        driver
            .predicate_values
            .push_back(PredicateValue::Bool(false));
        driver.verify_values.push_back(Verdict::Confirmed);
        let cfg = cfg();
        let mut store = store;
        let mut state = RunState {
            journal: store.load(&book_hash(&book)).unwrap(),
            tier1_calls: 0,
            strikes_on: None,
            shot_ring: VecDeque::new(),
        };
        let outcome = resume(&book, &cfg, &mut state, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Complete));
        // Exactly one fresh dispatch: the pre-crash dispatch never
        // observably happened, so this is not a repeat.
        assert_eq!(driver.dispatches, vec!["s0"]);
    }

    #[test]
    fn restart_after_dispatched_without_emit_already_satisfied_skips() {
        // Crash point 1, but the step is already satisfied: the re-plan
        // must notice and never dispatch.
        let book = vec![step("s0")];
        let store = seeded_journal(&book, |j| {
            j.last_outcome = Some("dispatched:step0:attempt0".to_string());
        });
        let mut driver = ScriptDriver::new();
        driver
            .predicate_values
            .push_back(PredicateValue::Bool(true));
        let cfg = cfg();
        let mut store = store;
        let mut state = RunState {
            journal: store.load(&book_hash(&book)).unwrap(),
            tier1_calls: 0,
            strikes_on: None,
            shot_ring: VecDeque::new(),
        };
        let outcome = resume(&book, &cfg, &mut state, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Complete));
        assert!(driver.dispatches.is_empty(), "satisfied step was clicked!");
    }

    #[test]
    fn execute_step_journals_emit_confirmed_between_dispatch_and_verify() {
        // Regression pin for the crash-point ambiguity: the journal must
        // show dispatched -> emit_confirmed -> confirmed, in that order.
        // Without the fix, "dispatched" is followed directly by the
        // post-verify marker and the two crash points are indistinguishable.
        struct RecordingStore {
            inner: MemStore,
            seen: Vec<Option<String>>,
        }
        impl JournalStore for RecordingStore {
            fn load(&self, book_hash: &str) -> Option<RunJournal> {
                self.inner.load(book_hash)
            }
            fn save(&mut self, j: &RunJournal) -> Result<(), String> {
                self.seen.push(j.last_outcome.clone());
                self.inner.save(j)
            }
        }

        let book = vec![step("s0")];
        let mut store = RecordingStore {
            inner: MemStore::new(),
            seen: Vec::new(),
        };
        let mut driver = ScriptDriver::new();
        driver.verify_values.push_back(Verdict::Confirmed);
        let cfg = cfg();
        let ctx = ExecCtx {
            budget: 5,
            cfg: &cfg,
            book_len: 1,
        };
        let mut state = RunState {
            journal: RunJournal {
                run_id: "r".to_string(),
                book_hash: book_hash(&book),
                cursor: 0,
                last_outcome: None,
                profile_id: "p".to_string(),
                origin: "portal.example".to_string(),
                tree_generation: 0,
                committed_steps: Vec::new(),
                checkpoints: Vec::new(),
            },
            tier1_calls: 0,
            strikes_on: None,
            shot_ring: VecDeque::new(),
        };
        let result = execute_step(0, &book[0], &ctx, &mut state, &mut driver, &mut store);
        assert!(matches!(result, StepResult::Done));
        let seq: Vec<&str> = store.seen.iter().filter_map(|o| o.as_deref()).collect();
        assert_eq!(
            seq,
            [
                "dispatched:step0:attempt0",
                "emit_confirmed:step0:attempt0",
                "confirmed:step0:attempt0",
            ]
        );
    }

    #[test]
    fn restart_after_confirm_skips_committed_step() {
        let book = vec![step("s0"), step("s1")];
        let store = seeded_journal(&book, |j| {
            j.cursor = 1;
            j.committed_steps = vec![0];
            j.last_outcome = Some("confirmed:step0".to_string());
        });
        let mut driver = ScriptDriver::new();
        let cfg = cfg();
        let mut store = store;
        let mut state = RunState {
            journal: store.load(&book_hash(&book)).unwrap(),
            tier1_calls: 0,
            strikes_on: None,
            shot_ring: VecDeque::new(),
        };
        let outcome = resume(&book, &cfg, &mut state, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Complete));
        assert_eq!(driver.dispatches, vec!["s1"], "s0 must not repeat");
    }

    #[test]
    fn restart_at_checkpoint_executes_commit_step_once() {
        let s0 = step("s0");
        let mut s1 = step("s1");
        s1.commit = true;
        s1.idempotent = false;
        let book = vec![s0, s1];
        let store = seeded_journal(&book, |j| {
            j.cursor = 1;
            j.committed_steps = vec![0];
            j.checkpoints = vec![1];
            j.last_outcome = Some("checkpoint:step1".to_string());
        });
        let mut driver = ScriptDriver::new();
        let cfg = cfg();
        let mut store = store;
        let mut state = RunState {
            journal: store.load(&book_hash(&book)).unwrap(),
            tier1_calls: 0,
            strikes_on: None,
            shot_ring: VecDeque::new(),
        };
        let outcome = resume(&book, &cfg, &mut state, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Complete));
        assert_eq!(driver.dispatches, vec!["s1"]);
        assert_eq!(state.journal.committed_steps, vec![0, 1]);
    }

    #[test]
    fn resume_refuses_a_different_book() {
        let book = vec![step("s0")];
        let other = vec![step("other")];
        let store = seeded_journal(&book, |_| {});
        let mut driver = ScriptDriver::new();
        let cfg = cfg();
        let mut store = store;
        let mut state = RunState {
            journal: store.load(&book_hash(&book)).unwrap(),
            tier1_calls: 0,
            strikes_on: None,
            shot_ring: VecDeque::new(),
        };
        let outcome = resume(&other, &cfg, &mut state, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Parked { .. }));
        assert!(driver.dispatches.is_empty());
    }

    // --- retry rules ---

    #[test]
    fn unchanged_idempotent_retries_once_with_resnapshot() {
        let book = vec![step("s0")];
        let mut driver = ScriptDriver::new();
        driver.verify_values.push_back(Verdict::Unchanged);
        driver.verify_values.push_back(Verdict::Confirmed);
        let cfg = cfg();
        let mut store = MemStore::new();
        let (state, outcome) = run(&book, &cfg, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Complete));
        assert_eq!(driver.dispatches, vec!["s0", "s0"]);
        assert_eq!(driver.resnapshots, 1, "retry must re-snapshot first");
        assert_eq!(
            driver.failed,
            vec![7],
            "failed id inserted before escalate path"
        );
        assert_eq!(state.journal.committed_steps, vec![0]);
    }

    #[test]
    fn unchanged_commit_step_never_retries() {
        let mut s = step("s0");
        s.commit = true;
        s.idempotent = false;
        let book = vec![s];
        // Pre-seed the checkpoint so the step executes.
        let store = seeded_journal(&book, |j| {
            j.checkpoints = vec![0];
            j.last_outcome = Some("checkpoint:step0".to_string());
        });
        let mut driver = ScriptDriver::new();
        driver.verify_values.push_back(Verdict::Unchanged);
        let cfg = cfg();
        let mut store = store;
        let mut state = RunState {
            journal: store.load(&book_hash(&book)).unwrap(),
            tier1_calls: 0,
            strikes_on: None,
            shot_ring: VecDeque::new(),
        };
        let outcome = resume(&book, &cfg, &mut state, &mut driver, &mut store);
        assert!(matches!(
            outcome,
            RunOutcome::Escalated(Escalation {
                reason: EscalateReason::VerifyFailed,
                ..
            })
        ));
        assert_eq!(
            driver.dispatches.len(),
            1,
            "commit step must not auto-retry"
        );
    }

    #[test]
    fn unchanged_non_idempotent_never_retries() {
        let mut s = step("s0");
        s.idempotent = false;
        let book = vec![s];
        let mut driver = ScriptDriver::new();
        driver.verify_values.push_back(Verdict::Unchanged);
        let cfg = cfg();
        let mut store = MemStore::new();
        let (_, outcome) = run(&book, &cfg, &mut driver, &mut store);
        assert!(matches!(
            outcome,
            RunOutcome::Escalated(Escalation {
                reason: EscalateReason::VerifyFailed,
                ..
            })
        ));
        assert_eq!(driver.dispatches.len(), 1);
    }

    #[test]
    fn inconclusive_escalates_without_retry() {
        let book = vec![step("s0")];
        let mut driver = ScriptDriver::new();
        driver.verify_values.push_back(Verdict::Inconclusive);
        let cfg = cfg();
        let mut store = MemStore::new();
        let (_, outcome) = run(&book, &cfg, &mut driver, &mut store);
        assert!(matches!(
            outcome,
            RunOutcome::Escalated(Escalation {
                reason: EscalateReason::Inconclusive,
                ..
            })
        ));
        assert_eq!(driver.dispatches.len(), 1);
    }

    #[test]
    fn retry_failure_carries_failed_ids_into_escalation() {
        let book = vec![step("s0")];
        let mut driver = ScriptDriver::new();
        driver.verify_values.push_back(Verdict::Unchanged);
        driver.verify_values.push_back(Verdict::Unchanged);
        let cfg = cfg();
        let mut store = MemStore::new();
        let (_, outcome) = run(&book, &cfg, &mut driver, &mut store);
        match outcome {
            RunOutcome::Escalated(e) => {
                assert_eq!(e.reason, EscalateReason::VerifyFailed);
                assert_eq!(e.context.failed_ids, vec![7, 7]);
                assert_eq!(driver.failed, vec![7, 7]);
            }
            other => panic!("expected escalation, got {other:?}"),
        }
    }

    #[test]
    fn already_satisfied_never_dispatches() {
        let book = vec![step("s0")];
        let mut driver = ScriptDriver::new();
        driver
            .predicate_values
            .push_back(PredicateValue::Bool(true));
        let cfg = cfg();
        let mut store = MemStore::new();
        let (state, outcome) = run(&book, &cfg, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Complete));
        assert!(driver.dispatches.is_empty());
        assert_eq!(state.journal.committed_steps, vec![0]);
    }

    #[test]
    fn dispatch_rejected_escalates() {
        let book = vec![step("s0")];
        let mut driver = ScriptDriver::new();
        driver
            .dispatch_results
            .push_back(Err("cdp rejected".to_string()));
        let cfg = cfg();
        let mut store = MemStore::new();
        let (_, outcome) = run(&book, &cfg, &mut driver, &mut store);
        assert!(matches!(
            outcome,
            RunOutcome::Escalated(Escalation {
                reason: EscalateReason::DispatchRejected,
                ..
            })
        ));
    }

    // --- guardrails ---

    #[test]
    fn step_outside_domain_allowlist_escalates_without_dispatch() {
        let mut s = step("s0");
        s.domain = "evil.example".to_string();
        let book = vec![s];
        let mut driver = ScriptDriver::new();
        let cfg = cfg();
        let mut store = MemStore::new();
        let (_, outcome) = run(&book, &cfg, &mut driver, &mut store);
        assert!(matches!(
            outcome,
            RunOutcome::Escalated(Escalation {
                reason: EscalateReason::AllowlistViolation,
                ..
            })
        ));
        assert!(driver.dispatches.is_empty());
    }

    #[test]
    fn live_context_outside_allowlist_escalates() {
        let book = vec![step("s0")];
        let mut driver = ScriptDriver::new();
        driver.context = ("Chrome - Portal".to_string(), "other.example".to_string());
        let cfg = cfg();
        let mut store = MemStore::new();
        let (_, outcome) = run(&book, &cfg, &mut driver, &mut store);
        assert!(matches!(
            outcome,
            RunOutcome::Escalated(Escalation {
                reason: EscalateReason::AllowlistViolation,
                ..
            })
        ));
        assert!(driver.dispatches.is_empty());
    }

    #[test]
    fn per_run_step_ceiling_parks() {
        let book = vec![step("s0"), step("s1"), step("s2")];
        let mut cfg = cfg();
        cfg.max_steps = 1;
        let mut driver = ScriptDriver::new();
        let mut store = MemStore::new();
        let (_, outcome) = run(&book, &cfg, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Parked { .. }));
        assert_eq!(driver.dispatches, vec!["s0"]);
    }

    #[test]
    fn commit_step_checkpoints_for_review() {
        let mut s1 = step("s1");
        s1.commit = true;
        s1.idempotent = false;
        let book = vec![step("s0"), s1];
        let mut driver = ScriptDriver::new();
        let cfg = cfg();
        let mut store = MemStore::new();
        let (mut state, outcome) = run(&book, &cfg, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Checkpoint { step: 1 }));
        assert_eq!(
            driver.dispatches,
            vec!["s0"],
            "commit step must pause before dispatch"
        );
        assert_eq!(state.journal.checkpoints, vec![1]);

        // Resume executes the commit step exactly once.
        let outcome = resume(&book, &cfg, &mut state, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Complete));
        assert_eq!(driver.dispatches, vec!["s0", "s1"]);
    }

    // --- escalation budget + strikes ---

    #[test]
    fn two_strikes_on_same_step_park() {
        let book = vec![step("s0")];
        let mut driver = ScriptDriver::new();
        driver.verify_values.push_back(Verdict::Inconclusive);
        let cfg = cfg();
        let mut store = MemStore::new();
        let (mut state, outcome) = run(&book, &cfg, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Escalated(_)));
        // Tier-1 "answers", runner resumes the same step, escalates again.
        driver.verify_values.push_back(Verdict::Inconclusive);
        let outcome = resume(&book, &cfg, &mut state, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Parked { .. }), "{outcome:?}");
    }

    #[test]
    fn budget_exhaustion_parks() {
        // 25 steps -> budget 5. Six escalations must park on the sixth.
        let book: Vec<Step> = (0..25).map(|i| step(&format!("s{i}"))).collect();
        let cfg = cfg();
        let mut store = MemStore::new();
        let mut driver = ScriptDriver::new();
        // Consumption order (FIFO): escalate, confirm, escalate, confirm...
        for v in [
            Verdict::Inconclusive,
            Verdict::Confirmed,
            Verdict::Inconclusive,
            Verdict::Confirmed,
            Verdict::Inconclusive,
            Verdict::Confirmed,
            Verdict::Inconclusive,
            Verdict::Confirmed,
            Verdict::Inconclusive,
            Verdict::Confirmed,
            Verdict::Inconclusive,
        ] {
            driver.verify_values.push_back(v);
        }
        let (mut state, mut outcome) = run(&book, &cfg, &mut driver, &mut store);
        for _ in 0..5 {
            assert!(matches!(outcome, RunOutcome::Escalated(_)), "{outcome:?}");
            outcome = resume(&book, &cfg, &mut state, &mut driver, &mut store);
        }
        assert!(matches!(outcome, RunOutcome::Parked { .. }), "{outcome:?}");
        if let RunOutcome::Parked { reason } = outcome {
            assert!(reason.contains("budget"), "{reason}");
        }
    }

    #[test]
    fn escalation_context_ring_caps_at_three() {
        // 25 steps -> budget 5: four escalations fit under it.
        let book: Vec<Step> = (0..25).map(|i| step(&format!("s{i}"))).collect();
        let cfg = cfg();
        let mut store = MemStore::new();
        let mut driver = ScriptDriver::new();
        for k in 0..4u8 {
            driver.shots.push_back(vec![k; 8]);
        }
        // Four escalations interleaved with confirmations so the run advances.
        for v in [
            Verdict::Inconclusive,
            Verdict::Confirmed,
            Verdict::Inconclusive,
            Verdict::Confirmed,
            Verdict::Inconclusive,
            Verdict::Confirmed,
            Verdict::Inconclusive,
        ] {
            driver.verify_values.push_back(v);
        }
        let (mut state, mut outcome) = run(&book, &cfg, &mut driver, &mut store);
        let mut last: Option<Escalation> = None;
        for _ in 0..4 {
            match outcome {
                RunOutcome::Escalated(e) => {
                    last = Some(e);
                    outcome = resume(&book, &cfg, &mut state, &mut driver, &mut store);
                }
                other => panic!("expected escalation, got {other:?}"),
            }
        }
        let e = last.unwrap();
        assert_eq!(e.context.shots.len(), 3, "ring must cap at 3");
        // Most recent last: shots 1, 2, 3 (shot 0 pruned).
        assert_eq!(e.context.shots[0], vec![1u8; 8]);
        assert_eq!(e.context.shots[2], vec![3u8; 8]);
        assert_eq!(e.context.shortlist, vec!["shortlist item".to_string()]);
    }

    // --- journal failure ---

    #[test]
    fn journal_write_failure_parks_instead_of_lying() {
        let book = vec![step("s0")];
        let mut driver = ScriptDriver::new();
        let cfg = cfg();
        let mut store = MemStore::new();
        store.fail_save = true;
        let (_, outcome) = run(&book, &cfg, &mut driver, &mut store);
        assert!(matches!(outcome, RunOutcome::Parked { .. }), "{outcome:?}");
    }
}
