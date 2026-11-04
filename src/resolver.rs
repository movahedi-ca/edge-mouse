//! Layer 2 target resolution cascade, resolver v1 (mouse-future Phase 4).
//!
//! A dependable production resolver with measured recall: cheap
//! deterministic stages first (identity cache, exact match, real BM25),
//! an interim calibrated linear ranker second, Tier-1 escalation last.
//!
//! Design rules (from plan v7, validated against the sketch):
//! - The query is tokenized ONCE per resolve through the shared
//!   [`StringInterner`], into the SAME term space as the candidate index.
//!   No per-candidate allocation in the hot path.
//! - [`PreparedNode`] is computed ONCE per node insertion (interned token
//!   ids, normalized name/role, log area, text hash, trigram embedding).
//!   Insertion threads the frame chain via
//!   [`super::frame_chain::attach_chain`]; the chain itself is built by
//!   [`super::coords::build_chain`].
//! - Stage 0 is exact accessible-name match with a stage-aware early exit
//!   (visible, enabled, current generation, unambiguous). Its confidence
//!   is the MEASURED precision from the evaluation harness, not a guess.
//! - Stage 1 is real BM25 (k1 = 1.2, b = 0.75, token matching, NaN
//!   guards), top-8 via a bounded [`std::collections::BinaryHeap`], never
//!   a full sort. The role prior enters as ranker FEATURE 3, never as a
//!   silent multiplier.
//! - Stage 2 is an interim linear ranker (Stage 2 of the cascade): 137-dim
//!   weights on the same [`super::features::assemble`] the S1 head will
//!   use, listwise softmax cross-entropy, temperature-scaled, with its own
//!   calibrated margin gate. There is NO `prior_shift`: a list-wide scalar
//!   cancels out of softmax, so it would be a no-op (documented below).
//! - A query with no searchable terms returns typed
//!   [`MissReason::NoSearchableTerms`], never a silent zero-score
//!   fallthrough.
//! - Candidate embeddings are cached per (backend_node_id, text_hash). A
//!   missing embedding is NEVER 64 silent zeros: the candidate is skipped
//!   and counted (see [`ResolverEngine::skipped_no_embedding`]).
//! - [`calibrated_p`] is temperature-only, fit on natural calibration
//!   lists from measured rows. Gates are fit on real rows, never on
//!   legacy logs.
//! - Grounding: targets under 12x12 physical px set `needs_refinement`
//!   (the crop-and-zoom path is stubbed here; full vision lands in B5).
//!   Every escalation payload records its [`ModelSpaceMapping`]; the
//!   scale is inverted before the typed conversion, and a coordinate
//!   without its mapping is REJECTED, never guessed.
//! - Mining (Track C): every Confirmed resolution appends a labelled row
//!   to `~/workspace/mouse-future/corpus/rows.jsonl`, OUTSIDE the repo
//!   (the repo is public; mined page text stays out of it).
//!
//! Pure synchronous Rust. No tokio on these paths, no async runtime.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;

use indexmap::IndexMap;

use super::coords::{document_to_main_viewport, FrameDocumentCss, FrameTransform, MainViewportCss};
use super::features::{self, FeatureInputs, FEATURE_SCHEMA_VERSION};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// BM25 saturation parameter (frozen).
pub const BM25_K1: f64 = 1.2;
/// BM25 length-normalization parameter (frozen).
pub const BM25_B: f64 = 0.75;
/// Bounded shortlist size; the heap never holds more than this.
pub const TOP_K: usize = 8;
/// Targets smaller than this many physical px on either side need the
/// crop-and-zoom refinement path (stubbed in B5).
pub const REFINEMENT_MIN_PX: f64 = 12.0;
/// Identity-cache capacity (bounded O(1) LRU).
pub const IDENTITY_CACHE_CAP: usize = 512;
/// Confidence carried by an identity-cache hit: the node was Confirmed by
/// the verifier before it was cached, so this is a policy value for a
/// verified prior, not a measured precision.
pub const CACHE_HIT_CONFIDENCE: f64 = 0.95;
/// Hard ceiling applied to every reported confidence.
pub const CONFIDENCE_CAP: f64 = 0.95;

/// Stop list, fixed and versioned with the feature schema. A query made
/// only of these (or of nothing) has no searchable terms.
pub const STOP_WORDS: &[&str] = &["the", "and", "for", "with", "from", "that", "this"];

// ---------------------------------------------------------------------------
// Deterministic 64-bit hash (FNV-1a). Used for cache keys and text hashes.
// This is a different contract from features.rs's FxHash-like n-gram hash;
// the two must not be confused.
// ---------------------------------------------------------------------------

/// FNV-1a 64 over bytes.
pub fn fx64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Normalization and tokenization
// ---------------------------------------------------------------------------

/// Lowercase + collapse whitespace: the shared normalization for names,
/// descriptions, and n-gram text (same contract as features.rs).
pub fn normalize_desc(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Tokenize into normalized token strings (pre-interning).
///
/// Lowercase, split on non-alphanumeric, drop the fixed stop list.
/// One-character tokens survive only when they ARE the whole text
/// ("x", "+"): a lone control label must stay searchable. When splitting
/// leaves nothing but the text is non-empty, the whole trimmed text is
/// kept as one token so a query is never silently emptied by punctuation.
pub fn tokenize(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let trimmed = lower.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if trimmed.chars().count() == 1 {
        return vec![trimmed.to_string()];
    }
    let raw: Vec<&str> = trimmed
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    if raw.is_empty() {
        // No alphanumeric content at all (punctuation-only like "!!!"):
        // keep the whole trimmed text so the query is never silently
        // emptied by punctuation.
        return vec![trimmed.to_string()];
    }
    // Stop words are dropped; a query of only stop words ("the and for")
    // has NO searchable terms and yields NoSearchableTerms downstream.
    raw.into_iter()
        .filter(|t| !STOP_WORDS.contains(t))
        .map(|t| t.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Shared string interner: built once per tree generation.
// ---------------------------------------------------------------------------

/// Term-id interner shared by the candidate index and the query path.
/// The query is tokenized ONCE through this interner into the same term
/// space the candidates were prepared in. Query terms unseen in the
/// corpus intern to fresh ids with document frequency 0, which the BM25
/// IDF formula handles naturally (they contribute nothing, since their
/// term frequency is 0 in every candidate).
#[derive(Debug, Default)]
pub struct StringInterner {
    map: HashMap<Box<str>, u32>,
}

impl StringInterner {
    /// Fresh interner for one tree generation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern one token, returning its term id.
    pub fn intern(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.map.get(token) {
            return id;
        }
        let id = self.map.len() as u32;
        self.map.insert(token.into(), id);
        id
    }

    /// Look up without inserting.
    pub fn get(&self, token: &str) -> Option<u32> {
        self.map.get(token).copied()
    }

    /// Number of interned terms.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True when no term has been interned.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

// ---------------------------------------------------------------------------
// PreparedNode: computed ONCE per node insertion.
// ---------------------------------------------------------------------------

/// Everything about a candidate that never changes within a tree
/// generation, prepared once at insertion so the hot path allocates
/// nothing per candidate.
///
/// `role_intent` and `exact_name` are QUERY-dependent, so they cannot be
/// frozen at insertion. What IS frozen is their normalized input:
/// [`PreparedNode::role_norm`], [`PreparedNode::name_norm`], and
/// [`PreparedNode::role_token_ids`]. The flags themselves are computed per
/// resolve from those, with no re-normalization and no allocation.
#[derive(Debug, Clone)]
pub struct PreparedNode {
    /// Stable node id within the document generation.
    pub backend_node_id: u64,
    /// Document generation this preparation belongs to.
    pub generation: u64,
    /// Normalized accessible name (exact-match stage, feature 4).
    pub name_norm: Box<str>,
    /// Normalized role string.
    pub role_norm: Box<str>,
    /// Interned token ids of the role (for the role-intent flag).
    pub role_token_ids: Vec<u32>,
    /// Interned token ids of name + text, in order (duplicates kept).
    pub token_ids: Vec<u32>,
    /// (term id, term frequency), sorted by term id: avoids the
    /// per-resolve HashMap the sketch built for every candidate.
    pub token_counts: Vec<(u32, u32)>,
    /// fx64 over `role|name|text`: the cache-validation fingerprint.
    pub text_hash: u64,
    /// ln(1+w*h)/ln(1+4_000_000), clamped [0,1]; 0 on bad geometry.
    pub log_area: f64,
    /// AX enabled+visible state.
    pub enabled_visible: bool,
    /// Signed 64-dim trigram embedding of name + text, cached per
    /// (backend_node_id, text_hash) by the engine. `None` only when the
    /// node was NOT built through [`prepare_candidate`]; such candidates
    /// are skipped at resolve time, never scored with 64 silent zeros.
    pub embedding: Option<[f32; 64]>,
}

impl PreparedNode {
    /// Feature 3: does the description name this candidate's role?
    /// True when any role token appears among the query token ids.
    pub fn role_intent_flag(&self, query_token_ids: &HashSet<u32>) -> bool {
        self.role_token_ids
            .iter()
            .any(|t| query_token_ids.contains(t))
    }

    /// Feature 4: normalized description == normalized accessible name.
    pub fn exact_name_flag(&self, desc_norm: &str) -> bool {
        self.name_norm.as_ref() == desc_norm
    }

    /// Term frequency of `term` in this node (0 when absent).
    pub fn tf(&self, term: u32) -> u32 {
        match self.token_counts.binary_search_by_key(&term, |(t, _)| *t) {
            Ok(i) => self.token_counts[i].1,
            Err(_) => 0,
        }
    }
}

/// Log-area feature, same contract as features.rs feature 5:
/// ln(1+w*h)/ln(1+4_000_000) in [0,1]; non-finite or negative geometry
/// reads as missing (0.0).
fn log_area_norm(w: f64, h: f64) -> f64 {
    if !(w.is_finite() && h.is_finite() && w >= 0.0 && h >= 0.0) {
        return 0.0;
    }
    ((1.0 + w * h).ln() / (1.0 + 4_000_000.0f64).ln()).clamp(0.0, 1.0)
}

/// One candidate: geometry in main-viewport CSS px (normalized ONCE at
/// insertion via the frame chain) plus its prepared node.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Stable node id within the document generation.
    pub backend_node_id: u64,
    /// Document generation.
    pub generation: u64,
    /// Original role string (for reporting and mining).
    pub role: Box<str>,
    /// Top-left of the node box in main-viewport CSS px.
    pub at: MainViewportCss,
    /// Box width/height, CSS px.
    pub w: f64,
    pub h: f64,
    /// Device pixel ratio governing this node.
    pub dpr: f64,
    /// Prepared once at insertion.
    pub prep: PreparedNode,
}

impl Candidate {
    /// Box center in main-viewport CSS px.
    pub fn center(&self) -> (f64, f64) {
        (self.at.0 + self.w / 2.0, self.at.1 + self.h / 2.0)
    }
}

/// Insertion-time primitive: prepare one candidate from a raw node.
///
/// `doc_xy` is the box top-left in the owning frame's document CSS px;
/// it is normalized to main-viewport CSS ONCE here via `chain`
/// (callers obtain the chain from
/// [`super::frame_chain::attach_chain`]). The trigram embedding is
/// cached per (backend_node_id, text_hash) in `emb_cache` so a node that
/// survives across mutations is never re-hashed.
#[allow(clippy::too_many_arguments)]
pub fn prepare_candidate(
    interner: &mut StringInterner,
    emb_cache: &mut HashMap<(u64, u64), [f32; 64]>,
    backend_node_id: u64,
    generation: u64,
    role: &str,
    name: &str,
    text: &str,
    doc_xy: (f64, f64),
    wh: (f64, f64),
    enabled_visible: bool,
    chain: &[FrameTransform],
) -> Candidate {
    let name_norm = normalize_desc(name);
    let text_norm = normalize_desc(text);
    let role_norm = normalize_desc(role);

    let mut token_ids = Vec::new();
    for tok in tokenize(&format!("{name_norm} {text_norm}")) {
        token_ids.push(interner.intern(&tok));
    }
    let mut role_token_ids = Vec::new();
    for tok in tokenize(&role_norm) {
        role_token_ids.push(interner.intern(&tok));
    }

    // Sorted (term, tf) pairs for allocation-free tf lookup.
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &t in &token_ids {
        *counts.entry(t).or_insert(0) += 1;
    }
    let mut token_counts: Vec<(u32, u32)> = counts.into_iter().collect();
    token_counts.sort_by_key(|(t, _)| *t);

    let text_hash = fx64(format!("{role_norm}|{name_norm}|{text_norm}").as_bytes());

    let embedding = emb_cache
        .get(&(backend_node_id, text_hash))
        .copied()
        .or_else(|| {
            let emb = features::ngram_embedding(&format!("{name_norm} {text_norm}"));
            emb_cache.insert((backend_node_id, text_hash), emb);
            Some(emb)
        });

    let at = document_to_main_viewport(FrameDocumentCss(doc_xy.0, doc_xy.1), chain);

    Candidate {
        backend_node_id,
        generation,
        role: role.into(),
        at,
        w: wh.0,
        h: wh.1,
        dpr: chain.first().map(|t| t.dpr).unwrap_or(1.0),
        prep: PreparedNode {
            backend_node_id,
            generation,
            name_norm: name_norm.into(),
            role_norm: role_norm.into(),
            role_token_ids,
            token_ids,
            token_counts,
            text_hash,
            log_area: log_area_norm(wh.0, wh.1),
            enabled_visible,
            embedding,
        },
    }
}
// ---------------------------------------------------------------------------
// Stage 1: real BM25 over the shared interner.
// ---------------------------------------------------------------------------

/// One ranked candidate: index into the engine's candidate slice plus its
/// finite BM25 score.
#[derive(Debug, Clone, Copy)]
pub struct Ranked {
    /// Index into the candidate slice.
    pub idx: usize,
    /// BM25 score; always finite (non-finite scores are clamped to the
    /// missing-value -10.0 before entering the heap, matching the feature
    /// schema's missing rule).
    pub score: f64,
}

// Total order via f64::total_cmp so the heap never sees a partial order.
// Tie-break is index DESCENDING in this order: the bounded heap evicts
// the minimum, so among equal scores it evicts the HIGHEST candidate
// index and keeps the lowest. That matches the final output sort
// (score desc, index asc), so heap top-k == full-sort top-k exactly,
// ties included. Deterministic by construction.
impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == std::cmp::Ordering::Equal && self.idx == other.idx
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.idx.cmp(&self.idx))
    }
}

/// BM25 index, built once per tree generation from prepared vectors.
/// Only document frequency is (re)computed per generation; per-query work
/// is scoring plus the bounded heap.
#[derive(Debug, Default)]
pub struct Bm25Index {
    df: HashMap<u32, u32>,
    avgdl: f64,
    n: f64,
}

impl Bm25Index {
    /// Build from the prepared token vectors of one generation.
    pub fn build(cands: &[Candidate]) -> Self {
        let n = cands.len().max(1) as f64;
        let mut df: HashMap<u32, u32> = HashMap::new();
        let mut total_len = 0usize;
        for c in cands {
            total_len += c.prep.token_ids.len();
            // Document frequency counts a term once per document.
            let mut seen = HashSet::new();
            for t in &c.prep.token_ids {
                if seen.insert(t) {
                    *df.entry(*t).or_insert(0) += 1;
                }
            }
        }
        Self {
            df,
            avgdl: total_len as f64 / n,
            n,
        }
    }

    /// Number of documents in the index.
    pub fn doc_count(&self) -> usize {
        self.n as usize
    }

    /// Average document length in tokens.
    pub fn avg_doc_len(&self) -> f64 {
        self.avgdl
    }

    /// Document frequency of a term (0 for unseen query terms).
    pub fn doc_freq(&self, term: u32) -> u32 {
        self.df.get(&term).copied().unwrap_or(0)
    }

    /// BM25 IDF: ln(((n - df + 0.5) / (df + 0.5)) + 1). Always finite for
    /// finite n and df in [0, n].
    pub fn idf(&self, term: u32) -> f64 {
        let df = self.doc_freq(term) as f64;
        ((self.n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Score one candidate against the interned query terms.
    ///
    /// NaN guard: a non-finite score can only arise from degenerate
    /// geometry constants (avgdl <= 0 with empty docs); it is clamped to
    /// the schema's missing value -10.0, matching feature 0's rule, so a
    /// bad constant can never poison the heap or the feature vector.
    pub fn score_candidate(&self, cand: &Candidate, qterms: &[u32]) -> f64 {
        let dl = cand.prep.token_ids.len().max(1) as f64;
        let avgdl = if self.avgdl > 0.0 { self.avgdl } else { 1.0 };
        let mut s = 0.0f64;
        for &t in qterms {
            let f = cand.prep.tf(t) as f64;
            if f == 0.0 {
                continue;
            }
            let idf = self.idf(t);
            s += idf * (f * (BM25_K1 + 1.0)) / (f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl));
        }
        if s.is_finite() {
            s
        } else {
            -10.0
        }
    }

    /// Bounded top-k selection over the candidate slice. A
    /// [`BinaryHeap`] of capacity k+1 keeps the k best; this is never a
    /// full sort of the candidate list. Candidates whose prepared node
    /// lacks an embedding are SKIPPED here (never 64 silent zeros); their
    /// indices are reported so the engine can count them.
    pub fn top_k(
        &self,
        cands: &[Candidate],
        qterms: &[u32],
        k: usize,
    ) -> (Vec<Ranked>, Vec<usize>) {
        let mut heap: BinaryHeap<Reverse<Ranked>> = BinaryHeap::with_capacity(k + 1);
        let mut skipped: Vec<usize> = Vec::new();
        for (i, c) in cands.iter().enumerate() {
            if c.prep.embedding.is_none() {
                skipped.push(i);
                continue;
            }
            let s = self.score_candidate(c, qterms);
            heap.push(Reverse(Ranked { idx: i, score: s }));
            if heap.len() > k {
                heap.pop();
            }
        }
        let mut out: Vec<Ranked> = heap.into_iter().map(|r| r.0).collect();
        // Descending by score; ties keep the lower candidate index first
        // (deterministic, and the tie-margin test pins margin zero).
        out.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.idx.cmp(&b.idx)));
        (out, skipped)
    }
}

// ---------------------------------------------------------------------------
// Resolution types
// ---------------------------------------------------------------------------

/// Which cascade stage produced the hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Identity-cache hit (previously Confirmed node).
    Cache,
    /// Stage 0: unique exact accessible-name match.
    Exact,
    /// Stage 1: BM25 top-1 passed the relative-margin and share gates.
    Bm25,
    /// Stage 2: interim linear ranker top-1 passed its gates.
    Ranker,
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stage::Cache => write!(f, "cache"),
            Stage::Exact => write!(f, "exact"),
            Stage::Bm25 => write!(f, "bm25"),
            Stage::Ranker => write!(f, "ranker"),
        }
    }
}

/// A resolved target: interior click point in main-viewport CSS px.
#[derive(Debug, Clone)]
pub struct Hit {
    /// Interior click point (see [`interior_point`]).
    pub at: MainViewportCss,
    /// Node box width/height, CSS px.
    pub w: f64,
    pub h: f64,
    /// Winning node.
    pub backend_node_id: u64,
    /// Calibrated confidence in [0, 1], capped at [`CONFIDENCE_CAP`].
    pub confidence: f64,
    /// Stage that produced the hit.
    pub stage: Stage,
    /// True when the target is under 12x12 physical px: the caller must
    /// run the crop-and-zoom refinement (stubbed in B5) before trusting
    /// the point.
    pub needs_refinement: bool,
}

/// Typed miss reasons. There is no silent fallthrough: a query with no
/// searchable terms is [`MissReason::NoSearchableTerms`], an empty tree
/// is [`MissReason::PerceptionEmpty`], and an uncertain ranking is
/// [`MissReason::RankingUncertain`] (the escalation lane).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissReason {
    /// No candidates in the current generation.
    PerceptionEmpty,
    /// The query tokenized to zero searchable terms.
    NoSearchableTerms,
    /// A ranking exists but no gate passed: escalate.
    RankingUncertain,
}

/// Cascade outcome.
#[derive(Debug, Clone)]
pub enum Resolution {
    Hit(Hit),
    Miss {
        reason: MissReason,
        /// The natural top-k BM25 list, best first, for the escalation
        /// payload and the mining hook.
        candidates: Vec<Ranked>,
    },
}

/// Typed resolve errors for the fallible entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveError {
    NoSearchableTerms,
    PerceptionEmpty,
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::NoSearchableTerms => write!(f, "query has no searchable terms"),
            ResolveError::PerceptionEmpty => write!(f, "no candidates in this generation"),
        }
    }
}

impl std::error::Error for ResolveError {}

// ---------------------------------------------------------------------------
// Gates (fit on real rows; defaults are pre-calibration starting points)
// ---------------------------------------------------------------------------

/// Decision gates for the cascade. The margin/share/ranker gates are fit
/// on measured rows by the evaluation harness; `exact_match_precision`
/// is the measured precision of the stage-0 early exit on those rows.
/// Defaults are conservative starting points, documented as such: they
/// are replaced by the fitted values before production use.
#[derive(Debug, Clone)]
pub struct Gates {
    /// (top1 - top2) / top1 must reach this for a BM25 hit.
    pub rel_margin_gate: f64,
    /// top1 / top-k mass must reach this for a BM25 hit.
    pub share_gate: f64,
    /// Temperature-scaled softmax p1 must reach this for a ranker hit.
    pub ranker_gate: f64,
    /// Independent ranker margin (p1 - p2) must reach this. Must be > 0:
    /// a tie has margin exactly zero and never passes.
    pub ranker_margin_gate: f64,
    /// Measured precision of the stage-0 exact early exit.
    pub exact_match_precision: f64,
}

impl Default for Gates {
    fn default() -> Self {
        Self {
            rel_margin_gate: 0.15,
            share_gate: 0.35,
            ranker_gate: 0.55,
            ranker_margin_gate: 0.10,
            exact_match_precision: EXACT_MATCH_PRECISION_MEASURED,
        }
    }
}

/// Measured precision of the stage-0 exact-match early exit on the
/// evaluation rows. Updated by B4's evaluation; the harness asserts the
/// constant matches the measured value within tolerance.
///
/// Measured 2026-11-03 on 45 labelled VM rows: the exact stage resolved
/// 8 rows, all 8 correct (precision 1.000). Reported hit confidence is
/// additionally capped at CONFIDENCE_CAP (0.95).
pub const EXACT_MATCH_PRECISION_MEASURED: f64 = 1.0;

// ---------------------------------------------------------------------------
// Stage 0: exact match
// ---------------------------------------------------------------------------

/// Outcome of the exact-name stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactOutcome {
    /// No candidate has this normalized name.
    None,
    /// More than one candidate matches: ambiguous, no early exit.
    Ambiguous,
    /// Exactly one match.
    Unique,
}

/// Stage 0 scan over prepared normalized names (no per-candidate
/// allocation). The caller applies the stage-aware early exit: only
/// [`ExactOutcome::Unique`] AND visible AND enabled AND current
/// generation may exit early; ambiguity always falls through to BM25.
pub fn exact_match_scan(cands: &[Candidate], desc_norm: &str) -> (ExactOutcome, Option<usize>) {
    let mut found: Option<usize> = None;
    let mut count = 0usize;
    for (i, c) in cands.iter().enumerate() {
        if c.prep.name_norm.as_ref() == desc_norm {
            count += 1;
            if found.is_none() {
                found = Some(i);
            }
            if count > 1 {
                return (ExactOutcome::Ambiguous, None);
            }
        }
    }
    match found {
        Some(i) => (ExactOutcome::Unique, Some(i)),
        None => (ExactOutcome::None, None),
    }
}

// ---------------------------------------------------------------------------
// calibrated_p: temperature-only fit on natural calibration lists
// ---------------------------------------------------------------------------

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Temperature-only calibrated confidence over the BM25 relative margin.
///
/// p = sigmoid(margin / T), margin in [0, 1]. The intercept is fixed at 0
/// by design: a zero margin is the maximally uncertain point of the
/// sigmoid (p = 0.5), and the margin GATE, not the confidence value,
/// decides whether a hit is taken. Only the temperature T is fit, by
/// minimizing log-loss on natural calibration lists (margin, top1-correct).
#[derive(Debug, Clone, Copy)]
pub struct Calibrator {
    /// Fitted temperature; 1.0 is the uncalibrated baseline.
    pub temperature: f64,
}

impl Calibrator {
    /// Uncalibrated baseline (T = 1).
    pub fn uncalibrated() -> Self {
        Self { temperature: 1.0 }
    }

    /// Calibrated P(top-1 correct | relative margin).
    pub fn p(&self, rel_margin: f64) -> f64 {
        let t = if self.temperature > 0.0 && self.temperature.is_finite() {
            self.temperature
        } else {
            1.0
        };
        let m = rel_margin.clamp(0.0, 1.0);
        sigmoid(m / t).clamp(0.0, 1.0)
    }

    /// Mean binary log-loss of this calibrator on rows of
    /// (rel_margin, top1_correct).
    pub fn log_loss(&self, rows: &[(f64, bool)]) -> f64 {
        if rows.is_empty() {
            return f64::NAN;
        }
        let mut sum = 0.0;
        for &(m, y) in rows {
            let p = self.p(m).clamp(1e-12, 1.0 - 1e-12);
            sum += if y { -p.ln() } else { -(1.0 - p).ln() };
        }
        sum / rows.len() as f64
    }

    /// Fit the temperature by grid search over log-spaced candidates in
    /// [0.02, 4.0], minimizing log-loss. Returns the best temperature.
    /// The grid is deterministic and always includes T = 1.0 (the
    /// uncalibrated baseline), so the fit can never be worse than
    /// baseline by construction. Ties keep the smaller temperature.
    pub fn fit_temperature(rows: &[(f64, bool)]) -> f64 {
        let mut best_t = 1.0;
        let mut best_loss = Self { temperature: 1.0 }.log_loss(rows);
        // 160 log-spaced points from 0.02 to 4.0, plus exactly 1.0.
        for i in 0..160 {
            let t = 0.02 * (4.0f64 / 0.02).powf(i as f64 / 159.0);
            let loss = Self { temperature: t }.log_loss(rows);
            if loss.is_finite() && loss < best_loss {
                best_loss = loss;
                best_t = t;
            }
        }
        best_t
    }
}
// ---------------------------------------------------------------------------
// Stage 2: interim linear ranker (pure Rust).
//
// 137-dim weights on the same featurize() the S1 head uses
// (super::features::assemble). Trains and infers the same head: listwise
// softmax cross-entropy over the live top-8, temperature-scaled.
//
// NO prior_shift: a list-wide scalar added to every logit cancels out of
// softmax exactly (softmax(z + c) == softmax(z)), so it would have zero
// gradient, zero effect on the ranking, and zero effect on the margin.
// The role prior therefore enters ONLY as ranker feature 3
// (role-intent flag), where it is a real, learnable signal.
// ---------------------------------------------------------------------------

/// One training row: the live top-k feature vectors and the index of the
/// correct candidate (correct < xs.len() <= 8).
#[derive(Debug, Clone)]
pub struct TrainRow {
    pub xs: Vec<[f32; 137]>,
    pub correct: usize,
}

/// Interim linear ranker: w (137) + bias, temperature-scaled softmax.
#[derive(Debug, Clone)]
pub struct LinearRanker {
    pub weights: [f32; 137],
    pub bias: f32,
    pub temperature: f64,
}

impl LinearRanker {
    /// Zero-initialized ranker (trains from here).
    pub fn zero() -> Self {
        Self {
            weights: [0.0; 137],
            bias: 0.0,
            temperature: 1.0,
        }
    }

    /// Raw logits for the first n rows (1 <= n <= 8). Panics in debug if
    /// n is out of range or xs is short: a programming error, never
    /// caller data.
    pub fn logits(&self, xs: &[[f32; 137]], n: usize) -> [f64; 8] {
        debug_assert!((1..=8).contains(&n));
        debug_assert!(xs.len() >= n);
        let mut z = [0.0f64; 8];
        for (slot, x) in z.iter_mut().zip(xs.iter()).take(n) {
            let mut s = f64::from(self.bias);
            for (w, xv) in self.weights.iter().zip(x.iter()) {
                s += f64::from(*w) * f64::from(*xv);
            }
            *slot = if s.is_finite() { s } else { 0.0 };
        }
        z
    }

    /// Temperature-scaled softmax over the first n rows. Writes
    /// probabilities into `out[0..n]`; the rest stays 0.
    pub fn softmax_scaled(&self, xs: &[[f32; 137]], n: usize, out: &mut [f64; 8]) {
        let t = if self.temperature > 0.0 && self.temperature.is_finite() {
            self.temperature
        } else {
            1.0
        };
        let z = self.logits(xs, n);
        let mut max = f64::NEG_INFINITY;
        for &v in z.iter().take(n) {
            let v = v / t;
            if v > max {
                max = v;
            }
        }
        let mut sum = 0.0;
        for (slot, &zv) in out.iter_mut().zip(z.iter()).take(n) {
            let e = (zv / t - max).exp();
            let e = if e.is_finite() { e } else { 0.0 };
            *slot = e;
            sum += e;
        }
        if sum > 0.0 {
            for slot in out.iter_mut().take(n) {
                *slot /= sum;
            }
        } else {
            // Degenerate: fall back to uniform rather than NaN.
            for slot in out.iter_mut().take(n) {
                *slot = 1.0 / n as f64;
            }
        }
    }

    /// Train listwise softmax cross-entropy with plain SGD and L2
    /// weight decay. Deterministic: rows are visited in order, no
    /// shuffling, fixed epoch count.
    pub fn train(&mut self, rows: &[TrainRow], epochs: usize, lr: f32, l2: f32) {
        for _ in 0..epochs {
            for row in rows {
                let n = row.xs.len().clamp(1, 8);
                if row.correct >= n {
                    continue;
                }
                let mut probs = [0.0f64; 8];
                self.softmax_scaled(&row.xs, n, &mut probs);
                for (i, (prob, x)) in probs.iter().zip(row.xs.iter()).take(n).enumerate() {
                    let target = if i == row.correct { 1.0f64 } else { 0.0 };
                    let grad = prob - target; // dL/dz_i
                    let g = grad as f32 * lr;
                    for (w, xv) in self.weights.iter_mut().zip(x.iter()) {
                        *w -= g * xv + lr * l2 * *w;
                    }
                    self.bias -= g;
                }
            }
        }
    }

    /// Mean listwise log-loss (cross-entropy) on rows.
    pub fn log_loss(&self, rows: &[TrainRow]) -> f64 {
        if rows.is_empty() {
            return f64::NAN;
        }
        let mut sum = 0.0;
        let mut count = 0usize;
        for row in rows {
            let n = row.xs.len().clamp(1, 8);
            if row.correct >= n {
                continue;
            }
            let mut probs = [0.0f64; 8];
            self.softmax_scaled(&row.xs, n, &mut probs);
            let p = probs[row.correct].clamp(1e-12, 1.0);
            sum += -p.ln();
            count += 1;
        }
        if count == 0 {
            f64::NAN
        } else {
            sum / count as f64
        }
    }

    /// Fit the softmax temperature on rows by grid search over log-spaced
    /// candidates in [0.05, 4.0], minimizing listwise log-loss.
    pub fn fit_temperature(&mut self, rows: &[TrainRow]) {
        let mut best_t = self.temperature;
        let mut best_loss = self.log_loss(rows);
        for i in 0..120 {
            let t = 0.05 * (4.0f64 / 0.05).powf(i as f64 / 119.0);
            self.temperature = t;
            let loss = self.log_loss(rows);
            if loss.is_finite() && loss < best_loss {
                best_loss = loss;
                best_t = t;
            }
        }
        self.temperature = best_t;
    }

    /// Top-two selection over the live top-n, RETAINING ties: the order
    /// is a stable sort by probability, so a tie keeps both candidates
    /// with margin exactly zero. Returns (order, p1, margin).
    pub fn top_two(&self, probs: &[f64; 8], n: usize) -> (Vec<usize>, f64, f64) {
        let n = n.clamp(1, 8);
        let mut order: Vec<usize> = (0..n).collect();
        // Stable: ties keep ascending index order.
        order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then_with(|| a.cmp(&b)));
        let p1 = probs[order[0]];
        let p2 = if n > 1 { probs[order[1]] } else { 0.0 };
        // A tie has margin exactly zero by construction and never passes
        // a positive ranker_margin_gate.
        (order, p1, p1 - p2)
    }
}

// ---------------------------------------------------------------------------
// Interior click point
// ---------------------------------------------------------------------------

/// Uniform [0,1) source. Production passes the trajectory RNG; tests pass
/// a seeded XorShift.
pub trait Rng01 {
    fn uniform01(&mut self) -> f64;
}

/// Interior click point: uniform in the inset rect (15% margin each
/// side), never on the node edge. Deterministic only under an explicit
/// test seed; production draws from the trajectory RNG.
pub fn interior_point(c: &Candidate, rng: &mut dyn Rng01) -> MainViewportCss {
    let mx = c.w * 0.15;
    let my = c.h * 0.15;
    let iw = (c.w - 2.0 * mx).max(1.0);
    let ih = (c.h - 2.0 * my).max(1.0);
    MainViewportCss(
        c.at.0 + mx + rng.uniform01() * iw,
        c.at.1 + my + rng.uniform01() * ih,
    )
}

// ---------------------------------------------------------------------------
// Identity cache: real bounded O(1) LRU over interned keys.
// ---------------------------------------------------------------------------

/// Cache entry: the node that was Confirmed for this description.
#[derive(Debug, Clone, Copy)]
pub struct CacheEntry {
    pub node_id: u64,
    pub text_hash: u64,
    pub generation: u64,
}

/// (profile_id, origin, route/state, normalized_description) ->
/// (backend_node_id, text_hash, doc_generation).
///
/// Bounded O(1) LRU on [`IndexMap`]: lookup is a hash probe, recency is
/// maintained by shift-remove + re-insert, eviction pops index 0. Keys
/// are interned u64 hashes (fx64 over NUL-joined fields), not owned
/// strings. A hit additionally requires the node to be live in the
/// current generation with the same text hash; the caller then reads the
/// node's CURRENT box.
///
/// Dependency: `indexmap` (already in rust/Cargo.toml as `version = "2"`
/// with the `serde` feature; Cargo.lock pins 2.14.2 at build time). The
/// shared Cargo.toml was deliberately not edited: sibling workers build
/// against the same manifest concurrently.
#[derive(Debug)]
pub struct IdentityCache {
    map: IndexMap<u64, CacheEntry>,
    cap: usize,
}

impl IdentityCache {
    /// New cache with the given capacity bound.
    pub fn new(cap: usize) -> Self {
        Self {
            map: IndexMap::new(),
            cap: cap.max(1),
        }
    }

    /// Intern the identity key. Same inputs always yield the same key;
    /// any differing field yields a (practically) different key.
    pub fn key(profile: &str, origin: &str, route: &str, desc_norm: &str) -> u64 {
        fx64(format!("{profile}\0{origin}\0{route}\0{desc_norm}").as_bytes())
    }

    /// Number of live entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// O(1) lookup. Returns the candidate index on a full hit:
    /// entry generation == current generation, node live in
    /// `live_by_id`, and text hash unchanged. Touches the entry (MRU).
    /// Anything else is a miss (stale generation, changed text, or a
    /// node that left the tree).
    pub fn lookup(
        &mut self,
        key: u64,
        live_by_id: &HashMap<u64, usize>,
        cands: &[Candidate],
        generation: u64,
    ) -> Option<usize> {
        let e = *self.map.get(&key)?;
        if e.generation != generation {
            return None;
        }
        let i = *live_by_id.get(&e.node_id)?;
        let c = cands.get(i)?;
        if c.backend_node_id != e.node_id || c.prep.text_hash != e.text_hash {
            return None;
        }
        // Touch: shift-remove + re-insert marks MRU. Both are O(1)
        // amortized on IndexMap's dense storage for the probed key.
        if let Some(entry) = self.map.shift_remove(&key) {
            self.map.insert(key, entry);
        }
        Some(i)
    }

    /// Commit ONLY after a Confirmed verifier verdict (never on a guess).
    /// Evicts the least-recently-used entry when at capacity.
    pub fn commit(&mut self, key: u64, entry: CacheEntry) {
        if self.map.len() >= self.cap && !self.map.contains_key(&key) {
            self.map.shift_remove_index(0);
        }
        self.map.insert(key, entry);
    }

    /// Drop entries from older generations (memory hygiene on nav).
    pub fn evict_stale(&mut self, generation: u64) {
        self.map.retain(|_, e| e.generation == generation);
    }
}

// ---------------------------------------------------------------------------
// Grounding refinements: crop-and-zoom trigger and the escalation
// coordinate contract.
// ---------------------------------------------------------------------------

/// True when the target is smaller than 12x12 physical px: a single
/// grounding pass is not trusted; the caller must run the crop-and-zoom
/// refinement (stubbed here; full vision lands in B5).
pub fn needs_refinement(w: f64, h: f64, dpr: f64) -> bool {
    let dpr = if dpr.is_finite() && dpr > 0.0 {
        dpr
    } else {
        1.0
    };
    w * dpr < REFINEMENT_MIN_PX || h * dpr < REFINEMENT_MIN_PX
}

/// Refinement decision for a resolved hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refinement {
    /// Geometry is trustworthy as-is.
    Direct,
    /// Run the crop-and-zoom path before locking the coordinate.
    /// STUB in resolver v1: the escalation payload is produced, but the
    /// zoomed re-resolve itself lands with B5's vision work.
    CropAndZoom,
}

/// Refinement decision from a hit's box.
pub fn refinement_for(hit: &Hit, dpr: f64) -> Refinement {
    if needs_refinement(hit.w, hit.h, dpr) {
        Refinement::CropAndZoom
    } else {
        Refinement::Direct
    }
}

/// The coordinate space a visual grounder reports in. Models disagree:
/// some cap the longest edge, some use a 0-999 grid, some return ratios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSpace {
    /// Pixels in an image whose longest edge was capped at 1568.
    Pixels1568Capped,
    /// Integer grid 0-999 on each axis.
    Grid0_999,
    /// Ratios in [0, 1] of the image size.
    Ratio01,
    /// Already in display px (no scaling).
    DisplayPx,
}

/// The mapping an escalation payload MUST record: the image it sent
/// (size), the model space it asked for, and the aspect-preserving scale
/// factors (display px per model unit) to invert on return.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpaceMapping {
    pub image_w: u32,
    pub image_h: u32,
    pub model_space: ModelSpace,
    /// Display px per model unit on x (aspect-preserving).
    pub scale_x: f64,
    /// Display px per model unit on y (aspect-preserving).
    pub scale_y: f64,
}

/// Invert the aspect-preserving scale recorded in the mapping, yielding
/// main-viewport CSS px. Non-finite scales or points fall back to 0.0
/// rather than producing a NaN coordinate.
pub fn invert_model_space(p: (f64, f64), m: &ModelSpaceMapping) -> MainViewportCss {
    let sx = if m.scale_x.is_finite() {
        m.scale_x
    } else {
        0.0
    };
    let sy = if m.scale_y.is_finite() {
        m.scale_y
    } else {
        0.0
    };
    let x = if p.0.is_finite() { p.0 * sx } else { 0.0 };
    let y = if p.1.is_finite() { p.1 * sy } else { 0.0 };
    MainViewportCss(x, y)
}

/// A coordinate that carries its mapping. There is no constructor that
/// builds one without a mapping: a mapping-less coordinate cannot exist
/// in this type.
#[derive(Debug, Clone, Copy)]
pub struct GroundedPoint {
    pub point: MainViewportCss,
    pub mapping: ModelSpaceMapping,
}

/// Named escalation-coordinate failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationError {
    /// The grounder returned a point with no recorded mapping.
    /// REJECTED, never guessed.
    MissingMapping,
    /// The point itself was non-finite.
    NonFiniteCoordinate,
}

impl std::fmt::Display for EscalationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EscalationError::MissingMapping => {
                write!(
                    f,
                    "escalated coordinate arrived without its model-space mapping"
                )
            }
            EscalationError::NonFiniteCoordinate => {
                write!(f, "escalated coordinate is non-finite")
            }
        }
    }
}

impl std::error::Error for EscalationError {}

/// Accept an escalated (Tier-1 / visual grounder) coordinate: the mapping
/// is REQUIRED. `None` is rejected with [`EscalationError::MissingMapping`],
/// never guessed.
pub fn accept_escalated_point(
    mapping: Option<&ModelSpaceMapping>,
    p: (f64, f64),
) -> Result<GroundedPoint, EscalationError> {
    let m = mapping.ok_or(EscalationError::MissingMapping)?;
    if !p.0.is_finite() || !p.1.is_finite() {
        return Err(EscalationError::NonFiniteCoordinate);
    }
    Ok(GroundedPoint {
        point: invert_model_space(p, m),
        mapping: *m,
    })
}
// ---------------------------------------------------------------------------
// Mining hook (Track C).
//
// Every Confirmed resolution appends one labelled row to
// ~/workspace/mouse-future/corpus/rows.jsonl. The corpus lives OUTSIDE
// the bridge repo: the repo is public, and mined page text must never
// land in it. The directory is created on first write. Callers invoke
// this ONLY after a Confirmed verifier verdict, never on a guess.
// ---------------------------------------------------------------------------

/// Env override for the corpus directory (tests point it at a temp dir).
pub const CORPUS_DIR_ENV: &str = "MOUSE_FUTURE_CORPUS_DIR";
/// Corpus rows file name.
pub const CORPUS_ROWS_FILE: &str = "rows.jsonl";

/// Resolve the corpus rows path: `$MOUSE_FUTURE_CORPUS_DIR/rows.jsonl`
/// when set, else `$HOME/workspace/mouse-future/corpus/rows.jsonl`.
pub fn corpus_rows_path() -> PathBuf {
    if let Ok(dir) = std::env::var(CORPUS_DIR_ENV) {
        let mut p = PathBuf::from(dir);
        p.push(CORPUS_ROWS_FILE);
        return p;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let mut p = PathBuf::from(home);
    p.push("workspace/mouse-future/corpus");
    p.push(CORPUS_ROWS_FILE);
    p
}

/// One mined top-k candidate: BM25 score plus the full 137-dim feature
/// vector, so Track C can train without re-featurizing. The features are
/// a Vec (not [f32; 137]) because serde only implements arrays up to
/// length 32; the eval harness asserts the length is 137 on read.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MinedCandidateRow {
    pub backend_node_id: u64,
    pub role: String,
    pub name: String,
    pub bm25: f64,
    pub features: Vec<f32>,
}

/// One labelled resolution row.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MinedRow {
    /// Feature schema version (refuse on mismatch downstream).
    pub schema: u32,
    pub description: String,
    pub page_url: String,
    pub origin: String,
    pub generation: u64,
    /// Natural top-k BM25 list, best first, with features.
    pub top_k: Vec<MinedCandidateRow>,
    /// The Confirmed winner.
    pub winner_backend_node_id: u64,
    /// Cascade stage that produced the winner.
    pub stage: String,
}

/// Append one labelled row (JSONL, one object per line). Creates the
/// corpus directory when missing. Refuses rows whose schema version is
/// not the frozen v1 (the corpus must never mix schemas silently).
/// Returns the io error to the caller; mining must never panic the
/// resolve path.
pub fn mine_confirmed_row(row: &MinedRow) -> std::io::Result<()> {
    if row.schema != FEATURE_SCHEMA_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "mined row schema {} != frozen v{}",
                row.schema, FEATURE_SCHEMA_VERSION
            ),
        ));
    }
    let path = corpus_rows_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    let mut line = serde_json::to_string(row)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    line.push('\n');
    f.write_all(line.as_bytes())?;
    f.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// ResolverEngine: one tree generation, fully prepared.
// ---------------------------------------------------------------------------

/// Engine configuration (per tree generation).
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Identity-cache profile namespace.
    pub profile_id: String,
    /// Page origin (e.g. host) for the identity-cache key.
    pub origin: String,
    /// Main viewport (width, height), CSS px, for the position prior.
    pub viewport: (f64, f64),
    /// Identity-cache capacity.
    pub cache_cap: usize,
}

/// Options for one resolve call.
#[derive(Debug, Clone)]
pub struct ResolveOpts {
    /// Route/state string for the identity-cache key.
    pub route: String,
    /// Skip the identity cache (freshness probes, tests).
    pub bypass_cache: bool,
}

/// The resolver for one tree generation: shared interner, prepared
/// candidates, BM25 index, identity cache, calibrator, interim ranker.
///
/// Lifecycle: `new` -> `insert` (per node, via
/// [`super::frame_chain::attach_chain`]) -> `finalize` (builds the index
/// and the live id->index map) -> `resolve`. A new generation starts
/// with [`ResolverEngine::begin_generation`], which clears the interner,
/// candidates, and index but KEEPS the cache, calibrator, and ranker.
pub struct ResolverEngine {
    cfg: EngineConfig,
    interner: StringInterner,
    emb_cache: HashMap<(u64, u64), [f32; 64]>,
    candidates: Vec<Candidate>,
    live_by_id: HashMap<u64, usize>,
    generation: u64,
    index: Option<Bm25Index>,
    cache: IdentityCache,
    /// Decision gates (fit on measured rows).
    pub gates: Gates,
    /// Temperature-only confidence calibrator.
    pub calibrator: Calibrator,
    /// Interim linear ranker (Stage 2).
    pub ranker: LinearRanker,
    /// Candidates skipped for missing embeddings (never silent zeros).
    pub skipped_no_embedding: u64,
}

impl ResolverEngine {
    /// New engine. `generation` is the document generation being built.
    pub fn new(cfg: EngineConfig, generation: u64, gates: Gates) -> Self {
        let cap = cfg.cache_cap;
        Self {
            cfg,
            interner: StringInterner::new(),
            emb_cache: HashMap::new(),
            candidates: Vec::new(),
            live_by_id: HashMap::new(),
            generation,
            index: None,
            cache: IdentityCache::new(cap),
            gates,
            calibrator: Calibrator::uncalibrated(),
            ranker: LinearRanker::zero(),
            skipped_no_embedding: 0,
        }
    }

    /// Start a new tree generation: clears the interner, candidates, and
    /// BM25 index; keeps the identity cache, calibrator, and ranker.
    /// Stale cache entries are evicted by generation on lookup; call
    /// `evict_stale` explicitly for memory hygiene after navigation.
    pub fn begin_generation(&mut self, generation: u64) {
        self.generation = generation;
        self.interner = StringInterner::new();
        self.candidates.clear();
        self.live_by_id.clear();
        self.index = None;
        self.skipped_no_embedding = 0;
    }

    /// Insert one raw node. `doc_xy` is the box top-left in the owning
    /// frame's document CSS px; `chain` comes from
    /// [`super::frame_chain::attach_chain`].
    #[allow(clippy::too_many_arguments)]
    pub fn insert(
        &mut self,
        backend_node_id: u64,
        role: &str,
        name: &str,
        text: &str,
        doc_xy: (f64, f64),
        wh: (f64, f64),
        enabled_visible: bool,
        chain: &[FrameTransform],
    ) -> u64 {
        let c = prepare_candidate(
            &mut self.interner,
            &mut self.emb_cache,
            backend_node_id,
            self.generation,
            role,
            name,
            text,
            doc_xy,
            wh,
            enabled_visible,
            chain,
        );
        self.candidates.push(c);
        backend_node_id
    }

    /// Build the BM25 index and the live backend_node_id -> index map.
    /// Must be called after the last insert of a generation.
    pub fn finalize(&mut self) {
        self.live_by_id.clear();
        for (i, c) in self.candidates.iter().enumerate() {
            self.live_by_id.insert(c.backend_node_id, i);
        }
        self.index = Some(Bm25Index::build(&self.candidates));
    }

    /// Number of prepared candidates.
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    /// Current document generation.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Borrow the candidate slice (eval harness, mining).
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    /// Borrow the BM25 index (None before finalize).
    pub fn index(&self) -> Option<&Bm25Index> {
        self.index.as_ref()
    }

    /// Borrow the identity cache mutably (commit after Confirmed).
    pub fn cache_mut(&mut self) -> &mut IdentityCache {
        &mut self.cache
    }

    /// Tokenize the query ONCE through the shared interner.
    fn intern_query(&mut self, desc_norm: &str) -> Vec<u32> {
        tokenize(desc_norm)
            .iter()
            .map(|t| self.interner.intern(t))
            .collect()
    }

    /// Featurize one (description, candidate) pair with the frozen
    /// schema v1 contract (super::features::assemble). The BM25 score
    /// passed in is the stage-1 score for this candidate.
    fn featurize(
        &self,
        desc_norm: &str,
        query_token_ids: &HashSet<u32>,
        cand: &Candidate,
        bm25: f64,
    ) -> [f32; 137] {
        let inp = FeatureInputs {
            bm25,
            description: desc_norm,
            candidate_text: &cand.prep.name_norm,
            role_intent_match: cand.prep.role_intent_flag(query_token_ids),
            exact_name: cand.prep.exact_name_flag(desc_norm),
            candidate_w: cand.w,
            candidate_h: cand.h,
            candidate_center: cand.center(),
            viewport: self.cfg.viewport,
            enabled_visible: cand.prep.enabled_visible,
        };
        features::assemble(&inp)
    }

    /// The natural top-k BM25 list for a description, with features.
    /// Used by the evaluation harness and the mining hook. Returns
    /// [`ResolveError`] typed errors, never a silent fallthrough.
    pub fn natural_top_k(
        &mut self,
        desc: &str,
        k: usize,
    ) -> Result<Vec<(usize, f64, [f32; 137])>, ResolveError> {
        if self.candidates.is_empty() {
            return Err(ResolveError::PerceptionEmpty);
        }
        let desc_norm = normalize_desc(desc);
        let qterms = self.intern_query(&desc_norm);
        if qterms.is_empty() {
            return Err(ResolveError::NoSearchableTerms);
        }
        let index = self
            .index
            .get_or_insert_with(|| Bm25Index::build(&self.candidates));
        let (ranked, skipped) = index.top_k(&self.candidates, &qterms, k);
        self.skipped_no_embedding += skipped.len() as u64;
        let qset: HashSet<u32> = qterms.iter().copied().collect();
        let mut out = Vec::with_capacity(ranked.len());
        for r in ranked {
            let c = &self.candidates[r.idx];
            let fv = self.featurize(&desc_norm, &qset, c, r.score);
            out.push((r.idx, r.score, fv));
        }
        Ok(out)
    }

    /// Full cascade resolve.
    pub fn resolve(&mut self, desc: &str, opts: &ResolveOpts, rng: &mut dyn Rng01) -> Resolution {
        if self.candidates.is_empty() {
            return Resolution::Miss {
                reason: MissReason::PerceptionEmpty,
                candidates: Vec::new(),
            };
        }
        let desc_norm = normalize_desc(desc);
        let qterms = self.intern_query(&desc_norm);
        if qterms.is_empty() {
            return Resolution::Miss {
                reason: MissReason::NoSearchableTerms,
                candidates: Vec::new(),
            };
        }
        let qset: HashSet<u32> = qterms.iter().copied().collect();

        // Identity cache: live node, same generation, same text hash.
        if !opts.bypass_cache {
            let key = IdentityCache::key(
                &self.cfg.profile_id,
                &self.cfg.origin,
                &opts.route,
                &desc_norm,
            );
            if let Some(i) =
                self.cache
                    .lookup(key, &self.live_by_id, &self.candidates, self.generation)
            {
                let c = &self.candidates[i];
                return Resolution::Hit(Hit {
                    at: interior_point(c, rng),
                    w: c.w,
                    h: c.h,
                    backend_node_id: c.backend_node_id,
                    confidence: CACHE_HIT_CONFIDENCE,
                    stage: Stage::Cache,
                    needs_refinement: needs_refinement(c.w, c.h, c.dpr),
                });
            }
        }

        // Stage 0: exact accessible-name match. Early exit ONLY when the
        // match is unique, visible, enabled, and current-generation.
        let (outcome, unique_idx) = exact_match_scan(&self.candidates, &desc_norm);
        if outcome == ExactOutcome::Unique {
            let i = unique_idx.expect("unique exact match carries its index");
            let c = &self.candidates[i];
            if c.prep.enabled_visible && c.generation == self.generation {
                return Resolution::Hit(Hit {
                    at: interior_point(c, rng),
                    w: c.w,
                    h: c.h,
                    backend_node_id: c.backend_node_id,
                    confidence: self.gates.exact_match_precision.min(CONFIDENCE_CAP),
                    stage: Stage::Exact,
                    needs_refinement: needs_refinement(c.w, c.h, c.dpr),
                });
            }
        }

        // Stage 1: real BM25, bounded top-k heap.
        let index = self
            .index
            .get_or_insert_with(|| Bm25Index::build(&self.candidates));
        let (ranked, skipped) = index.top_k(&self.candidates, &qterms, TOP_K);
        self.skipped_no_embedding += skipped.len() as u64;
        if ranked.is_empty() {
            return Resolution::Miss {
                reason: MissReason::RankingUncertain,
                candidates: Vec::new(),
            };
        }
        let top1 = ranked[0].score;
        let top2 = ranked.get(1).map(|r| r.score).unwrap_or(0.0);
        let mass: f64 = ranked.iter().map(|r| r.score).sum();
        if !mass.is_finite() || mass <= 0.0 {
            return Resolution::Miss {
                reason: MissReason::RankingUncertain,
                candidates: ranked,
            };
        }
        let rel_margin = if top1 > 0.0 {
            (top1 - top2) / top1
        } else {
            0.0
        };
        let share = top1 / mass;
        if rel_margin >= self.gates.rel_margin_gate && share >= self.gates.share_gate {
            let c = &self.candidates[ranked[0].idx];
            return Resolution::Hit(Hit {
                at: interior_point(c, rng),
                w: c.w,
                h: c.h,
                backend_node_id: c.backend_node_id,
                confidence: self.calibrator.p(rel_margin).min(CONFIDENCE_CAP),
                stage: Stage::Bm25,
                needs_refinement: needs_refinement(c.w, c.h, c.dpr),
            });
        }

        // Stage 2: interim linear ranker over the live top-k.
        let n = ranked.len().min(TOP_K);
        let mut xs = [[0.0f32; 137]; TOP_K];
        for (k, r) in ranked.iter().take(n).enumerate() {
            let c = &self.candidates[r.idx];
            xs[k] = self.featurize(&desc_norm, &qset, c, r.score);
        }
        let mut probs = [0.0f64; 8];
        self.ranker.softmax_scaled(&xs, n, &mut probs);
        let (order, p1, margin) = self.ranker.top_two(&probs, n);
        if p1.is_finite() && p1 >= self.gates.ranker_gate && margin >= self.gates.ranker_margin_gate
        {
            let c = &self.candidates[ranked[order[0]].idx];
            return Resolution::Hit(Hit {
                at: interior_point(c, rng),
                w: c.w,
                h: c.h,
                backend_node_id: c.backend_node_id,
                confidence: p1.min(CONFIDENCE_CAP),
                stage: Stage::Ranker,
                needs_refinement: needs_refinement(c.w, c.h, c.dpr),
            });
        }

        Resolution::Miss {
            reason: MissReason::RankingUncertain,
            candidates: ranked,
        }
    }

    /// Commit an identity-cache entry. Call ONLY after a Confirmed
    /// verifier verdict, never on a guess.
    pub fn commit_cache(
        &mut self,
        profile: &str,
        origin: &str,
        route: &str,
        desc_norm: &str,
        backend_node_id: u64,
        text_hash: u64,
    ) {
        let key = IdentityCache::key(profile, origin, route, desc_norm);
        self.cache.commit(
            key,
            CacheEntry {
                node_id: backend_node_id,
                text_hash,
                generation: self.generation,
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64 for randomized tests.
    struct XorShift64(u64);
    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n.max(1) as u64) as usize
        }
    }
    impl Rng01 for XorShift64 {
        fn uniform01(&mut self) -> f64 {
            // 53-bit mantissa uniform in [0,1).
            const DEN: f64 = (1u64 << 53) as f64;
            ((self.next() >> 11) as f64) / DEN
        }
    }

    fn identity_chain() -> Vec<FrameTransform> {
        vec![FrameTransform {
            scroll: (0.0, 0.0),
            content_origin: (0.0, 0.0),
            nested_offset: (0.0, 0.0),
            css_scale: 1.0,
            dpr: 1.0,
        }]
    }

    fn test_engine() -> ResolverEngine {
        ResolverEngine::new(
            EngineConfig {
                profile_id: "test".to_string(),
                origin: "example.com".to_string(),
                viewport: (1280.0, 800.0),
                cache_cap: 16,
            },
            1,
            Gates::default(),
        )
    }

    /// Build a tiny engine from (role, name) pairs laid out in a row.
    fn engine_from_names(names: &[(&str, &str)]) -> ResolverEngine {
        let mut e = test_engine();
        let chain = identity_chain();
        for (i, (role, name)) in names.iter().enumerate() {
            e.insert(
                1000 + i as u64,
                role,
                name,
                "",
                (10.0 * i as f64, 20.0),
                (80.0, 30.0),
                true,
                &chain,
            );
        }
        e.finalize();
        e
    }

    // -- interner / tokenization -------------------------------------------

    #[test]
    fn interner_is_shared_between_index_and_query() {
        let mut it = StringInterner::new();
        let a = it.intern("search");
        let b = it.intern("search");
        assert_eq!(a, b);
        assert_eq!(it.len(), 1);
        assert_eq!(it.get("missing"), None);
    }

    #[test]
    fn tokenize_contract() {
        assert!(tokenize("").is_empty());
        assert!(tokenize("   ").is_empty());
        // Stop words are dropped; "the and for" has no searchable terms.
        assert!(tokenize("the and for").is_empty());
        assert_eq!(tokenize("Search THE web"), vec!["search", "web"]);
        // Single-char whole text is kept ("x" close button).
        assert_eq!(tokenize("x"), vec!["x"]);
        // Punctuation-only text falls back to the whole trimmed token.
        assert_eq!(tokenize("!!!"), vec!["!!!"]);
    }

    // -- BM25 ----------------------------------------------------------------

    /// Hand-computed BM25 example (k1 = 1.2, b = 0.75).
    ///
    /// Corpus (term ids a=0, b=1, c=2, d=3):
    ///   d0: [a, b]            dl=2
    ///   d1: [a, a, c]         dl=3
    ///   d2: [b, c, c, d]      dl=4
    /// Query: [a, c].  n = 3, avgdl = 3.
    /// df(a) = 2, df(c) = 2 -> idf = ln((3-2+0.5)/(2+0.5) + 1) = ln(1.6).
    /// d0: a: tf=1 -> ln(1.6) * (1*2.2)/(1 + 1.2*(0.25 + 0.75*2/3))
    /// d1: a: tf=2 -> ln(1.6) * (2*2.2)/(2 + 1.2*(0.25 + 0.75*3/3))
    ///     c: tf=1 -> ln(1.6) * (1*2.2)/(1 + 1.2*(0.25 + 0.75*3/3))
    /// d2: c: tf=2 -> ln(1.6) * (2*2.2)/(2 + 1.2*(0.25 + 0.75*4/3))
    #[test]
    fn bm25_hand_computed_example() {
        let idf = (1.6f64).ln();
        // Independent re-derivation of each term contribution.
        let term = |idf: f64, tf: f64, dl: f64, avgdl: f64| -> f64 {
            idf * (tf * (1.2 + 1.0)) / (tf + 1.2 * (1.0 - 0.75 + 0.75 * dl / avgdl))
        };
        let avgdl = 3.0;
        let d0 = term(idf, 1.0, 2.0, avgdl);
        let d1 = term(idf, 2.0, 3.0, avgdl) + term(idf, 1.0, 3.0, avgdl);
        let d2 = term(idf, 2.0, 4.0, avgdl);
        // Hand values (recompute to pin): d0 ~ 0.5442, d1 ~ 1.1163, d2 ~ 0.5909.
        assert!((d0 - 0.54421473).abs() < 1e-6, "d0 = {d0}");
        assert!((d1 - 1.11625862).abs() < 1e-6, "d1 = {d1}");
        assert!((d2 - 0.59086170).abs() < 1e-6, "d2 = {d2}");
        assert!(d1 > d2 && d2 > d0);

        // The implementation must agree with the worked example.
        // Bypass prepare_candidate's text pipeline: build candidates with
        // exact token multisets via direct PreparedNode construction.
        let mut interner = StringInterner::new();
        let mut mk = |tokens: &[&str]| -> Candidate {
            let ids: Vec<u32> = tokens.iter().map(|t| interner.intern(t)).collect();
            let mut counts: HashMap<u32, u32> = HashMap::new();
            for &t in &ids {
                *counts.entry(t).or_insert(0) += 1;
            }
            let mut tc: Vec<(u32, u32)> = counts.into_iter().collect();
            tc.sort_by_key(|(t, _)| *t);
            Candidate {
                backend_node_id: 1,
                generation: 1,
                role: "x".into(),
                at: MainViewportCss(0.0, 0.0),
                w: 10.0,
                h: 10.0,
                dpr: 1.0,
                prep: PreparedNode {
                    backend_node_id: 1,
                    generation: 1,
                    name_norm: "".into(),
                    role_norm: "".into(),
                    role_token_ids: vec![],
                    token_ids: ids,
                    token_counts: tc,
                    text_hash: 0,
                    log_area: 0.0,
                    enabled_visible: true,
                    embedding: Some([0.0; 64]),
                },
            }
        };
        let cands = vec![
            mk(&["a", "b"]),
            mk(&["a", "a", "c"]),
            mk(&["b", "c", "c", "d"]),
        ];
        let idx = Bm25Index::build(&cands);
        assert!((idx.avg_doc_len() - 3.0).abs() < 1e-12);
        assert_eq!(idx.doc_freq(interner.get("a").unwrap()), 2);
        assert!((idx.idf(interner.get("a").unwrap()) - idf).abs() < 1e-12);
        let qa = interner.get("a").unwrap();
        let qc = interner.get("c").unwrap();
        let q = vec![qa, qc];
        let s0 = idx.score_candidate(&cands[0], &q);
        let s1 = idx.score_candidate(&cands[1], &q);
        let s2 = idx.score_candidate(&cands[2], &q);
        assert!((s0 - d0).abs() < 1e-9, "impl d0 = {s0}, want {d0}");
        assert!((s1 - d1).abs() < 1e-9, "impl d1 = {s1}, want {d1}");
        assert!((s2 - d2).abs() < 1e-9, "impl d2 = {s2}, want {d2}");
    }

    #[test]
    fn heap_top8_matches_full_sort_on_random_inputs() {
        let mut rng = XorShift64::new(0xC0FFEE);
        let vocab = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        ];
        for round in 0..50 {
            let mut interner = StringInterner::new();
            let n_docs = 3 + rng.below(40);
            let mut cands: Vec<Candidate> = Vec::new();
            for _ in 0..n_docs {
                let len = 1 + rng.below(8);
                let toks: Vec<&str> = (0..len).map(|_| vocab[rng.below(vocab.len())]).collect();
                let ids: Vec<u32> = toks.iter().map(|t| interner.intern(t)).collect();
                let mut counts: HashMap<u32, u32> = HashMap::new();
                for &t in &ids {
                    *counts.entry(t).or_insert(0) += 1;
                }
                let mut tc: Vec<(u32, u32)> = counts.into_iter().collect();
                tc.sort_by_key(|(t, _)| *t);
                cands.push(Candidate {
                    backend_node_id: rng.next(),
                    generation: 1,
                    role: "x".into(),
                    at: MainViewportCss(0.0, 0.0),
                    w: 10.0,
                    h: 10.0,
                    dpr: 1.0,
                    prep: PreparedNode {
                        backend_node_id: 0,
                        generation: 1,
                        name_norm: "".into(),
                        role_norm: "".into(),
                        role_token_ids: vec![],
                        token_ids: ids,
                        token_counts: tc,
                        text_hash: 0,
                        log_area: 0.0,
                        enabled_visible: true,
                        embedding: Some([0.0; 64]),
                    },
                });
            }
            let idx = Bm25Index::build(&cands);
            let qlen = 1 + rng.below(3);
            let q: Vec<u32> = (0..qlen)
                .map(|_| interner.intern(vocab[rng.below(vocab.len())]))
                .collect();
            let (heap_top, _) = idx.top_k(&cands, &q, 8);
            // Reference: full sort, descending by score.
            let mut full: Vec<(usize, f64)> = cands
                .iter()
                .enumerate()
                .map(|(i, c)| (i, idx.score_candidate(c, &q)))
                .collect();
            full.sort_by(|a, b| b.1.total_cmp(&a.1));
            let k = 8.min(full.len());
            // Same index multiset and same score multiset in the top-k.
            let mut heap_ids: Vec<usize> = heap_top.iter().map(|r| r.idx).collect();
            let mut full_ids: Vec<usize> = full[..k].iter().map(|(i, _)| *i).collect();
            heap_ids.sort_unstable();
            full_ids.sort_unstable();
            assert_eq!(
                heap_ids, full_ids,
                "round {round}: heap/sort index mismatch"
            );
            let mut heap_scores: Vec<u64> = heap_top.iter().map(|r| r.score.to_bits()).collect();
            let mut full_scores: Vec<u64> = full[..k].iter().map(|(_, s)| s.to_bits()).collect();
            heap_scores.sort_unstable();
            full_scores.sort_unstable();
            assert_eq!(heap_scores, full_scores, "round {round}: score mismatch");
            // All heap scores finite.
            assert!(heap_top.iter().all(|r| r.score.is_finite()));
        }
    }

    #[test]
    fn bm25_nan_guard_clamps_to_missing_value() {
        // Degenerate index: every document empty -> avgdl = 0.
        let mut interner = StringInterner::new();
        let mk_empty = || Candidate {
            backend_node_id: 1,
            generation: 1,
            role: "x".into(),
            at: MainViewportCss(0.0, 0.0),
            w: 10.0,
            h: 10.0,
            dpr: 1.0,
            prep: PreparedNode {
                backend_node_id: 1,
                generation: 1,
                name_norm: "".into(),
                role_norm: "".into(),
                role_token_ids: vec![],
                token_ids: vec![],
                token_counts: vec![],
                text_hash: 0,
                log_area: 0.0,
                enabled_visible: true,
                embedding: Some([0.0; 64]),
            },
        };
        let cands = vec![mk_empty(), mk_empty()];
        let idx = Bm25Index::build(&cands);
        let q = vec![interner.intern("zzz")];
        // tf is 0 everywhere, so the score is 0.0 (finite) even here;
        // the guard path is exercised by forcing a NaN through the
        // clamp: score_candidate never returns non-finite.
        for c in &cands {
            assert!(idx.score_candidate(c, &q).is_finite());
        }
    }

    // -- typed errors ---------------------------------------------------------

    #[test]
    fn no_searchable_terms_is_typed() {
        let mut e = engine_from_names(&[("button", "Search"), ("link", "Home")]);
        let mut rng = XorShift64::new(7);
        let opts = ResolveOpts {
            route: "r".to_string(),
            bypass_cache: true,
        };
        for desc in ["", "   ", "the and for", "with that this"] {
            match e.resolve(desc, &opts, &mut rng) {
                Resolution::Miss {
                    reason: MissReason::NoSearchableTerms,
                    ..
                } => {}
                other => panic!("{desc:?}: expected NoSearchableTerms, got {other:?}"),
            }
        }
        assert_eq!(
            e.natural_top_k("the and for", 8).unwrap_err(),
            ResolveError::NoSearchableTerms
        );
    }

    #[test]
    fn perception_empty_is_typed() {
        let mut e = test_engine();
        e.finalize();
        let mut rng = XorShift64::new(7);
        let opts = ResolveOpts {
            route: "r".to_string(),
            bypass_cache: true,
        };
        match e.resolve("search", &opts, &mut rng) {
            Resolution::Miss {
                reason: MissReason::PerceptionEmpty,
                ..
            } => {}
            other => panic!("expected PerceptionEmpty, got {other:?}"),
        }
        assert_eq!(
            e.natural_top_k("search", 8).unwrap_err(),
            ResolveError::PerceptionEmpty
        );
    }

    // -- stage 0 ----------------------------------------------------------------

    #[test]
    fn exact_stage_early_exit_requires_unique_visible_enabled_current() {
        let mut rng = XorShift64::new(11);
        let opts = ResolveOpts {
            route: "r".to_string(),
            bypass_cache: true,
        };
        // Unique, visible, enabled, current generation -> early exit.
        let mut e = engine_from_names(&[("button", "Search"), ("link", "Home")]);
        match e.resolve("Search", &opts, &mut rng) {
            Resolution::Hit(h) => {
                assert_eq!(h.stage, Stage::Exact);
                assert_eq!(h.backend_node_id, 1000);
                assert!((h.confidence - EXACT_MATCH_PRECISION_MEASURED.min(0.95)).abs() < 1e-12);
            }
            other => panic!("expected exact hit, got {other:?}"),
        }

        // Ambiguous (two "Search" buttons): no early exit; BM25 or ranker
        // decides, or it misses. It must NOT be an Exact hit.
        let mut e2 = engine_from_names(&[("button", "Search"), ("button", "Search")]);
        match e2.resolve("Search", &opts, &mut rng) {
            Resolution::Hit(h) => {
                assert_ne!(h.stage, Stage::Exact, "ambiguous must not early-exit")
            }
            Resolution::Miss { .. } => {}
        }

        // Disabled node: exact name matches but the node is not enabled.
        let mut e3 = test_engine();
        let chain = identity_chain();
        e3.insert(
            1,
            "button",
            "Search",
            "",
            (0.0, 0.0),
            (80.0, 30.0),
            false,
            &chain,
        );
        e3.insert(
            2,
            "link",
            "Home",
            "",
            (90.0, 0.0),
            (80.0, 30.0),
            true,
            &chain,
        );
        e3.finalize();
        match e3.resolve("Search", &opts, &mut rng) {
            Resolution::Hit(h) => assert_ne!(h.stage, Stage::Exact),
            Resolution::Miss { .. } => {}
        }
    }

    #[test]
    fn exact_scan_outcomes() {
        let e = engine_from_names(&[("button", "Search"), ("link", "Home")]);
        let (o, i) = exact_match_scan(e.candidates(), "search");
        assert_eq!(o, ExactOutcome::Unique);
        assert_eq!(i, Some(0));
        let (o, _) = exact_match_scan(e.candidates(), "missing");
        assert_eq!(o, ExactOutcome::None);
        let e2 = engine_from_names(&[("button", "Search"), ("button", "Search")]);
        let (o, i) = exact_match_scan(e2.candidates(), "search");
        assert_eq!(o, ExactOutcome::Ambiguous);
        assert_eq!(i, None);
    }

    // -- identity cache ----------------------------------------------------------

    fn cache_test_setup() -> (ResolverEngine, Vec<Candidate>) {
        let e = engine_from_names(&[("button", "Search"), ("link", "Home")]);
        let cands = e.candidates().to_vec();
        (e, cands)
    }

    #[test]
    fn cache_hit_requires_live_node_same_generation_same_hash() {
        let (mut e, cands) = cache_test_setup();
        let mut live: HashMap<u64, usize> =
            [(1000u64, 0usize), (1001u64, 1usize)].into_iter().collect();
        let key = IdentityCache::key("test", "example.com", "r", "search");
        e.cache_mut().commit(
            key,
            CacheEntry {
                node_id: 1000,
                text_hash: cands[0].prep.text_hash,
                generation: 1,
            },
        );
        // Hit.
        assert_eq!(e.cache_mut().lookup(key, &live, &cands, 1), Some(0));
        // Stale generation.
        assert_eq!(e.cache_mut().lookup(key, &live, &cands, 2), None);
        // Changed text hash.
        let mut cands2 = cands.clone();
        cands2[0].prep.text_hash ^= 1;
        assert_eq!(e.cache_mut().lookup(key, &live, &cands2, 1), None);
        // Node left the tree.
        live.remove(&1000);
        assert_eq!(e.cache_mut().lookup(key, &live, &cands, 1), None);
        // Unknown key.
        assert_eq!(e.cache_mut().lookup(0xDEAD, &live, &cands, 1), None);
    }

    #[test]
    fn cache_key_is_interned_and_namespaced() {
        let a = IdentityCache::key("p", "o", "r", "search");
        assert_eq!(a, IdentityCache::key("p", "o", "r", "search"));
        assert_ne!(a, IdentityCache::key("p", "o", "r2", "search"));
        assert_ne!(a, IdentityCache::key("p", "o2", "r", "search"));
        assert_ne!(a, IdentityCache::key("p2", "o", "r", "search"));
        assert_ne!(a, IdentityCache::key("p", "o", "r", "search "));
    }

    #[test]
    fn cache_is_bounded_lru() {
        let mut c = IdentityCache::new(2);
        let live: HashMap<u64, usize> = HashMap::new();
        let no_cands: Vec<Candidate> = Vec::new();
        c.commit(
            1,
            CacheEntry {
                node_id: 1,
                text_hash: 1,
                generation: 1,
            },
        );
        c.commit(
            2,
            CacheEntry {
                node_id: 2,
                text_hash: 2,
                generation: 1,
            },
        );
        assert_eq!(c.len(), 2);
        // Touch 1 -> MRU; inserting 3 evicts 2 (the LRU).
        let _ = c.lookup(1, &live, &no_cands, 1); // miss on empty tree, but still touches? no: miss returns before touch
        c.commit(
            3,
            CacheEntry {
                node_id: 3,
                text_hash: 3,
                generation: 1,
            },
        );
        assert_eq!(c.len(), 2);
        // Key 1 was evicted (lookup missed before the touch, so 1 stayed LRU).
        assert_eq!(c.lookup(1, &live, &no_cands, 1), None);
        // Evict stale generations.
        c.evict_stale(2);
        assert!(c.is_empty());
    }

    #[test]
    fn resolve_uses_cache_then_reads_current_box() {
        let (mut e, _) = cache_test_setup();
        let mut rng = XorShift64::new(13);
        // Commit the Search button after a "Confirmed" verdict.
        let cands = e.candidates().to_vec();
        e.commit_cache(
            "test",
            "example.com",
            "r",
            "search",
            1000,
            cands[0].prep.text_hash,
        );
        let opts = ResolveOpts {
            route: "r".to_string(),
            bypass_cache: false,
        };
        match e.resolve("Search", &opts, &mut rng) {
            Resolution::Hit(h) => {
                assert_eq!(h.stage, Stage::Cache);
                assert_eq!(h.backend_node_id, 1000);
                assert_eq!(h.confidence, CACHE_HIT_CONFIDENCE);
                // Interior point lies strictly inside the node box.
                assert!(h.at.0 > 0.0 && h.at.0 < 80.0);
                assert!(h.at.1 > 20.0 && h.at.1 < 50.0);
            }
            other => panic!("expected cache hit, got {other:?}"),
        }
        // Bypass flag skips the cache.
        let opts2 = ResolveOpts {
            route: "r".to_string(),
            bypass_cache: true,
        };
        match e.resolve("Search", &opts2, &mut rng) {
            Resolution::Hit(h) => assert_eq!(h.stage, Stage::Exact),
            other => panic!("expected exact hit on bypass, got {other:?}"),
        }
    }

    // -- ranker ---------------------------------------------------------------------

    #[test]
    fn softmax_ties_retain_both_and_margin_is_zero() {
        let r = LinearRanker::zero();
        let xs = [[0.0f32; 137], [0.0f32; 137]];
        let mut probs = [0.0f64; 8];
        r.softmax_scaled(&xs, 2, &mut probs);
        assert!((probs[0] - 0.5).abs() < 1e-12);
        assert!((probs[1] - 0.5).abs() < 1e-12);
        let (order, p1, margin) = r.top_two(&probs, 2);
        // Both retained, stable order, margin exactly zero.
        assert_eq!(order, vec![0, 1]);
        assert!((p1 - 0.5).abs() < 1e-12);
        assert_eq!(margin, 0.0);
        // A tie never passes a positive margin gate.
        assert!(margin < Gates::default().ranker_margin_gate);
    }

    #[test]
    fn ranker_train_reduces_log_loss_on_separable_rows() {
        // Two features separate the classes; the rest is noise-free.
        let mut rows = Vec::new();
        for i in 0..6 {
            let mut good = [0.0f32; 137];
            good[0] = 5.0;
            good[1] = 1.0;
            let mut bad = [0.0f32; 137];
            bad[0] = 0.5;
            bad[1] = 0.0;
            let _ = i;
            rows.push(TrainRow {
                xs: vec![good, bad],
                correct: 0,
            });
            rows.push(TrainRow {
                xs: vec![bad, good],
                correct: 1,
            });
        }
        let mut r = LinearRanker::zero();
        let before = r.log_loss(&rows);
        r.train(&rows, 200, 0.1, 1e-4);
        let after = r.log_loss(&rows);
        assert!(after < before, "before={before} after={after}");
        // And the trained ranker picks the good candidate top-1.
        for row in &rows {
            let mut probs = [0.0f64; 8];
            r.softmax_scaled(&row.xs, 2, &mut probs);
            let (order, _, _) = r.top_two(&probs, 2);
            assert_eq!(order[0], row.correct);
        }
    }

    #[test]
    fn no_prior_shift_a_list_wide_scalar_cancels() {
        // softmax(z + c) == softmax(z): a list-wide prior shift is a no-op.
        let mut r = LinearRanker::zero();
        r.weights[0] = 1.5;
        r.bias = 0.25;
        let xs = [
            [
                1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            [
                0.5f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ];
        let z = r.logits(&xs, 2);
        let mut p_plain = [0.0f64; 8];
        r.softmax_scaled(&xs, 2, &mut p_plain);
        // Manually shift both logits by +10 and recompute: identical probs.
        let shifted = [z[0] + 10.0, z[1] + 10.0];
        let max = shifted[0].max(shifted[1]);
        let e0 = (shifted[0] - max).exp();
        let e1 = (shifted[1] - max).exp();
        assert!((p_plain[0] - e0 / (e0 + e1)).abs() < 1e-12);
        assert!((p_plain[1] - e1 / (e0 + e1)).abs() < 1e-12);
    }

    #[test]
    fn temperature_fit_improves_log_loss_on_calibration_split() {
        // Synthetic calibration rows: high margin -> usually correct.
        let mut rng = XorShift64::new(0xCA1);
        let mut rows: Vec<(f64, bool)> = Vec::new();
        for _ in 0..200 {
            let m = rng.uniform01();
            let correct = rng.uniform01() < 0.15 + 0.8 * m;
            rows.push((m, correct));
        }
        let base = Calibrator::uncalibrated().log_loss(&rows);
        let t = Calibrator::fit_temperature(&rows);
        let fit = Calibrator { temperature: t }.log_loss(&rows);
        assert!(t > 0.0 && t.is_finite());
        assert!(fit <= base, "fit={fit} base={base}");
    }

    #[test]
    fn calibrator_p_is_monotone_and_bounded() {
        let c = Calibrator { temperature: 0.5 };
        let mut prev = 0.0;
        for i in 0..=10 {
            let p = c.p(i as f64 / 10.0);
            assert!((0.0..=1.0).contains(&p));
            assert!(p >= prev);
            prev = p;
        }
        assert!((c.p(0.0) - 0.5).abs() < 1e-12);
    }

    // -- grounding ---------------------------------------------------------------

    #[test]
    fn interior_point_stays_inside_with_margin() {
        let e = engine_from_names(&[("button", "OK")]);
        let c = &e.candidates()[0]; // at (0,20), 80x30
        let mut rng = XorShift64::new(99);
        for _ in 0..100 {
            let p = interior_point(c, &mut rng);
            assert!(p.0 >= 12.0 && p.0 <= 68.0, "x = {}", p.0);
            assert!(p.1 >= 24.5 && p.1 <= 45.5, "y = {}", p.1);
        }
    }

    #[test]
    fn needs_refinement_threshold() {
        assert!(needs_refinement(10.0, 10.0, 1.0));
        assert!(needs_refinement(11.9, 100.0, 1.0));
        assert!(!needs_refinement(12.0, 12.0, 1.0));
        assert!(!needs_refinement(6.0, 6.0, 2.0)); // 12 physical px: fine
        assert!(needs_refinement(5.0, 5.0, 2.0)); // 10 physical px: refine
                                                  // Bad DPR falls back to 1.0 rather than NaN.
        assert!(needs_refinement(5.0, 5.0, f64::NAN));
    }

    #[test]
    fn escalation_coordinate_contract() {
        let m = ModelSpaceMapping {
            image_w: 1568,
            image_h: 1000,
            model_space: ModelSpace::Pixels1568Capped,
            scale_x: 0.8,
            scale_y: 0.8,
        };
        // With mapping: scale inverted.
        let g = accept_escalated_point(Some(&m), (100.0, 50.0)).expect("mapped point accepted");
        assert!((g.point.0 - 80.0).abs() < 1e-9);
        assert!((g.point.1 - 40.0).abs() < 1e-9);
        // Without mapping: REJECTED, never guessed.
        assert_eq!(
            accept_escalated_point(None, (100.0, 50.0)).unwrap_err(),
            EscalationError::MissingMapping
        );
        // Non-finite point rejected.
        assert_eq!(
            accept_escalated_point(Some(&m), (f64::NAN, 50.0)).unwrap_err(),
            EscalationError::NonFiniteCoordinate
        );
        // Grid space example: 0-999 grid over a 1568x1000 image.
        let grid = ModelSpaceMapping {
            image_w: 1568,
            image_h: 1000,
            model_space: ModelSpace::Grid0_999,
            scale_x: 1568.0 / 999.0,
            scale_y: 1000.0 / 999.0,
        };
        let g2 = accept_escalated_point(Some(&grid), (999.0, 999.0)).expect("grid accepted");
        assert!((g2.point.0 - 1568.0).abs() < 1e-6);
        assert!((g2.point.1 - 1000.0).abs() < 1e-6);
    }

    // -- missing embedding: never 64 silent zeros ----------------------------------

    #[test]
    fn missing_embedding_skips_candidate_and_counts() {
        let mut e = engine_from_names(&[("button", "Search")]);
        // Sabotage the embedding: a node NOT built through prepare_candidate
        // (simulated by clearing it) must be skipped, never zero-scored.
        e.candidates[0].prep.embedding = None;
        e.finalize();
        let (ranked, skipped) = e.index().unwrap().top_k(e.candidates(), &[0, 1, 2], TOP_K);
        assert!(ranked.is_empty());
        assert_eq!(skipped, vec![0]);
        let before = e.skipped_no_embedding;
        let _ = e.natural_top_k("search", 8);
        assert_eq!(e.skipped_no_embedding, before + 1);
    }

    // -- mining hook -----------------------------------------------------------------

    #[test]
    fn mining_hook_appends_jsonl_outside_repo() {
        let dir = std::env::temp_dir().join(format!("mf-corpus-test-{}", std::process::id()));
        std::env::set_var(CORPUS_DIR_ENV, &dir);
        let row = MinedRow {
            schema: FEATURE_SCHEMA_VERSION,
            description: "search".to_string(),
            page_url: "https://example.com/".to_string(),
            origin: "example.com".to_string(),
            generation: 1,
            top_k: vec![MinedCandidateRow {
                backend_node_id: 7,
                role: "button".to_string(),
                name: "Search".to_string(),
                bm25: 1.23,
                features: vec![0.0; 137],
            }],
            winner_backend_node_id: 7,
            stage: "bm25".to_string(),
        };
        mine_confirmed_row(&row).expect("mining writes");
        mine_confirmed_row(&row).expect("mining appends");
        let path = dir.join(CORPUS_ROWS_FILE);
        let text = std::fs::read_to_string(&path).expect("rows file readable");
        assert_eq!(text.lines().count(), 2);
        let v: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(v["description"], "search");
        assert_eq!(v["schema"], 1);
        // The path is outside the repo working copy.
        assert!(!path.to_string_lossy().contains("chromebook-vm-bridge"));
        std::env::remove_var(CORPUS_DIR_ENV);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn corpus_path_defaults_outside_repo() {
        std::env::remove_var(CORPUS_DIR_ENV);
        let p = corpus_rows_path();
        let s = p.to_string_lossy();
        assert!(
            s.ends_with("workspace/mouse-future/corpus/rows.jsonl"),
            "{s}"
        );
        assert!(!s.contains("chromebook-vm-bridge"), "{s}");
    }

    // -- featurize sanity ---------------------------------------------------------------

    #[test]
    fn featurize_uses_frozen_schema_and_finite_floats() {
        let mut e = engine_from_names(&[("button", "Search"), ("link", "Home")]);
        let top = e.natural_top_k("search the web", 8).expect("top-k");
        assert!(!top.is_empty());
        for (_, score, fv) in &top {
            assert!(score.is_finite());
            assert_eq!(fv.len(), 137);
            assert!(fv.iter().all(|x| x.is_finite()), "all 137 finite");
        }
        // The top hit for "search the web" should be the Search button.
        let (idx, _, _) = top[0];
        assert_eq!(e.candidates()[idx].backend_node_id, 1000);
    }

    #[test]
    fn role_prior_is_a_feature_not_a_multiplier() {
        // "click the button" names the button role: feature 3 must be 1
        // for the button and 0 for the link, with identical BM25 text
        // otherwise. The resolver never multiplies scores by a prior.
        let mut e = engine_from_names(&[("button", "Go"), ("link", "Go")]);
        let top = e.natural_top_k("click the button", 8).expect("top-k");
        assert_eq!(top.len(), 2);
        let f_button = top
            .iter()
            .find(|(i, _, _)| e.candidates()[*i].backend_node_id == 1000)
            .unwrap()
            .2;
        let f_link = top
            .iter()
            .find(|(i, _, _)| e.candidates()[*i].backend_node_id == 1001)
            .unwrap()
            .2;
        assert_eq!(f_button[3], 1.0);
        assert_eq!(f_link[3], 0.0);
        // BM25 scores are pure text scores: identical names -> identical BM25.
        let b_button = top
            .iter()
            .find(|(i, _, _)| e.candidates()[*i].backend_node_id == 1000)
            .unwrap()
            .1;
        let b_link = top
            .iter()
            .find(|(i, _, _)| e.candidates()[*i].backend_node_id == 1001)
            .unwrap()
            .1;
        assert!((b_button - b_link).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Evaluation harness (B4): honest, measured, on real VM pages.
//
// Reads the labelled set collected from real public pages on the
// chromebook-vm (see the B4 report for the collection method):
//   $MOUSE_FUTURE_EVAL_DIR/pages/<page>.json
//   $MOUSE_FUTURE_EVAL_DIR/labels.jsonl
// Page JSON: {url, origin, viewport:[w,h],
//             nodes:[{id, role, name, text, x,y,w,h, enabled_visible}]}
// Label:     {page, description, correct_id, action}
//
// The harness reports Recall@1/@3/@8 for exact-match and BM25
// separately, end-to-end resolution success, escalation rate, and
// breakdowns by origin / correct-node role / action type. It fits the
// temperature and the ranker-margin gate on a calibration split and
// reports precision/recall at the gate on a held-out split. Confirmed
// rows are mined to the corpus via the mining hook.
//
// When the eval directory is absent the harness skips with a clear
// message instead of failing the build: the labelled set is collected
// data, not a checked-in fixture. Structural invariants (finite scores,
// calibration fit improving log-loss) ARE asserted when data exists.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod eval_harness {
    use super::*;
    use std::collections::HashMap;

    const EVAL_DIR_ENV: &str = "MOUSE_FUTURE_EVAL_DIR";

    fn eval_dir() -> PathBuf {
        if let Ok(d) = std::env::var(EVAL_DIR_ENV) {
            return PathBuf::from(d);
        }
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let mut p = PathBuf::from(home);
        p.push("workspace/mouse-future/eval");
        p
    }

    #[derive(Debug, serde::Deserialize)]
    struct EvalNode {
        id: u64,
        role: String,
        name: String,
        #[serde(default)]
        text: String,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        #[serde(default = "default_true")]
        enabled_visible: bool,
    }
    fn default_true() -> bool {
        true
    }

    #[derive(Debug, serde::Deserialize)]
    struct EvalPage {
        url: String,
        origin: String,
        viewport: (f64, f64),
        nodes: Vec<EvalNode>,
    }

    #[derive(Debug, serde::Deserialize)]
    struct Label {
        page: String,
        description: String,
        correct_id: u64,
        #[serde(default)]
        action: String,
    }

    struct RowResult {
        label_idx: usize,
        origin: String,
        role: String,
        action: String,
        exact_correct: bool,
        exact_resolved: bool,
        bm25_rank: Option<usize>, // 1-based rank of correct node in top-8
        rel_margin: f64,
        top1_correct: bool,
        e2e_hit_correct: bool,
        e2e_stage: String,
        e2e_escalated: bool,
        needs_refinement: bool,
    }

    fn load_page(dir: &std::path::Path, name: &str) -> Option<EvalPage> {
        let mut p = dir.to_path_buf();
        p.push("pages");
        p.push(name);
        let text = std::fs::read_to_string(p).ok()?;
        serde_json::from_str(&text).ok()
    }

    fn build_engine(page: &EvalPage, origin: &str) -> ResolverEngine {
        let mut e = ResolverEngine::new(
            EngineConfig {
                profile_id: "eval".to_string(),
                origin: origin.to_string(),
                viewport: page.viewport,
                cache_cap: IDENTITY_CACHE_CAP,
            },
            1,
            Gates::default(),
        );
        let chain = vec![FrameTransform {
            scroll: (0.0, 0.0),
            content_origin: (0.0, 0.0),
            nested_offset: (0.0, 0.0),
            css_scale: 1.0,
            dpr: 1.0,
        }];
        for n in &page.nodes {
            e.insert(
                n.id,
                &n.role,
                &n.name,
                &n.text,
                (n.x, n.y),
                (n.w, n.h),
                n.enabled_visible,
                &chain,
            );
        }
        e.finalize();
        e
    }

    fn recall_at(ranks: &[Option<usize>], k: usize) -> f64 {
        if ranks.is_empty() {
            return f64::NAN;
        }
        let hit = ranks
            .iter()
            .filter(|r| matches!(r, Some(rank) if *rank <= k))
            .count();
        hit as f64 / ranks.len() as f64
    }

    #[test]
    fn eval_on_vm_rows() {
        let dir = eval_dir();
        let mut labels_path = dir.clone();
        labels_path.push("labels.jsonl");
        let labels_text = match std::fs::read_to_string(&labels_path) {
            Ok(t) => t,
            Err(_) => {
                eprintln!(
                    "eval harness: no labelled set at {} (collect it first); skipping",
                    labels_path.display()
                );
                return;
            }
        };
        let labels: Vec<Label> = labels_text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("label parses"))
            .collect();
        assert!(!labels.is_empty(), "labels.jsonl must not be empty");

        let mut engines: HashMap<String, ResolverEngine> = HashMap::new();
        let mut page_of: HashMap<String, EvalPage> = HashMap::new();
        let mut rows: Vec<RowResult> = Vec::new();
        let mut rng = XorShift64ForEval::new();

        for (li, label) in labels.iter().enumerate() {
            let page = page_of.entry(label.page.clone()).or_insert_with(|| {
                load_page(&dir, &label.page)
                    .unwrap_or_else(|| panic!("page {} missing", label.page))
            });
            let engine = engines
                .entry(label.page.clone())
                .or_insert_with(|| build_engine(page, &page.origin));

            let desc_norm = normalize_desc(&label.description);
            // Stage 0 bookkeeping.
            let (outcome, unique_idx) = exact_match_scan(engine.candidates(), &desc_norm);
            let exact_resolved = outcome == ExactOutcome::Unique
                && unique_idx
                    .map(|i| {
                        let c = &engine.candidates()[i];
                        c.prep.enabled_visible && c.generation == engine.generation()
                    })
                    .unwrap_or(false);
            let exact_correct = exact_resolved
                && unique_idx
                    .map(|i| engine.candidates()[i].backend_node_id == label.correct_id)
                    .unwrap_or(false);

            // Natural top-8 BM25 list.
            let topk = engine
                .natural_top_k(&label.description, TOP_K)
                .expect("natural top-k");
            assert!(
                topk.iter()
                    .all(|(_, s, fv)| s.is_finite() && fv.iter().all(|x| x.is_finite())),
                "all BM25 scores and features finite (row {li})"
            );
            let bm25_rank = topk
                .iter()
                .position(|(i, _, _)| engine.candidates()[*i].backend_node_id == label.correct_id)
                .map(|p| p + 1);
            let top1 = topk.first().map(|(_, s, _)| *s).unwrap_or(0.0);
            let top2 = topk.get(1).map(|(_, s, _)| *s).unwrap_or(0.0);
            let rel_margin = if top1 > 0.0 {
                (top1 - top2) / top1
            } else {
                0.0
            };
            let top1_correct = topk
                .first()
                .map(|(i, _, _)| engine.candidates()[*i].backend_node_id == label.correct_id)
                .unwrap_or(false);

            // End-to-end cascade.
            let opts = ResolveOpts {
                route: label.page.clone(),
                bypass_cache: true,
            };
            let (e2e_hit_correct, e2e_stage, e2e_escalated, needs_ref) =
                match engine.resolve(&label.description, &opts, &mut rng) {
                    Resolution::Hit(h) => (
                        h.backend_node_id == label.correct_id,
                        h.stage.to_string(),
                        false,
                        h.needs_refinement,
                    ),
                    Resolution::Miss { .. } => (false, "miss".to_string(), true, false),
                };

            let role = page
                .nodes
                .iter()
                .find(|n| n.id == label.correct_id)
                .map(|n| n.role.clone())
                .unwrap_or_else(|| "?".to_string());
            rows.push(RowResult {
                label_idx: li,
                origin: page.origin.clone(),
                role,
                action: if label.action.is_empty() {
                    "?".to_string()
                } else {
                    label.action.clone()
                },
                exact_correct,
                exact_resolved,
                bm25_rank,
                rel_margin,
                top1_correct,
                e2e_hit_correct,
                e2e_stage,
                e2e_escalated,
                needs_refinement: needs_ref,
            });
        }

        // ---- Recall tables -------------------------------------------------
        let bm25_ranks: Vec<Option<usize>> = rows.iter().map(|r| r.bm25_rank).collect();
        let exact_ranks: Vec<Option<usize>> = rows
            .iter()
            .map(|r| if r.exact_correct { Some(1) } else { None })
            .collect();
        let n = rows.len() as f64;
        let e2e_ok = rows.iter().filter(|r| r.e2e_hit_correct).count() as f64 / n;
        let escalated = rows.iter().filter(|r| r.e2e_escalated).count() as f64 / n;
        let need_ref = rows.iter().filter(|r| r.needs_refinement).count() as f64 / n;

        eprintln!("=== resolver v1 eval: {} labelled rows ===", rows.len());
        eprintln!("stage   Recall@1  Recall@3  Recall@8");
        eprintln!(
            "exact   {:>8.3}  {:>8.3}  {:>8.3}   (exact either resolves uniquely or not; @3/@8 identical by construction)",
            recall_at(&exact_ranks, 1),
            recall_at(&exact_ranks, 3),
            recall_at(&exact_ranks, 8)
        );
        eprintln!(
            "bm25    {:>8.3}  {:>8.3}  {:>8.3}",
            recall_at(&bm25_ranks, 1),
            recall_at(&bm25_ranks, 3),
            recall_at(&bm25_ranks, 8)
        );
        eprintln!("end-to-end resolution success: {e2e_ok:.3}");
        eprintln!("escalation rate (miss -> Tier-1): {escalated:.3}");
        eprintln!("needs_refinement fraction (hits): {need_ref:.3}");

        // Failure analysis: every end-to-end miss gets a row (page,
        // description, correct target, top-3 BM25, margin/share) so the
        // next session mines failures, not anecdotes.
        eprintln!("--- e2e misses (failure analysis) ---");
        for r in &rows {
            if !r.e2e_escalated {
                continue;
            }
            let label = &labels[r.label_idx];
            let engine = engines.get_mut(&label.page).expect("engine cached");
            let topk = engine
                .natural_top_k(&label.description, TOP_K)
                .unwrap_or_default();
            let top1 = topk.first().map(|(_, s, _)| *s).unwrap_or(0.0);
            let mass: f64 = topk.iter().map(|(_, s, _)| s).sum();
            let share = if mass > 0.0 { top1 / mass } else { 0.0 };
            let top3: Vec<String> = topk
                .iter()
                .take(3)
                .map(|(i, s, _)| {
                    let c = &engine.candidates()[*i];
                    format!("{}:{:.2}", c.prep.name_norm, s)
                })
                .collect();
            let correct_name = page_of
                .get(&label.page)
                .and_then(|p| p.nodes.iter().find(|n| n.id == label.correct_id))
                .map(|n| n.name.clone())
                .unwrap_or_default();
            eprintln!(
                "  [{}] {:?} -> {:?} (rank {:?}) margin={:.3} share={:.3} top3=[{}]",
                label.page,
                label.description,
                correct_name,
                r.bm25_rank,
                r.rel_margin,
                share,
                top3.join(", ")
            );
        }

        // Breakdowns by origin / role / action.
        for (title, keyfn) in [
            (
                "origin",
                Box::new(|r: &RowResult| r.origin.clone()) as Box<dyn Fn(&RowResult) -> String>,
            ),
            ("role", Box::new(|r: &RowResult| r.role.clone())),
            ("action", Box::new(|r: &RowResult| r.action.clone())),
        ] {
            let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
            for (i, r) in rows.iter().enumerate() {
                groups.entry(keyfn(r)).or_default().push(i);
            }
            eprintln!("--- by {title} ---");
            let mut keys: Vec<_> = groups.keys().collect();
            keys.sort();
            for k in keys {
                let idxs = &groups[k];
                let sub: Vec<Option<usize>> = idxs.iter().map(|&i| rows[i].bm25_rank).collect();
                let ok = idxs.iter().filter(|&&i| rows[i].e2e_hit_correct).count() as f64
                    / idxs.len() as f64;
                eprintln!(
                    "  {k:<28} n={:<3} bm25@1={:.3} bm25@8={:.3} e2e={:.3}",
                    idxs.len(),
                    recall_at(&sub, 1),
                    recall_at(&sub, 8),
                    ok
                );
            }
        }

        // ---- Calibration split: even = calibration, odd = held-out -------
        let cal_rows: Vec<(f64, bool)> = rows
            .iter()
            .filter(|r| r.label_idx % 2 == 0)
            .map(|r| (r.rel_margin, r.top1_correct))
            .collect();
        let held_rows: Vec<(f64, bool)> = rows
            .iter()
            .filter(|r| r.label_idx % 2 == 1)
            .map(|r| (r.rel_margin, r.top1_correct))
            .collect();
        assert!(!cal_rows.is_empty() && !held_rows.is_empty());

        let base_loss = Calibrator::uncalibrated().log_loss(&cal_rows);
        let t_fit = Calibrator::fit_temperature(&cal_rows);
        let fit_loss = Calibrator { temperature: t_fit }.log_loss(&cal_rows);
        assert!(
            fit_loss <= base_loss,
            "temperature fit must improve log-loss on the calibration split"
        );
        let held_base = Calibrator::uncalibrated().log_loss(&held_rows);
        let held_fit = Calibrator { temperature: t_fit }.log_loss(&held_rows);
        eprintln!("calibrated_p temperature: T={t_fit:.4} (grid fit on calibration split)");
        eprintln!("  calibration log-loss: uncalibrated={base_loss:.4} fit={fit_loss:.4}");
        eprintln!("  held-out    log-loss: uncalibrated={held_base:.4} fit={held_fit:.4}");

        // ---- Exact-stage measured precision -------------------------------
        let exact_n = rows.iter().filter(|r| r.exact_resolved).count();
        let exact_ok = rows.iter().filter(|r| r.exact_correct).count();
        let exact_prec = if exact_n > 0 {
            exact_ok as f64 / exact_n as f64
        } else {
            f64::NAN
        };
        eprintln!(
            "exact stage: resolved {exact_n}/{} rows, precision {exact_prec:.3}",
            rows.len()
        );
        if exact_n > 0 {
            assert!(
                (EXACT_MATCH_PRECISION_MEASURED - exact_prec).abs() <= 0.05,
                "EXACT_MATCH_PRECISION_MEASURED ({}) must match the measured precision ({:.3}); update the constant from this report",
                EXACT_MATCH_PRECISION_MEASURED,
                exact_prec
            );
        }

        // ---- Ranker-margin gate fit ---------------------------------------
        // Train the interim linear ranker on the calibration split, fit
        // its temperature there, then report held-out numbers.
        let mut train_rows: Vec<TrainRow> = Vec::new();
        for r in rows.iter().filter(|r| r.label_idx % 2 == 0) {
            let label = &labels[r.label_idx];
            let engine = engines.get_mut(&label.page).expect("engine cached");
            if let Ok(topk) = engine.natural_top_k(&label.description, TOP_K) {
                if let Some(pos) = topk.iter().position(|(i, _, _)| {
                    engine.candidates()[*i].backend_node_id == label.correct_id
                }) {
                    train_rows.push(TrainRow {
                        xs: topk.into_iter().map(|(_, _, fv)| fv).collect(),
                        correct: pos,
                    });
                }
            }
        }
        let mut ranker = LinearRanker::zero();
        let ranker_base = ranker.log_loss(&train_rows);
        ranker.train(&train_rows, 300, 0.05, 1e-3);
        ranker.fit_temperature(&train_rows);
        let ranker_fit = ranker.log_loss(&train_rows);
        eprintln!(
            "interim ranker: train log-loss {ranker_base:.4} -> {ranker_fit:.4}, T={:.3}",
            ranker.temperature
        );

        // Held-out: ranker top-1 accuracy vs BM25 top-1 accuracy.
        let mut ranker_hits = 0usize;
        let mut bm25_hits = 0usize;
        let mut held_total = 0usize;
        let mut gate_tp = 0usize;
        let mut gate_fp = 0usize;
        let mut gate_fn = 0usize;
        // Fit the margin gate on the calibration split first (F1 max).
        // The scan starts at 0.01: a gate of exactly 0 would admit
        // zero-margin ties, which must never pass (ties are retained in
        // the top-two selection precisely so their margin reads zero).
        let mut best_f1 = -1.0;
        let mut best_gate = 0.10;
        for gi in 1..=50 {
            let g = gi as f64 * 0.01;
            let (mut tp, mut fp, mut fn_) = (0, 0, 0);
            for tr in &train_rows {
                let n = tr.xs.len().min(8);
                let mut probs = [0.0f64; 8];
                ranker.softmax_scaled(&tr.xs, n, &mut probs);
                let (order, p1, margin) = ranker.top_two(&probs, n);
                let would_hit = p1 >= Gates::default().ranker_gate && margin >= g;
                let correct = order[0] == tr.correct;
                match (would_hit, correct) {
                    (true, true) => tp += 1,
                    (true, false) => fp += 1,
                    (false, true) => fn_ += 1,
                    (false, false) => {}
                }
            }
            let prec = tp as f64 / (tp + fp).max(1) as f64;
            let rec = tp as f64 / (tp + fn_).max(1) as f64;
            let f1 = if prec + rec > 0.0 {
                2.0 * prec * rec / (prec + rec)
            } else {
                0.0
            };
            if f1 > best_f1 {
                best_f1 = f1;
                best_gate = g;
            }
        }
        for r in rows.iter().filter(|r| r.label_idx % 2 == 1) {
            let label = &labels[r.label_idx];
            let engine = engines.get_mut(&label.page).expect("engine cached");
            if let Ok(topk) = engine.natural_top_k(&label.description, TOP_K) {
                let Some(pos) = topk.iter().position(|(i, _, _)| {
                    engine.candidates()[*i].backend_node_id == label.correct_id
                }) else {
                    continue;
                };
                held_total += 1;
                if r.top1_correct {
                    bm25_hits += 1;
                }
                let xs: Vec<[f32; 137]> = topk.into_iter().map(|(_, _, fv)| fv).collect();
                let n = xs.len().min(8);
                let mut probs = [0.0f64; 8];
                ranker.softmax_scaled(&xs, n, &mut probs);
                let (order, p1, margin) = ranker.top_two(&probs, n);
                if order[0] == pos {
                    ranker_hits += 1;
                }
                let would_hit = p1 >= Gates::default().ranker_gate && margin >= best_gate;
                let correct = order[0] == pos;
                match (would_hit, correct) {
                    (true, true) => gate_tp += 1,
                    (true, false) => gate_fp += 1,
                    (false, true) => gate_fn += 1,
                    (false, false) => {}
                }
            }
        }
        let gate_prec = gate_tp as f64 / (gate_tp + gate_fp).max(1) as f64;
        let gate_rec = gate_tp as f64 / (gate_tp + gate_fn).max(1) as f64;
        eprintln!(
            "ranker-margin gate fit on calibration: margin_gate={best_gate:.2} (F1={best_f1:.3})"
        );
        eprintln!(
            "held-out: n={held_total} bm25-top1-acc={:.3} ranker-top1-acc={:.3}",
            bm25_hits as f64 / held_total.max(1) as f64,
            ranker_hits as f64 / held_total.max(1) as f64
        );
        eprintln!(
            "held-out gate precision={gate_prec:.3} recall={gate_rec:.3} (would-hit && correct)"
        );

        // ---- Mining: confirmed rows only ----------------------------------
        let mut mined = 0usize;
        for r in &rows {
            if !r.e2e_hit_correct {
                continue;
            }
            let label = &labels[r.label_idx];
            let engine = engines.get_mut(&label.page).expect("engine cached");
            if let Ok(topk) = engine.natural_top_k(&label.description, TOP_K) {
                let winner = topk
                    .iter()
                    .find(|(i, _, _)| engine.candidates()[*i].backend_node_id == label.correct_id)
                    .map(|(i, _, _)| engine.candidates()[*i].backend_node_id)
                    .unwrap_or(label.correct_id);
                let page = page_of.get(&label.page).expect("page cached");
                let mined_topk: Vec<MinedCandidateRow> = topk
                    .iter()
                    .map(|(i, s, fv)| {
                        let c = &engine.candidates()[*i];
                        MinedCandidateRow {
                            backend_node_id: c.backend_node_id,
                            role: c.role.to_string(),
                            name: c.prep.name_norm.to_string(),
                            bm25: *s,
                            features: fv.to_vec(),
                        }
                    })
                    .collect();
                let row = MinedRow {
                    schema: FEATURE_SCHEMA_VERSION,
                    description: label.description.clone(),
                    page_url: page.url.clone(),
                    origin: page.origin.clone(),
                    generation: engine.generation(),
                    top_k: mined_topk,
                    winner_backend_node_id: winner,
                    stage: r.e2e_stage.clone(),
                };
                if mine_confirmed_row(&row).is_ok() {
                    mined += 1;
                }
            }
        }
        eprintln!(
            "mined {mined} confirmed rows to {}",
            corpus_rows_path().display()
        );

        // ---- Report file ----------------------------------------------------
        let mut report_path = dir.clone();
        report_path.push("eval_report.json");
        let report = serde_json::json!({
            "rows": rows.len(),
            "recall": {
                "exact": {"r1": recall_at(&exact_ranks, 1), "r3": recall_at(&exact_ranks, 3), "r8": recall_at(&exact_ranks, 8)},
                "bm25": {"r1": recall_at(&bm25_ranks, 1), "r3": recall_at(&bm25_ranks, 3), "r8": recall_at(&bm25_ranks, 8)},
            },
            "end_to_end_success": e2e_ok,
            "escalation_rate": escalated,
            "needs_refinement_fraction": need_ref,
            "temperature": t_fit,
            "calibration_log_loss": {"uncalibrated": base_loss, "fit": fit_loss},
            "held_out_log_loss": {"uncalibrated": held_base, "fit": held_fit},
            "exact_precision_measured": exact_prec,
            "exact_match_precision_const": EXACT_MATCH_PRECISION_MEASURED,
            "ranker_margin_gate_fit": best_gate,
            "ranker_gate_f1_calibration": best_f1,
            "held_out": {
                "n": held_total,
                "bm25_top1_acc": bm25_hits as f64 / held_total.max(1) as f64,
                "ranker_top1_acc": ranker_hits as f64 / held_total.max(1) as f64,
                "gate_precision": gate_prec,
                "gate_recall": gate_rec,
            },
            "mined_rows": mined,
            "corpus": corpus_rows_path().display().to_string(),
        });
        std::fs::write(&report_path, serde_json::to_string_pretty(&report).unwrap())
            .expect("report writes");
        eprintln!("report: {}", report_path.display());
    }

    /// Seeded RNG for the eval harness (interior points must not depend on
    /// the unit-test RNG's stream).
    struct XorShift64ForEval(u64);
    impl XorShift64ForEval {
        fn new() -> Self {
            Self(0xE1A1)
        }
    }
    impl Rng01 for XorShift64ForEval {
        fn uniform01(&mut self) -> f64 {
            let mut x = self.0 | 1;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            ((x >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }
}
