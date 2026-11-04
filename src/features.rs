//! Frozen feature schema v1 for the resolution cascade (mouse-future Phase 0).
//!
//! The 137-dim vector is laid out as:
//!
//! ```text
//! [0]    BM25 score, finite; non-finite -> -10.0, clamped to >= -10.0
//! [1]    Jaccard token overlap between description and candidate, [0,1]
//! [2]    edit similarity = 1 - levenshtein/max_len on normalized text, [0,1]
//! [3]    role-intent flag: description names this candidate role, 0/1
//! [4]    exact-name flag: normalized description == normalized name, 0/1
//! [5]    log area: ln(1+w*h) / ln(1+4_000_000), [0,1]
//! [6]    position prior: 1 - distance(viewport center, candidate center)
//!        / half viewport diagonal, [0,1]
//! [7]    enabled and visible from AX state, 0/1
//! [8..72]   64-dim signed hashed char-trigram embedding of the description
//! [72..136] 64-dim signed hashed char-trigram embedding of the candidate
//! [136]  dot(desc_emb, cand_emb)
//! ```
//!
//! Signed hashing contract (pinned by the golden pairs below):
//! text is lowercased with collapsed whitespace; unigrams, bigrams, and
//! trigrams are taken over its chars (short strings keep their unigrams and
//! bigrams, so "OK" is not the zero vector). Each n-gram unit is hashed
//! with the FxHash-like u64: h = 0, then per UTF-8 byte
//! `h = ((h rotated left 5) XOR byte) * 0x517cc1b727220a95` (wrapping).
//! `index = (h >> 1) % 64`, `sign = +1` when `h & 1 == 0` else `-1`.
//! The signed count accumulates in i32 at `index` (exact for integer
//! counts); the L2 norm is computed in f64 and the 64-vector is converted
//! to f32 once at the end (a zero vector stays zero).
//!
//! Any change to this layout bumps FEATURE_SCHEMA_VERSION and rebuilds the
//! golden pairs. The engine refuses to start on a schema mismatch.
//! No `unwrap` on caller/attacker-influenced input; finite guards on all
//! floats; the O(n*m) edit distance is capped (long strings read as
//! missing, feature value 0).

use std::collections::HashSet;

/// Frozen feature schema version. Bump on ANY layout change.
pub const FEATURE_SCHEMA_VERSION: u32 = 1;
/// Total feature dimension.
pub const FEATURE_DIM: usize = 137;
/// Signed trigram embedding dimension.
pub const EMBED_DIM: usize = 64;
/// FxHash-64 multiply constant used by the signed-hashing contract.
const FX_MULT: u64 = 0x51_7c_c1_b7_27_22_0a_95;
/// Area normalization: feature [5] = ln(1+area) / ln(1+4_000_000).
const MAX_AREA: f64 = 4_000_000.0;
/// Max chars per side for the edit similarity; longer reads as missing (0).
const MAX_EDIT_CHARS: usize = 512;

/// Schema mismatch error returned by [`check_schema_version`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemaMismatch {
    /// The version this build expects.
    pub expected: u32,
    /// The version that was offered.
    pub got: u32,
}

impl std::fmt::Display for SchemaMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "feature schema mismatch: expected v{}, got v{}",
            self.expected, self.got
        )
    }
}

impl std::error::Error for SchemaMismatch {}

/// Refuse to run when the offered schema version is not the frozen v1.
pub fn check_schema_version(version: u32) -> Result<(), SchemaMismatch> {
    if version == FEATURE_SCHEMA_VERSION {
        Ok(())
    } else {
        Err(SchemaMismatch {
            expected: FEATURE_SCHEMA_VERSION,
            got: version,
        })
    }
}

/// Inputs to [`assemble`]. Strings are borrowed; normalization happens inside.
#[derive(Debug, Clone, Copy)]
pub struct FeatureInputs<'a> {
    /// Raw BM25 score from stage 1 (may be non-finite; guarded to -10.0).
    pub bm25: f64,
    /// The step's natural-language target description.
    pub description: &'a str,
    /// The candidate's accessible name / text.
    pub candidate_text: &'a str,
    /// Whether the description names this candidate's role.
    pub role_intent_match: bool,
    /// Whether the normalized description equals the normalized name.
    pub exact_name: bool,
    /// Candidate bounding-box width, CSS px.
    pub candidate_w: f64,
    /// Candidate bounding-box height, CSS px.
    pub candidate_h: f64,
    /// Candidate center in main-viewport CSS px.
    pub candidate_center: (f64, f64),
    /// Main viewport (width, height), CSS px.
    pub viewport: (f64, f64),
    /// Enabled and visible per AX state.
    pub enabled_visible: bool,
}

fn finite_f32(v: f64, fallback: f32) -> f32 {
    if v.is_finite() {
        v as f32
    } else {
        fallback
    }
}

/// Lowercase + collapse whitespace, the shared normalization for tokens,
/// names, and n-gram text.
fn normalize_text(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// The FxHash-like u64 contract pinned by the golden pairs:
/// h = 0; per byte, h = ((h <<< 5) ^ byte) * FX_MULT, wrapping.
fn fx_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0;
    for &b in bytes {
        h = (h.rotate_left(5) ^ u64::from(b)).wrapping_mul(FX_MULT);
    }
    h
}

fn l2_normalize_f64(acc: &[i32; EMBED_DIM]) -> [f32; EMBED_DIM] {
    // Exact integer counts in i32 (f32's 24-bit mantissa would quantize
    // large counts); the norm accumulates in f64; conversion to f32
    // happens once, at the end.
    let mut sum = 0.0f64;
    for &v in acc {
        let x = v as f64;
        sum += x * x;
    }
    let n = sum.sqrt();
    let mut out = [0.0f32; EMBED_DIM];
    if n.is_finite() && n > 0.0 {
        for (o, &v) in out.iter_mut().zip(acc.iter()) {
            *o = (v as f64 / n) as f32;
        }
    }
    out
}

/// 64-dim signed hashed char-trigram embedding of `text`.
///
/// Unigrams, bigrams, and trigrams over the normalized chars; signed counts
/// hashed per the module contract; L2-normalized. Empty text yields the
/// zero vector; short text like "OK" yields a non-zero vector.
pub fn ngram_embedding(text: &str) -> [f32; EMBED_DIM] {
    let norm = normalize_text(text);
    let chars: Vec<char> = norm.chars().collect();
    let mut acc = [0i32; EMBED_DIM];
    for n in 1..=3usize {
        if chars.len() < n {
            continue;
        }
        for i in 0..=(chars.len() - n) {
            let unit: String = chars[i..i + n].iter().collect();
            let h = fx_hash(unit.as_bytes());
            let idx = ((h >> 1) % EMBED_DIM as u64) as usize;
            let sign = if h & 1 == 0 { 1i32 } else { -1i32 };
            acc[idx] += sign;
        }
    }
    l2_normalize_f64(&acc)
}

fn jaccard_token_overlap(a: &str, b: &str) -> f64 {
    let na = normalize_text(a);
    let nb = normalize_text(b);
    let sa: HashSet<&str> = na.split(' ').filter(|t| !t.is_empty()).collect();
    let sb: HashSet<&str> = nb.split(' ').filter(|t| !t.is_empty()).collect();
    if sa.is_empty() && sb.is_empty() {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    inter / union
}

fn levenshtein_chars(a: &[char], b: &[char]) -> usize {
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let mut prev: Vec<usize> = (0..=short.len()).collect();
    let mut curr = vec![0usize; short.len() + 1];
    for (i, &lc) in long.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &sc) in short.iter().enumerate() {
            let cost = usize::from(lc != sc);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[short.len()]
}

/// Edit similarity = 1 - levenshtein / max(len), on normalized text, [0,1].
/// Strings over MAX_EDIT_CHARS per side read as missing (0.0).
fn edit_similarity(a: &str, b: &str) -> f64 {
    let na = normalize_text(a);
    let nb = normalize_text(b);
    let ca: Vec<char> = na.chars().collect();
    let cb: Vec<char> = nb.chars().collect();
    if ca.len() > MAX_EDIT_CHARS || cb.len() > MAX_EDIT_CHARS {
        return 0.0;
    }
    let max_len = ca.len().max(cb.len());
    if max_len == 0 {
        return 0.0;
    }
    1.0 - (levenshtein_chars(&ca, &cb) as f64) / (max_len as f64)
}

/// Log area feature: ln(1+w*h) / ln(1+MAX_AREA), [0,1].
/// Non-finite or negative dims read as missing (0.0).
fn log_area_norm(w: f64, h: f64) -> f64 {
    if !(w.is_finite() && h.is_finite() && w >= 0.0 && h >= 0.0) {
        return 0.0;
    }
    ((1.0 + w * h).ln() / (1.0 + MAX_AREA).ln()).clamp(0.0, 1.0)
}

/// Position prior: 1 - dist(center, viewport center) / half-diagonal, [0,1].
/// Degenerate viewports read as missing (0.0).
fn position_prior(center: (f64, f64), viewport: (f64, f64)) -> f64 {
    let (cx, cy) = center;
    let (vw, vh) = viewport;
    if !(cx.is_finite() && cy.is_finite() && vw.is_finite() && vh.is_finite()) {
        return 0.0;
    }
    if vw <= 0.0 || vh <= 0.0 {
        return 0.0;
    }
    let dist = (cx - vw / 2.0).hypot(cy - vh / 2.0);
    let half_diag = (vw / 2.0).hypot(vh / 2.0);
    if half_diag <= 0.0 {
        return 0.0;
    }
    (1.0 - dist / half_diag).clamp(0.0, 1.0)
}

fn dot(a: &[f32; EMBED_DIM], b: &[f32; EMBED_DIM]) -> f32 {
    let mut d = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        d += x * y;
    }
    d
}

/// Assemble the frozen 137-dim feature vector for one
/// (description, candidate) pair. Every float is finite-guarded.
pub fn assemble(inp: &FeatureInputs) -> [f32; FEATURE_DIM] {
    let mut out = [0.0f32; FEATURE_DIM];
    out[0] = finite_f32(inp.bm25, -10.0).max(-10.0);
    out[1] = finite_f32(
        jaccard_token_overlap(inp.description, inp.candidate_text),
        0.0,
    )
    .clamp(0.0, 1.0);
    out[2] = finite_f32(edit_similarity(inp.description, inp.candidate_text), 0.0).clamp(0.0, 1.0);
    out[3] = f32::from(inp.role_intent_match);
    out[4] = f32::from(inp.exact_name);
    out[5] = finite_f32(log_area_norm(inp.candidate_w, inp.candidate_h), 0.0).clamp(0.0, 1.0);
    out[6] = finite_f32(position_prior(inp.candidate_center, inp.viewport), 0.0).clamp(0.0, 1.0);
    out[7] = f32::from(inp.enabled_visible);
    let desc_emb = ngram_embedding(inp.description);
    let cand_emb = ngram_embedding(inp.candidate_text);
    out[8..72].copy_from_slice(&desc_emb);
    out[72..136].copy_from_slice(&cand_emb);
    out[136] = dot(&desc_emb, &cand_emb);
    out
}

#[cfg(test)]
#[allow(clippy::excessive_precision)] // golden test vectors: digits are exact, trimming would corrupt them
mod tests {
    use super::*;

    // Hardcoded golden vectors, generated by an independent Python reference
    // implementation written straight from the documented algorithm (the
    // FxHash-like contract above). They pin the schema; any layout change
    // must rebuild them, not edit them to match new output.
    // OK_PAIR: desc='ok' cand='OK'
    const OK_PAIR: [f32; 137] = [
        2.50000000e+00,
        1.00000000e+00,
        1.00000000e+00,
        0.00000000e+00,
        1.00000000e+00,
        4.78436112e-01,
        3.67544472e-01,
        1.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -5.77350259e-01,
        0.00000000e+00,
        0.00000000e+00,
        -5.77350259e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -5.77350259e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -5.77350259e-01,
        0.00000000e+00,
        0.00000000e+00,
        -5.77350259e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -5.77350259e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        9.99999940e-01,
    ];
    // SEARCH_PAIR: desc='search the web' cand='Search'
    const SEARCH_PAIR: [f32; 137] = [
        4.09999990e+00,
        3.33333343e-01,
        4.28571433e-01,
        1.00000000e+00,
        0.00000000e+00,
        5.38683832e-01,
        5.22989452e-01,
        1.00000000e+00,
        0.00000000e+00,
        1.37360558e-01,
        1.37360558e-01,
        0.00000000e+00,
        2.74721116e-01,
        1.37360558e-01,
        -1.37360558e-01,
        0.00000000e+00,
        1.37360558e-01,
        0.00000000e+00,
        0.00000000e+00,
        1.37360558e-01,
        -1.37360558e-01,
        0.00000000e+00,
        0.00000000e+00,
        -1.37360558e-01,
        2.74721116e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -2.74721116e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -1.37360558e-01,
        0.00000000e+00,
        -1.37360558e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        1.37360558e-01,
        0.00000000e+00,
        -1.37360558e-01,
        -1.37360558e-01,
        0.00000000e+00,
        -4.12081689e-01,
        1.37360558e-01,
        0.00000000e+00,
        1.37360558e-01,
        0.00000000e+00,
        0.00000000e+00,
        1.37360558e-01,
        0.00000000e+00,
        0.00000000e+00,
        2.74721116e-01,
        -1.37360558e-01,
        0.00000000e+00,
        0.00000000e+00,
        1.37360558e-01,
        0.00000000e+00,
        1.37360558e-01,
        0.00000000e+00,
        1.37360558e-01,
        0.00000000e+00,
        -1.37360558e-01,
        -1.37360558e-01,
        2.74721116e-01,
        -1.37360558e-01,
        0.00000000e+00,
        1.37360558e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        2.42535636e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        2.42535636e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -2.42535636e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -2.42535636e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -2.42535636e-01,
        0.00000000e+00,
        0.00000000e+00,
        2.42535636e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        2.42535636e-01,
        -2.42535636e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        2.42535636e-01,
        0.00000000e+00,
        2.42535636e-01,
        0.00000000e+00,
        -2.42535636e-01,
        0.00000000e+00,
        4.85071272e-01,
        -2.42535636e-01,
        0.00000000e+00,
        2.42535636e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        7.32926250e-01,
    ];
    // SETTINGS_PAIR: desc='open settings' cand='Settings'
    const SETTINGS_PAIR: [f32; 137] = [
        1.70000005e+00,
        5.00000000e-01,
        6.15384638e-01,
        1.00000000e+00,
        0.00000000e+00,
        5.15228987e-01,
        4.18810785e-01,
        0.00000000e+00,
        1.36082768e-01,
        0.00000000e+00,
        5.44331074e-01,
        2.72165537e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        1.36082768e-01,
        1.36082768e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -1.36082768e-01,
        -2.72165537e-01,
        -1.36082768e-01,
        1.36082768e-01,
        1.36082768e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -1.36082768e-01,
        0.00000000e+00,
        1.36082768e-01,
        1.36082768e-01,
        -1.36082768e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        1.36082768e-01,
        0.00000000e+00,
        0.00000000e+00,
        -1.36082768e-01,
        1.36082768e-01,
        -2.72165537e-01,
        0.00000000e+00,
        0.00000000e+00,
        1.36082768e-01,
        0.00000000e+00,
        1.36082768e-01,
        0.00000000e+00,
        0.00000000e+00,
        1.36082768e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        1.36082768e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -2.72165537e-01,
        0.00000000e+00,
        -1.36082768e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -1.36082768e-01,
        1.36082768e-01,
        -1.36082768e-01,
        1.64398983e-01,
        0.00000000e+00,
        6.57595932e-01,
        1.64398983e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        1.64398983e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -3.28797966e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -1.64398983e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -1.64398983e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -1.64398983e-01,
        0.00000000e+00,
        0.00000000e+00,
        1.64398983e-01,
        0.00000000e+00,
        1.64398983e-01,
        0.00000000e+00,
        0.00000000e+00,
        1.64398983e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -3.28797966e-01,
        0.00000000e+00,
        -1.64398983e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -1.64398983e-01,
        1.64398983e-01,
        -1.64398983e-01,
        8.72503042e-01,
    ];

    #[allow(clippy::too_many_arguments)] // test helper; mirrors the bridge
    fn inputs_for(
        bm25: f64,
        description: &'static str,
        candidate_text: &'static str,
        role_intent_match: bool,
        exact_name: bool,
        candidate_w: f64,
        candidate_h: f64,
        candidate_center: (f64, f64),
        viewport: (f64, f64),
        enabled_visible: bool,
    ) -> FeatureInputs<'static> {
        FeatureInputs {
            bm25,
            description,
            candidate_text,
            role_intent_match,
            exact_name,
            candidate_w,
            candidate_h,
            candidate_center,
            viewport,
            enabled_visible,
        }
    }

    fn assert_close(a: &[f32; FEATURE_DIM], b: &[f32; FEATURE_DIM]) {
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= 1e-6,
                "golden mismatch at index {i}: got {x}, want {y}"
            );
        }
    }

    #[test]
    fn golden_ok_pair() {
        let inp = inputs_for(
            2.5,
            "ok",
            "OK",
            false,
            true,
            60.0,
            24.0,
            (100.0, 200.0),
            (800.0, 600.0),
            true,
        );
        assert_close(&assemble(&inp), &OK_PAIR);
    }

    #[test]
    fn golden_search_pair() {
        let inp = inputs_for(
            4.1,
            "search the web",
            "Search",
            true,
            false,
            120.0,
            30.0,
            (400.0, 100.0),
            (1024.0, 768.0),
            true,
        );
        assert_close(&assemble(&inp), &SEARCH_PAIR);
    }

    #[test]
    fn golden_settings_pair() {
        let inp = inputs_for(
            1.7,
            "open settings",
            "Settings",
            true,
            false,
            90.0,
            28.0,
            (960.0, 700.0),
            (1280.0, 800.0),
            false,
        );
        assert_close(&assemble(&inp), &SETTINGS_PAIR);
    }

    /// Independent re-derivation of the trigram indices/signs straight from
    /// the documented algorithm, written separately from `ngram_embedding`:
    /// h = 0; per byte h = ((h rotated left 5) XOR byte) * K (wrapping);
    /// index = (h >> 1) % 64; sign = +1 when h & 1 == 0 else -1; i32
    /// accumulation, then f64 L2 normalization. This pins the hardcoded
    /// goldens, not just the production code.
    fn ref_embedding(text: &str) -> [f32; EMBED_DIM] {
        const K: u64 = 0x51_7c_c1_b7_27_22_0a_95;
        let norm: String = text
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let chars: Vec<char> = norm.chars().collect();
        let mut acc = [0i32; EMBED_DIM];
        for n in 1..=3usize {
            if chars.len() < n {
                continue;
            }
            for i in 0..=(chars.len() - n) {
                let unit: String = chars[i..i + n].iter().collect();
                let mut h: u64 = 0;
                for &byte in unit.as_bytes() {
                    h = (h.rotate_left(5) ^ u64::from(byte)).wrapping_mul(K);
                }
                let idx = ((h >> 1) % EMBED_DIM as u64) as usize;
                let sign = if h & 1 == 0 { 1i32 } else { -1i32 };
                acc[idx] += sign;
            }
        }
        let mut sum = 0.0f64;
        for &v in acc.iter() {
            let x = v as f64;
            sum += x * x;
        }
        let n = sum.sqrt();
        let mut out = [0.0f32; EMBED_DIM];
        if n.is_finite() && n > 0.0 {
            for (o, &v) in out.iter_mut().zip(acc.iter()) {
                *o = (v as f64 / n) as f32;
            }
        }
        out
    }

    #[test]
    fn independent_hash_rederivation_pins_goldens() {
        // Hand-verified from the algorithm: "ok" normalizes to "ok";
        // units o -> (idx 13, -1), k -> (idx 35, -1), ok -> (idx 10, -1);
        // L2 norm of three unit counts is sqrt(3), so each is -1/sqrt(3).
        let emb = ref_embedding("ok");
        let neg = -(1.0f32 / 3.0f32.sqrt());
        assert!((emb[10] - neg).abs() < 1e-6, "idx 10: {}", emb[10]);
        assert!((emb[13] - neg).abs() < 1e-6, "idx 13: {}", emb[13]);
        assert!((emb[35] - neg).abs() < 1e-6, "idx 35: {}", emb[35]);
        for (i, &v) in emb.iter().enumerate() {
            if i != 10 && i != 13 && i != 35 {
                assert_eq!(v, 0.0, "idx {i} should be zero");
            }
        }
        // The independent re-derivation must match the hardcoded golden
        // slices for all three pairs (description and candidate sides).
        let cases: &[(&str, &[f32; FEATURE_DIM], usize)] = &[
            ("ok", &OK_PAIR, 8),
            ("search the web", &SEARCH_PAIR, 8),
            ("Settings", &SETTINGS_PAIR, 72),
        ];
        for (text, golden, base) in cases {
            let r = ref_embedding(text);
            for i in 0..EMBED_DIM {
                assert!(
                    (r[i] - golden[base + i]).abs() < 1e-6,
                    "{text:?} emb idx {i}: got {}, want {}",
                    r[i],
                    golden[base + i]
                );
            }
        }
        // And the production embedding must agree with the re-derivation.
        let p = ngram_embedding("search the web");
        let r = ref_embedding("search the web");
        for i in 0..EMBED_DIM {
            assert!((p[i] - r[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn short_strings_are_not_zero_vectors() {
        for s in ["OK", "a", "Go", "x y"] {
            let e = ngram_embedding(s);
            assert!(
                e.iter().any(|&v| v != 0.0),
                "{s:?} must not be the zero vector"
            );
            let mut sum = 0.0f32;
            for &v in &e {
                sum += v * v;
            }
            assert!((sum.sqrt() - 1.0).abs() < 1e-5, "{s:?} not L2-normalized");
        }
        // Empty text is the documented zero vector.
        assert_eq!(ngram_embedding(""), [0.0f32; EMBED_DIM]);
    }

    #[test]
    fn schema_version_refusal() {
        assert!(check_schema_version(FEATURE_SCHEMA_VERSION).is_ok());
        assert!(check_schema_version(0).is_err());
        assert!(check_schema_version(FEATURE_SCHEMA_VERSION + 1).is_err());
        assert!(check_schema_version(u32::MAX).is_err());
        let err = check_schema_version(2).unwrap_err();
        assert_eq!(
            err,
            SchemaMismatch {
                expected: FEATURE_SCHEMA_VERSION,
                got: 2
            }
        );
    }

    #[test]
    fn hand_feature_finite_guards() {
        let base = || {
            inputs_for(
                2.5,
                "ok",
                "OK",
                false,
                true,
                60.0,
                24.0,
                (100.0, 200.0),
                (800.0, 600.0),
                true,
            )
        };
        // Non-finite BM25 guards to -10.0.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut inp = base();
            inp.bm25 = bad;
            assert_eq!(assemble(&inp)[0], -10.0);
        }
        // Non-finite dims guard the log-area feature to 0.
        let mut inp = base();
        inp.candidate_w = f64::NAN;
        assert_eq!(assemble(&inp)[5], 0.0);
        // Degenerate viewport guards the position prior to 0.
        let mut inp = base();
        inp.viewport = (0.0, 0.0);
        assert_eq!(assemble(&inp)[6], 0.0);
        // Every output float is finite on ordinary input.
        let v = assemble(&base());
        assert!(v.iter().all(|x| x.is_finite()));
        // Feature 136 is the dot of the two embedding slices.
        let mut d = 0.0f32;
        for i in 0..EMBED_DIM {
            d += v[8 + i] * v[72 + i];
        }
        assert!((d - v[136]).abs() < 1e-6);
    }
}
