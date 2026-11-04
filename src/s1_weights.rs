//! S1-mouse weight artifact loader (mouse-future Phase 6).
//!
//! The production weight type [`S1MouseWeights`] is defined EXACTLY ONCE,
//! here. Nothing else in the crate may define or duplicate it.
//!
//! Load contract (frozen, plan section 4):
//! real SHA-256 -> exact length check (45,592) -> header/schema/version
//! check -> parse into a TEMPORARY struct -> finite-value checks ->
//! atomic publish. Production weights are NEVER constructed from zeros;
//! any malformed input is an error, never a silent default.
//!
//! `.bin` layout (written FROM THE STATE DICT, never round-tripped
//! through an exporter's transpose):
//! ```text
//! header 16B LE: "S1MW" | format_version u16=1 | feature_schema u16=1
//!                | n_train_params u32=10945 | pad_dim u16=144 | reserved u16=0
//! body f32 LE: w1 64x144 padded rows (9216) | b1 64 | w2 32x64 (2048)
//!              | b2 32 | w3 32 | b3 1 | inv_temp 1
//! total: 16 + 11394*4 = 45592 bytes exact. Any other length is rejected.
//! ```
//! The temperature is the ONLY calibration scalar in the artifact. There
//! is no prior_shift: a list-wide scalar cancels out of its own softmax.
//!
//! Inference is pure synchronous Rust GEMV on the padded 144-wide rows,
//! into a caller stack array. No async runtime anywhere near this path.

use std::sync::{Arc, RwLock};

use super::features;

/// Magic bytes opening every artifact.
pub const S1MW_MAGIC: [u8; 4] = *b"S1MW";
/// Artifact format version.
pub const FORMAT_VERSION: u16 = 1;
/// Trainable parameter count: 64*137+64 + 32*64+32 + 32+1.
pub const N_TRAIN_PARAMS: u32 = 10_945;
/// Padded GEMV row width (144*4 = 576 bytes, a multiple of 32).
pub const PAD_DIM: usize = 144;
/// Hidden widths.
pub const H1: usize = 64;
pub const H2: usize = 32;
/// Stored floats: 9216 + 64 + 2048 + 32 + 32 + 1 + 1.
pub const N_STORED_FLOATS: usize = 11_394;
/// Exact artifact length in bytes: 16 + 11394*4.
pub const BIN_LEN: usize = 45_592;

/// Calibration bounds for inv_temp: below the minimum the softmax
/// flattens toward uniform (random ranking); above the maximum,
/// logits * inv_temp risks overflow. Enforced at load time.
pub const INV_TEMP_MIN: f32 = 1e-3;
pub const INV_TEMP_MAX: f32 = 1e3;

const OFF_W1: usize = 16;
const OFF_B1: usize = OFF_W1 + 9216 * 4;
const OFF_W2: usize = OFF_B1 + 64 * 4;
const OFF_B2: usize = OFF_W2 + 2048 * 4;
const OFF_W3: usize = OFF_B2 + 32 * 4;
const OFF_B3: usize = OFF_W3 + 32 * 4;
const OFF_TEMP: usize = OFF_B3 + 4;

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4, hand-rolled: no new dependency for one load-time hash)
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

/// Real SHA-256 over `data`. Pinned by known-answer tests below.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    #[allow(clippy::chunks_exact_to_as_chunks)]
    // verified against known answers; restyle risks drift
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            let b: [u8; 4] = chunk[4 * i..4 * i + 4].try_into().expect("chunk slice");
            w[i] = u32::from_be_bytes(b);
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
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

fn hex_of(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for &b in digest {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

// ---------------------------------------------------------------------------
// Load errors
// ---------------------------------------------------------------------------

/// Every way an artifact can fail to load. Malformed input is always an
/// error; there is no silent default and no zero-constructed weights.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadError {
    /// Byte length is not exactly 45,592.
    BadLength { got: usize },
    /// First four bytes are not "S1MW".
    BadMagic { got: [u8; 4] },
    /// Format version is not 1.
    BadFormatVersion { got: u16 },
    /// Feature schema is not the frozen v1.
    BadSchema { got: u16 },
    /// Declared trainable param count is not 10,945.
    BadParamCount { got: u32 },
    /// Padded row width is not 144.
    BadPadDim { got: u16 },
    /// A stored float is NaN or infinite (index into the 11,394 body floats).
    NonFinite { index: usize },
    /// inv_temp is not finite or not in [INV_TEMP_MIN, INV_TEMP_MAX]
    /// (it scales the shipped softmax).
    BadInvTemp { got: f32 },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::BadLength { got } => {
                write!(f, "s1 artifact bad length: got {got}, want {BIN_LEN}")
            }
            LoadError::BadMagic { got } => {
                write!(f, "s1 artifact bad magic: {got:?}, want {S1MW_MAGIC:?}")
            }
            LoadError::BadFormatVersion { got } => {
                write!(
                    f,
                    "s1 artifact bad format version: got {got}, want {FORMAT_VERSION}"
                )
            }
            LoadError::BadSchema { got } => write!(
                f,
                "s1 artifact bad feature schema: got {got}, want {}",
                features::FEATURE_SCHEMA_VERSION
            ),
            LoadError::BadParamCount { got } => {
                write!(
                    f,
                    "s1 artifact bad param count: got {got}, want {N_TRAIN_PARAMS}"
                )
            }
            LoadError::BadPadDim { got } => {
                write!(f, "s1 artifact bad pad dim: got {got}, want {PAD_DIM}")
            }
            LoadError::NonFinite { index } => {
                write!(f, "s1 artifact non-finite float at body index {index}")
            }
            LoadError::BadInvTemp { got } => {
                write!(
                    f,
                    "s1 artifact bad inv_temp: {got} (must be finite and in [{INV_TEMP_MIN}, {INV_TEMP_MAX}])"
                )
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// Inference-time errors: programming errors (bad count), never data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferError {
    /// n not in 1..=8.
    BadCount(usize),
    /// fewer than n rows supplied.
    ShortInput { need: usize, got: usize },
    /// inv_temp not finite or not in [INV_TEMP_MIN, INV_TEMP_MAX].
    BadInvTemp,
}

impl std::fmt::Display for InferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferError::BadCount(n) => write!(f, "s1 inference: n={n} not in 1..=8"),
            InferError::ShortInput { need, got } => {
                write!(f, "s1 inference: need {need} rows, got {got}")
            }
            InferError::BadInvTemp => write!(f, "s1 inference: inv_temp not finite/positive"),
        }
    }
}

impl std::error::Error for InferError {}

// ---------------------------------------------------------------------------
// Staged parse (temporary) -> validated production weights
// ---------------------------------------------------------------------------

/// Temporary parsed form. Never escapes this module; only
/// [`StagedWeights::validate`] can promote it to [`S1MouseWeights`].
struct StagedWeights {
    w1: Box<[[f32; PAD_DIM]; H1]>,
    b1: Box<[f32; H1]>,
    w2: Box<[[f32; H1]; H2]>,
    b2: Box<[f32; H2]>,
    w3: Box<[f32; H2]>,
    b3: f32,
    inv_temp: f32,
}

fn read_f32_le(bytes: &[u8], at: usize) -> f32 {
    let b: [u8; 4] = bytes[at..at + 4].try_into().expect("body slice");
    f32::from_le_bytes(b)
}

impl StagedWeights {
    fn parse(bytes: &[u8]) -> Result<Self, LoadError> {
        if bytes.len() != BIN_LEN {
            return Err(LoadError::BadLength { got: bytes.len() });
        }
        let magic: [u8; 4] = bytes[0..4].try_into().expect("header slice");
        if magic != S1MW_MAGIC {
            return Err(LoadError::BadMagic { got: magic });
        }
        let fmt = u16::from_le_bytes(bytes[4..6].try_into().expect("header slice"));
        if fmt != FORMAT_VERSION {
            return Err(LoadError::BadFormatVersion { got: fmt });
        }
        let schema = u16::from_le_bytes(bytes[6..8].try_into().expect("header slice"));
        if u32::from(schema) != features::FEATURE_SCHEMA_VERSION {
            return Err(LoadError::BadSchema { got: schema });
        }
        let n_params = u32::from_le_bytes(bytes[8..12].try_into().expect("header slice"));
        if n_params != N_TRAIN_PARAMS {
            return Err(LoadError::BadParamCount { got: n_params });
        }
        let pad = u16::from_le_bytes(bytes[12..14].try_into().expect("header slice"));
        if usize::from(pad) != PAD_DIM {
            return Err(LoadError::BadPadDim { got: pad });
        }

        let mut w1 = Box::new([[0.0f32; PAD_DIM]; H1]);
        for (i, row) in w1.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = read_f32_le(bytes, OFF_W1 + (i * PAD_DIM + j) * 4);
            }
        }
        let mut b1 = Box::new([0.0f32; H1]);
        for (i, slot) in b1.iter_mut().enumerate() {
            *slot = read_f32_le(bytes, OFF_B1 + i * 4);
        }
        let mut w2 = Box::new([[0.0f32; H1]; H2]);
        for (i, row) in w2.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = read_f32_le(bytes, OFF_W2 + (i * H1 + j) * 4);
            }
        }
        let mut b2 = Box::new([0.0f32; H2]);
        for (i, slot) in b2.iter_mut().enumerate() {
            *slot = read_f32_le(bytes, OFF_B2 + i * 4);
        }
        let mut w3 = Box::new([0.0f32; H2]);
        for (i, slot) in w3.iter_mut().enumerate() {
            *slot = read_f32_le(bytes, OFF_W3 + i * 4);
        }
        let b3 = read_f32_le(bytes, OFF_B3);
        let inv_temp = read_f32_le(bytes, OFF_TEMP);
        Ok(StagedWeights {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            inv_temp,
        })
    }

    /// Finite-value checks, then promotion. This is the only path that
    /// constructs an [`S1MouseWeights`].
    fn validate(self) -> Result<S1MouseWeights, LoadError> {
        let mut idx = 0usize;
        let mut check = |v: f32| -> Result<(), LoadError> {
            if !v.is_finite() {
                return Err(LoadError::NonFinite { index: idx });
            }
            idx += 1;
            Ok(())
        };
        for row in self.w1.iter() {
            for &v in row.iter() {
                check(v)?;
            }
        }
        for &v in self.b1.iter() {
            check(v)?;
        }
        for row in self.w2.iter() {
            for &v in row.iter() {
                check(v)?;
            }
        }
        for &v in self.b2.iter() {
            check(v)?;
        }
        for &v in self.w3.iter() {
            check(v)?;
        }
        check(self.b3)?;
        check(self.inv_temp)?;
        // NaN/inf were rejected by the finite check above, so this
        // comparison is total here.
        if self.inv_temp <= 0.0 {
            return Err(LoadError::BadInvTemp { got: self.inv_temp });
        }
        // Calibration bounds: near-zero inv_temp flattens the softmax
        // toward uniform (the ranker degenerates into random selection);
        // huge inv_temp risks overflow in logits * inv_temp. Both are
        // rejected rather than silently accepted.
        if !(INV_TEMP_MIN..=INV_TEMP_MAX).contains(&self.inv_temp) {
            return Err(LoadError::BadInvTemp { got: self.inv_temp });
        }
        Ok(S1MouseWeights {
            w1: self.w1,
            b1: self.b1,
            w2: self.w2,
            b2: self.b2,
            w3: self.w3,
            b3: self.b3,
            inv_temp: self.inv_temp,
        })
    }
}

// ---------------------------------------------------------------------------
// Production weights (defined exactly once, here)
// ---------------------------------------------------------------------------

/// Shipped S1-mouse weights: 137->64 (ReLU) ->32 (ReLU) ->1 logit per
/// candidate, temperature-scaled softmax over the live top-8.
///
/// Fields are private and there is no `Default`/`zero()` constructor:
/// the only way to obtain one is [`S1MouseWeights::load_from_bytes`],
/// which parses, validates, and only then returns.
#[derive(Debug, Clone)]
pub struct S1MouseWeights {
    w1: Box<[[f32; PAD_DIM]; H1]>,
    b1: Box<[f32; H1]>,
    w2: Box<[[f32; H1]; H2]>,
    b2: Box<[f32; H2]>,
    w3: Box<[f32; H2]>,
    b3: f32,
    inv_temp: f32,
}

impl S1MouseWeights {
    /// Load from artifact bytes: length -> header -> staged parse ->
    /// finite checks -> validated weights. The caller is expected to
    /// verify [`sha256`] against the pinned golden hash first.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        StagedWeights::parse(bytes)?.validate()
    }

    /// Fitted inverse temperature (calibration pages only).
    pub fn inv_temp(&self) -> f32 {
        self.inv_temp
    }

    /// Raw logits for the first n rows (1 <= n <= 8), into the caller's
    /// stack array. Rows are 137-dim; the 7 pad columns are zero.
    pub fn logits(
        &self,
        xs: &[[f32; 137]],
        n: usize,
        out: &mut [f32; 8],
    ) -> Result<(), InferError> {
        if !(1..=8).contains(&n) {
            return Err(InferError::BadCount(n));
        }
        if xs.len() < n {
            return Err(InferError::ShortInput {
                need: n,
                got: xs.len(),
            });
        }
        for i in 0..n {
            let mut x = [0.0f32; PAD_DIM];
            x[..137].copy_from_slice(&xs[i]);
            let mut h1 = [0.0f32; H1];
            for ((acc, wrow), &b) in h1.iter_mut().zip(self.w1.iter()).zip(self.b1.iter()) {
                let mut s = b;
                for (w, &xv) in wrow.iter().zip(x.iter()) {
                    s += w * xv;
                }
                // ReLU; a non-finite pre-activation reads as 0, matching
                // the interim ranker's non-finite guard.
                *acc = if s > 0.0 { s } else { 0.0 };
            }
            let mut h2 = [0.0f32; H2];
            for ((acc, wrow), &b) in h2.iter_mut().zip(self.w2.iter()).zip(self.b2.iter()) {
                let mut s = b;
                for (w, &hv) in wrow.iter().zip(h1.iter()) {
                    s += w * hv;
                }
                *acc = if s > 0.0 { s } else { 0.0 };
            }
            let mut z = self.b3;
            for (w, &hv) in self.w3.iter().zip(h2.iter()) {
                z += w * hv;
            }
            out[i] = if z.is_finite() { z } else { 0.0 };
        }
        Ok(())
    }

    /// Temperature-scaled softmax over the first n logits, into the
    /// caller's stack array. `p = softmax(logits * inv_temp)`.
    pub fn softmax_batch(
        logits: &[f32; 8],
        n: usize,
        inv_temp: f32,
        out: &mut [f64; 8],
    ) -> Result<(), InferError> {
        if !(1..=8).contains(&n) {
            return Err(InferError::BadCount(n));
        }
        if !(inv_temp.is_finite() && inv_temp > 0.0) {
            return Err(InferError::BadInvTemp);
        }
        let t = f64::from(inv_temp);
        let mut max = f64::NEG_INFINITY;
        for &z in logits.iter().take(n) {
            let v = f64::from(z) * t;
            if v > max {
                max = v;
            }
        }
        let mut sum = 0.0f64;
        for (slot, &z) in out.iter_mut().zip(logits.iter()).take(n) {
            let e = (f64::from(z) * t - max).exp();
            let e = if e.is_finite() { e } else { 0.0 };
            *slot = e;
            sum += e;
        }
        if sum > 0.0 {
            for slot in out.iter_mut().take(n) {
                *slot /= sum;
            }
        } else {
            for slot in out.iter_mut().take(n) {
                *slot = 1.0 / n as f64;
            }
        }
        Ok(())
    }

    /// One call: logits then temperature-scaled softmax into `probs[0..n]`.
    pub fn infer(
        &self,
        xs: &[[f32; 137]],
        n: usize,
        probs: &mut [f64; 8],
    ) -> Result<(), InferError> {
        let mut z = [0.0f32; 8];
        self.logits(xs, n, &mut z)?;
        Self::softmax_batch(&z, n, self.inv_temp, probs)
    }
}

// ---------------------------------------------------------------------------
// Atomic publish
// ---------------------------------------------------------------------------

static ACTIVE: RwLock<Option<Arc<S1MouseWeights>>> = RwLock::new(None);

/// Atomically publish validated weights as the active S1-mouse model.
/// Readers see either the old weights or the new ones, never a half-parse.
pub fn publish(weights: S1MouseWeights) {
    let mut guard = ACTIVE.write().expect("s1 weights lock");
    *guard = Some(Arc::new(weights));
}

/// The currently active weights, if any. `None` until the first publish:
/// the resolver keeps its existing ranker (BM25 / interim linear) then.
pub fn active() -> Option<Arc<S1MouseWeights>> {
    ACTIVE.read().expect("s1 weights lock").clone()
}

/// Clear the active weights (tests and controlled rollback only).
pub fn clear_active() {
    let mut guard = ACTIVE.write().expect("s1 weights lock");
    *guard = None;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN_BIN: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/s1_mouse/golden.bin"
    ));
    const GOLDEN_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/s1_mouse/golden.json"
    ));

    #[test]
    fn sha256_known_answers() {
        assert_eq!(
            hex_of(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex_of(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn consts_match_frozen_contract() {
        assert_eq!(BIN_LEN, 45_592);
        assert_eq!(16 + N_STORED_FLOATS * 4, BIN_LEN);
        assert_eq!(OFF_TEMP + 4, BIN_LEN);
        assert_eq!(u32::from(FORMAT_VERSION), 1);
        assert_eq!(
            64 * 137 + 64 + 32 * 64 + 32 + 32 + 1,
            N_TRAIN_PARAMS as usize
        );
    }

    fn golden_inputs() -> (Vec<[f32; 137]>, Vec<f64>, f64) {
        let g: serde_json::Value = serde_json::from_str(GOLDEN_JSON).expect("golden json");
        let inputs: Vec<[f32; 137]> = g["inputs"]
            .as_array()
            .expect("inputs")
            .iter()
            .map(|row| {
                let v: Vec<f32> = row
                    .as_array()
                    .expect("row")
                    .iter()
                    .map(|x| x.as_f64().expect("float") as f32)
                    .collect();
                assert_eq!(v.len(), 137);
                let mut a = [0.0f32; 137];
                a.copy_from_slice(&v);
                a
            })
            .collect();
        let expected: Vec<f64> = g["expected_logits"]
            .as_array()
            .expect("logits")
            .iter()
            .map(|x| x.as_f64().expect("float"))
            .collect();
        let inv_temp = g["inv_temp"].as_f64().expect("inv_temp");
        (inputs, expected, inv_temp)
    }

    /// Golden artifact: pinned SHA-256, pinned logits from the independent
    /// float64 reference. Native f32 GEMV must match within 1e-4.
    #[test]
    fn golden_artifact_parity() {
        let g: serde_json::Value = serde_json::from_str(GOLDEN_JSON).expect("golden json");
        assert_eq!(GOLDEN_BIN.len(), BIN_LEN);
        assert_eq!(
            hex_of(&sha256(GOLDEN_BIN)),
            g["sha256"].as_str().expect("sha256")
        );
        let w = S1MouseWeights::load_from_bytes(GOLDEN_BIN).expect("golden loads");
        let (inputs, expected, inv_temp) = golden_inputs();
        assert_eq!(inputs.len(), 8);
        assert!((f64::from(w.inv_temp()) - inv_temp).abs() < 1e-6);
        let mut logits = [0.0f32; 8];
        w.logits(&inputs, 8, &mut logits).expect("infer");
        for (i, (got, exp)) in logits.iter().zip(expected.iter()).enumerate() {
            let d = (f64::from(*got) - *exp).abs();
            assert!(
                d <= 1e-4,
                "logit {i}: native {got} vs golden {exp} (diff {d})"
            );
        }
        // Softmax path on the same logits.
        let mut probs = [0.0f64; 8];
        S1MouseWeights::softmax_batch(&logits, 8, w.inv_temp(), &mut probs).expect("softmax");
        let sum: f64 = probs.iter().take(8).sum();
        assert!((sum - 1.0).abs() < 1e-9, "probs sum to {sum}");
        assert!(probs.iter().take(8).all(|&p| p.is_finite() && p >= 0.0));
    }

    #[test]
    fn reject_bad_length() {
        let err = S1MouseWeights::load_from_bytes(&GOLDEN_BIN[..BIN_LEN - 1]).unwrap_err();
        assert_eq!(err, LoadError::BadLength { got: BIN_LEN - 1 });
        let mut longer = GOLDEN_BIN.to_vec();
        longer.push(0);
        let err = S1MouseWeights::load_from_bytes(&longer).unwrap_err();
        assert_eq!(err, LoadError::BadLength { got: BIN_LEN + 1 });
    }

    #[test]
    fn reject_bad_magic() {
        let mut bad = GOLDEN_BIN.to_vec();
        bad[0] = b'X';
        let err = S1MouseWeights::load_from_bytes(&bad).unwrap_err();
        assert!(matches!(err, LoadError::BadMagic { .. }), "{err:?}");
    }

    #[test]
    fn reject_bad_schema() {
        let mut bad = GOLDEN_BIN.to_vec();
        bad[6] = 2; // feature_schema u16 LE low byte
        let err = S1MouseWeights::load_from_bytes(&bad).unwrap_err();
        assert_eq!(err, LoadError::BadSchema { got: 2 });
    }

    #[test]
    fn reject_bad_format_version() {
        let mut bad = GOLDEN_BIN.to_vec();
        bad[4] = 9;
        let err = S1MouseWeights::load_from_bytes(&bad).unwrap_err();
        assert_eq!(err, LoadError::BadFormatVersion { got: 9 });
    }

    #[test]
    fn reject_bad_param_count() {
        let mut bad = GOLDEN_BIN.to_vec();
        bad[8] = 0; // n_train_params u32 LE low byte -> 10944
        let err = S1MouseWeights::load_from_bytes(&bad).unwrap_err();
        assert!(matches!(err, LoadError::BadParamCount { .. }), "{err:?}");
    }

    #[test]
    fn reject_nonfinite_weight() {
        let mut bad = GOLDEN_BIN.to_vec();
        // First body float -> NaN bits.
        bad[OFF_W1..OFF_W1 + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let err = S1MouseWeights::load_from_bytes(&bad).unwrap_err();
        assert_eq!(err, LoadError::NonFinite { index: 0 });
    }

    #[test]
    fn reject_nonpositive_inv_temp() {
        let mut bad = GOLDEN_BIN.to_vec();
        bad[OFF_TEMP..OFF_TEMP + 4].copy_from_slice(&0.0f32.to_le_bytes());
        let err = S1MouseWeights::load_from_bytes(&bad).unwrap_err();
        assert_eq!(err, LoadError::BadInvTemp { got: 0.0 });
        let mut bad = GOLDEN_BIN.to_vec();
        bad[OFF_TEMP..OFF_TEMP + 4].copy_from_slice(&(-1.5f32).to_le_bytes());
        let err = S1MouseWeights::load_from_bytes(&bad).unwrap_err();
        assert_eq!(err, LoadError::BadInvTemp { got: -1.5 });
    }

    #[test]
    fn reject_degenerate_inv_temp_bounds() {
        // Near-zero flattens the softmax to uniform (random ranking);
        // huge values risk overflow. Both must be rejected, not just
        // non-finite and non-positive.
        for &v in &[1e-30f32, 1e-4f32, 1e30f32] {
            let mut bad = GOLDEN_BIN.to_vec();
            bad[OFF_TEMP..OFF_TEMP + 4].copy_from_slice(&v.to_le_bytes());
            let err = S1MouseWeights::load_from_bytes(&bad).unwrap_err();
            assert_eq!(err, LoadError::BadInvTemp { got: v }, "inv_temp={v}");
        }
        // Non-finite is still rejected by the finite check (inv_temp is
        // the last of the 11,394 stored floats).
        for &v in &[f32::INFINITY, f32::NAN] {
            let mut bad = GOLDEN_BIN.to_vec();
            bad[OFF_TEMP..OFF_TEMP + 4].copy_from_slice(&v.to_le_bytes());
            let err = S1MouseWeights::load_from_bytes(&bad).unwrap_err();
            assert_eq!(
                err,
                LoadError::NonFinite {
                    index: N_STORED_FLOATS - 1
                },
                "inv_temp={v}"
            );
        }
        // The bounds themselves are accepted.
        for &v in &[INV_TEMP_MIN, 1.7f32, INV_TEMP_MAX] {
            let mut good = GOLDEN_BIN.to_vec();
            good[OFF_TEMP..OFF_TEMP + 4].copy_from_slice(&v.to_le_bytes());
            assert!(
                S1MouseWeights::load_from_bytes(&good).is_ok(),
                "inv_temp={v} should load"
            );
        }
    }

    #[test]
    fn infer_rejects_bad_counts() {
        let w = S1MouseWeights::load_from_bytes(GOLDEN_BIN).expect("golden loads");
        let (inputs, _, _) = golden_inputs();
        let mut z = [0.0f32; 8];
        assert_eq!(w.logits(&inputs, 0, &mut z), Err(InferError::BadCount(0)));
        assert_eq!(w.logits(&inputs, 9, &mut z), Err(InferError::BadCount(9)));
        assert_eq!(
            w.logits(&inputs[..3], 4, &mut z),
            Err(InferError::ShortInput { need: 4, got: 3 })
        );
        // n < 8 is fine.
        w.logits(&inputs, 3, &mut z).expect("n=3");
    }

    #[test]
    fn softmax_rejects_bad_temp() {
        let z = [0.1f32; 8];
        let mut p = [0.0f64; 8];
        assert_eq!(
            S1MouseWeights::softmax_batch(&z, 8, 0.0, &mut p),
            Err(InferError::BadInvTemp)
        );
        assert_eq!(
            S1MouseWeights::softmax_batch(&z, 8, f32::NAN, &mut p),
            Err(InferError::BadInvTemp)
        );
    }

    #[test]
    fn temperature_sharpens_distribution() {
        let z = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut cold = [0.0f64; 8];
        let mut hot = [0.0f64; 8];
        S1MouseWeights::softmax_batch(&z, 8, 2.0, &mut cold).unwrap();
        S1MouseWeights::softmax_batch(&z, 8, 0.5, &mut hot).unwrap();
        assert!(cold[0] > hot[0], "higher inv_temp sharpens the top-1 mass");
    }

    #[test]
    fn publish_roundtrip_is_atomic() {
        clear_active();
        assert!(active().is_none());
        let w = S1MouseWeights::load_from_bytes(GOLDEN_BIN).expect("golden loads");
        publish(w);
        let a = active().expect("published");
        assert!((a.inv_temp() - 1.7).abs() < 1e-6);
        clear_active();
        assert!(active().is_none());
    }
}
