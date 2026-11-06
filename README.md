# edge-mouse

[![Architecture diagram](https://gitdiagram.com/diagram-badge.svg)](https://gitdiagram.com/movahedi-ca/edge-mouse?utm_source=readme&utm_medium=badge)

**Edge-local mouse control for X11:** calibrated target resolution, Fitts-timed movement, and predicate-verified actuation. Pure synchronous Rust; no async runtime on the actuation path.

Version: **0.3.0** · License: **PolyForm Noncommercial 1.0.0**

---

## Overview

`edge-mouse` resolves a natural-language target (for example, “click the Data Protection option”) to a screen coordinate, drives the pointer there with honest Fitts timing, and then **proves** what happened using predicate-based visual checks over X11 captures.

It is designed for agents and automation that must click real UI on a headed Linux desktop without guessing coordinates, clamping off-window points, or shipping password pixels off the machine.

Three stages:

1. **Resolution** (`resolver`) — exact match, verified identity cache, real BM25, and an optional S1 neural ranker (`s1_weights`) over a frozen 137-dim feature schema (`features`). Confidence is temperature-calibrated; low confidence escalates instead of guessing.
2. **Movement** (`scheduler`) — one scheduler for move / click / drag / scroll with honest Fitts timing, asymmetric minimum-jerk easing, OU tremor, and a Legacy branch that reproduces a measured baseline. A dedicated input thread owns the X11 connection over a lock-free ring and never touches an async runtime.
3. **Verification** (`vision`, `predicates`) — tri-state verdicts (`Confirmed` / `Unchanged` / `Inconclusive`) over X11 captures. Only `Confirmed` actions are cached; only `Unchanged` actions are retried. Password fields are zeroed before any pixel data leaves the machine.

The **SEC-1 gate** holds everywhere: coordinates outside the verified window rect are **rejected**, never clamped.

Origin: extracted from the `rust/src/edge/` tree of [chromebook-vm-bridge](https://github.com/movahedi-ca/chromebook-vm-bridge). The `legacy` module is adapted from that repo’s `x11_mouse.rs`.

---

## Key features

- **Natural-language target resolution** with a multi-stage cascade (exact → identity cache → BM25 → optional S1 ranker).
- **Temperature-calibrated confidence**; low scores escalate rather than click the wrong control.
- **Typed coordinate spaces** (`coords`) with finite checks and Result-based conversion — no silent NaN → origin clicks.
- **Fitts-timed gestures** with Legacy (default, KS-validated), plus provisional Human/Fast profiles gated until available.
- **Dedicated input thread** + lock-free SPSC ring (`rtrb`); actuation path is sync-only (no tokio).
- **Predicate verification** with damage-driven settle timing; pre-press baselines for honest post-action compare.
- **Password redaction** before escalation screenshots; password nodes never enter the resolver cascade.
- **Durable run journal** (`runbook`) with SQLite, restart-safe resume, escalation budgets, and allowlists.
- **Versioned candidate tree** (`perceive`), iframe frame chains (`frame_chain`), integer-nanosecond stage timers (`telemetry`).
- Optional **`cdp` feature** for CDP frame-chain capture via chromiumoxide (runs on the caller’s async runtime only).

---

## Project status

| Area | Status |
|------|--------|
| Core library (13 modules) | Shipped (v0.3.0 / v1.2 audit line) |
| Unit + integration tests | 187 unit + 8 integration (as of latest audit commit) |
| Clippy / fmt | Clean |
| S1 ranker weights | Fixture + golden vectors under `tests/fixtures/s1_mouse/` |
| Human / Fast speed profiles | Provisional — refused until KS gate marks them available |
| Crop-and-zoom refinement for sub-12px targets | Stubbed (`needs_refinement`) |
| CDP frame capture | Optional feature `cdp` |

---

## Requirements

- **Rust** 2021 edition (toolchain recent enough for the dependency set).
- **Linux + X11** for real actuation (`x11rb` with xinput / xtest / damage).
- Optional: Chrome/Chromium + CDP for perception backends that use the `cdp` feature.

This is a **library crate**, not a standalone daemon. Host applications (for example chromebook-vm-bridge) provide perception, capture, and X11 connection wiring via traits.

CPU note: `.cargo/config.toml` targets `x86-64-v3` (AVX2) for f32 GEMV / embedding loops. Adjust or remove that flag if you need broader CPU support.

---

## Installation

Add as a path or git dependency from a host crate:

```toml
[dependencies]
edge-mouse = { git = "https://github.com/movahedi-ca/edge-mouse", tag = "v0.3.0" }
# optional CDP support:
# edge-mouse = { git = "https://github.com/movahedi-ca/edge-mouse", features = ["cdp"] }
```

Or clone and build locally:

```bash
git clone https://github.com/movahedi-ca/edge-mouse.git
cd edge-mouse
cargo build --release
```

With CDP:

```bash
cargo build --release --features cdp
```

---

## Quick start (library usage)

The integrated pipeline lives in `engine` and is driven through the runbook’s `StepDriver` contract for a single **(page, natural-language target, declared predicate)** task:

```text
perceive  →  CDP/DOM snapshot → versioned tree
resolve   →  resolver cascade → target box or typed escalation
actuate   →  typed coords → scheduler (Legacy) → X11 click
verify    →  predicate verdict + damage-driven settle
journal   →  SQLite run record (no double-execute on resume)
```

Host code implements three traits and supplies geometry:

| Trait | Role |
|-------|------|
| `Perception` | Snapshot → versioned tree + predicate eval context + page text |
| `Actuator` | Typed viewport point → click (e.g. `X11Actuator`) |
| `Capturer` | Region BGRX captures + redacted full-viewport shot |

```rust
use edge_mouse::engine::{EngineDriver, Geometry, X11Actuator, check_profile_available};
use edge_mouse::scheduler::SpeedProfile;
// … Perception + Capturer implementations from the host …

// Default profile is Legacy (KS-validated).
// Human/Fast are refused until check_profile_available succeeds.
assert!(check_profile_available(SpeedProfile::Legacy).is_ok());
```

Unit tests use fakes; production wires live CDP (off the input thread) and a real X11 connection owned by the actuation path.

---

## Architecture

```
┌─────────────┐     snapshot      ┌──────────────┐     hit/box     ┌─────────────┐
│  perceive   │ ───────────────►  │   resolver   │ ─────────────►  │  scheduler  │
│  (tree)     │                   │  (+ S1 opt.) │                 │  (X11 in)   │
└─────────────┘                   └──────────────┘                 └──────┬──────┘
       ▲                                                                  │ click
       │                                                           ┌──────▼──────┐
┌──────┴──────┐     verdicts       ┌──────────────┐                │   vision    │
│   runbook   │ ◄───────────────── │  predicates  │ ◄───────────── │  (capture)  │
│  (journal)  │                    └──────────────┘                └─────────────┘
└─────────────┘
       ▲
       │ orchestrates
┌──────┴──────┐
│   engine    │  StepDriver over perceive → resolve → actuate → verify
└─────────────┘
```

**Modules**

| Module | Responsibility |
|--------|----------------|
| `coords` | Typed coordinate spaces, frame transforms, SEC-1-safe conversion |
| `features` | Frozen 137-dim feature schema for ranking |
| `s1_weights` | Optional neural ranker weights / calibrated scores |
| `resolver` | Cascade: exact, identity cache, BM25, S1; escalation on low confidence |
| `scheduler` | Gesture planning (Fitts, easing, tremor), input thread, emit |
| `legacy` | Trajectory primitives vendored from the bridge input driver |
| `vision` | Captures, dHash / luma baselines, damage probe, verdicts |
| `predicates` | Declared success conditions (tri-state evaluation) |
| `perceive` | Versioned candidate tree, feed into resolver |
| `frame_chain` | Iframe frame chains |
| `engine` | Integration runner implementing `StepDriver` |
| `runbook` | Durable SQLite journal, budgets, allowlists, resume |
| `telemetry` | Integer-nanosecond stage timers |

**Invariants**

- Only `Verdict::Confirmed` commits to the identity cache.
- Only `Verdict::Unchanged` triggers retry of the same action.
- Coordinates outside the verified window rect are rejected (SEC-1).
- Password fields never enter the cascade; escalation shots must be redacted.
- No tokio on the actuation path; CDP (if enabled) blocks on the caller’s runtime outside the input thread.

---

## Project structure

```
.
├── Cargo.toml
├── LICENSE                 # PolyForm Noncommercial 1.0.0
├── README.md
├── .cargo/config.toml      # target-cpu=x86-64-v3
├── src/
│   ├── lib.rs
│   ├── coords.rs
│   ├── engine.rs
│   ├── features.rs
│   ├── frame_chain.rs
│   ├── legacy.rs
│   ├── perceive.rs
│   ├── predicates.rs
│   ├── resolver.rs
│   ├── runbook.rs
│   ├── s1_weights.rs
│   ├── scheduler.rs
│   ├── telemetry.rs
│   └── vision.rs
└── tests/
    ├── coords_golden.rs
    └── fixtures/s1_mouse/  # golden.bin / golden.json
```

---

## Development

```bash
cargo build --release
cargo test
cargo test --release
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Release profile (from `Cargo.toml`):

- `opt-level = 3`
- fat LTO
- `codegen-units = 1`
- strip symbols
- panic = **unwind** (required so OCR workers can `catch_unwind`; abort was removed after it defeated that isolation)

---

## Testing

- Unit tests live next to modules; integration coverage includes coords golden vectors and S1 fixtures under `tests/fixtures/s1_mouse/`.
- Latest recorded suite: **187 unit + 8 integration** tests green on both dev and release profiles (see commit history for the v1.2 audit line).
- Clippy and fmt are expected clean.

Live X11 actuation and CDP perception are the responsibility of the host application’s integration tests; this crate’s CI surface is pure library tests.

---

## Security and privacy

- **SEC-1**: out-of-window coordinates are rejected, never clamped to the edge.
- **Password handling**: password nodes are filtered at feed time; values are not copied into accessible names; escalation screenshots must zero password boxes before encode (`Capturer::redacted_full_shot`).
- **Allowlists**: runbook allowlist matching is exact/suffix (not substring).
- **Escalation budgets / strikes** persist across restarts so a crash cannot reset limits.
- Mined ranking rows (if enabled by the host) are intended to stay **outside** public repos; this repository does not ship page text corpora.

Treat any process that can call the library as able to move the local pointer and read the screen — same threat model as other desktop automation.

---

## Limitations

- **X11 only** for real input (no Wayland backend in-tree).
- **Library**, not a service: no HTTP/MCP surface here; host crates supply that.
- Human/Fast profiles are provisional until KS gates pass.
- Sub-12 physical-px targets set `needs_refinement`; full crop-and-zoom vision is not complete.
- Default build targets `x86-64-v3`; older CPUs need a local `.cargo/config.toml` change.

---

## License

**PolyForm Noncommercial License 1.0.0**

Free for noncommercial use (personal, research, education, charitable, government, etc., as defined in the license text). **Any commercial use requires a separate commercial license from Movahedi Solutions.**

See [LICENSE](LICENSE) and https://polyformproject.org/licenses/noncommercial/1.0.0/

---

## Contributing

1. Open an issue describing the change.
2. Keep the actuation path synchronous and free of tokio.
3. Do not weaken SEC-1, password redaction, or allowlist exactness.
4. Run `cargo test`, `cargo test --release`, `cargo clippy --all-targets`, and `cargo fmt --check`.
5. Submit a PR against `main`.

Commercial licensing inquiries: contact Movahedi Solutions via the repository owner.

---

## Support and references

| Resource | Link |
|----------|------|
| Source | https://github.com/movahedi-ca/edge-mouse |
| Issues | https://github.com/movahedi-ca/edge-mouse/issues |
| Parent project | [chromebook-vm-bridge](https://github.com/movahedi-ca/chromebook-vm-bridge) |
| License text | [LICENSE](LICENSE) |

Module-level rustdoc in `src/` is the detailed API reference for integrators.
