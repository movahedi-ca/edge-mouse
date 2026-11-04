//! edge-mouse: edge-local mouse control for X11.
//!
//! A self-contained Rust library for calibrated, verified mouse actuation:
//!
//! - **Resolution** ([`resolver`]): exact match, verified identity cache,
//!   real BM25, and an optional S1 neural ranker ([`s1_weights`]) over a
//!   frozen 137-dim feature schema ([`features`]).
//! - **Movement** ([`scheduler`]): one scheduler for move/click/drag/scroll
//!   with honest Fitts timing, a Legacy branch reproducing the measured
//!   baseline, and a dedicated input thread that never touches an async
//!   runtime.
//! - **Verification** ([`vision`], [`predicates`]): predicate-based,
//!   tri-state verdicts (`Confirmed` / `Unchanged` / `Inconclusive`) over
//!   X11 captures, with password-field redaction before any pixel data
//!   leaves the machine.
//! - **Orchestration** ([`engine`], [`runbook`]): run journal with
//!   restart-safe resume, escalation budgets, and per-run allowlists.
//! - **Perception** ([`perceive`], [`coords`], [`frame_chain`]):
//!   versioned candidate tree, typed coordinate spaces, iframe frame
//!   chains. CDP frame capture lives behind the `cdp` feature and runs on
//!   the caller's async runtime, never on the input thread.
//!
//! [`legacy`] holds the vendored trajectory primitives (window rect and
//! its SEC-1 gate, easing, Bezier builders, arc-length cursor) adapted
//! from the chromebook-vm-bridge input driver.
//!
//! Pure synchronous Rust throughout. No tokio on the actuation path.

#![allow(dead_code)]

pub mod coords;
pub mod engine;
pub mod features;
pub mod frame_chain;
pub mod legacy;
pub mod perceive;
pub mod predicates;
pub mod resolver;
pub mod runbook;
pub mod s1_weights;
pub mod scheduler;
pub mod telemetry;
pub mod vision;
