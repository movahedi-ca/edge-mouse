# edge-mouse

Edge-local mouse control for X11: calibrated target resolution, Fitts-timed
movement, and predicate-verified actuation. Pure synchronous Rust; no async
runtime on the actuation path.

## What it is

`edge-mouse` resolves a natural-language target ("click the Data Protection
option") to a screen coordinate and drives the pointer there, then proves
what happened. Three stages:

1. **Resolution** (`resolver`): exact match, verified identity cache, real
   BM25, and an optional S1 neural ranker (`s1_weights`) over a frozen
   137-dim feature schema (`features`). Confidence is temperature-calibrated;
   low confidence escalates instead of guessing.
2. **Movement** (`scheduler`): one scheduler for move/click/drag/scroll with
   honest Fitts timing, asymmetric minimum-jerk easing, OU tremor, and a
   Legacy branch reproducing the measured baseline. A dedicated input thread
   owns the X11 connection and communicates over a lock-free ring; it never
   touches tokio.
3. **Verification** (`vision`, `predicates`): predicate-based, tri-state
   verdicts (`Confirmed` / `Unchanged` / `Inconclusive`) over X11 captures.
   Only `Confirmed` actions are cached; only `Unchanged` actions are
   retried. Password fields are zeroed before any pixel data leaves the
   machine.

Supporting modules: `engine` (integration runner), `runbook` (durable
journal with restart-safe resume, escalation budgets, allowlists),
`perceive` (versioned candidate tree), `coords` (typed coordinate spaces),
`frame_chain` (iframe frame chains), `telemetry` (integer-nanosecond stage
timers), and `legacy` (trajectory primitives vendored from the
chromebook-vm-bridge input driver).

The SEC-1 gate holds everywhere: coordinates outside the verified window
rect are rejected, never clamped.

## Build and test

```sh
cargo build --release
cargo test
cargo clippy --all-targets
cargo fmt --check
```

The `cdp` feature enables CDP frame-chain capture via chromiumoxide. It
runs on the caller's async runtime, never on the input thread:

```sh
cargo build --features cdp
```

## Origin

Extracted from the `rust/src/edge/` tree of
[chromebook-vm-bridge](https://github.com/movahedi-ca/chromebook-vm-bridge).
The `legacy` module is adapted from that repo's `x11_mouse.rs`.

## License

MIT. See [LICENSE](LICENSE).
