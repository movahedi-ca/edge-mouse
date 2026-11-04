//! Layer 1: versioned event-driven perception tree (mouse-future Phase 5).
//!
//! The target from plan v7 section 3, Layer 1: an event-driven candidate
//! tree with a concrete lifecycle. `DOM.getDocument` once, then DOM
//! child/attribute events applied as ATOMIC batches to a versioned tree
//! state (document generation, frame id, node identity, last mutation
//! sequence, AX/DOM merge status). The resolver sees IMMUTABLE generation
//! snapshots. Navigation, frame attach/detach, execution-context reset,
//! node removal, and event-stream desync are explicit invalidation paths.
//! In test mode the incremental tree is periodically diffed against a
//! full DOM/AX rebuild; drift forces a rebuild. A backend node id alone
//! is not a durable identity: identity is (node id, document generation).
//!
//! # Measured 2026-11-03: DOM mutation events do NOT flow on Chrome 156
//!
//! Live evaluation on the chromebook-vm (Chrome 156, dedicated desktop
//! vd8) injected 45 real DOM mutations (20 childNodeInserted, 5
//! childNodeRemoved, 10 attributeModified, 5 attributeRemoved, 10
//! characterDataModified; verified present in the DOM afterwards) with
//! `DOM.enable` + `DOM.getDocument` priming on the current document.
//! Result: **0 of 45 events delivered** (`DOM.childNodeInserted`,
//! `DOM.childNodeRemoved`, `DOM.attributeModified`,
//! `DOM.attributeRemoved`, `DOM.characterDataModified`,
//! `DOM.setChildNodes`, `DOM.documentUpdated` all silent), while
//! `Page.frameNavigated` delivered 1/1 through the same listener path.
//! The vendored chromiumoxide 0.7 dispatch was traced and is correct
//! (session-scoped events reach the target's listeners); Chrome simply
//! does not emit the DOM mutation events.
//!
//! Per the task's interim rule, perception therefore ships
//! **DOMSnapshot-first**: every perceive step rebuilds from a fresh
//! snapshot, the drift detector runs as the correctness gate, and the
//! event-batch machinery below (`EventBatch`, `apply_batch`) stays as
//! real, unit-tested code that is NOT the default path until event
//! delivery is re-proven on a future Chrome. Nothing here fakes an
//! event layer.
//!
//! # Snapshot model: Arc-swap (documented choice)
//!
//! [`VersionedTree`] owns the mutable state. Every committed mutation
//! (full rebuild or applied event batch) publishes a new immutable
//! [`GenerationSnapshot`] behind an `Arc`; readers (the resolver, the
//! Tier-0 index, the OCR worker's generation checks) hold the `Arc` and
//! never touch the mutable tree. The alternative considered was
//! clone-on-write of individual subtrees (persistent-map style); it was
//! rejected because at this scale (hundreds of nodes per page) a full
//! snapshot rebuild per committed batch is single-digit microseconds and
//! keeps the read path lock-free with no aliasing hazards. There is no
//! mutex on the read path: `Arc::clone` is the only synchronization.
//!
//! # Atomic batches
//!
//! A [`EventBatch`] carries a stream sequence number and the generation
//! it applies to. Application is two-phase: every event is validated
//! against a scratch copy of the state first, and only when ALL events
//! validate does the batch commit. A single invalid event rejects the
//! WHOLE batch: the committed state is never left half-applied, and the
//! tree is flagged desynced so the caller runs the explicit invalidation
//! path (full rebuild, generation bump). Partial application is the
//! silent-drift bug this design exists to prevent.
//!
//! # CDP event names (verified, not guessed)
//!
//! Subscription identifiers in [`recommended_subscriptions`] are the exact
//! `IDENTIFIER` constants from the generated CDP bindings the vendored
//! chromiumoxide 0.7 links against (chromiumoxide_cdp 0.7.0, registry
//! source read 2026-11-03), NOT upstream documentation:
//!
//! ```text
//! DOM.documentUpdated            EventDocumentUpdated        (no payload)
//! DOM.setChildNodes              EventSetChildNodes          { parent_node_id, nodes[] }
//! DOM.childNodeInserted          EventChildNodeInserted      { parent_node_id, previous_node_id, node }
//! DOM.childNodeRemoved           EventChildNodeRemoved       { parent_node_id, node_id }
//! DOM.attributeModified          EventAttributeModified      { node_id, name, value }
//! DOM.attributeRemoved           EventAttributeRemoved       { node_id, name }
//! DOM.characterDataModified      EventCharacterDataModified  { node_id, characterData }
//! DOM.childNodeCountUpdated      EventChildNodeCountUpdated  { node_id, childNodeCount }
//! Page.frameNavigated            EventFrameNavigated         { frame }
//! Page.frameAttached             EventFrameAttached          { frameId, parentFrameId }
//! Page.frameDetached             EventFrameDetached          { frameId }
//! Page.documentOpened            EventDocumentOpened         { frame }
//! Runtime.executionContextsCleared  EventExecutionContextsCleared  (no payload)
//! ```
//!
//! Two load-bearing facts about these events:
//!
//! 1. DOM mutation events carry the session-scoped `NodeId`, NOT
//!    `backendNodeId`. The tree maintains the `NodeId -> backendNodeId`
//!    map from `getDocument`/`setChildNodes` payloads (both carry
//!    `backendNodeId` per node). An event referencing an unknown `NodeId`
//!    is a desync signal: the batch is rejected, never partially applied.
//! 2. The Accessibility domain exposes NO change events in cdp 0.7
//!    (verified: the generated `accessibility` module contains only
//!    commands and types). AX name/role freshness is therefore
//!    poll-based; [`VersionedTree`] tracks the AX merge status per
//!    generation and surfaces staleness instead of pretending the AX
//!    side is event-driven.
//!
//! # Interim path: DOMSnapshot-first
//!
//! Until the event layer is proven against Chrome 156 on the VM,
//! perception runs DOMSnapshot-first: one `DOMSnapshot.captureSnapshot`
//! per page, parsed into [`DomSnapshotNode`]s (main-viewport CSS px, via
//! the B1 frame chain at the call site), built into a fresh generation
//! with the incremental `(backendNodeId, text_hash)` index. The event
//! batch machinery above is real and tested, but it is NOT the default
//! path until the Chrome 156 reliability evaluation (see the B5 report)
//! clears it. This module never fakes an event layer: if the events
//! prove flaky, the interim stays and the docs say so.
//!
//! # Threading contract
//!
//! No tokio on these paths. The tree is plain synchronous Rust behind
//! `Arc`. The OCR worker owns two dedicated blocking `std::thread`
//! workers; each owns one Tesseract execution slot for its lifetime.
//! The resolver runs on the calling thread. Gesture dispatch stays on
//! the rtrb ring owned by the input thread (B3).

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use super::coords::FrameTransform;
use super::resolver::{fx64, normalize_desc, Hit};
use super::vision::{validate_capture, Capture, OcrOutcome, OcrRequest};

// ---------------------------------------------------------------------------
// Generations and node identity.
// ---------------------------------------------------------------------------

/// Document generation: bumped on every explicit invalidation (navigation,
/// frame attach/detach, execution-context reset, event-stream desync,
/// `DOM.documentUpdated`, full rebuild). A backend node id alone is NOT a
/// durable identity: identity is (backend node id, generation).
pub type Generation = u64;

/// Generation before any document exists.
pub const NO_DOCUMENT: Generation = 0;

/// Session-scoped CDP DOM `NodeId` (i64 on the wire). Distinct from
/// `backendNodeId`: NodeIds die with the document, backend ids are the
/// stable-per-document handle the rest of the engine uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeRef(pub i64);

// ---------------------------------------------------------------------------
// TreeNode: one structural candidate in one generation.
// ---------------------------------------------------------------------------

/// One node in the versioned tree. Coordinates are MAIN-VIEWPORT CSS px,
/// normalized ONCE at insertion through the B1 frame chain by the caller
/// (the same contract as the resolver's `Candidate.at`); the tree never
/// stores frame-document coordinates, so coordinates are never
/// transformed twice.
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Stable node id within the document generation.
    pub backend_node_id: u64,
    /// Document generation this node belongs to.
    pub generation: Generation,
    /// Normalized role string (same normalization as the resolver).
    pub role: Box<str>,
    /// Normalized accessible name.
    pub name: Box<str>,
    /// Normalized text content.
    pub text: Box<str>,
    /// fx64(role|name|text): the cache-validation fingerprint, same
    /// contract as `PreparedNode.text_hash`.
    pub text_hash: u64,
    /// Owning frame id (frame lifecycle events key off this).
    pub frame_id: Box<str>,
    /// Box top-left, main-viewport CSS px.
    pub at: (f64, f64),
    /// Box width/height, CSS px.
    pub wh: (f64, f64),
    /// Device pixel ratio governing this node.
    pub dpr: f64,
    /// AX enabled+visible state at insertion.
    pub enabled: bool,
    pub visible: bool,
    /// `input[type=password]`: excluded from Tier-0 clicks, included in
    /// the password-box list for screenshot redaction.
    pub is_password: bool,
    /// Session-scoped CDP DOM NodeId. `None` for nodes built from
    /// DOMSnapshot (the interim path has no NodeIds); `Some` for nodes
    /// inserted or refreshed through the event stream.
    pub node_ref: Option<NodeRef>,
}

impl TreeNode {
    /// Build a node, normalizing role/name/text and computing the text
    /// hash. `at`/`wh` are main-viewport CSS px (caller normalizes).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend_node_id: u64,
        generation: Generation,
        role: &str,
        name: &str,
        text: &str,
        frame_id: &str,
        at: (f64, f64),
        wh: (f64, f64),
        dpr: f64,
        enabled: bool,
        visible: bool,
        is_password: bool,
    ) -> Self {
        let role_n = normalize_desc(role);
        let name_n = normalize_desc(name);
        let text_n = normalize_desc(text);
        let text_hash = fx64(format!("{role_n}|{name_n}|{text_n}").as_bytes());
        Self {
            backend_node_id,
            generation,
            role: role_n.into(),
            name: name_n.into(),
            text: text_n.into(),
            text_hash,
            frame_id: frame_id.into(),
            at,
            wh,
            dpr: if dpr.is_finite() && dpr > 0.0 {
                dpr
            } else {
                1.0
            },
            enabled,
            visible,
            is_password,
            node_ref: None,
        }
    }

    /// Attach the session NodeId (event-stream inserts).
    pub fn with_node_ref(mut self, node_ref: NodeRef) -> Self {
        self.node_ref = Some(node_ref);
        self
    }

    /// Box center, main-viewport CSS px.
    pub fn center(&self) -> (f64, f64) {
        (self.at.0 + self.wh.0 / 2.0, self.at.1 + self.wh.1 / 2.0)
    }

    /// Zero-area nodes never enter the spatial index.
    pub fn has_area(&self) -> bool {
        self.wh.0 >= 1.0 && self.wh.1 >= 1.0
    }
}

// ---------------------------------------------------------------------------
// Events and atomic batches.
// ---------------------------------------------------------------------------

/// The AX-side refresh for one node. CDP offers no AX change events, so
/// these arrive from the poll-based AX merge, not from the event stream.
#[derive(Debug, Clone)]
pub struct AxMerge {
    pub backend_node_id: u64,
    pub role: String,
    pub name: String,
}

/// One tree mutation, in CDP-event terms. The wire payloads this models
/// are listed in the module docs; field names mirror the generated
/// `Event*` structs.
#[derive(Debug, Clone)]
pub enum TreeEvent {
    /// `DOM.setChildNodes`: authoritative child list for a parent.
    SetChildNodes {
        parent: NodeRef,
        nodes: Vec<NodeWire>,
    },
    /// `DOM.childNodeInserted`.
    ChildInserted {
        parent: NodeRef,
        previous: NodeRef,
        node: NodeWire,
    },
    /// `DOM.childNodeRemoved`.
    ChildRemoved { parent: NodeRef, node: NodeRef },
    /// `DOM.attributeModified`. Only attributes that affect the tree
    /// (class/style/hidden/disabled/aria-*) are applied; the rest are
    /// recorded and ignored.
    AttrModified {
        node: NodeRef,
        name: Box<str>,
        value: Box<str>,
    },
    /// `DOM.attributeRemoved`.
    AttrRemoved { node: NodeRef, name: Box<str> },
    /// `DOM.characterDataModified`.
    CharDataModified { node: NodeRef, text: Box<str> },
    /// `DOM.documentUpdated`: the document is gone; the whole tree is
    /// stale. Applying this event invalidates the generation; it never
    /// patches nodes.
    DocumentUpdated,
    /// `Page.frameNavigated` on the main frame.
    MainFrameNavigated,
    /// `Page.frameAttached`.
    FrameAttached { frame_id: Box<str> },
    /// `Page.frameDetached`: every node owned by the frame leaves the tree.
    FrameDetached { frame_id: Box<str> },
    /// `Runtime.executionContextsCleared`.
    ExecutionContextsCleared,
    /// Poll-based AX merge refresh (not a CDP event; see module docs).
    AxRefreshed { nodes: Vec<AxMerge> },
}

/// The wire shape of one DOM node in `setChildNodes`/`childNodeInserted`
/// payloads: session NodeId plus the stable backendNodeId, with the
/// structural fields the tree tracks. Geometry arrives separately (the
/// interim path) or via `DOM.getBoxModel` on demand; `at`/`wh` here are
/// main-viewport CSS px when known, else the previous values are kept.
#[derive(Debug, Clone)]
pub struct NodeWire {
    pub node_ref: NodeRef,
    pub backend_node_id: u64,
    pub frame_id: String,
    pub at: Option<(f64, f64)>,
    pub wh: Option<(f64, f64)>,
    pub visible: bool,
    pub enabled: bool,
}

/// One atomic batch from the CDP event stream.
#[derive(Debug, Clone)]
pub struct EventBatch {
    /// Stream sequence: must be exactly `last_seq + 1`, else the batch
    /// is rejected as a desync (events were dropped in transit).
    pub seq: u64,
    /// Generation the batch applies to; a batch for any other generation
    /// is stale and rejected.
    pub generation: Generation,
    pub events: Vec<TreeEvent>,
}

/// Why a batch was rejected. Every variant leaves the committed state
/// untouched and flags the stream desynced: the caller MUST run the
/// explicit invalidation path (full rebuild, generation bump).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// `batch.seq != last_seq + 1`: events were dropped; the incremental
    /// state can no longer be trusted.
    SequenceGap { want: u64, got: u64 },
    /// The batch targets a generation that is not current (stale).
    WrongGeneration { want: Generation, got: Generation },
    /// An event referenced an unknown session NodeId.
    UnknownNode(NodeRef),
    /// An event referenced an unknown frame.
    UnknownFrame,
    /// A structural event was internally inconsistent.
    Inconsistent(String),
}

/// Outcome of applying a batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchOutcome {
    /// All events validated and committed.
    Applied { events: usize },
    /// The whole batch was rejected; committed state untouched, stream
    /// flagged desynced.
    Rejected { reason: RejectReason },
}

/// Explicit invalidation paths. Each bumps the generation and drops the
/// tree back to empty; the next perception cycle rebuilds from a fresh
/// `DOM.getDocument` / DOMSnapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidationReason {
    /// `Page.frameNavigated` / `Page.documentOpened` on the main frame,
    /// or `TreeEvent::MainFrameNavigated`.
    Navigation,
    /// `Page.frameAttached` / `Page.frameDetached`, or the corresponding
    /// tree events.
    FrameLifecycle,
    /// `Runtime.executionContextsCleared`, or the corresponding event.
    ExecutionContextReset,
    /// `DOM.documentUpdated`, or a rejected event batch.
    EventStreamDesync,
    /// The drift check found the incremental tree diverged from a full
    /// rebuild.
    DriftDetected,
    /// Caller asked for a clean rebuild (test mode, profile switch).
    Manual,
}

// ---------------------------------------------------------------------------
// VersionedTree: mutable state; GenerationSnapshot: immutable Arc payload.
// ---------------------------------------------------------------------------

/// Mutable per-generation node storage. Never shared across threads
/// directly; readers get [`GenerationSnapshot`] behind an `Arc`.
#[derive(Debug, Default)]
struct TreeState {
    nodes: Vec<TreeNode>,
    /// Session NodeId -> node index (the CDP event key space).
    by_node_ref: HashMap<i64, usize>,
    /// backendNodeId -> node index (the engine key space).
    by_backend: HashMap<u64, usize>,
    /// Parent NodeId -> child NodeIds (structural bookkeeping for
    /// authoritative `setChildNodes` lists).
    children: HashMap<i64, Vec<i64>>,
    /// Child NodeId -> parent NodeId.
    parent_of: HashMap<i64, i64>,
    /// Known frame ids (frame lifecycle validation).
    frames: HashMap<Box<str>, ()>,
    /// Last applied batch sequence.
    last_seq: u64,
}

/// The immutable snapshot the resolver sees. Constructed only by
/// [`VersionedTree`]; all fields are read-only after construction.
#[derive(Debug)]
pub struct GenerationSnapshot {
    /// Document generation this snapshot belongs to.
    pub generation: Generation,
    /// Nodes in document order.
    pub nodes: Vec<TreeNode>,
    /// backendNodeId -> node index.
    pub by_backend: HashMap<u64, usize>,
    /// (backendNodeId, text_hash) -> node index: the interim incremental
    /// index (plan v7: DOMSnapshot-first interim).
    pub by_text: HashMap<(u64, u64), usize>,
    /// Tier-0 spatial index over the visible, non-zero-area nodes.
    pub spatial: SpatialIndex,
    /// `input[type=password]` boxes, main-viewport CSS px (x, y, w, h),
    /// for screenshot redaction before bytes leave the VM.
    pub password_boxes: Vec<(u32, u32, u32, u32)>,
    /// AX merge status: true when the AX side was refreshed for this
    /// generation after the last structural change.
    pub ax_merged: bool,
    /// Main viewport size, CSS px, that the spatial index was built for.
    pub viewport: (f64, f64),
}

/// The versioned tree: owns the mutable state, publishes immutable
/// snapshots. Single-threaded mutation; snapshots are `Arc` for
/// lock-free readers.
pub struct VersionedTree {
    state: TreeState,
    generation: Generation,
    current: Arc<GenerationSnapshot>,
    desynced: bool,
    ax_merged: bool,
    viewport: (f64, f64),
}

impl VersionedTree {
    /// Empty tree, no document. The first `rebuild_from_snapshot` moves
    /// to generation 1.
    pub fn new(viewport: (f64, f64)) -> Self {
        Self {
            state: TreeState::default(),
            generation: NO_DOCUMENT,
            current: Arc::new(GenerationSnapshot::empty(NO_DOCUMENT, viewport)),
            desynced: false,
            ax_merged: false,
            viewport,
        }
    }

    /// Current document generation.
    pub fn generation(&self) -> Generation {
        self.generation
    }

    /// The current immutable snapshot (cheap `Arc` clone, no lock).
    pub fn snapshot(&self) -> Arc<GenerationSnapshot> {
        Arc::clone(&self.current)
    }

    /// True after a rejected batch: the event stream can no longer be
    /// trusted. The caller MUST call [`VersionedTree::invalidate`] with
    /// [`InvalidationReason::EventStreamDesync`] and rebuild.
    pub fn is_desynced(&self) -> bool {
        self.desynced
    }

    /// Number of nodes in the mutable state (tests, telemetry).
    pub fn node_count(&self) -> usize {
        self.state.nodes.len()
    }

    /// Last applied batch sequence (tests, telemetry).
    pub fn last_seq(&self) -> u64 {
        self.state.last_seq
    }

    /// Full rebuild from a fresh document: drops all state, bumps the
    /// generation, clears the desync flag, publishes a new snapshot.
    /// This is the DOMSnapshot-first interim path and the recovery path
    /// for every invalidation.
    pub fn rebuild_from_snapshot(
        &mut self,
        nodes: Vec<TreeNode>,
        frames: Vec<String>,
    ) -> Generation {
        self.generation = self.generation.wrapping_add(1).max(1);
        let mut state = TreeState::default();
        for n in &nodes {
            let mut n = n.clone();
            n.generation = self.generation;
            let idx = state.nodes.len();
            state.by_backend.insert(n.backend_node_id, idx);
            state.nodes.push(n);
        }
        for f in frames {
            state.frames.insert(f.into(), ());
        }
        // Rebuilds reset the stream sequence: the next batch must start
        // at last_seq + 1 = 1.
        state.last_seq = 0;
        self.state = state;
        self.desynced = false;
        self.ax_merged = false;
        self.publish();
        self.generation
    }

    /// Explicit invalidation: bumps the generation, drops the tree,
    /// flags no desync (the invalidation IS the recovery), publishes an
    /// empty snapshot. The caller rebuilds from a fresh document next.
    pub fn invalidate(&mut self, reason: InvalidationReason) -> Generation {
        let _ = reason;
        self.generation = self.generation.wrapping_add(1).max(1);
        self.state = TreeState::default();
        self.desynced = false;
        self.ax_merged = false;
        self.publish();
        self.generation
    }

    /// Apply one event batch ATOMICALLY: validate every event against a
    /// scratch copy first; commit only when all validate. Any failure
    /// rejects the whole batch, leaves committed state untouched, and
    /// flags the stream desynced.
    pub fn apply_batch(&mut self, batch: &EventBatch) -> BatchOutcome {
        if batch.generation != self.generation {
            self.desynced = true;
            return BatchOutcome::Rejected {
                reason: RejectReason::WrongGeneration {
                    want: self.generation,
                    got: batch.generation,
                },
            };
        }
        if batch.seq != self.state.last_seq + 1 {
            self.desynced = true;
            return BatchOutcome::Rejected {
                reason: RejectReason::SequenceGap {
                    want: self.state.last_seq + 1,
                    got: batch.seq,
                },
            };
        }
        // Phase 1: validate + apply onto a scratch clone.
        let mut scratch = self.state.clone_for_batch();
        for ev in &batch.events {
            if let Err(reason) = apply_event(&mut scratch, ev, self.generation) {
                // Phase 1 failed: discard the scratch, flag desync.
                self.desynced = true;
                return BatchOutcome::Rejected { reason };
            }
        }
        // Phase 2: all events validated; commit.
        scratch.last_seq = batch.seq;
        self.state = scratch;
        self.publish();
        BatchOutcome::Applied {
            events: batch.events.len(),
        }
    }

    /// Record that the poll-based AX merge refreshed this generation.
    /// Until this is called after structural changes, `ax_merged` is
    /// false and consumers must treat names/roles as potentially stale.
    pub fn note_ax_merged(&mut self) {
        self.ax_merged = true;
        self.publish();
    }

    /// Rebuild the immutable snapshot from the mutable state and swap
    /// the `Arc`. Readers holding the old `Arc` keep seeing the old
    /// generation, untouched.
    fn publish(&mut self) {
        let snap = GenerationSnapshot::build(
            self.generation,
            &self.state.nodes,
            self.ax_merged,
            self.viewport,
        );
        self.current = Arc::new(snap);
    }
}

impl TreeState {
    /// Scratch copy for batch validation. `Vec` clone is O(n); at this
    /// scale (hundreds of nodes) it is microseconds, and it is what
    /// makes the batch atomic without a persistent data structure.
    fn clone_for_batch(&self) -> TreeState {
        TreeState {
            nodes: self.nodes.clone(),
            by_node_ref: self.by_node_ref.clone(),
            by_backend: self.by_backend.clone(),
            children: self.children.clone(),
            parent_of: self.parent_of.clone(),
            frames: self.frames.clone(),
            last_seq: self.last_seq,
        }
    }

    /// Rebuild the backend-id index from the node vec. The NodeRef map is
    /// maintained incrementally by the mutation helpers (NodeRefs live on
    /// the nodes, so it could be rebuilt too, but incremental updates are
    /// O(1) and obviously correct).
    fn reindex_backend(&mut self) {
        self.by_backend.clear();
        for (i, n) in self.nodes.iter().enumerate() {
            self.by_backend.insert(n.backend_node_id, i);
        }
    }
}

/// Apply one event to scratch state. `Err` means the batch must be
/// rejected; the scratch is discarded by the caller.
fn apply_event(
    state: &mut TreeState,
    ev: &TreeEvent,
    generation: Generation,
) -> Result<(), RejectReason> {
    match ev {
        TreeEvent::SetChildNodes { parent, nodes } => {
            // NodeRef(0) is the implicit document root: the initial
            // DOM.getDocument load arrives as setChildNodes under it.
            if parent.0 != 0 && !state.by_node_ref.contains_key(&parent.0) {
                return Err(RejectReason::UnknownNode(*parent));
            }
            // Authoritative child list: children present before but absent
            // now leave the tree; every listed child is upserted.
            let mut new_set = std::collections::HashSet::new();
            for w in nodes {
                new_set.insert(w.node_ref.0);
            }
            let old_children: Vec<i64> = state.children.get(&parent.0).cloned().unwrap_or_default();
            for old in old_children {
                if !new_set.contains(&old) {
                    if let Some(&idx) = state.by_node_ref.get(&old) {
                        remove_node_at(state, idx);
                    }
                }
            }
            for w in nodes {
                upsert_wire_node(state, *parent, w, generation)?;
            }
            Ok(())
        }
        TreeEvent::ChildInserted {
            parent,
            previous,
            node,
        } => {
            if parent.0 != 0 && !state.by_node_ref.contains_key(&parent.0) {
                return Err(RejectReason::UnknownNode(*parent));
            }
            if previous.0 != 0 && !state.by_node_ref.contains_key(&previous.0) {
                return Err(RejectReason::UnknownNode(*previous));
            }
            upsert_wire_node(state, *parent, node, generation)
        }
        TreeEvent::ChildRemoved { parent, node } => {
            if !state.by_node_ref.contains_key(&parent.0) {
                return Err(RejectReason::UnknownNode(*parent));
            }
            let idx = *state
                .by_node_ref
                .get(&node.0)
                .ok_or(RejectReason::UnknownNode(*node))?;
            remove_node_at(state, idx);
            Ok(())
        }
        TreeEvent::AttrModified { node, name, value } => {
            let idx = *state
                .by_node_ref
                .get(&node.0)
                .ok_or(RejectReason::UnknownNode(*node))?;
            let n = &mut state.nodes[idx];
            apply_attr(n, name, value, true);
            Ok(())
        }
        TreeEvent::AttrRemoved { node, name } => {
            let idx = *state
                .by_node_ref
                .get(&node.0)
                .ok_or(RejectReason::UnknownNode(*node))?;
            let n = &mut state.nodes[idx];
            apply_attr(n, name, "", false);
            Ok(())
        }
        TreeEvent::CharDataModified { node, text } => {
            let idx = *state
                .by_node_ref
                .get(&node.0)
                .ok_or(RejectReason::UnknownNode(*node))?;
            let n = &mut state.nodes[idx];
            n.text = normalize_desc(text).into();
            n.text_hash = fx64(format!("{}|{}|{}", n.role, n.name, n.text).as_bytes());
            Ok(())
        }
        TreeEvent::DocumentUpdated
        | TreeEvent::MainFrameNavigated
        | TreeEvent::ExecutionContextsCleared => {
            // These are INVALIDATION signals, not patches. A batch that
            // carries one is rejected with Inconsistent so the caller
            // takes the explicit invalidation path; the event must never
            // be "applied" as a node mutation.
            Err(RejectReason::Inconsistent(
                "document-level event is an invalidation signal, not a patch".to_string(),
            ))
        }
        TreeEvent::FrameAttached { frame_id } => {
            state.frames.insert(frame_id.clone(), ());
            Ok(())
        }
        TreeEvent::FrameDetached { frame_id } => {
            if !state.frames.contains_key(frame_id) {
                return Err(RejectReason::UnknownFrame);
            }
            state.frames.remove(frame_id);
            // Every node owned by the detached frame leaves the tree.
            let owned: Vec<usize> = state
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.frame_id.as_ref() == frame_id.as_ref())
                .map(|(i, _)| i)
                .collect();
            for i in owned.into_iter().rev() {
                remove_node_at(state, i);
            }
            Ok(())
        }
        TreeEvent::AxRefreshed { nodes } => {
            for m in nodes {
                if let Some(&idx) = state.by_backend.get(&m.backend_node_id) {
                    let n = &mut state.nodes[idx];
                    n.role = normalize_desc(&m.role).into();
                    n.name = normalize_desc(&m.name).into();
                    n.text_hash = fx64(format!("{}|{}|{}", n.role, n.name, n.text).as_bytes());
                }
                // Unknown backend ids in an AX refresh are tolerated
                // (the AX tree and DOM tree enumerate slightly different
                // node sets); they are not a desync signal.
            }
            Ok(())
        }
    }
}

/// Insert or replace a node from a wire payload. Geometry falls back to
/// the previous values when the payload carries none.
fn upsert_wire_node(
    state: &mut TreeState,
    parent: NodeRef,
    w: &NodeWire,
    generation: Generation,
) -> Result<(), RejectReason> {
    link_child(state, parent.0, w.node_ref.0);
    if let Some(&idx) = state.by_node_ref.get(&w.node_ref.0) {
        let n = &mut state.nodes[idx];
        if let Some(at) = w.at {
            n.at = at;
        }
        if let Some(wh) = w.wh {
            n.wh = wh;
        }
        n.visible = w.visible;
        n.enabled = w.enabled;
        n.generation = generation;
        return Ok(());
    }
    let idx = state.nodes.len();
    let node = TreeNode::new(
        w.backend_node_id,
        generation,
        "",
        "",
        "",
        &w.frame_id,
        w.at.unwrap_or((0.0, 0.0)),
        w.wh.unwrap_or((0.0, 0.0)),
        1.0,
        w.enabled,
        w.visible,
        false,
    )
    .with_node_ref(w.node_ref);
    state.by_node_ref.insert(w.node_ref.0, idx);
    state.by_backend.insert(w.backend_node_id, idx);
    state.nodes.push(node);
    Ok(())
}

/// Record a parent -> child edge. Idempotent: re-linking an existing
/// edge is a no-op.
fn link_child(state: &mut TreeState, parent: i64, child: i64) {
    let entry = state.children.entry(parent).or_default();
    if !entry.contains(&child) {
        entry.push(child);
    }
    state.parent_of.insert(child, parent);
}

/// Remove the node at `idx`, keeping every map consistent. Swap-remove is
/// O(1); the moved node's NodeRef entry is fixed from the node now sitting
/// at `idx`; structural edges are unlinked; the backend index is rebuilt
/// (O(n), rare path).
fn remove_node_at(state: &mut TreeState, idx: usize) {
    debug_assert!(idx < state.nodes.len());
    let last = state.nodes.len() - 1;
    let dead_ref = state.nodes[idx].node_ref.map(|r| r.0);
    if let Some(r) = dead_ref {
        state.by_node_ref.remove(&r);
        // Unlink the structural edges.
        if let Some(p) = state.parent_of.remove(&r) {
            if let Some(kids) = state.children.get_mut(&p) {
                kids.retain(|&k| k != r);
            }
        }
        state.children.remove(&r);
    }
    state.nodes.swap_remove(idx);
    if idx != last {
        // The node that was at `last` now sits at `idx`: fix its entry.
        if let Some(r) = state.nodes[idx].node_ref {
            state.by_node_ref.insert(r.0, idx);
        }
    }
    state.reindex_backend();
}

/// Apply an attribute mutation that affects the tree. Returns nothing;
/// unknown attributes are ignored (they do not affect targeting).
fn apply_attr(n: &mut TreeNode, name: &str, value: &str, present: bool) {
    let lname = name.to_ascii_lowercase();
    match lname.as_str() {
        "hidden" => {
            n.visible = !present;
        }
        "disabled" => {
            n.enabled = !present;
        }
        "aria-hidden" => {
            n.visible = !(present && value.trim() == "true");
        }
        "aria-disabled" => {
            n.enabled = !(present && value.trim() == "true");
        }
        "aria-label" | "aria-labelledby" | "alt" | "title" | "placeholder" | "value" => {
            // Name-affecting attributes: recompute the hash. The full
            // accessible-name computation needs the document; the event
            // layer records the attribute value as the name candidate and
            // the next AX merge fixes it authoritatively.
            if present {
                n.name = normalize_desc(value).into();
                n.text_hash = fx64(format!("{}|{}|{}", n.role, n.name, n.text).as_bytes());
            }
        }
        "type" => {
            n.is_password = present && value.trim().eq_ignore_ascii_case("password");
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// GenerationSnapshot construction.
// ---------------------------------------------------------------------------

impl GenerationSnapshot {
    /// Empty snapshot (pre-document or post-invalidation).
    fn empty(generation: Generation, viewport: (f64, f64)) -> Self {
        Self {
            generation,
            nodes: Vec::new(),
            by_backend: HashMap::new(),
            by_text: HashMap::new(),
            spatial: SpatialIndex::empty(viewport),
            password_boxes: Vec::new(),
            ax_merged: false,
            viewport,
        }
    }

    /// Build the immutable snapshot from the mutable node vec. All
    /// indices are derived here, in one pass each, so they can never
    /// disagree with the node vec they describe.
    fn build(
        generation: Generation,
        nodes: &[TreeNode],
        ax_merged: bool,
        viewport: (f64, f64),
    ) -> Self {
        let mut by_backend = HashMap::with_capacity(nodes.len());
        let mut by_text = HashMap::with_capacity(nodes.len());
        let mut password_boxes = Vec::new();
        for (i, n) in nodes.iter().enumerate() {
            by_backend.insert(n.backend_node_id, i);
            by_text.insert((n.backend_node_id, n.text_hash), i);
            if n.is_password {
                let (x, y) = n.at;
                let (w, h) = n.wh;
                if x >= 0.0 && y >= 0.0 && w > 0.0 && h > 0.0 {
                    password_boxes.push((x as u32, y as u32, w as u32, h as u32));
                }
            }
        }
        let spatial = SpatialIndex::build(nodes, viewport);
        Self {
            generation,
            nodes: nodes.to_vec(),
            by_backend,
            by_text,
            spatial,
            password_boxes,
            ax_merged,
            viewport,
        }
    }

    /// Look up a node by backend id.
    pub fn node(&self, backend_node_id: u64) -> Option<&TreeNode> {
        self.by_backend
            .get(&backend_node_id)
            .map(|&i| &self.nodes[i])
    }
}

// ---------------------------------------------------------------------------
// Interim path: DOMSnapshot-first.
// ---------------------------------------------------------------------------

/// One node parsed from `DOMSnapshot.captureSnapshot`. Coordinates are
/// main-viewport CSS px: the call site normalizes through the B1 frame
/// chain (per-document scroll offsets and owner origins arrive in the
/// same snapshot response), exactly as the resolver's `insert` does via
/// `attach_chain`.
#[derive(Debug, Clone)]
pub struct DomSnapshotNode {
    pub backend_node_id: u64,
    pub frame_id: String,
    pub role: String,
    pub name: String,
    pub text: String,
    pub at: (f64, f64),
    pub wh: (f64, f64),
    pub dpr: f64,
    pub visible: bool,
    pub enabled: bool,
    pub is_password: bool,
}

/// Build a fresh versioned tree from one DOMSnapshot capture: full
/// rebuild, generation bump, the `(backendNodeId, text_hash)` interim
/// index. This is the DEFAULT perception path until the event layer is
/// proven against Chrome 156 (see the B5 evaluation report).
pub fn build_from_dom_snapshot(
    viewport: (f64, f64),
    nodes: Vec<DomSnapshotNode>,
    frames: Vec<String>,
) -> VersionedTree {
    let mut tree = VersionedTree::new(viewport);
    let tree_nodes: Vec<TreeNode> = nodes
        .into_iter()
        .map(|n| {
            TreeNode::new(
                n.backend_node_id,
                NO_DOCUMENT,
                &n.role,
                &n.name,
                &n.text,
                &n.frame_id,
                n.at,
                n.wh,
                n.dpr,
                n.enabled,
                n.visible,
                n.is_password,
            )
        })
        .collect();
    tree.rebuild_from_snapshot(tree_nodes, frames);
    tree
}

// ---------------------------------------------------------------------------
// Drift check: incremental tree vs full rebuild.
// ---------------------------------------------------------------------------

/// What the drift check found. `drifted` is true when ANY list is
/// non-empty; the caller then forces a full rebuild (generation bump via
/// [`InvalidationReason::DriftDetected`]).
#[derive(Debug, Default, Clone)]
pub struct DriftReport {
    pub drifted: bool,
    /// backendNodeIds in the full rebuild but missing from the
    /// incremental tree (dropped removals / missed inserts).
    pub missing: Vec<u64>,
    /// backendNodeIds in the incremental tree but not the rebuild
    /// (phantom nodes / missed removals).
    pub extra: Vec<u64>,
    /// Nodes whose text_hash disagrees (missed text/attribute events).
    pub changed: Vec<u64>,
    /// Nodes whose geometry disagrees by more than 0.5 CSS px (missed
    /// layout-affecting changes).
    pub geo_changed: Vec<u64>,
}

/// Diff the incremental tree against a full DOM/AX rebuild. Both
/// snapshots must belong to the same generation; cross-generation diffs
/// are a caller bug and report drifted (fail closed).
pub fn drift_check(incremental: &GenerationSnapshot, full: &GenerationSnapshot) -> DriftReport {
    let mut r = DriftReport::default();
    if incremental.generation != full.generation {
        r.drifted = true;
        return r;
    }
    for (id, &fi) in &full.by_backend {
        match incremental.by_backend.get(id) {
            None => r.missing.push(*id),
            Some(&ii) => {
                let a = &incremental.nodes[ii];
                let b = &full.nodes[fi];
                if a.text_hash != b.text_hash {
                    r.changed.push(*id);
                }
                if (a.at.0 - b.at.0).abs() > 0.5
                    || (a.at.1 - b.at.1).abs() > 0.5
                    || (a.wh.0 - b.wh.0).abs() > 0.5
                    || (a.wh.1 - b.wh.1).abs() > 0.5
                {
                    r.geo_changed.push(*id);
                }
            }
        }
    }
    for id in incremental.by_backend.keys() {
        if !full.by_backend.contains_key(id) {
            r.extra.push(*id);
        }
    }
    r.drifted = !(r.missing.is_empty()
        && r.extra.is_empty()
        && r.changed.is_empty()
        && r.geo_changed.is_empty());
    r
}

// ---------------------------------------------------------------------------
// Tier-0 spatial short-circuit: uniform grid over the tree.
// ---------------------------------------------------------------------------

/// Grid cell size, main-viewport CSS px. At 64px a 1920x1080 viewport is
/// 30x17 cells; a node overlaps a handful of cells, so occlusion checks
/// stay O(cells * occupants) with tiny constants. No new dependency.
pub const SPATIAL_CELL_PX: f64 = 64.0;

/// Incrementally built spatial index over one generation's nodes.
/// Zero-area and hidden nodes are culled in the single build pass, so
/// they can never short-circuit a click.
#[derive(Debug, Clone)]
pub struct SpatialIndex {
    cell: f64,
    cols: usize,
    rows: usize,
    viewport: (f64, f64),
    /// Per-cell node indices into the snapshot's node vec.
    cells: Vec<Vec<usize>>,
    /// Number of indexed nodes (post-cull).
    indexed: usize,
}

impl SpatialIndex {
    /// Empty index.
    pub fn empty(viewport: (f64, f64)) -> Self {
        let cols = ((viewport.0 / SPATIAL_CELL_PX).ceil() as usize).max(1);
        let rows = ((viewport.1 / SPATIAL_CELL_PX).ceil() as usize).max(1);
        Self {
            cell: SPATIAL_CELL_PX,
            cols,
            rows,
            viewport,
            cells: vec![Vec::new(); cols * rows],
            indexed: 0,
        }
    }

    /// Build in one pass. Culled (never indexed): zero-area nodes,
    /// invisible nodes, password fields (Tier-0 never clicks a password
    /// field on a structural box alone; those go through the cascade).
    pub fn build(nodes: &[TreeNode], viewport: (f64, f64)) -> Self {
        let mut idx = Self::empty(viewport);
        for (i, n) in nodes.iter().enumerate() {
            if !n.has_area() || !n.visible || !n.enabled || n.is_password {
                continue;
            }
            idx.insert(i, n.at.0, n.at.1, n.wh.0, n.wh.1);
        }
        idx
    }

    fn cell_range(&self, x: f64, y: f64, w: f64, h: f64) -> (usize, usize, usize, usize) {
        let c0 = (x / self.cell).floor().max(0.0) as usize;
        let r0 = (y / self.cell).floor().max(0.0) as usize;
        let c1 = ((x + w) / self.cell).floor().max(0.0) as usize;
        let r1 = ((y + h) / self.cell).floor().max(0.0) as usize;
        (
            c0.min(self.cols - 1),
            r0.min(self.rows - 1),
            c1.min(self.cols - 1),
            r1.min(self.rows - 1),
        )
    }

    fn insert(&mut self, node_idx: usize, x: f64, y: f64, w: f64, h: f64) {
        let (c0, r0, c1, r1) = self.cell_range(x, y, w, h);
        for r in r0..=r1 {
            for c in c0..=c1 {
                self.cells[r * self.cols + c].push(node_idx);
            }
        }
        self.indexed += 1;
    }

    /// Nodes whose boxes touch the query rect (cell pre-filter, exact
    /// box test). The query node itself is included when it overlaps.
    pub fn overlapping(&self, nodes: &[TreeNode], x: f64, y: f64, w: f64, h: f64) -> Vec<usize> {
        let (c0, r0, c1, r1) = self.cell_range(x, y, w, h);
        let mut out = Vec::new();
        let mut seen = vec![false; nodes.len()];
        for r in r0..=r1 {
            for c in c0..=c1 {
                for &i in &self.cells[r * self.cols + c] {
                    if seen[i] {
                        continue;
                    }
                    seen[i] = true;
                    let n = &nodes[i];
                    if boxes_overlap((x, y, w, h), (n.at.0, n.at.1, n.wh.0, n.wh.1)) {
                        out.push(i);
                    }
                }
            }
        }
        out
    }

    /// Nodes indexed (post-cull).
    pub fn indexed(&self) -> usize {
        self.indexed
    }
}

/// Box overlap test (edges touching counts as overlap: conservative).
/// Boxes are (x, y, w, h).
fn boxes_overlap(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> bool {
    a.0 <= b.0 + b.2 && b.0 <= a.0 + a.2 && a.1 <= b.1 + b.3 && b.1 <= a.1 + a.3
}

/// True when the node is occluded: ANY other indexed node overlaps its
/// box. This is deliberately order-free and conservative: paint order
/// and z-index subtleties vanish, because any overlap at all disqualifies
/// the short-circuit. The cost is a lower short-circuit rate on dense
/// pages; the benefit is that "unoccluded" is never wrong.
pub fn is_occluded(snap: &GenerationSnapshot, node_idx: usize) -> bool {
    let n = &snap.nodes[node_idx];
    let others = snap
        .spatial
        .overlapping(&snap.nodes, n.at.0, n.at.1, n.wh.0, n.wh.1);
    others.iter().any(|&i| i != node_idx)
}

/// Why Tier-0 fell through to the cascade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier0Miss {
    /// No node has this normalized accessible name.
    NoMatch,
    /// More than one node matches: ambiguous, the cascade decides.
    Ambiguous,
    /// The unique match is occluded by another node.
    Occluded,
    /// The unique match is not enabled+visible, or is a password field.
    NotClickable,
}

/// Tier-0 verdict: either the structural box goes straight through (no
/// screenshot, no ranking) or the cascade runs.
#[derive(Debug, Clone)]
pub enum Tier0Verdict {
    ShortCircuit {
        backend_node_id: u64,
        /// Box top-left, main-viewport CSS px.
        at: (f64, f64),
        w: f64,
        h: f64,
    },
    FallThrough {
        reason: Tier0Miss,
    },
}

/// Tier-0 lookup: exact normalized-name match that is unique, enabled,
/// visible, non-password, and unoccluded short-circuits with the
/// structural bounding box. Everything else falls through to the
/// resolver cascade. This runs BEFORE BM25: on a warm page it is the
/// reason steady-state resolve stays under 1ms.
pub fn tier0_lookup(snap: &GenerationSnapshot, desc_norm: &str) -> Tier0Verdict {
    let mut found: Option<usize> = None;
    let mut count = 0usize;
    for (i, n) in snap.nodes.iter().enumerate() {
        if n.name.as_ref() == desc_norm {
            count += 1;
            if found.is_none() {
                found = Some(i);
            }
            if count > 1 {
                return Tier0Verdict::FallThrough {
                    reason: Tier0Miss::Ambiguous,
                };
            }
        }
    }
    let i = match found {
        Some(i) => i,
        None => {
            return Tier0Verdict::FallThrough {
                reason: Tier0Miss::NoMatch,
            }
        }
    };
    let n = &snap.nodes[i];
    if !n.enabled || !n.visible || n.is_password || !n.has_area() {
        return Tier0Verdict::FallThrough {
            reason: Tier0Miss::NotClickable,
        };
    }
    if is_occluded(snap, i) {
        return Tier0Verdict::FallThrough {
            reason: Tier0Miss::Occluded,
        };
    }
    Tier0Verdict::ShortCircuit {
        backend_node_id: n.backend_node_id,
        at: n.at,
        w: n.wh.0,
        h: n.wh.1,
    }
}

/// Validate a resolver-produced hit against the spatial index: the
/// winner must be present, enabled, visible, and unoccluded, in the same
/// generation. Used by the engine before trusting a structural box.
pub fn check_hit_unoccluded(snap: &GenerationSnapshot, backend_node_id: u64) -> bool {
    match snap.by_backend.get(&backend_node_id) {
        None => false,
        Some(&i) => {
            let n = &snap.nodes[i];
            n.enabled && n.visible && !is_occluded(snap, i)
        }
    }
}

// ---------------------------------------------------------------------------
// Bounded OCR worker: capacity 2, per-job deadlines, generation ids.
// ---------------------------------------------------------------------------

/// Grace added to the submitter-side wait beyond the job deadline: the
/// worker enforces the deadline itself (kills the child), so the reply
/// should already be queued; the grace only covers scheduling jitter.
const OCR_REPLY_GRACE_MS: u64 = 500;

/// Poll cadence while waiting for the tesseract child.
const OCR_POLL_MS: u64 = 5;

/// stderr bytes kept for an Error message (truncated, never the image).
const OCR_STDERR_SNIP: usize = 200;

/// PNG magic bytes: the worker validates input before spawning tesseract.
const PNG_MAGIC: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// One OCR job: PNG bytes in, outcome out. The worker drops the job
/// BEFORE spawning tesseract when the generation is stale or the
/// deadline already passed.
struct OcrJob {
    png: Vec<u8>,
    deadline: Instant,
    generation: Generation,
    reply: mpsc::SyncSender<OcrOutcome>,
}

/// Bounded OCR worker: two slots, per-job deadlines, generation ids that
/// invalidate stale results, structured Timeout/Busy/Error states.
///
/// # Why the `tesseract` CLI as a child process (documented choice)
///
/// A native binding (tesseract-sys/leptess) would add a C++ link surface,
/// build-time native dependencies, and an in-process Tesseract whose
/// worst case cannot be killed without killing the engine. The CLI gives:
/// no new native deps, deadline enforcement as process kill (SIGKILL is
/// the only honest timeout for a stuck OCR), and one process per job so
/// a wedged job can never poison later ones. The cost is process spawn
/// (~30-80ms on the VM); OCR is the last-resort lane and jobs are rare,
/// so the tradeoff favors robustness. "Initialized once per worker" is
/// satisfied structurally: each of the two worker threads owns its
/// execution slot for its lifetime; the tesseract process is per-job by
/// design, not per-worker, because the deadline IS the process lifetime.
///
/// # Threading
///
/// Two dedicated blocking `std::thread` workers, each owning one
/// `mpsc::Receiver`. No tokio, no async, no shared mutable state: the
/// only cross-thread values are the job channels and the atomic
/// generation counter. `submit` is synchronous and never blocks longer
/// than the job deadline plus [`OCR_REPLY_GRACE_MS`].
pub struct OcrWorker {
    slots: [mpsc::SyncSender<OcrJob>; 2],
    generation: Arc<AtomicU64>,
    tesseract: Option<PathBuf>,
    /// Jobs accepted but not yet finished (queued or executing), 0..=2.
    /// This counter IS the capacity gate: try_send timing races are
    /// irrelevant because the gate is checked before queueing.
    in_flight: Arc<AtomicUsize>,
    // The two worker threads are intentionally detached (their
    // JoinHandles are dropped at spawn): they live exactly as long as
    // the job senders, exiting when the last sender is dropped. Keeping
    // no handles also keeps OcrWorker Sync, so &OcrWorker can be shared
    // across threads for the Busy-path test and for production callers.
}

impl OcrWorker {
    /// New worker, probing `PATH` for the `tesseract` binary. When the
    /// binary is absent every submit returns
    /// `OcrOutcome::Error("tesseract binary not found ...")`: a
    /// structured error, never a panic, never silent.
    pub fn new() -> Self {
        Self::with_binary_opt(probe_tesseract())
    }

    /// New worker with an explicit binary path. Tests point this at a
    /// fake `tesseract` script to exercise the Timeout/Busy/Error paths
    /// deterministically without the real binary.
    pub fn with_binary(path: PathBuf) -> Self {
        Self::with_binary_opt(Some(path))
    }

    fn with_binary_opt(tesseract: Option<PathBuf>) -> Self {
        let generation = Arc::new(AtomicU64::new(1));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let mut slots = Vec::with_capacity(2);
        for _ in 0..2 {
            let (tx, rx) = mpsc::sync_channel::<OcrJob>(1);
            slots.push(tx);
            let gen = Arc::clone(&generation);
            let bin = tesseract.clone();
            let flying = Arc::clone(&in_flight);
            // Detached: the JoinHandle is dropped; the thread exits when
            // the job senders are dropped.
            thread::Builder::new()
                .name("ocr-worker".to_string())
                .spawn(move || ocr_worker_loop(rx, gen, bin, flying))
                .expect("ocr worker thread spawns");
        }
        Self {
            slots: [slots.remove(0), slots.remove(0)],
            generation,
            tesseract,
            in_flight,
        }
    }

    /// Move the worker to a new document generation. Jobs submitted with
    /// an older generation are dropped before OCR begins (stale results
    /// can never be produced).
    pub fn set_generation(&self, generation: Generation) {
        self.generation.store(generation, Ordering::SeqCst);
    }

    /// Current generation.
    pub fn generation(&self) -> Generation {
        self.generation.load(Ordering::SeqCst)
    }

    /// Jobs currently accepted but not finished (0..=2). Telemetry for
    /// the engine; also lets tests wait for slot occupancy
    /// deterministically instead of racing thread startup.
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// Submit one OCR job. Returns immediately with:
    /// - `Error` when the tesseract binary is missing, the PNG is
    ///   malformed, or the request generation is stale;
    /// - `Busy` when both worker slots are occupied;
    /// - otherwise blocks until the worker replies or the deadline plus
    ///   grace expires (`Timeout` on expiry; the worker kills the child
    ///   at the deadline independently).
    pub fn submit(&self, req: &OcrRequest, png: &[u8]) -> OcrOutcome {
        let bin = match &self.tesseract {
            Some(b) => b,
            None => {
                return OcrOutcome::Error(
                    "tesseract binary not found on PATH (install tesseract-ocr)".to_string(),
                )
            }
        };
        // Fail fast when the configured binary is not there at all; the
        // spawn-error path below remains the backup for a TOCTOU race.
        if !bin.is_file() {
            return OcrOutcome::Error(format!("tesseract binary not found at {}", bin.display()));
        }
        if req.generation != self.generation() {
            // Dropped before OCR begins: stale-generation results are
            // never produced.
            return OcrOutcome::Error(format!(
                "stale OCR generation: job {} != worker {}",
                req.generation,
                self.generation()
            ));
        }
        if png.len() < PNG_MAGIC.len() || png[..PNG_MAGIC.len()] != PNG_MAGIC {
            return OcrOutcome::Error("OCR input is not a PNG".to_string());
        }
        // Capacity gate: at most 2 jobs in flight, ever. fetch_add
        // returns the previous value atomically, so the check is exact
        // under contention; an overshoot is undone immediately.
        // Validation runs BEFORE the gate so caller bugs (stale
        // generation, malformed input, missing binary) report
        // specifically even under load.
        if self.in_flight.fetch_add(1, Ordering::SeqCst) >= 2 {
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            return OcrOutcome::Busy;
        }
        let deadline = Instant::now() + Duration::from_millis(req.deadline_ms.max(1));
        let (reply_tx, reply_rx) = mpsc::sync_channel::<OcrOutcome>(1);
        let job = OcrJob {
            png: png.to_vec(),
            deadline,
            generation: req.generation,
            reply: reply_tx,
        };
        // The gate guarantees a free slot; try_send never blocks, so
        // submit can never wedge behind a stuck worker. A Disconnected
        // slot means its worker thread died: that is an Error, not Busy
        // (Busy is strictly "both slots occupied").
        let queued = match self.slots[0].try_send(job) {
            Ok(()) => true,
            Err(mpsc::TrySendError::Full(job)) => match self.slots[1].try_send(job) {
                Ok(()) => true,
                Err(mpsc::TrySendError::Full(_)) => {
                    self.in_flight.fetch_sub(1, Ordering::SeqCst);
                    return OcrOutcome::Busy;
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    self.in_flight.fetch_sub(1, Ordering::SeqCst);
                    return OcrOutcome::Error("ocr worker thread died".to_string());
                }
            },
            Err(mpsc::TrySendError::Disconnected(_)) => {
                self.in_flight.fetch_sub(1, Ordering::SeqCst);
                return OcrOutcome::Error("ocr worker thread died".to_string());
            }
        };
        debug_assert!(queued);
        let wait = Duration::from_millis(req.deadline_ms.max(1) + OCR_REPLY_GRACE_MS);
        match reply_rx.recv_timeout(wait) {
            Ok(out) => out,
            Err(_) => OcrOutcome::Timeout,
        }
    }
}

impl Default for OcrWorker {
    fn default() -> Self {
        Self::new()
    }
}

/// Walk `PATH` for an executable `tesseract`.
fn probe_tesseract() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let cand = dir.join("tesseract");
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

/// The worker thread body: one slot, blocking recv, deadline-enforced
/// child process per job.
/// Decrements the in-flight counter when dropped: the worker's
/// panic-safety net, so a wedged or panicked job can never leak a slot
/// and wedge the worker at Busy forever.
struct InFlightGuard {
    count: Arc<AtomicUsize>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.count.fetch_sub(1, Ordering::SeqCst);
    }
}

fn ocr_worker_loop(
    rx: mpsc::Receiver<OcrJob>,
    generation: Arc<AtomicU64>,
    tesseract: Option<PathBuf>,
    in_flight: Arc<AtomicUsize>,
) {
    while let Ok(job) = rx.recv() {
        let _guard = InFlightGuard {
            count: Arc::clone(&in_flight),
        };
        // A panicking job must not kill the slot: the submitter gets a
        // structured Error and the worker keeps serving.
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_ocr_job(&job, &generation, tesseract.as_ref())
        }))
        .unwrap_or_else(|_| OcrOutcome::Error("ocr worker panicked".to_string()));
        // The submitter may have timed out and dropped the receiver;
        // a failed send is fine (the result is simply unobserved).
        let _ = job.reply.try_send(out);
    }
}

fn run_ocr_job(
    job: &OcrJob,
    generation: &Arc<AtomicU64>,
    tesseract: Option<&PathBuf>,
) -> OcrOutcome {
    // Dropped before OCR begins: stale generation.
    if job.generation != generation.load(Ordering::SeqCst) {
        return OcrOutcome::Error(format!(
            "stale OCR generation: job {} != worker {}",
            job.generation,
            generation.load(Ordering::SeqCst)
        ));
    }
    if Instant::now() >= job.deadline {
        return OcrOutcome::Timeout;
    }
    let bin = match tesseract {
        Some(b) => b,
        None => {
            return OcrOutcome::Error(
                "tesseract binary not found on PATH (install tesseract-ocr)".to_string(),
            )
        }
    };
    match run_tesseract(bin, &job.png, job.deadline) {
        Ok(text) => OcrOutcome::Text(text),
        Err(RunError::Timeout) => OcrOutcome::Timeout,
        Err(RunError::Failed(msg)) => OcrOutcome::Error(msg),
    }
}

#[derive(Debug)]
enum RunError {
    Timeout,
    Failed(String),
}

/// Spawn `tesseract stdin stdout --psm 6 -l eng`, pipe the PNG in, kill
/// the child at the absolute deadline. `--psm 6` (uniform text block)
/// suits UI regions; `-l eng` is the language installed on the VM
/// (verified: `tesseract --list-langs` shows eng, osd).
fn run_tesseract(bin: &PathBuf, png: &[u8], deadline: Instant) -> Result<String, RunError> {
    let mut child = Command::new(bin)
        .arg("stdin")
        .arg("stdout")
        .arg("--psm")
        .arg("6")
        .arg("-l")
        .arg("eng")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| RunError::Failed(format!("tesseract spawn failed: {e}")))?;

    // Write the PNG, then close stdin so tesseract starts reading.
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        if stdin.write_all(png).is_err() {
            let _ = child.kill();
            return Err(RunError::Failed("tesseract stdin write failed".to_string()));
        }
        // stdin dropped here: EOF to the child.
    }

    // Deadline enforcement: poll, kill at expiry. There is no
    // wait_timeout on stable std, so this is the honest implementation.
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout = Vec::new();
                let mut stderr = Vec::new();
                if let Some(mut out) = child.stdout.take() {
                    use std::io::Read;
                    let _ = out.read_to_end(&mut stdout);
                }
                if let Some(mut err) = child.stderr.take() {
                    use std::io::Read;
                    let _ = err.read_to_end(&mut stderr);
                }
                if status.success() {
                    let text = String::from_utf8_lossy(&stdout).trim().to_string();
                    return Ok(text);
                }
                let snip = String::from_utf8_lossy(&stderr);
                let snip: String = snip.chars().take(OCR_STDERR_SNIP).collect();
                return Err(RunError::Failed(format!(
                    "tesseract exited {}: {}",
                    status,
                    snip.trim()
                )));
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(RunError::Timeout);
                }
                thread::sleep(Duration::from_millis(OCR_POLL_MS));
            }
            Err(e) => {
                let _ = child.kill();
                return Err(RunError::Failed(format!("tesseract wait failed: {e}")));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PNG encoding: validated Capture (BGRX) -> PNG bytes for the OCR worker.
// ---------------------------------------------------------------------------

/// Encode a validated [`Capture`] as PNG (RGBA8) for piping to
/// tesseract. BGRX -> RGBA per pixel, honoring the real stride; the
/// capture is validated FIRST (failures are errors, never silent zeros).
pub fn capture_to_png(cap: &Capture) -> Result<Vec<u8>, String> {
    validate_capture(cap).map_err(|e| format!("capture_to_png: {e:#}"))?;
    let w = cap.w as usize;
    let h = cap.h as usize;
    let mut rgba = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let i = y * cap.stride + x * 4;
            // BGRX -> RGBA.
            rgba.push(cap.buf[i + 2]);
            rgba.push(cap.buf[i + 1]);
            rgba.push(cap.buf[i]);
            rgba.push(255);
        }
    }
    let mut out = Vec::new();
    {
        use image::codecs::png::PngEncoder;
        use image::ImageEncoder;
        let enc = PngEncoder::new(&mut out);
        enc.write_image(&rgba, cap.w, cap.h, image::ExtendedColorType::Rgba8)
            .map_err(|e| format!("capture_to_png: PNG encode failed: {e}"))?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// CDP event subscriptions (verified identifiers).
// ---------------------------------------------------------------------------

/// The exact CDP event identifiers to subscribe for the event-driven
/// layer. Every string is the `IDENTIFIER` constant of the corresponding
/// generated `Event*` struct in chromiumoxide_cdp 0.7.0 (the CDP layer
/// the vendored chromiumoxide 0.7 links), verified against the registry
/// source 2026-11-03. Do not "correct" these from upstream docs: the
/// generated bindings are what deserialize the wire.
pub struct EventSubscriptions {
    /// DOM domain: the mutation stream.
    pub dom: &'static [&'static str],
    /// Page domain: navigation and frame lifecycle.
    pub page: &'static [&'static str],
    /// Runtime domain: execution-context resets.
    pub runtime: &'static [&'static str],
}

/// Subscription table for the event layer. Wire-up (calling
/// `Page::event_listener` / session subscribe per identifier) lives on
/// the route/handler side; this table is the contract it subscribes to.
pub fn recommended_subscriptions() -> EventSubscriptions {
    EventSubscriptions {
        dom: &[
            "DOM.documentUpdated",
            "DOM.setChildNodes",
            "DOM.childNodeInserted",
            "DOM.childNodeRemoved",
            "DOM.attributeModified",
            "DOM.attributeRemoved",
            "DOM.characterDataModified",
            "DOM.childNodeCountUpdated",
        ],
        page: &[
            "Page.frameNavigated",
            "Page.frameAttached",
            "Page.frameDetached",
            "Page.documentOpened",
        ],
        runtime: &["Runtime.executionContextsCleared"],
    }
}

// ---------------------------------------------------------------------------
// Crop-and-zoom refinement for sub-12px targets (wires B4's
// needs_refinement).
// ---------------------------------------------------------------------------

/// Target rendered side, device px, after zoom (default 48: a 6px CSS
/// target at dpr 1 zooms 8x into a 48px working image).
pub const ZOOM_TARGET_SIDE_DEV_PX: f64 = 48.0;
/// Zoom clamp: never magnify more than 8x (beyond that is interpolation
/// noise, not information).
pub const ZOOM_MAX: f64 = 8.0;
/// CSS px padding around the hit box for the capture region.
pub const ZOOM_PAD_CSS: f64 = 8.0;
/// Render-check heuristic: the winner's box in the zoomed capture must
/// have luma stddev >= this, else the tree's box is treated as stale
/// (blank region) and the refinement fails rather than locking a lie.
pub const RENDER_BLANK_STDDEV: f64 = 3.0;

/// Crop-and-zoom pass configuration.
#[derive(Debug, Clone, Copy)]
pub struct CropZoomPass {
    pub target_side_dev_px: f64,
    pub max_zoom: f64,
    pub pad_css: f64,
}

impl Default for CropZoomPass {
    fn default() -> Self {
        Self {
            target_side_dev_px: ZOOM_TARGET_SIDE_DEV_PX,
            max_zoom: ZOOM_MAX,
            pad_css: ZOOM_PAD_CSS,
        }
    }
}

/// The locked refined hit: center in main-viewport CSS px.
#[derive(Debug, Clone, Copy)]
pub struct RefinedHit {
    pub at: (f64, f64),
    pub w: f64,
    pub h: f64,
    pub backend_node_id: u64,
    pub zoom_used: f64,
}

/// Why the crop-and-zoom pass refused to lock.
#[derive(Debug, Clone, PartialEq)]
pub enum RefineError {
    /// The target is not small enough to need refinement.
    NotSmallEnough,
    /// The zoomed region capture failed.
    RegionCapture(String),
    /// Re-resolution inside the region found no unique match.
    ReresolveMiss,
    /// Re-resolution found a DIFFERENT node: never lock a guess.
    NodeMismatch { want: u64, got: u64 },
    /// The winner's box is blank in the zoomed capture: the tree's
    /// geometry is stale.
    RenderCheckFailed { stddev: f64 },
}

impl std::fmt::Display for RefineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefineError::NotSmallEnough => write!(f, "target does not need refinement"),
            RefineError::RegionCapture(e) => write!(f, "region capture failed: {e}"),
            RefineError::ReresolveMiss => write!(f, "no unique re-resolve match in region"),
            RefineError::NodeMismatch { want, got } => {
                write!(f, "re-resolve found node {got}, want {want}")
            }
            RefineError::RenderCheckFailed { stddev } => {
                write!(f, "winner box blank in zoomed capture (stddev {stddev:.2})")
            }
        }
    }
}

impl std::error::Error for RefineError {}

/// Crop-and-zoom refinement for a resolver hit with `needs_refinement`
/// (target under 12x12 physical px). Instead of a blind grounding:
///
/// 1. Capture the hit's region (padded, clamped to the viewport).
/// 2. Re-resolve WITHIN the region: exact normalized-name match among
///    nodes overlapping the region; the winner must be the hit's node.
/// 3. Render check: the winner's box, digitally zoomed, must show real
///    pixels (luma variance), not a blank stale box.
/// 4. Lock: return the winner's center in main-viewport CSS px.
///
/// `capture` takes a device-px region (x, y, w, h) and returns the
/// validated frame. `dpr` is the device pixel ratio for the hit's frame.
pub fn refine_small_target(
    pass: &CropZoomPass,
    hit: &Hit,
    snap: &GenerationSnapshot,
    desc_norm: &str,
    capture: &dyn Fn((u32, u32, u32, u32)) -> Result<Capture, String>,
    viewport: (f64, f64),
    dpr: f64,
) -> Result<RefinedHit, RefineError> {
    use super::resolver::needs_refinement;
    if !needs_refinement(hit.w, hit.h, dpr) {
        return Err(RefineError::NotSmallEnough);
    }
    let node = snap
        .node(hit.backend_node_id)
        .ok_or(RefineError::ReresolveMiss)?;
    let dpr = if dpr.is_finite() && dpr > 0.0 {
        dpr
    } else {
        1.0
    };

    // Zoom so the target's smaller side renders at target_side_dev_px.
    let min_side = node.wh.0.min(node.wh.1).max(0.5);
    let zoom = (pass.target_side_dev_px / (min_side * dpr)).clamp(1.0, pass.max_zoom);

    // Region: hit box expanded by pad, clamped to the viewport.
    let rx = (node.at.0 - pass.pad_css).max(0.0);
    let ry = (node.at.1 - pass.pad_css).max(0.0);
    let rx1 = (node.at.0 + node.wh.0 + pass.pad_css).min(viewport.0);
    let ry1 = (node.at.1 + node.wh.1 + pass.pad_css).min(viewport.1);
    let rw = (rx1 - rx).max(1.0);
    let rh = (ry1 - ry).max(1.0);
    let region_dev = (
        (rx * dpr) as u32,
        (ry * dpr) as u32,
        (rw * dpr).max(1.0) as u32,
        (rh * dpr).max(1.0) as u32,
    );
    let cap = capture(region_dev).map_err(RefineError::RegionCapture)?;
    validate_capture(&cap).map_err(|e| RefineError::RegionCapture(format!("{e:#}")))?;

    // Re-resolve within the region: exact unique name match among nodes
    // overlapping the region. The winner MUST be the hit's node.
    let overlapping = snap.spatial.overlapping(&snap.nodes, rx, ry, rw, rh);
    let mut found: Option<u64> = None;
    let mut count = 0usize;
    for &i in &overlapping {
        let n = &snap.nodes[i];
        if n.name.as_ref() == desc_norm && n.enabled && n.visible {
            count += 1;
            if found.is_none() {
                found = Some(n.backend_node_id);
            }
            if count > 1 {
                break;
            }
        }
    }
    match (count, found) {
        (1, Some(id)) if id == hit.backend_node_id => {}
        (1, Some(id)) => {
            return Err(RefineError::NodeMismatch {
                want: hit.backend_node_id,
                got: id,
            })
        }
        _ => return Err(RefineError::ReresolveMiss),
    }

    // Render check: digitally zoom the capture, measure luma variance in
    // the winner's box. A blank box means the tree's geometry is stale.
    let stddev =
        zoomed_box_stddev(&cap, node, (rx, ry), dpr, zoom).map_err(RefineError::RegionCapture)?;
    if stddev < RENDER_BLANK_STDDEV {
        return Err(RefineError::RenderCheckFailed { stddev });
    }

    let c = node.center();
    Ok(RefinedHit {
        at: c,
        w: node.wh.0,
        h: node.wh.1,
        backend_node_id: node.backend_node_id,
        zoom_used: zoom,
    })
}

/// Luma stddev of the node's box inside the zoomed capture. The capture
/// covers `region_origin` (main-viewport CSS px) at `dpr`; the box is
/// mapped into capture-px, the image is digitally zoomed, and BT.601
/// luma variance is measured over the zoomed box.
fn zoomed_box_stddev(
    cap: &Capture,
    node: &TreeNode,
    region_origin: (f64, f64),
    dpr: f64,
    zoom: f64,
) -> Result<f64, String> {
    // BGRX -> RgbImage.
    let w = cap.w as usize;
    let h = cap.h as usize;
    let mut rgb = image::RgbImage::new(cap.w, cap.h);
    for y in 0..h {
        for x in 0..w {
            let i = y * cap.stride + x * 4;
            rgb.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([cap.buf[i + 2], cap.buf[i + 1], cap.buf[i]]),
            );
        }
    }
    let zw = ((cap.w as f64) * zoom).round().max(1.0) as u32;
    let zh = ((cap.h as f64) * zoom).round().max(1.0) as u32;
    let zoomed = image::imageops::resize(&rgb, zw, zh, image::imageops::FilterType::Triangle);

    // Winner box in zoomed-px.
    let bx = (node.at.0 - region_origin.0) * dpr * zoom;
    let by = (node.at.1 - region_origin.1) * dpr * zoom;
    let bw = (node.wh.0 * dpr * zoom).max(1.0);
    let bh = (node.wh.1 * dpr * zoom).max(1.0);
    let x0 = bx.max(0.0) as u32;
    let y0 = by.max(0.0) as u32;
    let x1 = ((bx + bw) as u32).min(zw);
    let y1 = ((by + bh) as u32).min(zh);
    if x1 <= x0 || y1 <= y0 {
        return Err("winner box outside the zoomed capture".to_string());
    }
    let (mut sum, mut sum2, mut n) = (0.0f64, 0.0f64, 0u64);
    for y in y0..y1 {
        for x in x0..x1 {
            let p = zoomed.get_pixel(x, y);
            // BT.601 luma, same weights as the dHash path.
            let luma = (29.0 * p[2] as f64 + 150.0 * p[1] as f64 + 77.0 * p[0] as f64) / 256.0;
            sum += luma;
            sum2 += luma * luma;
            n += 1;
        }
    }
    if n == 0 {
        return Err("empty winner box".to_string());
    }
    let mean = sum / n as f64;
    let var = (sum2 / n as f64 - mean * mean).max(0.0);
    Ok(var.sqrt())
}

// ---------------------------------------------------------------------------
// Resolver feed: an immutable generation snapshot becomes resolver input.
// ---------------------------------------------------------------------------

/// Feed one immutable generation snapshot into a [`ResolverEngine`].
/// Starts a new resolver generation, inserts every node with the
/// IDENTITY frame chain (coordinates are already main-viewport CSS px,
/// normalized once at tree-insertion time; they are never transformed
/// twice), then finalizes the BM25 index.
pub fn feed_resolver(engine: &mut super::resolver::ResolverEngine, snap: &GenerationSnapshot) {
    engine.begin_generation(snap.generation);
    for n in &snap.nodes {
        // Identity chain: at/wh are already main-viewport CSS px.
        let chain = [FrameTransform {
            scroll: (0.0, 0.0),
            content_origin: (0.0, 0.0),
            nested_offset: (0.0, 0.0),
            css_scale: 1.0,
            transform_origin: (0.0, 0.0),
            dpr: n.dpr,
        }];
        if engine
            .insert(
                n.backend_node_id,
                &n.role,
                &n.name,
                &n.text,
                n.at,
                n.wh,
                n.enabled && n.visible,
                &chain,
            )
            .is_err()
        {
            engine.skipped_bad_coords += 1;
        }
    }
    engine.finalize();
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::resolver::{EngineConfig, Gates, Resolution, ResolveOpts, Rng01};
    use super::*;
    use std::path::PathBuf;

    const VIEWPORT: (f64, f64) = (1280.0, 800.0);

    fn tnode(backend: u64, name: &str, x: f64, y: f64, w: f64, h: f64) -> TreeNode {
        TreeNode::new(
            backend,
            1,
            "button",
            name,
            "",
            "main",
            (x, y),
            (w, h),
            1.0,
            true,
            true,
            false,
        )
    }

    fn wire(ref_id: i64, backend: u64) -> NodeWire {
        NodeWire {
            node_ref: NodeRef(ref_id),
            backend_node_id: backend,
            frame_id: "main".to_string(),
            at: Some((10.0, 10.0)),
            wh: Some((80.0, 30.0)),
            visible: true,
            enabled: true,
        }
    }

    /// Load three nodes (refs 1..3, backends 101..103) through the event
    /// stream: the honest initial-document path.
    fn tree_with_three() -> VersionedTree {
        let mut tree = VersionedTree::new(VIEWPORT);
        tree.rebuild_from_snapshot(Vec::new(), vec!["main".to_string()]);
        let gen = tree.generation();
        let batch = EventBatch {
            seq: 1,
            generation: gen,
            events: vec![TreeEvent::SetChildNodes {
                parent: NodeRef(0),
                nodes: vec![wire(1, 101), wire(2, 102), wire(3, 103)],
            }],
        };
        assert_eq!(
            tree.apply_batch(&batch),
            BatchOutcome::Applied { events: 1 }
        );
        tree
    }

    // --- atomic batches ----------------------------------------------------

    #[test]
    fn batch_atomicity_partial_never_visible() {
        let mut tree = tree_with_three();
        let snap_before = tree.snapshot();
        let gen = tree.generation();
        // Second event is valid (inserts ref 4); third references an
        // unknown node: the WHOLE batch must be rejected.
        let batch = EventBatch {
            seq: 2,
            generation: gen,
            events: vec![
                TreeEvent::ChildInserted {
                    parent: NodeRef(0),
                    previous: NodeRef(3),
                    node: wire(4, 104),
                },
                TreeEvent::AttrModified {
                    node: NodeRef(999),
                    name: "hidden".into(),
                    value: "".into(),
                },
            ],
        };
        let outcome = tree.apply_batch(&batch);
        assert_eq!(
            outcome,
            BatchOutcome::Rejected {
                reason: RejectReason::UnknownNode(NodeRef(999))
            }
        );
        // Committed state untouched: the valid insert is NOT visible.
        assert_eq!(tree.node_count(), 3);
        assert!(!tree.snapshot().by_backend.contains_key(&104));
        assert!(tree.is_desynced());
        // The pre-batch snapshot is unchanged (immutability).
        assert_eq!(snap_before.nodes.len(), 3);
    }

    #[test]
    fn batch_sequence_gap_rejected() {
        let mut tree = tree_with_three();
        let gen = tree.generation();
        let batch = EventBatch {
            seq: 7, // want 2: events were dropped in transit
            generation: gen,
            events: vec![],
        };
        assert_eq!(
            tree.apply_batch(&batch),
            BatchOutcome::Rejected {
                reason: RejectReason::SequenceGap { want: 2, got: 7 }
            }
        );
        assert!(tree.is_desynced());
        assert_eq!(tree.node_count(), 3, "rejected batch changes nothing");
    }

    #[test]
    fn batch_wrong_generation_rejected() {
        let mut tree = tree_with_three();
        let batch = EventBatch {
            seq: 2,
            generation: 999,
            events: vec![],
        };
        assert_eq!(
            tree.apply_batch(&batch),
            BatchOutcome::Rejected {
                reason: RejectReason::WrongGeneration {
                    want: tree.generation(),
                    got: 999
                }
            }
        );
        assert!(tree.is_desynced());
    }

    #[test]
    fn document_level_events_are_invalidation_signals_not_patches() {
        let mut tree = tree_with_three();
        for ev in [
            TreeEvent::DocumentUpdated,
            TreeEvent::MainFrameNavigated,
            TreeEvent::ExecutionContextsCleared,
        ] {
            let gen = tree.generation();
            let batch = EventBatch {
                seq: tree.last_seq() + 1,
                generation: gen,
                events: vec![ev],
            };
            match tree.apply_batch(&batch) {
                BatchOutcome::Rejected {
                    reason: RejectReason::Inconsistent(_),
                } => {}
                other => panic!("expected Inconsistent rejection, got {other:?}"),
            }
            assert!(tree.is_desynced());
            // Recover through the explicit invalidation path.
            let g2 = tree.invalidate(InvalidationReason::EventStreamDesync);
            assert!(g2 > gen);
            assert_eq!(tree.node_count(), 0);
            assert!(!tree.is_desynced());
            // Rebuild for the next iteration.
            tree_with_three_rebuild(&mut tree);
        }
    }

    fn tree_with_three_rebuild(tree: &mut VersionedTree) {
        tree.rebuild_from_snapshot(Vec::new(), vec!["main".to_string()]);
        let gen = tree.generation();
        let batch = EventBatch {
            seq: 1,
            generation: gen,
            events: vec![TreeEvent::SetChildNodes {
                parent: NodeRef(0),
                nodes: vec![wire(1, 101), wire(2, 102), wire(3, 103)],
            }],
        };
        assert_eq!(
            tree.apply_batch(&batch),
            BatchOutcome::Applied { events: 1 }
        );
    }

    #[test]
    fn generation_invalidation_on_navigation() {
        let mut tree = tree_with_three();
        let snap_old = tree.snapshot();
        let g1 = tree.generation();
        let g2 = tree.invalidate(InvalidationReason::Navigation);
        assert!(g2 > g1);
        assert_eq!(tree.node_count(), 0);
        assert_eq!(tree.snapshot().nodes.len(), 0);
        // The old snapshot is immutable: still generation g1, 3 nodes.
        assert_eq!(snap_old.generation, g1);
        assert_eq!(snap_old.nodes.len(), 3);
    }

    #[test]
    fn node_identity_is_backend_plus_generation() {
        let mut tree = VersionedTree::new(VIEWPORT);
        let g1 = tree.rebuild_from_snapshot(
            vec![tnode(101, "search", 10.0, 10.0, 80.0, 30.0)],
            vec!["main".to_string()],
        );
        let snap1 = tree.snapshot();
        let g2 = tree.rebuild_from_snapshot(
            vec![tnode(101, "search", 10.0, 10.0, 80.0, 30.0)],
            vec!["main".to_string()],
        );
        let snap2 = tree.snapshot();
        assert_ne!(g1, g2, "rebuild bumps the generation");
        assert_eq!(snap1.nodes[0].generation, g1);
        assert_eq!(snap2.nodes[0].generation, g2);
        // Same backend id, different generation: distinct identities.
        assert_eq!(
            snap1.nodes[0].backend_node_id,
            snap2.nodes[0].backend_node_id
        );
        assert_ne!(snap1.nodes[0].generation, snap2.nodes[0].generation);
    }

    #[test]
    fn snapshot_immutable_across_batches() {
        let mut tree = tree_with_three();
        let before = tree.snapshot();
        let before_nodes = before.nodes.clone();
        let gen = tree.generation();
        let batch = EventBatch {
            seq: 2,
            generation: gen,
            events: vec![TreeEvent::CharDataModified {
                node: NodeRef(1),
                text: "changed".into(),
            }],
        };
        assert_eq!(
            tree.apply_batch(&batch),
            BatchOutcome::Applied { events: 1 }
        );
        // Old Arc untouched.
        assert_eq!(before.nodes.len(), before_nodes.len());
        assert_eq!(before.nodes[0].text.as_ref(), before_nodes[0].text.as_ref());
        // New snapshot sees the change.
        let after = tree.snapshot();
        assert_eq!(after.nodes[0].text.as_ref(), "changed");
        assert_ne!(after.nodes[0].text_hash, before.nodes[0].text_hash);
    }

    #[test]
    fn frame_detached_removes_owned_nodes() {
        let mut tree = VersionedTree::new(VIEWPORT);
        tree.rebuild_from_snapshot(Vec::new(), vec!["main".to_string(), "sub".to_string()]);
        let gen = tree.generation();
        let mut w = wire(1, 101);
        w.frame_id = "sub".to_string();
        let batch = EventBatch {
            seq: 1,
            generation: gen,
            events: vec![
                TreeEvent::SetChildNodes {
                    parent: NodeRef(0),
                    nodes: vec![wire(2, 102), w],
                },
                TreeEvent::FrameDetached {
                    frame_id: "sub".into(),
                },
            ],
        };
        assert_eq!(
            tree.apply_batch(&batch),
            BatchOutcome::Applied { events: 2 }
        );
        assert_eq!(tree.node_count(), 1);
        assert!(!tree.snapshot().by_backend.contains_key(&101));
        assert!(tree.snapshot().by_backend.contains_key(&102));
    }

    #[test]
    fn set_child_nodes_is_authoritative() {
        let mut tree = tree_with_three();
        let gen = tree.generation();
        // New list drops ref 3 and adds ref 4.
        let batch = EventBatch {
            seq: 2,
            generation: gen,
            events: vec![TreeEvent::SetChildNodes {
                parent: NodeRef(0),
                nodes: vec![wire(1, 101), wire(2, 102), wire(4, 104)],
            }],
        };
        assert_eq!(
            tree.apply_batch(&batch),
            BatchOutcome::Applied { events: 1 }
        );
        let snap = tree.snapshot();
        assert_eq!(snap.nodes.len(), 3);
        assert!(!snap.by_backend.contains_key(&103), "dropped child removed");
        assert!(snap.by_backend.contains_key(&104), "new child present");
    }

    // --- drift check ---------------------------------------------------------

    #[test]
    fn drift_check_catches_injected_desync() {
        // Ground truth: the full rebuild.
        let full_nodes = vec![
            tnode(101, "search", 10.0, 10.0, 80.0, 30.0),
            tnode(102, "home", 100.0, 10.0, 80.0, 30.0),
            tnode(103, "extra", 200.0, 10.0, 80.0, 30.0),
        ];
        let full = GenerationSnapshot::build(7, &full_nodes, true, VIEWPORT);

        // Incremental tree missed a text event on 101, missed the removal
        // of 103, and has a phantom 104: three desync classes at once.
        let mut inc_nodes = vec![
            tnode(101, "stale", 10.0, 10.0, 80.0, 30.0), // text differs
            tnode(102, "home", 100.0, 10.0, 80.0, 30.0),
            tnode(104, "phantom", 300.0, 10.0, 80.0, 30.0), // extra
        ];
        // Geometry drift on 102.
        inc_nodes[1].at = (110.0, 10.0);
        let inc = GenerationSnapshot::build(7, &inc_nodes, false, VIEWPORT);

        let r = drift_check(&inc, &full);
        assert!(r.drifted);
        assert_eq!(r.changed, vec![101]);
        assert_eq!(r.missing, vec![103]);
        assert_eq!(r.extra, vec![104]);
        assert_eq!(r.geo_changed, vec![102]);
    }

    #[test]
    fn drift_check_clean_when_in_sync() {
        let nodes = vec![
            tnode(101, "search", 10.0, 10.0, 80.0, 30.0),
            tnode(102, "home", 100.0, 10.0, 80.0, 30.0),
        ];
        let a = GenerationSnapshot::build(7, &nodes, true, VIEWPORT);
        let b = GenerationSnapshot::build(7, &nodes, true, VIEWPORT);
        let r = drift_check(&a, &b);
        assert!(!r.drifted);
        assert!(r.missing.is_empty() && r.extra.is_empty());
        assert!(r.changed.is_empty() && r.geo_changed.is_empty());
    }

    #[test]
    fn drift_check_fails_closed_on_generation_mismatch() {
        let nodes = vec![tnode(101, "search", 10.0, 10.0, 80.0, 30.0)];
        let a = GenerationSnapshot::build(7, &nodes, true, VIEWPORT);
        let b = GenerationSnapshot::build(8, &nodes, true, VIEWPORT);
        assert!(drift_check(&a, &b).drifted);
    }

    // --- Tier-0 ------------------------------------------------------------------

    fn tier0_snap() -> GenerationSnapshot {
        let nodes = vec![
            tnode(101, "search", 10.0, 10.0, 80.0, 30.0),
            tnode(102, "home", 200.0, 10.0, 80.0, 30.0),
        ];
        GenerationSnapshot::build(3, &nodes, true, VIEWPORT)
    }

    #[test]
    fn tier0_short_circuits_unambiguous_unoccluded() {
        let snap = tier0_snap();
        match tier0_lookup(&snap, "search") {
            Tier0Verdict::ShortCircuit {
                backend_node_id,
                at,
                w,
                h,
            } => {
                assert_eq!(backend_node_id, 101);
                // The STRUCTURAL box goes straight through.
                assert_eq!(at, (10.0, 10.0));
                assert_eq!((w, h), (80.0, 30.0));
            }
            Tier0Verdict::FallThrough { reason } => {
                panic!("expected short-circuit, fell through: {reason:?}")
            }
        }
    }

    #[test]
    fn tier0_occluded_does_not_short_circuit() {
        // A popup overlapping the search button: the unique match is
        // occluded, so no short-circuit.
        let nodes = vec![
            tnode(101, "search", 10.0, 10.0, 80.0, 30.0),
            tnode(102, "popup", 50.0, 20.0, 200.0, 200.0),
        ];
        let snap = GenerationSnapshot::build(3, &nodes, true, VIEWPORT);
        assert!(is_occluded(&snap, 0));
        // The popup is ALSO flagged: the rule is order-free and
        // conservative (any box overlap disqualifies the short-circuit),
        // so overlapping nodes are mutually occluded. The cascade, not
        // Tier-0, decides those.
        assert!(is_occluded(&snap, 1));
        match tier0_lookup(&snap, "search") {
            Tier0Verdict::FallThrough {
                reason: Tier0Miss::Occluded,
            } => {}
            other => panic!("expected Occluded fall-through, got {other:?}"),
        }
    }

    #[test]
    fn tier0_ambiguous_falls_through() {
        let nodes = vec![
            tnode(101, "go", 10.0, 10.0, 80.0, 30.0),
            tnode(102, "go", 200.0, 10.0, 80.0, 30.0),
        ];
        let snap = GenerationSnapshot::build(3, &nodes, true, VIEWPORT);
        match tier0_lookup(&snap, "go") {
            Tier0Verdict::FallThrough {
                reason: Tier0Miss::Ambiguous,
            } => {}
            other => panic!("expected Ambiguous fall-through, got {other:?}"),
        }
    }

    #[test]
    fn tier0_zero_area_hidden_and_password_culled() {
        let mut zero = tnode(101, "zero", 10.0, 10.0, 0.0, 30.0);
        let _ = &mut zero;
        let nodes = vec![
            TreeNode::new(
                101,
                1,
                "button",
                "zero",
                "",
                "main",
                (10.0, 10.0),
                (0.0, 30.0),
                1.0,
                true,
                true,
                false,
            ),
            TreeNode::new(
                102,
                1,
                "button",
                "hid",
                "",
                "main",
                (10.0, 50.0),
                (80.0, 30.0),
                1.0,
                true,
                false,
                false,
            ),
            TreeNode::new(
                103,
                1,
                "button",
                "pw",
                "",
                "main",
                (10.0, 90.0),
                (80.0, 30.0),
                1.0,
                true,
                true,
                true,
            ),
        ];
        let _ = zero;
        let snap = GenerationSnapshot::build(3, &nodes, true, VIEWPORT);
        assert_eq!(snap.spatial.indexed(), 0, "all three culled in one pass");
        for desc in ["zero", "hid", "pw"] {
            match tier0_lookup(&snap, desc) {
                Tier0Verdict::FallThrough {
                    reason: Tier0Miss::NotClickable,
                } => {}
                other => panic!("{desc}: expected NotClickable, got {other:?}"),
            }
        }
    }

    #[test]
    fn check_hit_unoccluded_validates_resolver_winner() {
        let snap = tier0_snap();
        assert!(check_hit_unoccluded(&snap, 101));
        assert!(!check_hit_unoccluded(&snap, 999), "unknown node fails");
        let nodes = vec![
            tnode(101, "search", 10.0, 10.0, 80.0, 30.0),
            tnode(102, "popup", 50.0, 20.0, 200.0, 200.0),
        ];
        let snap2 = GenerationSnapshot::build(3, &nodes, true, VIEWPORT);
        assert!(!check_hit_unoccluded(&snap2, 101), "occluded winner fails");
    }

    // --- subscriptions ---------------------------------------------------------

    #[test]
    fn subscriptions_use_verified_identifiers() {
        let s = recommended_subscriptions();
        // Exact IDENTIFIER constants from the generated cdp 0.7.0
        // bindings (registry source, 2026-11-03). Any deviation is a
        // subscription that silently receives nothing.
        assert_eq!(
            s.dom,
            &[
                "DOM.documentUpdated",
                "DOM.setChildNodes",
                "DOM.childNodeInserted",
                "DOM.childNodeRemoved",
                "DOM.attributeModified",
                "DOM.attributeRemoved",
                "DOM.characterDataModified",
                "DOM.childNodeCountUpdated",
            ]
        );
        assert_eq!(
            s.page,
            &[
                "Page.frameNavigated",
                "Page.frameAttached",
                "Page.frameDetached",
                "Page.documentOpened",
            ]
        );
        assert_eq!(s.runtime, &["Runtime.executionContextsCleared"]);
    }

    // --- capture -> PNG ----------------------------------------------------------

    fn solid_capture(w: u32, h: u32, v: u8) -> Capture {
        let stride = w as usize * 4;
        let mut buf = vec![0u8; stride * h as usize];
        #[allow(clippy::chunks_exact_to_as_chunks)]
        // verified against known answers; restyle risks drift
        for px in buf.chunks_exact_mut(4) {
            px[0] = v;
            px[1] = v;
            px[2] = v;
            px[3] = 255;
        }
        Capture { buf, stride, w, h }
    }

    #[test]
    fn capture_to_png_validates_and_encodes() {
        let png = capture_to_png(&solid_capture(16, 16, 200)).expect("encodes");
        assert_eq!(&png[..8], &PNG_MAGIC, "real PNG magic");
        assert!(png.len() > 8);
        // Decode round-trip through the image crate.
        let img = image::load_from_memory(&png).expect("decodes");
        assert_eq!((img.width(), img.height()), (16, 16));
    }

    #[test]
    fn capture_validation_failures_are_errors() {
        // Malformed captures are errors, never silent zeros: the OCR
        // worker must never hash an empty or truncated frame.
        let mut c = solid_capture(16, 16, 200);
        c.w = 0;
        assert!(capture_to_png(&c).is_err());
        let mut c = solid_capture(16, 16, 200);
        c.stride = 16 * 4 - 1;
        assert!(capture_to_png(&c).is_err());
        let mut c = solid_capture(16, 16, 200);
        c.buf.truncate(c.buf.len() - 8);
        assert!(capture_to_png(&c).is_err());
    }

    // --- OCR worker --------------------------------------------------------------

    fn fake_script(name: &str, body: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("b5-ocr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let p = dir.join(name);
        std::fs::write(&p, body).expect("write fake tesseract");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).expect("chmod +x");
        }
        p
    }

    fn tiny_png() -> Vec<u8> {
        capture_to_png(&solid_capture(16, 16, 200)).expect("png")
    }

    fn ocr_req(deadline_ms: u64, generation: u64) -> OcrRequest {
        OcrRequest {
            region: (0, 0, 16, 16),
            deadline_ms,
            generation,
        }
    }

    #[test]
    fn ocr_text_path() {
        let bin = fake_script(
            "fake-echo.sh",
            "#!/bin/sh\ncat > /dev/null\nprintf 'hello world'",
        );
        let worker = OcrWorker::with_binary(bin);
        match worker.submit(&ocr_req(2000, 1), &tiny_png()) {
            OcrOutcome::Text(t) => assert_eq!(t, "hello world"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn ocr_timeout_path() {
        let bin = fake_script(
            "fake-slow.sh",
            "#!/bin/sh\ncat > /dev/null\nsleep 5\nprintf 'too late'",
        );
        let worker = OcrWorker::with_binary(bin);
        let t0 = Instant::now();
        let out = worker.submit(&ocr_req(300, 1), &tiny_png());
        let elapsed = t0.elapsed();
        assert_eq!(out, OcrOutcome::Timeout);
        assert!(
            elapsed < Duration::from_millis(2500),
            "deadline enforced, took {elapsed:?}"
        );
    }

    #[test]
    fn ocr_busy_path() {
        // Capacity 2: two slow jobs occupy both slots; the third submit
        // gets Busy immediately instead of queueing behind them.
        let bin = fake_script(
            "fake-slow2.sh",
            "#!/bin/sh\ncat > /dev/null\nsleep 5\nprintf 'too late'",
        );
        let worker = std::sync::Arc::new(OcrWorker::with_binary(bin));
        let png = tiny_png();
        let spawn_slow = |w: std::sync::Arc<OcrWorker>, png: Vec<u8>| {
            std::thread::spawn(move || w.submit(&ocr_req(1500, 1), &png))
        };
        let h1 = spawn_slow(worker.clone(), png.clone());
        let h2 = spawn_slow(worker.clone(), png.clone());
        // Wait until both slow jobs are actually in flight (no thread-
        // startup race), then the third submit must see Busy at once.
        let t0 = Instant::now();
        while worker.in_flight() < 2 {
            assert!(
                t0.elapsed() < Duration::from_secs(5),
                "slow jobs never occupied both slots"
            );
            thread::sleep(Duration::from_millis(10));
        }
        let t1 = Instant::now();
        assert_eq!(
            worker.submit(&ocr_req(1500, 1), &png),
            OcrOutcome::Busy,
            "third submit must see Busy while both slots held"
        );
        assert!(
            t1.elapsed() < Duration::from_millis(500),
            "Busy must return immediately, not block"
        );
        // The slow jobs time out at their own deadlines; join them.
        assert_eq!(h1.join().unwrap(), OcrOutcome::Timeout);
        assert_eq!(h2.join().unwrap(), OcrOutcome::Timeout);
    }

    #[test]
    fn ocr_error_on_nonzero_exit() {
        let bin = fake_script(
            "fake-fail.sh",
            "#!/bin/sh\ncat > /dev/null\necho 'boom happened' >&2\nexit 1",
        );
        let worker = OcrWorker::with_binary(bin);
        match worker.submit(&ocr_req(2000, 1), &tiny_png()) {
            OcrOutcome::Error(msg) => assert!(
                msg.contains("boom happened"),
                "stderr snippet carried: {msg}"
            ),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn ocr_error_when_binary_missing() {
        let worker = OcrWorker::with_binary(PathBuf::from("/nonexistent/tesseract-xyz"));
        match worker.submit(&ocr_req(500, 1), &tiny_png()) {
            OcrOutcome::Error(msg) => assert!(msg.contains("not found"), "{msg}"),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn ocr_stale_generation_dropped_before_spawn() {
        // The slow script would take 5s; a stale-generation submit must
        // return in microseconds without spawning anything.
        let bin = fake_script(
            "fake-slow3.sh",
            "#!/bin/sh\ncat > /dev/null\nsleep 5\nprintf 'too late'",
        );
        let worker = OcrWorker::with_binary(bin);
        worker.set_generation(5);
        let t0 = Instant::now();
        let out = worker.submit(&ocr_req(2000, 3), &tiny_png());
        let elapsed = t0.elapsed();
        match out {
            OcrOutcome::Error(msg) => assert!(msg.contains("stale"), "{msg}"),
            other => panic!("expected stale-generation Error, got {other:?}"),
        }
        assert!(
            elapsed < Duration::from_millis(1000),
            "dropped before spawn, took {elapsed:?}"
        );
    }

    #[test]
    fn ocr_rejects_non_png_input() {
        let bin = fake_script("fake-unused.sh", "#!/bin/sh\nexit 0");
        let worker = OcrWorker::with_binary(bin);
        match worker.submit(&ocr_req(500, 1), b"definitely not a png") {
            OcrOutcome::Error(msg) => assert!(msg.contains("PNG"), "{msg}"),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    // --- crop-and-zoom -------------------------------------------------------------

    struct TestRng(u64);
    impl Rng01 for TestRng {
        fn uniform01(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn small_hit(backend: u64, w: f64, h: f64) -> Hit {
        Hit {
            at: super::super::coords::MainViewportCss(103.0, 103.0),
            w,
            h,
            backend_node_id: backend,
            confidence: 0.9,
            stage: super::super::resolver::Stage::Bm25,
            needs_refinement: true,
        }
    }

    /// Capture closure: the requested region size, with a checkerboard
    /// over the node's box (high luma variance) on mid-gray elsewhere.
    fn checker_capture(node_at: (f64, f64), node_wh: (f64, f64), pad: f64) -> Capture {
        let w = (node_wh.0 + 2.0 * pad) as u32;
        let h = (node_wh.1 + 2.0 * pad) as u32;
        let stride = w as usize * 4;
        let mut buf = vec![128u8; stride * h as usize];
        let ox = pad as usize;
        let oy = pad as usize;
        for y in 0..node_wh.1 as usize {
            for x in 0..node_wh.0 as usize {
                let v = if (x + y) % 2 == 0 { 0u8 } else { 255u8 };
                let i = (oy + y) * stride + (ox + x) * 4;
                buf[i] = v;
                buf[i + 1] = v;
                buf[i + 2] = v;
            }
        }
        let _ = node_at;
        Capture { buf, stride, w, h }
    }

    #[test]
    fn refine_small_target_locks() {
        // 6x6 CSS target at dpr 1: needs_refinement, zoom = 48/6 = 8.
        let node = TreeNode::new(
            201,
            1,
            "button",
            "x",
            "",
            "main",
            (100.0, 100.0),
            (6.0, 6.0),
            1.0,
            true,
            true,
            false,
        );
        let snap = GenerationSnapshot::build(1, &[node], true, VIEWPORT);
        let pass = CropZoomPass::default();
        let capture = |_r: (u32, u32, u32, u32)| -> Result<Capture, String> {
            Ok(checker_capture((100.0, 100.0), (6.0, 6.0), ZOOM_PAD_CSS))
        };
        let hit = small_hit(201, 6.0, 6.0);
        let out = refine_small_target(&pass, &hit, &snap, "x", &capture, VIEWPORT, 1.0)
            .expect("refinement locks");
        assert_eq!(out.backend_node_id, 201);
        assert_eq!(out.at, (103.0, 103.0), "locked to the node center");
        assert_eq!(out.zoom_used, 8.0);
    }

    #[test]
    fn refine_rejects_node_mismatch() {
        // Two same-named nodes in the region: re-resolve is ambiguous.
        let nodes = vec![
            TreeNode::new(
                201,
                1,
                "button",
                "x",
                "",
                "main",
                (100.0, 100.0),
                (6.0, 6.0),
                1.0,
                true,
                true,
                false,
            ),
            TreeNode::new(
                202,
                1,
                "button",
                "x",
                "",
                "main",
                (108.0, 100.0),
                (6.0, 6.0),
                1.0,
                true,
                true,
                false,
            ),
        ];
        let snap = GenerationSnapshot::build(1, &nodes, true, VIEWPORT);
        let pass = CropZoomPass::default();
        let capture = |_r: (u32, u32, u32, u32)| -> Result<Capture, String> {
            Ok(checker_capture((100.0, 100.0), (6.0, 6.0), ZOOM_PAD_CSS))
        };
        let hit = small_hit(201, 6.0, 6.0);
        match refine_small_target(&pass, &hit, &snap, "x", &capture, VIEWPORT, 1.0) {
            Err(RefineError::ReresolveMiss) => {}
            other => panic!("expected ReresolveMiss, got {other:?}"),
        }
    }

    #[test]
    fn refine_render_check_fails_on_blank() {
        let node = TreeNode::new(
            201,
            1,
            "button",
            "x",
            "",
            "main",
            (100.0, 100.0),
            (6.0, 6.0),
            1.0,
            true,
            true,
            false,
        );
        let snap = GenerationSnapshot::build(1, &[node], true, VIEWPORT);
        let pass = CropZoomPass::default();
        // Blank region: the tree's box is stale.
        let capture = |r: (u32, u32, u32, u32)| -> Result<Capture, String> {
            Ok(solid_capture(r.2, r.3, 128))
        };
        let hit = small_hit(201, 6.0, 6.0);
        match refine_small_target(&pass, &hit, &snap, "x", &capture, VIEWPORT, 1.0) {
            Err(RefineError::RenderCheckFailed { stddev }) => {
                assert!(stddev < RENDER_BLANK_STDDEV)
            }
            other => panic!("expected RenderCheckFailed, got {other:?}"),
        }
    }

    #[test]
    fn refine_rejects_big_target() {
        let node = TreeNode::new(
            201,
            1,
            "button",
            "ok",
            "",
            "main",
            (100.0, 100.0),
            (80.0, 30.0),
            1.0,
            true,
            true,
            false,
        );
        let snap = GenerationSnapshot::build(1, &[node], true, VIEWPORT);
        let pass = CropZoomPass::default();
        let capture = |_r: (u32, u32, u32, u32)| -> Result<Capture, String> {
            panic!("capture must not run for a target that needs no refinement")
        };
        let hit = small_hit(201, 80.0, 30.0);
        assert_eq!(
            refine_small_target(&pass, &hit, &snap, "ok", &capture, VIEWPORT, 1.0).unwrap_err(),
            RefineError::NotSmallEnough
        );
    }

    // --- resolver feed ---------------------------------------------------------------

    #[test]
    fn feed_resolver_roundtrip() {
        let nodes = vec![
            tnode(101, "search", 10.0, 10.0, 80.0, 30.0),
            tnode(102, "home", 200.0, 10.0, 80.0, 30.0),
        ];
        let snap = GenerationSnapshot::build(9, &nodes, true, VIEWPORT);
        let mut engine = super::super::resolver::ResolverEngine::new(
            EngineConfig {
                profile_id: "test".to_string(),
                origin: "example.com".to_string(),
                viewport: VIEWPORT,
                cache_cap: 16,
            },
            1,
            Gates::default(),
        );
        feed_resolver(&mut engine, &snap);
        assert_eq!(
            engine.generation(),
            9,
            "resolver tracks the tree generation"
        );
        assert_eq!(engine.candidate_count(), 2);
        let mut rng = TestRng(42);
        let opts = ResolveOpts {
            route: "r".to_string(),
            bypass_cache: true,
        };
        match engine.resolve("Search", &opts, &mut rng) {
            Resolution::Hit(h) => {
                assert_eq!(h.backend_node_id, 101);
                assert_eq!(h.stage, super::super::resolver::Stage::Exact);
            }
            other => panic!("expected exact hit, got {other:?}"),
        }
    }
}
