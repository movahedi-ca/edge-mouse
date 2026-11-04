//! Frame-chain capture over CDP (mouse-future Phase 1).
//!
//! Captures, per page: per-frame ids, per-frame scroll offsets, and iframe
//! content-box origins; threads frame origins through every candidate at
//! insertion via [`attach_chain`]. Transform *construction* lives in
//! [`super::coords::build_chain`]; this module only captures raw
//! [`FrameLink`] records and the per-node frame index.
//!
//! Capture sources (call names verified against the vendored
//! chromiumoxide 0.7 + chromiumoxide_cdp 0.7.0):
//! - `Page::frames` / `Page::frame_parent` / `Page::mainframe` (Page
//!   domain frame tree; covers OOPIFs, which appear in the page's frame
//!   tree with their own frame ids).
//! - `DOMSnapshot.captureSnapshot` (via `Page::execute`): one document
//!   per frame, with per-document `scrollOffsetX`/`scrollOffsetY` and a
//!   `backendNodeId` array per document, which yields the per-node frame
//!   index in a single round trip.
//! - `DOM.getFrameOwner` + `DOM.resolveNode` + `DOM.getBoxModel`: the
//!   iframe element lives in the parent frame's DOM even for OOPIFs, so
//!   the content quad's top-left is the frame's content-box origin in the
//!   parent frame's document coordinates, same- and out-of-process alike.
//!
//! OOPIF note: chromiumoxide_cdp 0.7's `TargetInfo` carries
//! `openerFrameId` but no frame id, so the browser target tree cannot
//! map OOPIF targets back onto frame ids. OOPIF *geometry* does not need
//! that mapping (see above); the target tree is inventoried
//! best-effort for future mapping work and to record whether target
//! info was available at all.
//!
//! Fallback (documented): when `Target.getTargets` fails or the snapshot
//! is unavailable (`target_tree_available == false`, or
//! [`attach_chain`] returns [`FrameChainError::UnknownNode`]), callers
//! fall back to a FULL-TREE REFRESH: re-run [`capture_frame_state`] from
//! scratch instead of patching the stale state incrementally.
//!
//! Async boundary: capture runs on the route/handler side (tokio), never
//! on the input/X11 thread. Blocking X11 calls stay behind
//! `spawn_blocking`; this module makes no X11 calls at all.

use std::collections::{HashMap, HashSet};

#[cfg(feature = "cdp")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "cdp")]
use chromiumoxide::cdp::browser_protocol::dom::{
    BackendNodeId, GetBoxModelParams, GetFrameOwnerParams, Quad, ResolveNodeParams,
};
#[cfg(feature = "cdp")]
use chromiumoxide::cdp::browser_protocol::dom_snapshot::CaptureSnapshotParams;
#[cfg(feature = "cdp")]
use chromiumoxide::cdp::browser_protocol::page::FrameId;
#[cfg(feature = "cdp")]
use chromiumoxide::cdp::browser_protocol::target::GetTargetsParams;
#[cfg(feature = "cdp")]
use chromiumoxide::{Browser, Page};

use super::coords::{build_chain, FrameChainError, FrameLink, FrameTransform};

/// One full frame capture for a page.
#[derive(Debug, Clone, Default)]
pub struct FrameState {
    /// Raw per-frame records, in `Page::frames` order.
    pub links: Vec<FrameLink>,
    /// Per-node frame index: `backendDOMNodeId` -> owning frame id
    /// (from `DOMSnapshot`: one document per frame).
    pub node_frames: HashMap<i64, String>,
    /// Frame ids whose owner origin could not be captured (detached
    /// mid-capture, box model unavailable). Their `content_origin` is
    /// (0, 0): consumers that need exact geometry must treat the state
    /// as stale and run a full-tree refresh.
    pub degraded_frames: Vec<String>,
    /// `window.devicePixelRatio` of the main frame at capture time.
    pub dpr: f64,
    /// Whether `Target.getTargets` succeeded. When false, callers must
    /// not rely on incremental updates: full-tree refresh only.
    pub target_tree_available: bool,
    /// Target ids of type `"iframe"` seen in the browser target tree
    /// (best-effort OOPIF inventory; not mapped to frame ids: cdp 0.7's
    /// `TargetInfo` has no frame id field).
    pub oopif_target_ids: Vec<String>,
}

impl FrameState {
    /// Frame ids currently known to be out-of-process, by the only
    /// mapping cdp 0.7 supports: none. Returns the empty set and
    /// documents why: `TargetInfo` carries `openerFrameId` but no frame
    /// id, so per-frame OOPIF marking is unavailable. OOPIF chains are
    /// still correct: the Page-domain frame tree includes OOPIFs, scroll
    /// offsets arrive per-document from `DOMSnapshot`, and owner origins
    /// come from `DOM.getBoxModel` on the frame owner in the parent DOM.
    pub fn oopif_frames(&self) -> HashSet<String> {
        HashSet::new()
    }
}

/// Capture the full frame state for `page`.
///
/// `browser` is `Some` when the caller holds the browser handle: then
/// `Target.getTargets` runs on the browser session for the OOPIF target
/// inventory. `None` (or a failed command) sets
/// `target_tree_available = false` and the caller falls back to
/// full-tree refreshes.
///
/// Available only with the `cdp` feature: capture runs on the caller's
/// async runtime (tokio), never on the input/X11 thread.
#[cfg(feature = "cdp")]
pub async fn capture_frame_state(page: &Page, browser: Option<&Browser>) -> Result<FrameState> {
    // 1. Frame tree (Page domain; includes OOPIFs with their own ids).
    let frames: Vec<FrameId> = page.frames().await.context("Page.frames")?;
    let main_id: Option<String> = page
        .mainframe()
        .await
        .context("Page.mainframe")?
        .map(|f| f.inner().clone());
    let mut parents: HashMap<String, Option<String>> = HashMap::with_capacity(frames.len());
    for fid in &frames {
        let parent = page
            .frame_parent(fid.clone())
            .await
            .context("Page.frame_parent")?;
        parents.insert(fid.inner().clone(), parent.map(|p| p.inner().clone()));
    }

    // 2. DOMSnapshot: per-document scroll offsets + per-node frame index,
    // one round trip.
    //
    // Chrome 156 quirk (verified live 2026-11-03 against the supervised
    // browser): DOMSnapshot.captureSnapshot returns -32602 unless
    // `computedStyles` is a NON-EMPTY array. `{}` and
    // `{"includeDOMRects":true}` both fail; `{"computedStyles":
    // ["display"]}` succeeds. The vendored chromiumoxide_cdp 0.7.0 marks
    // the field optional, but the browser disagrees, so always request
    // the one cheap style (its values are ignored).
    let snap = page
        .execute(CaptureSnapshotParams::new(vec!["display".to_string()]))
        .await
        .context("DOMSnapshot.captureSnapshot")?
        .result;
    let mut scroll_by_frame: HashMap<String, (f64, f64)> = HashMap::new();
    let mut node_frames: HashMap<i64, String> = HashMap::new();
    for doc in &snap.documents {
        let idx = *doc.frame_id.inner() as usize;
        let fid = snap.strings.get(idx).cloned().unwrap_or_default();
        scroll_by_frame.insert(
            fid.clone(),
            (
                doc.scroll_offset_x.unwrap_or(0.0),
                doc.scroll_offset_y.unwrap_or(0.0),
            ),
        );
        if let Some(ids) = &doc.nodes.backend_node_id {
            for b in ids {
                node_frames.insert(*b.inner(), fid.clone());
            }
        }
    }

    // 3. DPR from the main frame. Page zoom is applied at `to_screen`
    // time by the caller, so per-frame `css_scale` stays 1.0.
    let dpr = page
        .evaluate("window.devicePixelRatio")
        .await
        .ok()
        .and_then(|r| r.into_value::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1.0);

    // 4. Per-frame owner origins via DOM.getFrameOwner + DOM.getBoxModel.
    let mut links = Vec::with_capacity(frames.len());
    let mut degraded = Vec::new();
    for fid in &frames {
        let id = fid.inner().clone();
        let is_main = Some(&id) == main_id.as_ref();
        let content_origin = if is_main {
            (0.0, 0.0)
        } else {
            match frame_owner_origin(page, fid.clone()).await {
                Ok(origin) => origin,
                Err(_) => {
                    degraded.push(id.clone());
                    (0.0, 0.0)
                }
            }
        };
        links.push(FrameLink {
            frame_id: id.clone(),
            parent_id: parents.get(&id).cloned().flatten(),
            scroll: scroll_by_frame.get(&id).copied().unwrap_or((0.0, 0.0)),
            content_origin,
            nested_offset: (0.0, 0.0),
            css_scale: 1.0,
            // CDP does not report the CSS transform origin: pass (0,0)
            // explicitly per the documented assumption (transformed
            // frames are rare in the target lanes).
            transform_origin: (0.0, 0.0),
            dpr,
        });
    }

    // 5. Browser target tree: best-effort OOPIF target inventory.
    let (target_tree_available, oopif_target_ids) = match browser {
        Some(b) => match b.execute(GetTargetsParams::builder().build()).await {
            Ok(resp) => {
                let ids = resp
                    .result
                    .target_infos
                    .iter()
                    .filter(|t| t.r#type == "iframe")
                    .map(|t| t.target_id.inner().clone())
                    .collect();
                (true, ids)
            }
            Err(_) => (false, Vec::new()),
        },
        None => (false, Vec::new()),
    };

    Ok(FrameState {
        links,
        node_frames,
        degraded_frames: degraded,
        dpr,
        target_tree_available,
        oopif_target_ids,
    })
}

/// Content-box origin of the iframe element that owns `frame_id`, in the
/// parent frame's document coordinates: `DOM.getFrameOwner` yields the
/// owner's backend node id, `DOM.resolveNode` the object id,
/// `DOM.getBoxModel` the content quad whose top-left is the origin.
#[cfg(feature = "cdp")]
async fn frame_owner_origin(page: &Page, frame_id: FrameId) -> Result<(f64, f64)> {
    let owner = page
        .execute(GetFrameOwnerParams::new(frame_id))
        .await
        .context("DOM.getFrameOwner")?
        .result;
    let backend = *owner.backend_node_id.inner();
    let object_id = page
        .execute(
            ResolveNodeParams::builder()
                .backend_node_id(BackendNodeId::new(backend))
                .build(),
        )
        .await
        .context("DOM.resolveNode (frame owner)")?
        .result
        .object
        .object_id
        .ok_or_else(|| anyhow::anyhow!("DOM.resolveNode: no objectId for frame owner"))?;
    let model = page
        .execute(GetBoxModelParams::builder().object_id(object_id).build())
        .await
        .context("DOM.getBoxModel (frame owner)")?
        .result
        .model;
    let q = model.content.inner();
    if q.len() < 8 {
        anyhow::bail!("DOM.getBoxModel: short content quad for frame owner");
    }
    Ok(content_origin_of(&model.content))
}

/// Top-left of a content quad `[x1,y1, x2,y2, x3,y3, x4,y4]`.
#[cfg(feature = "cdp")]
fn content_origin_of(quad: &Quad) -> (f64, f64) {
    let q = quad.inner();
    let min_x = q.iter().step_by(2).copied().fold(f64::INFINITY, f64::min);
    let min_y = q
        .iter()
        .skip(1)
        .step_by(2)
        .copied()
        .fold(f64::INFINITY, f64::min);
    (min_x, min_y)
}

/// Insertion-time primitive for the resolver (Phase 4): given a backend
/// node id, return its owning frame id plus the ready-to-use chain
/// (target frame first, main frame last), built with
/// [`super::coords::build_chain`].
///
/// Returns [`FrameChainError::UnknownNode`] when the node is not in the
/// captured index, or the build error when its frame left the tree: the
/// caller falls back to a full-tree refresh (re-run
/// [`capture_frame_state`]) instead of patching incrementally.
pub fn attach_chain(
    backend_node_id: i64,
    state: &FrameState,
) -> Result<(String, Vec<FrameTransform>), FrameChainError> {
    let frame_id = state
        .node_frames
        .get(&backend_node_id)
        .ok_or(FrameChainError::UnknownNode { backend_node_id })?;
    let chain = build_chain(&state.links, frame_id)?;
    Ok((frame_id.clone(), chain))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cdp")]
    fn content_origin_takes_quad_top_left() {
        // Clockwise quad starting bottom-right: min must still win.
        let quad = Quad::new(vec![210.0, 160.0, 200.0, 160.0, 200.0, 150.0, 210.0, 150.0]);
        assert_eq!(content_origin_of(&quad), (200.0, 150.0));
    }

    fn state_with_two_frames() -> FrameState {
        FrameState {
            links: vec![
                FrameLink {
                    frame_id: "main".to_string(),
                    parent_id: None,
                    scroll: (0.0, 40.0),
                    content_origin: (0.0, 0.0),
                    nested_offset: (0.0, 0.0),
                    css_scale: 1.0,
                    transform_origin: (0.0, 0.0),
                    dpr: 1.0,
                },
                FrameLink {
                    frame_id: "sub".to_string(),
                    parent_id: Some("main".to_string()),
                    scroll: (10.0, 0.0),
                    content_origin: (100.0, 200.0),
                    nested_offset: (0.0, 0.0),
                    css_scale: 1.0,
                    transform_origin: (0.0, 0.0),
                    dpr: 1.0,
                },
            ],
            node_frames: [(42, "sub".to_string()), (7, "main".to_string())]
                .into_iter()
                .collect(),
            degraded_frames: Vec::new(),
            dpr: 1.0,
            target_tree_available: true,
            oopif_target_ids: Vec::new(),
        }
    }

    #[test]
    fn attach_chain_threads_frame_origin_at_insertion() {
        let state = state_with_two_frames();
        let (frame_id, chain) = attach_chain(42, &state).expect("chain attaches");
        assert_eq!(frame_id, "sub");
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].content_origin, (100.0, 200.0));
        assert_eq!(chain[0].scroll, (10.0, 0.0));
        assert_eq!(chain[1].scroll, (0.0, 40.0));
    }

    #[test]
    fn attach_chain_unknown_node_is_named() {
        let state = state_with_two_frames();
        assert_eq!(
            attach_chain(999, &state).unwrap_err(),
            FrameChainError::UnknownNode {
                backend_node_id: 999
            }
        );
    }

    #[test]
    fn oopif_frames_documents_the_cdp_limitation() {
        // cdp 0.7 TargetInfo has no frame id: no per-frame OOPIF marking.
        let state = state_with_two_frames();
        assert!(state.oopif_frames().is_empty());
    }
}
