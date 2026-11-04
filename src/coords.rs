//! Typed coordinate spaces for the edge engine (mouse-future Phase 0).
//!
//! Coordinate spaces are types, not comments: a viewport point cannot be
//! passed where a document point is expected, and no bare (i32, i32) pair
//! crosses a module boundary. The conversion chain is:
//!
//!   FrameDocumentCss --document_to_main_viewport--> MainViewportCss
//!       --to_screen--> RootDevicePx
//!
//! `to_screen` is the ONE viewport-to-device function. No other function in
//! edge/ performs viewport-to-device math.

/// CSS px, relative to the document of the frame that owns the node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameDocumentCss(pub f64, pub f64);

/// CSS px, relative to the main frame viewport (post scroll and frame chain).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MainViewportCss(pub f64, pub f64);

/// Device px in the X11 root window (XTest space). Values are whole pixels;
/// rounding happens explicitly inside `to_screen`, the single conversion site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootDevicePx(pub f64, pub f64);

/// One frame's contribution to the document-to-main-viewport chain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTransform {
    /// (scroll_x, scroll_y) of this frame's document, CSS px.
    pub scroll: (f64, f64),
    /// Origin of this frame's content box in the parent frame's coordinates,
    /// CSS px.
    pub content_origin: (f64, f64),
    /// Extra offset of this frame inside the parent viewport (for example the
    /// OOPIF offset reported by the browser target tree), CSS px.
    pub nested_offset: (f64, f64),
    /// CSS transform scale applied to this frame's document.
    pub css_scale: f64,
    /// Device pixel ratio reported for this frame. Carried so the escalation
    /// payload can record the full coordinate mapping; the CSS-space chain
    /// above does not consume it.
    pub dpr: f64,
}

fn finite_or(v: f64, fallback: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        fallback
    }
}

/// Walk the frame chain from the owning frame to the main frame.
///
/// `chain[0]` is the frame that owns the point; the last entry is the main
/// frame (usually the identity transform). Each step maps the point from the
/// frame's document coordinates into its parent's coordinates: subtract the
/// scroll, apply the CSS scale, then add the content-box origin and the
/// nested offset.
pub fn document_to_main_viewport(
    doc: FrameDocumentCss,
    chain: &[FrameTransform],
) -> MainViewportCss {
    let (mut x, mut y) = (finite_or(doc.0, 0.0), finite_or(doc.1, 0.0));
    for t in chain {
        let scale = if t.css_scale.is_finite() && t.css_scale != 0.0 {
            t.css_scale
        } else {
            1.0
        };
        x = (x - finite_or(t.scroll.0, 0.0)) * scale
            + finite_or(t.content_origin.0, 0.0)
            + finite_or(t.nested_offset.0, 0.0);
        y = (y - finite_or(t.scroll.1, 0.0)) * scale
            + finite_or(t.content_origin.1, 0.0)
            + finite_or(t.nested_offset.1, 0.0);
    }
    MainViewportCss(finite_or(x, 0.0), finite_or(y, 0.0))
}

/// The ONE viewport-to-device function, used by the driver and the hit test.
///
/// Applies page zoom and the device pixel ratio, then rounds to whole device
/// pixels. `window_origin` is the X11 client-window content origin in root
/// device px (typed, never a bare integer pair). A non-finite or
/// non-positive zoom*DPR falls back to 1.0; non-finite results fall back
/// to 0.0.
pub fn to_screen(
    v: MainViewportCss,
    zoom: f64,
    dpr: f64,
    window_origin: RootDevicePx,
) -> RootDevicePx {
    let scale = finite_or(zoom * dpr, 0.0);
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let x = finite_or(window_origin.0, 0.0) + finite_or(v.0, 0.0) * scale;
    let y = finite_or(window_origin.1, 0.0) + finite_or(v.1, 0.0) * scale;
    RootDevicePx(finite_or(x, 0.0).round(), finite_or(y, 0.0).round())
}

/// Inverse of `to_screen`: device point back into main-viewport CSS px.
/// Hit-test and test use only; the forward path stays `to_screen`.
pub fn screen_to_main_viewport(
    r: RootDevicePx,
    zoom: f64,
    dpr: f64,
    window_origin: RootDevicePx,
) -> MainViewportCss {
    let scale = finite_or(zoom * dpr, 0.0);
    let scale = if scale > 0.0 { scale } else { 1.0 };
    MainViewportCss(
        (finite_or(r.0, 0.0) - finite_or(window_origin.0, 0.0)) / scale,
        (finite_or(r.1, 0.0) - finite_or(window_origin.1, 0.0)) / scale,
    )
}

/// Inverse of `document_to_main_viewport`, for round-trip tests.
pub fn main_viewport_to_document(v: MainViewportCss, chain: &[FrameTransform]) -> FrameDocumentCss {
    let (mut x, mut y) = (finite_or(v.0, 0.0), finite_or(v.1, 0.0));
    for t in chain.iter().rev() {
        let scale = if t.css_scale.is_finite() && t.css_scale != 0.0 {
            t.css_scale
        } else {
            1.0
        };
        x = (x - finite_or(t.content_origin.0, 0.0) - finite_or(t.nested_offset.0, 0.0)) / scale
            + finite_or(t.scroll.0, 0.0);
        y = (y - finite_or(t.content_origin.1, 0.0) - finite_or(t.nested_offset.1, 0.0)) / scale
            + finite_or(t.scroll.1, 0.0);
    }
    FrameDocumentCss(finite_or(x, 0.0), finite_or(y, 0.0))
}

// ---------------------------------------------------------------------------
// Frame-chain construction (Phase 1).
//
// `FrameLink` is the raw per-frame record captured from CDP (see
// `edge::frame_chain`): the frame's own scroll offsets, the content-box
// origin of the iframe element that owns it in the parent frame's
// document coordinates, and the OOPIF nested offset from the browser
// target tree. `build_chain` turns a flat link list into the ordered
// `Vec<FrameTransform>` that `document_to_main_viewport` consumes:
// chain[0] is the frame that owns the point, the last entry is the main
// frame. This is the ONE place that assembles chains; the math itself
// stays in `document_to_main_viewport` / `to_screen` and is never
// duplicated.
// ---------------------------------------------------------------------------

/// One frame's raw record, as captured from CDP.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameLink {
    /// CDP `Page.FrameId` of this frame.
    pub frame_id: String,
    /// CDP `Page.FrameId` of the parent frame; `None` for the main frame.
    pub parent_id: Option<String>,
    /// `(scrollX, scrollY)` of this frame's own document, CSS px
    /// (from `DOMSnapshot` per-document scroll offsets).
    pub scroll: (f64, f64),
    /// Origin of this frame's content box in the parent frame's document
    /// coordinates, CSS px (from `DOM.getBoxModel` on the frame owner
    /// element). `(0, 0)` for the main frame.
    pub content_origin: (f64, f64),
    /// Extra offset inside the parent viewport for out-of-process frames,
    /// CSS px (from the browser target tree). `(0, 0)` for same-process
    /// frames; OOPIF geometry itself still comes from `DOM.getBoxModel`
    /// on the frame owner, which lives in the parent DOM.
    pub nested_offset: (f64, f64),
    /// CSS `zoom`/transform scale applied to this frame's document
    /// (1.0 unless the page zooms a frame).
    pub css_scale: f64,
    /// Device pixel ratio governing this frame.
    pub dpr: f64,
}

/// Named frame-chain assembly failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameChainError {
    /// No links were captured at all.
    EmptyLinks,
    /// The target frame id is not in the captured link set.
    UnknownFrame { frame_id: String },
    /// The backend node id is not in the captured per-node frame index
    /// (node from another document, or captured before the index).
    UnknownNode { backend_node_id: i64 },
    /// A link's parent id is not in the captured link set: the chain is
    /// incomplete (frame attached/detached mid-capture).
    OrphanedFrame { frame_id: String, parent_id: String },
    /// The parent walk cycled: the captured tree is corrupt.
    Cycle { frame_id: String },
}

impl std::fmt::Display for FrameChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameChainError::EmptyLinks => write!(f, "no frame links captured"),
            FrameChainError::UnknownNode { backend_node_id } => write!(
                f,
                "backend node {backend_node_id} not in captured per-node frame index"
            ),
            FrameChainError::UnknownFrame { frame_id } => {
                write!(f, "frame {frame_id} not in captured frame set")
            }
            FrameChainError::OrphanedFrame {
                frame_id,
                parent_id,
            } => write!(
                f,
                "frame {frame_id} has parent {parent_id} outside the captured frame set"
            ),
            FrameChainError::Cycle { frame_id } => {
                write!(f, "frame parent walk cycled at {frame_id}")
            }
        }
    }
}

impl std::error::Error for FrameChainError {}

/// Assemble the ordered [`FrameTransform`] chain for `target_frame` from
/// a flat CDP capture.
///
/// Walks `parent_id` from the target frame up to the root and returns the
/// transforms ordered target-first, main-frame-last, exactly as
/// [`document_to_main_viewport`] expects. Every link keeps its own
/// scroll, content origin, nested offset, CSS scale, and DPR: the walk
/// never merges or averages frames.
pub fn build_chain(
    links: &[FrameLink],
    target_frame: &str,
) -> Result<Vec<FrameTransform>, FrameChainError> {
    if links.is_empty() {
        return Err(FrameChainError::EmptyLinks);
    }
    let mut chain = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    let mut current = target_frame;
    loop {
        if seen.contains(&current) {
            return Err(FrameChainError::Cycle {
                frame_id: current.to_string(),
            });
        }
        seen.push(current);
        let link = links
            .iter()
            .find(|l| l.frame_id == current)
            .ok_or_else(|| {
                if current == target_frame {
                    FrameChainError::UnknownFrame {
                        frame_id: current.to_string(),
                    }
                } else {
                    // `seen` always holds at least the target frame here,
                    // so `seen[seen.len() - 2]` is the child that named
                    // this missing parent.
                    let child = seen[seen.len() - 2].to_string();
                    FrameChainError::OrphanedFrame {
                        frame_id: child,
                        parent_id: current.to_string(),
                    }
                }
            })?;
        chain.push(FrameTransform {
            scroll: link.scroll,
            content_origin: link.content_origin,
            nested_offset: link.nested_offset,
            css_scale: link.css_scale,
            dpr: link.dpr,
        });
        match &link.parent_id {
            Some(parent) => current = parent,
            None => break,
        }
    }
    Ok(chain)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_frame() -> FrameTransform {
        FrameTransform {
            scroll: (0.0, 0.0),
            content_origin: (0.0, 0.0),
            nested_offset: (0.0, 0.0),
            css_scale: 1.0,
            dpr: 1.0,
        }
    }

    #[test]
    fn chain_identity_passthrough() {
        let doc = FrameDocumentCss(120.0, 340.0);
        let vp = document_to_main_viewport(doc, &[identity_frame()]);
        assert!((vp.0 - 120.0).abs() < 1e-9);
        assert!((vp.1 - 340.0).abs() < 1e-9);
    }

    #[test]
    fn round_trip_dpr_1_and_2() {
        let doc = FrameDocumentCss(120.0, 340.0);
        let chain = [identity_frame()];
        let vp = document_to_main_viewport(doc, &chain);
        let origin = RootDevicePx(10.0, 20.0);
        for dpr in [1.0f64, 2.0f64] {
            let dev = to_screen(vp, 1.0, dpr, origin);
            // to_screen is the single conversion path: output is whole pixels.
            assert_eq!(dev.0.fract(), 0.0);
            assert_eq!(dev.1.fract(), 0.0);
            let back = screen_to_main_viewport(dev, 1.0, dpr, origin);
            assert!((back.0 - vp.0).abs() < 1e-9);
            assert!((back.1 - vp.1).abs() < 1e-9);
            let doc_back = main_viewport_to_document(back, &chain);
            assert!((doc_back.0 - doc.0).abs() < 1e-9);
            assert!((doc_back.1 - doc.1).abs() < 1e-9);
        }
    }

    #[test]
    fn scroll_and_nested_frame() {
        // Inner frame scrolled by (50, 80); its content box sits at (200, 150)
        // in the parent; OOPIF nested offset (5, 5); unit CSS scale.
        let inner = FrameTransform {
            scroll: (50.0, 80.0),
            content_origin: (200.0, 150.0),
            nested_offset: (5.0, 5.0),
            css_scale: 1.0,
            dpr: 2.0,
        };
        let chain = [inner, identity_frame()];
        let doc = FrameDocumentCss(100.0, 100.0);
        let vp = document_to_main_viewport(doc, &chain);
        // (100-50) + 200 + 5 = 255 ; (100-80) + 150 + 5 = 175
        assert!((vp.0 - 255.0).abs() < 1e-9);
        assert!((vp.1 - 175.0).abs() < 1e-9);

        // to_screen at zoom 1.25, DPR 2.0, origin (10, 20):
        // x = 10 + 255*2.5 = 647.5 -> 648 ; y = 20 + 175*2.5 = 457.5 -> 458
        let dev = to_screen(vp, 1.25, 2.0, RootDevicePx(10.0, 20.0));
        assert_eq!(dev.0, 648.0);
        assert_eq!(dev.1, 458.0);

        // Round trip stays within one document px (rounding loss only).
        let back = screen_to_main_viewport(dev, 1.25, 2.0, RootDevicePx(10.0, 20.0));
        let doc_back = main_viewport_to_document(back, &chain);
        assert!((doc_back.0 - 100.0).abs() < 1.0);
        assert!((doc_back.1 - 100.0).abs() < 1.0);
    }

    #[test]
    fn css_scale_walks_the_chain() {
        let scaled = FrameTransform {
            scroll: (0.0, 0.0),
            content_origin: (10.0, 10.0),
            nested_offset: (0.0, 0.0),
            css_scale: 2.0,
            dpr: 1.0,
        };
        let vp = document_to_main_viewport(FrameDocumentCss(5.0, 5.0), &[scaled, identity_frame()]);
        assert!((vp.0 - 20.0).abs() < 1e-9);
        assert!((vp.1 - 20.0).abs() < 1e-9);
        let doc_back = main_viewport_to_document(vp, &[scaled, identity_frame()]);
        assert!((doc_back.0 - 5.0).abs() < 1e-9);
        assert!((doc_back.1 - 5.0).abs() < 1e-9);
    }

    #[test]
    fn to_screen_guards_non_finite_scale() {
        let vp = MainViewportCss(10.0, 10.0);
        let origin = RootDevicePx(0.0, 0.0);
        for zoom in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            let d = to_screen(vp, zoom, 1.0, origin);
            assert!(d.0.is_finite() && d.1.is_finite());
            assert_eq!(d.0.fract(), 0.0);
        }
        // Non-finite inputs never produce non-finite output.
        let d = to_screen(MainViewportCss(f64::NAN, f64::NAN), 1.0, 1.0, origin);
        assert!(d.0.is_finite() && d.1.is_finite());
    }

    fn three_level_links() -> Vec<FrameLink> {
        vec![
            FrameLink {
                frame_id: "main".to_string(),
                parent_id: None,
                scroll: (0.0, 100.0),
                content_origin: (0.0, 0.0),
                nested_offset: (0.0, 0.0),
                css_scale: 1.0,
                dpr: 2.0,
            },
            FrameLink {
                frame_id: "child".to_string(),
                parent_id: Some("main".to_string()),
                scroll: (30.0, 0.0),
                content_origin: (200.0, 150.0),
                nested_offset: (0.0, 0.0),
                css_scale: 1.0,
                dpr: 2.0,
            },
            FrameLink {
                frame_id: "grandchild".to_string(),
                parent_id: Some("child".to_string()),
                scroll: (0.0, 0.0),
                content_origin: (50.0, 60.0),
                nested_offset: (7.0, 3.0), // OOPIF-style nested offset
                css_scale: 1.0,
                dpr: 2.0,
            },
        ]
    }

    #[test]
    fn build_chain_orders_target_first_main_last() {
        let links = three_level_links();
        let chain = build_chain(&links, "grandchild").expect("chain builds");
        assert_eq!(chain.len(), 3);
        // chain[0] is the owning frame, last is the main frame.
        assert_eq!(chain[0].scroll, (0.0, 0.0));
        assert_eq!(chain[0].content_origin, (50.0, 60.0));
        assert_eq!(chain[0].nested_offset, (7.0, 3.0));
        assert_eq!(chain[1].scroll, (30.0, 0.0));
        assert_eq!(chain[1].content_origin, (200.0, 150.0));
        assert_eq!(chain[2].scroll, (0.0, 100.0));
        assert_eq!(chain[2].content_origin, (0.0, 0.0));
    }

    #[test]
    fn build_chain_end_to_end_math() {
        // Hand-computed: point (100, 100) in the grandchild document.
        // grandchild: (100-0) + 50 + 7 = 157 ; (100-0) + 60 + 3 = 163
        // child:      (157-30) + 200 = 327 ; (163-0) + 150 = 313
        // main:       (327-0) = 327 ; (313-100) = 213
        let links = three_level_links();
        let chain = build_chain(&links, "grandchild").expect("chain builds");
        let vp = document_to_main_viewport(FrameDocumentCss(100.0, 100.0), &chain);
        assert!((vp.0 - 327.0).abs() < 1e-9);
        assert!((vp.1 - 213.0).abs() < 1e-9);
        let back = main_viewport_to_document(vp, &chain);
        assert!((back.0 - 100.0).abs() < 1e-9);
        assert!((back.1 - 100.0).abs() < 1e-9);
    }

    #[test]
    fn build_chain_named_errors() {
        let links = three_level_links();
        assert_eq!(
            build_chain(&links, "nope").unwrap_err(),
            FrameChainError::UnknownFrame {
                frame_id: "nope".to_string()
            }
        );
        assert_eq!(
            build_chain(&[], "main").unwrap_err(),
            FrameChainError::EmptyLinks
        );
        // Orphan: child names a parent outside the captured set.
        let mut orphan = three_level_links();
        orphan[1].parent_id = Some("ghost".to_string());
        assert_eq!(
            build_chain(&orphan, "grandchild").unwrap_err(),
            FrameChainError::OrphanedFrame {
                frame_id: "child".to_string(),
                parent_id: "ghost".to_string()
            }
        );
        // Cycle: main's parent points back at the grandchild.
        let mut cyclic = three_level_links();
        cyclic[0].parent_id = Some("grandchild".to_string());
        assert_eq!(
            build_chain(&cyclic, "grandchild").unwrap_err(),
            FrameChainError::Cycle {
                frame_id: "grandchild".to_string()
            }
        );
    }
}
