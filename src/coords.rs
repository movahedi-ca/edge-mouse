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
    /// Transform origin the CSS scale is applied around, in this frame's
    /// document coordinates, CSS px. Per the W3C CSS Transforms spec the
    /// scale applies as translate(origin) -> scale -> translate(-origin);
    /// assuming the origin is (0,0) produces a systematic offset for
    /// transformed frames. Callers that genuinely lack the origin must
    /// pass (0.0, 0.0) explicitly; the transform itself applies no silent
    /// default.
    pub transform_origin: (f64, f64),
    /// Device pixel ratio reported for this frame. Carried so the escalation
    /// payload can record the full coordinate mapping; the CSS-space chain
    /// above does not consume it.
    pub dpr: f64,
}

/// Named coordinate-mapping failures. Invalid inputs are reported, never
/// silently corrected: a NaN scroll mapping to the screen origin would
/// click the wrong place with no error signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoordError {
    /// A coordinate input was NaN or infinite.
    NonFiniteInput { field: &'static str },
    /// The CSS scale was NaN, infinite, zero, or negative.
    InvalidScale { field: &'static str },
}

impl std::fmt::Display for CoordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoordError::NonFiniteInput { field } => {
                write!(f, "non-finite coordinate input: {field}")
            }
            CoordError::InvalidScale { field } => {
                write!(
                    f,
                    "invalid CSS scale (NaN, infinite, zero, or negative): {field}"
                )
            }
        }
    }
}

impl std::error::Error for CoordError {}

fn finite(v: f64, field: &'static str) -> Result<f64, CoordError> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(CoordError::NonFiniteInput { field })
    }
}

/// Walk the frame chain from the owning frame to the main frame.
///
/// `chain[0]` is the frame that owns the point; the last entry is the main
/// frame (usually the identity transform). Each step maps the point from the
/// frame's document coordinates into its parent's coordinates: subtract the
/// scroll and the transform origin, apply the CSS scale around the origin,
/// then add the origin back plus the content-box origin and the nested
/// offset. Returns [`CoordError`] on non-finite inputs or a degenerate
/// scale instead of silently substituting values.
pub fn document_to_main_viewport(
    doc: FrameDocumentCss,
    chain: &[FrameTransform],
) -> Result<MainViewportCss, CoordError> {
    let (mut x, mut y) = (finite(doc.0, "doc.x")?, finite(doc.1, "doc.y")?);
    for t in chain {
        let scale = finite(t.css_scale, "css_scale")?;
        // A negative scale mirrors the coordinate to the wrong side of the
        // screen; only strictly positive scales are meaningful. (M1)
        if scale <= 0.0 {
            return Err(CoordError::InvalidScale { field: "css_scale" });
        }
        let (sx, sy) = (
            finite(t.scroll.0, "scroll.x")?,
            finite(t.scroll.1, "scroll.y")?,
        );
        let (ox, oy) = (
            finite(t.transform_origin.0, "transform_origin.x")?,
            finite(t.transform_origin.1, "transform_origin.y")?,
        );
        let (cx, cy) = (
            finite(t.content_origin.0, "content_origin.x")?,
            finite(t.content_origin.1, "content_origin.y")?,
        );
        let (nx, ny) = (
            finite(t.nested_offset.0, "nested_offset.x")?,
            finite(t.nested_offset.1, "nested_offset.y")?,
        );
        x = (x - sx - ox) * scale + ox + cx + nx;
        y = (y - sy - oy) * scale + oy + cy + ny;
    }
    Ok(MainViewportCss(x, y))
}

/// The ONE viewport-to-device function, used by the driver and the hit test.
///
/// Applies page zoom and the device pixel ratio, then rounds to whole device
/// pixels. `window_origin` is the X11 client-window content origin in root
/// device px (typed, never a bare integer pair). A non-finite input or a
/// non-positive zoom*DPR is a [`CoordError`], not a silent fallback.
pub fn to_screen(
    v: MainViewportCss,
    zoom: f64,
    dpr: f64,
    window_origin: RootDevicePx,
) -> Result<RootDevicePx, CoordError> {
    let (vx, vy) = (finite(v.0, "viewport.x")?, finite(v.1, "viewport.y")?);
    let zoom = finite(zoom, "zoom")?;
    let dpr = finite(dpr, "dpr")?;
    let (ox, oy) = (
        finite(window_origin.0, "window_origin.x")?,
        finite(window_origin.1, "window_origin.y")?,
    );
    let scale = zoom * dpr;
    if !(scale.is_finite() && scale > 0.0) {
        return Err(CoordError::InvalidScale { field: "zoom*dpr" });
    }
    // Rounding contract (M2): f64::round is round-half-away-from-zero, so a
    // CSS value landing exactly on a .5 device-px boundary rounds UP for
    // positive coordinates and DOWN for negative ones. Deterministic and
    // pinned by the golden tests; keep it in mind if a future CDP source
    // feeds sub-pixel viewport coordinates with negative window origins.
    Ok(RootDevicePx(
        (ox + vx * scale).round(),
        (oy + vy * scale).round(),
    ))
}

/// Inverse of `to_screen`: device point back into main-viewport CSS px.
/// Hit-test and test use only; the forward path stays `to_screen`.
pub fn screen_to_main_viewport(
    r: RootDevicePx,
    zoom: f64,
    dpr: f64,
    window_origin: RootDevicePx,
) -> Result<MainViewportCss, CoordError> {
    let (rx, ry) = (finite(r.0, "device.x")?, finite(r.1, "device.y")?);
    let zoom = finite(zoom, "zoom")?;
    let dpr = finite(dpr, "dpr")?;
    let (ox, oy) = (
        finite(window_origin.0, "window_origin.x")?,
        finite(window_origin.1, "window_origin.y")?,
    );
    let scale = zoom * dpr;
    if !(scale.is_finite() && scale > 0.0) {
        return Err(CoordError::InvalidScale { field: "zoom*dpr" });
    }
    Ok(MainViewportCss((rx - ox) / scale, (ry - oy) / scale))
}

/// Inverse of `document_to_main_viewport`, for round-trip tests.
pub fn main_viewport_to_document(
    v: MainViewportCss,
    chain: &[FrameTransform],
) -> Result<FrameDocumentCss, CoordError> {
    let (mut x, mut y) = (finite(v.0, "viewport.x")?, finite(v.1, "viewport.y")?);
    for t in chain.iter().rev() {
        let scale = finite(t.css_scale, "css_scale")?;
        // Same guard as the forward walk: negative scales mirror.
        if scale <= 0.0 {
            return Err(CoordError::InvalidScale { field: "css_scale" });
        }
        let (sx, sy) = (
            finite(t.scroll.0, "scroll.x")?,
            finite(t.scroll.1, "scroll.y")?,
        );
        let (ox, oy) = (
            finite(t.transform_origin.0, "transform_origin.x")?,
            finite(t.transform_origin.1, "transform_origin.y")?,
        );
        let (cx, cy) = (
            finite(t.content_origin.0, "content_origin.x")?,
            finite(t.content_origin.1, "content_origin.y")?,
        );
        let (nx, ny) = (
            finite(t.nested_offset.0, "nested_offset.x")?,
            finite(t.nested_offset.1, "nested_offset.y")?,
        );
        x = (x - cx - nx - ox) / scale + ox + sx;
        y = (y - cy - ny - oy) / scale + oy + sy;
    }
    Ok(FrameDocumentCss(x, y))
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
    /// Transform origin the CSS scale applies around, in this frame's
    /// document coordinates, CSS px. CDP does not report it; captures
    /// set (0.0, 0.0) explicitly with the documented assumption that
    /// transformed frames are rare in the target lanes.
    pub transform_origin: (f64, f64),
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
            transform_origin: link.transform_origin,
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
            transform_origin: (0.0, 0.0),
            dpr: 1.0,
        }
    }

    #[test]
    fn chain_identity_passthrough() -> Result<(), CoordError> {
        let doc = FrameDocumentCss(120.0, 340.0);
        let vp = document_to_main_viewport(doc, &[identity_frame()])?;
        assert!((vp.0 - 120.0).abs() < 1e-9);
        assert!((vp.1 - 340.0).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn round_trip_dpr_1_and_2() -> Result<(), CoordError> {
        let doc = FrameDocumentCss(120.0, 340.0);
        let chain = [identity_frame()];
        let vp = document_to_main_viewport(doc, &chain)?;
        let origin = RootDevicePx(10.0, 20.0);
        for dpr in [1.0f64, 2.0f64] {
            let dev = to_screen(vp, 1.0, dpr, origin)?;
            // to_screen is the single conversion path: output is whole pixels.
            assert_eq!(dev.0.fract(), 0.0);
            assert_eq!(dev.1.fract(), 0.0);
            let back = screen_to_main_viewport(dev, 1.0, dpr, origin)?;
            assert!((back.0 - vp.0).abs() < 1e-9);
            assert!((back.1 - vp.1).abs() < 1e-9);
            let doc_back = main_viewport_to_document(back, &chain)?;
            assert!((doc_back.0 - doc.0).abs() < 1e-9);
            assert!((doc_back.1 - doc.1).abs() < 1e-9);
        }
        Ok(())
    }

    #[test]
    fn scroll_and_nested_frame() -> Result<(), CoordError> {
        // Inner frame scrolled by (50, 80); its content box sits at (200, 150)
        // in the parent; OOPIF nested offset (5, 5); unit CSS scale.
        let inner = FrameTransform {
            scroll: (50.0, 80.0),
            content_origin: (200.0, 150.0),
            nested_offset: (5.0, 5.0),
            css_scale: 1.0,
            transform_origin: (0.0, 0.0),
            dpr: 2.0,
        };
        let chain = [inner, identity_frame()];
        let doc = FrameDocumentCss(100.0, 100.0);
        let vp = document_to_main_viewport(doc, &chain)?;
        // (100-50) + 200 + 5 = 255 ; (100-80) + 150 + 5 = 175
        assert!((vp.0 - 255.0).abs() < 1e-9);
        assert!((vp.1 - 175.0).abs() < 1e-9);

        // to_screen at zoom 1.25, DPR 2.0, origin (10, 20):
        // x = 10 + 255*2.5 = 647.5 -> 648 ; y = 20 + 175*2.5 = 457.5 -> 458
        let dev = to_screen(vp, 1.25, 2.0, RootDevicePx(10.0, 20.0))?;
        assert_eq!(dev.0, 648.0);
        assert_eq!(dev.1, 458.0);

        // Round trip stays within one document px (rounding loss only).
        let back = screen_to_main_viewport(dev, 1.25, 2.0, RootDevicePx(10.0, 20.0))?;
        let doc_back = main_viewport_to_document(back, &chain)?;
        assert!((doc_back.0 - 100.0).abs() < 1.0);
        assert!((doc_back.1 - 100.0).abs() < 1.0);
        Ok(())
    }

    #[test]
    fn css_scale_walks_the_chain() -> Result<(), CoordError> {
        let scaled = FrameTransform {
            scroll: (0.0, 0.0),
            content_origin: (10.0, 10.0),
            nested_offset: (0.0, 0.0),
            css_scale: 2.0,
            transform_origin: (0.0, 0.0),
            dpr: 1.0,
        };
        let vp =
            document_to_main_viewport(FrameDocumentCss(5.0, 5.0), &[scaled, identity_frame()])?;
        assert!((vp.0 - 20.0).abs() < 1e-9);
        assert!((vp.1 - 20.0).abs() < 1e-9);
        let doc_back = main_viewport_to_document(vp, &[scaled, identity_frame()])?;
        assert!((doc_back.0 - 5.0).abs() < 1e-9);
        assert!((doc_back.1 - 5.0).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn transform_origin_scales_around_origin() -> Result<(), CoordError> {
        // Regression test for the missing transform-origin: CSS scale
        // applies as translate(origin) -> scale -> translate(-origin).
        // Frame scaled 2x around origin (10, 10), no scroll, no offsets.
        let scaled = FrameTransform {
            scroll: (0.0, 0.0),
            content_origin: (0.0, 0.0),
            nested_offset: (0.0, 0.0),
            css_scale: 2.0,
            transform_origin: (10.0, 10.0),
            dpr: 1.0,
        };
        // (5-0-10)*2 + 10 = 0. The old origin-agnostic formula gave 10.
        let vp = document_to_main_viewport(FrameDocumentCss(5.0, 5.0), &[scaled])?;
        assert!((vp.0 - 0.0).abs() < 1e-9, "got {}", vp.0);
        assert!((vp.1 - 0.0).abs() < 1e-9, "got {}", vp.1);
        // The origin itself is the fixed point: (10-10)*2 + 10 = 10.
        let vp = document_to_main_viewport(FrameDocumentCss(10.0, 10.0), &[scaled])?;
        assert!((vp.0 - 10.0).abs() < 1e-9);
        // Inverse round-trips through the origin-aware formula.
        let doc_back = main_viewport_to_document(vp, &[scaled])?;
        assert!((doc_back.0 - 10.0).abs() < 1e-9);
        assert!((doc_back.1 - 10.0).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn non_finite_inputs_are_errors_not_silent_fallbacks() {
        // The old finite_or silently mapped these to 0.0/1.0; now they
        // are named errors. This is the regression test for the silent
        // correction: none of these may produce a coordinate.
        let good_chain = [identity_frame()];
        let origin = RootDevicePx(0.0, 0.0);
        assert!(document_to_main_viewport(FrameDocumentCss(f64::NAN, 0.0), &good_chain).is_err());
        assert!(
            document_to_main_viewport(FrameDocumentCss(0.0, f64::INFINITY), &good_chain).is_err()
        );
        let mut bad_scroll = identity_frame();
        bad_scroll.scroll = (f64::NAN, 0.0);
        assert!(document_to_main_viewport(FrameDocumentCss(1.0, 1.0), &[bad_scroll]).is_err());
        let mut bad_scale = identity_frame();
        bad_scale.css_scale = 0.0;
        assert_eq!(
            document_to_main_viewport(FrameDocumentCss(1.0, 1.0), &[bad_scale]),
            Err(CoordError::InvalidScale { field: "css_scale" })
        );
        let mut bad_scale = identity_frame();
        bad_scale.css_scale = f64::NAN;
        assert!(document_to_main_viewport(FrameDocumentCss(1.0, 1.0), &[bad_scale]).is_err());
        let vp = MainViewportCss(10.0, 10.0);
        for zoom in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            assert!(to_screen(vp, zoom, 1.0, origin).is_err(), "zoom={zoom}");
        }
        assert!(to_screen(MainViewportCss(f64::NAN, 1.0), 1.0, 1.0, origin).is_err());
        assert!(to_screen(vp, 1.0, 1.0, RootDevicePx(f64::INFINITY, 0.0)).is_err());
    }

    #[test]
    fn negative_css_scale_is_rejected() {
        // M1 regression: a negative scale mirrors the coordinate to the
        // wrong side of the screen. The old `scale == 0.0` guard let it
        // through; only strictly positive scales are meaningful.
        for bad in [-2.0f64, -0.5, -1e-9] {
            let mut fwd = identity_frame();
            fwd.css_scale = bad;
            assert_eq!(
                document_to_main_viewport(FrameDocumentCss(100.0, 100.0), &[fwd]),
                Err(CoordError::InvalidScale { field: "css_scale" }),
                "forward walk accepted css_scale={bad}"
            );
            assert_eq!(
                main_viewport_to_document(MainViewportCss(100.0, 100.0), &[fwd]),
                Err(CoordError::InvalidScale { field: "css_scale" }),
                "inverse walk accepted css_scale={bad}"
            );
        }
        // Sanity: small positive scales still pass.
        let mut tiny = identity_frame();
        tiny.css_scale = 1e-9;
        assert!(document_to_main_viewport(FrameDocumentCss(1.0, 1.0), &[tiny]).is_ok());
    }

    #[test]
    fn scale_with_nonzero_scroll_golden() {
        // T2: the scale-after-scroll ordering is the subtlest line of the
        // transform. frame: scroll (20, 30), css_scale 2, origin (0,0),
        // content origin (0,0). doc (100, 100):
        // x = (100 - 20 - 0) * 2 + 0 + 0 + 0 = 160
        // y = (100 - 30 - 0) * 2 + 0 + 0 + 0 = 140
        let scaled = FrameTransform {
            scroll: (20.0, 30.0),
            content_origin: (0.0, 0.0),
            nested_offset: (0.0, 0.0),
            css_scale: 2.0,
            transform_origin: (0.0, 0.0),
            dpr: 1.0,
        };
        let vp = document_to_main_viewport(FrameDocumentCss(100.0, 100.0), &[scaled])
            .expect("scale+scroll");
        assert!((vp.0 - 160.0).abs() < 1e-9, "got {}", vp.0);
        assert!((vp.1 - 140.0).abs() < 1e-9, "got {}", vp.1);
        // Inverse: x = (160 - 0 - 0 - 0)/2 + 0 + 20 = 100.
        let doc_back = main_viewport_to_document(vp, &[scaled]).expect("inverse");
        assert!((doc_back.0 - 100.0).abs() < 1e-9);
        assert!((doc_back.1 - 100.0).abs() < 1e-9);
    }

    #[test]
    fn half_pixel_rounding_boundaries() {
        // M2/T3: f64::round is round-half-away-from-zero. Pin the boundary
        // contract in both sign quadrants and at two DPRs.
        let origin = RootDevicePx(0.0, 0.0);
        // 0.25 CSS px at DPR 2.0 -> 0.5 device px -> rounds to 1.
        let dev = to_screen(MainViewportCss(0.25, 0.25), 1.0, 2.0, origin).expect("dpr2");
        assert_eq!(dev.0, 1.0);
        assert_eq!(dev.1, 1.0);
        // 0.25 CSS px at DPR 1.0 -> 0.25 device px -> rounds to 0.
        let dev = to_screen(MainViewportCss(0.25, 0.25), 1.0, 1.0, origin).expect("dpr1");
        assert_eq!(dev.0, 0.0);
        // Negative window origin: -0.5 device px rounds away from zero to -1.
        let neg_origin = RootDevicePx(-0.75, -0.75);
        let dev = to_screen(MainViewportCss(0.25, 0.25), 1.0, 1.0, neg_origin).expect("neg");
        assert_eq!(dev.0, -1.0, "got {}", dev.0);
        assert_eq!(dev.1, -1.0, "got {}", dev.1);
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
                transform_origin: (0.0, 0.0),
                dpr: 2.0,
            },
            FrameLink {
                frame_id: "child".to_string(),
                parent_id: Some("main".to_string()),
                scroll: (30.0, 0.0),
                content_origin: (200.0, 150.0),
                nested_offset: (0.0, 0.0),
                css_scale: 1.0,
                transform_origin: (0.0, 0.0),
                dpr: 2.0,
            },
            FrameLink {
                frame_id: "grandchild".to_string(),
                parent_id: Some("child".to_string()),
                scroll: (0.0, 0.0),
                content_origin: (50.0, 60.0),
                nested_offset: (7.0, 3.0), // OOPIF-style nested offset
                css_scale: 1.0,
                transform_origin: (0.0, 0.0),
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
        let vp =
            document_to_main_viewport(FrameDocumentCss(100.0, 100.0), &chain).expect("chain math");
        assert!((vp.0 - 327.0).abs() < 1e-9);
        assert!((vp.1 - 213.0).abs() < 1e-9);
        let back = main_viewport_to_document(vp, &chain).expect("inverse");
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
