//! Golden coordinate tests (mouse-future Phase 1).
//!
//! Every expectation below is hand-computed in the comments; a golden
//! failure fails the build. The tests drive the typed pipeline end to
//! end: `FrameLink` -> `build_chain` -> `document_to_main_viewport` ->
//! `to_screen` -> `screen_to_main_viewport` -> `main_viewport_to_document`,
//! plus the SEC-1 window gate (`check_in_window`) for moved windows and
//! the i16 range check.
//!
//! The modules under test are included by path: the bridge is a binary
//! crate, and these are the exact sources the binary compiles.

use edge_mouse::coords::{
    build_chain, document_to_main_viewport, main_viewport_to_document, screen_to_main_viewport,
    to_screen, FrameChainError, FrameDocumentCss, FrameLink, FrameTransform, MainViewportCss,
    RootDevicePx,
};
use edge_mouse::frame_chain::{attach_chain, FrameState};
use edge_mouse::legacy::{check_in_window, WindowRect};

fn main_frame(scroll: (f64, f64), dpr: f64) -> FrameLink {
    FrameLink {
        frame_id: "main".to_string(),
        parent_id: None,
        scroll,
        content_origin: (0.0, 0.0),
        nested_offset: (0.0, 0.0),
        css_scale: 1.0,
        dpr,
    }
}

fn sub_frame(
    id: &str,
    parent: &str,
    scroll: (f64, f64),
    content_origin: (f64, f64),
    nested_offset: (f64, f64),
) -> FrameLink {
    FrameLink {
        frame_id: id.to_string(),
        parent_id: Some(parent.to_string()),
        scroll,
        content_origin,
        nested_offset,
        css_scale: 1.0,
        dpr: 1.0,
    }
}

/// DPR 1.0 and 2.0 with page zoom and a scrolled main frame.
///
/// chain: main scrolled (0, 200). doc (300, 500) -> viewport (300, 300).
#[test]
fn golden_dpr_zoom_scroll() {
    let links = [main_frame((0.0, 200.0), 1.0)];
    let chain = build_chain(&links, "main").expect("chain builds");
    let vp = document_to_main_viewport(FrameDocumentCss(300.0, 500.0), &chain);
    assert_eq!(vp, MainViewportCss(300.0, 300.0));

    // zoom 1.0, DPR 1.0, origin (50, 80): (50+300, 80+300) = (350, 380).
    let d1 = to_screen(vp, 1.0, 1.0, RootDevicePx(50.0, 80.0));
    assert_eq!(d1, RootDevicePx(350.0, 380.0));

    // zoom 1.25, DPR 2.0, origin (50, 80): scale 2.5 ->
    // (50+750, 80+750) = (800, 830).
    let d2 = to_screen(vp, 1.25, 2.0, RootDevicePx(50.0, 80.0));
    assert_eq!(d2, RootDevicePx(800.0, 830.0));

    // Exact round trip at DPR 2 (no rounding loss: 800/830 divide evenly).
    let back = screen_to_main_viewport(d2, 1.25, 2.0, RootDevicePx(50.0, 80.0));
    assert_eq!(back, MainViewportCss(300.0, 300.0));
    let doc_back = main_viewport_to_document(back, &chain);
    assert_eq!(doc_back, FrameDocumentCss(300.0, 500.0));
}

/// Three nested same-process iframes.
///
/// grandchild doc (10, 10):
///   grandchild: (10-0)+40 = 50 ; (10-0)+50 = 60
///   child:      (50-20)+100 = 130 ; (60-30)+120 = 150
///   main:       (130, 150)
#[test]
fn golden_nested_iframes() {
    let links = [
        main_frame((0.0, 0.0), 1.0),
        sub_frame("child", "main", (20.0, 30.0), (100.0, 120.0), (0.0, 0.0)),
        sub_frame("grandchild", "child", (0.0, 0.0), (40.0, 50.0), (0.0, 0.0)),
    ];
    let chain = build_chain(&links, "grandchild").expect("chain builds");
    assert_eq!(chain.len(), 3);
    let vp = document_to_main_viewport(FrameDocumentCss(10.0, 10.0), &chain);
    assert_eq!(vp, MainViewportCss(130.0, 150.0));

    // 72px toolbar + 8px side chrome: origin (8, 80).
    // zoom 1, DPR 1: (8+130, 80+150) = (138, 230).
    let d1 = to_screen(vp, 1.0, 1.0, RootDevicePx(8.0, 80.0));
    assert_eq!(d1, RootDevicePx(138.0, 230.0));

    // zoom 2, DPR 1: (8+260, 80+300) = (268, 380).
    let d2 = to_screen(vp, 2.0, 1.0, RootDevicePx(8.0, 80.0));
    assert_eq!(d2, RootDevicePx(268.0, 380.0));

    let back = screen_to_main_viewport(d2, 2.0, 1.0, RootDevicePx(8.0, 80.0));
    assert_eq!(back, MainViewportCss(130.0, 150.0));
    let doc_back = main_viewport_to_document(back, &chain);
    assert_eq!(doc_back, FrameDocumentCss(10.0, 10.0));
}

/// OOPIF frame: the nested offset from the browser target tree walks the
/// chain alongside the content-box origin.
///
/// oopif doc (50, 60): (50-5)+300+12 = 357 ; (60-5)+200+8 = 263.
#[test]
fn golden_oopif_chain() {
    let links = [
        main_frame((0.0, 0.0), 2.0),
        sub_frame("oopif", "main", (5.0, 5.0), (300.0, 200.0), (12.0, 8.0)),
    ];
    let chain = build_chain(&links, "oopif").expect("chain builds");
    let vp = document_to_main_viewport(FrameDocumentCss(50.0, 60.0), &chain);
    assert_eq!(vp, MainViewportCss(357.0, 263.0));

    // zoom 1, DPR 2, origin (10, 70): (10+714, 70+526) = (724, 596).
    let d = to_screen(vp, 1.0, 2.0, RootDevicePx(10.0, 70.0));
    assert_eq!(d, RootDevicePx(724.0, 596.0));

    // Round trip lands within one document px (whole-pixel rounding).
    let back = screen_to_main_viewport(d, 1.0, 2.0, RootDevicePx(10.0, 70.0));
    let doc_back = main_viewport_to_document(back, &chain);
    assert!((doc_back.0 - 50.0).abs() < 1.0);
    assert!((doc_back.1 - 60.0).abs() < 1.0);
}

/// Moved window: the same viewport point maps to a device point shifted
/// by exactly the window delta, and SEC-1 rejects the stale point
/// against the new rect (reject, never clamp).
#[test]
fn golden_moved_window() {
    let chain = [FrameTransform {
        scroll: (0.0, 0.0),
        content_origin: (0.0, 0.0),
        nested_offset: (0.0, 0.0),
        css_scale: 1.0,
        dpr: 1.0,
    }];
    let vp = document_to_main_viewport(FrameDocumentCss(100.0, 100.0), &chain);

    let origin_a = RootDevicePx(200.0, 150.0);
    let origin_b = RootDevicePx(400.0, 150.0); // window moved +200 x
    let dev_a = to_screen(vp, 1.0, 1.0, origin_a);
    let dev_b = to_screen(vp, 1.0, 1.0, origin_b);
    assert_eq!(dev_a, RootDevicePx(300.0, 250.0));
    assert_eq!(dev_b, RootDevicePx(500.0, 250.0));
    // Shift equals the window delta exactly.
    assert_eq!((dev_b.0 - dev_a.0, dev_b.1 - dev_a.1), (200.0, 0.0));

    let rect_new = WindowRect {
        x: 400,
        y: 150,
        w: 800,
        h: 600,
    };
    // The stale point (300, 250) is OUTSIDE the new rect: rejected.
    assert!(check_in_window(Some(&rect_new), 300, 250).is_err());
    // The fresh point (500, 250) is inside: accepted.
    assert!(check_in_window(Some(&rect_new), 500, 250).is_ok());
}

/// SEC-1 keeps the i16 range check: XTest takes i16, so a coordinate the
/// rect would accept but XTest cannot represent is still rejected.
#[test]
fn golden_i16_range_check() {
    let huge = WindowRect {
        x: 0,
        y: 0,
        w: 70_000,
        h: 70_000,
    };
    assert!(check_in_window(Some(&huge), 40_000, 100).is_err());
    assert!(check_in_window(Some(&huge), 100, 100).is_ok());
    // No rect at all: refuse to drive the raw desktop.
    assert!(check_in_window(None, 100, 100).is_err());
}

/// Toolbar offset: `to_screen` adds the X11 content origin exactly;
/// device minus origin equals viewport times zoom*DPR, nothing else.
#[test]
fn golden_toolbar_offset() {
    let vp = MainViewportCss(100.0, 50.0);
    let origin = RootDevicePx(8.0, 80.0);
    // viewport (0,0) lands exactly on the content origin.
    assert_eq!(
        to_screen(MainViewportCss(0.0, 0.0), 1.0, 1.0, origin),
        RootDevicePx(8.0, 80.0)
    );
    // zoom 1.5, DPR 2 (scale 3): (8+300, 80+150) = (308, 230).
    assert_eq!(to_screen(vp, 1.5, 2.0, origin), RootDevicePx(308.0, 230.0));
}

/// Insertion threading: a candidate's backend node id resolves to its
/// owning frame plus the ready-to-use chain, and an unknown node is a
/// named error (the caller falls back to a full-tree refresh).
#[test]
fn golden_attach_chain_at_insertion() {
    let state = FrameState {
        links: vec![
            main_frame((0.0, 0.0), 1.0),
            sub_frame("child", "main", (20.0, 30.0), (100.0, 120.0), (0.0, 0.0)),
        ],
        node_frames: [(42, "child".to_string())].into_iter().collect(),
        degraded_frames: Vec::new(),
        dpr: 1.0,
        target_tree_available: true,
        oopif_target_ids: Vec::new(),
    };
    let (frame_id, chain) = attach_chain(42, &state).expect("chain attaches");
    assert_eq!(frame_id, "child");
    assert_eq!(chain.len(), 2);
    // Frame origins are threaded through at insertion: the chain carries
    // the child's content-box origin and both scroll offsets.
    let vp = document_to_main_viewport(FrameDocumentCss(10.0, 10.0), &chain);
    // child: (10-20)+100 = 90 ; (10-30)+120 = 100 ; main: (90, 100).
    assert_eq!(vp, MainViewportCss(90.0, 100.0));

    assert_eq!(
        attach_chain(999, &state).unwrap_err(),
        FrameChainError::UnknownNode {
            backend_node_id: 999
        }
    );
}
