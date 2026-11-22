//! Views: cuts, wedges and section planes, levels and ghosts, the slice's labels and picks, the ruler, measures and checks.

use super::*;

/// Cuts and slices: a wedge added from the Section menu and turned by its
/// handle (one undo step), a radial slice stepped with PageUp, the
/// orthographic view.
#[test]
fn wedges_radial_slices_and_ortho() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let size = egui::vec2(1400.0, 800.0);
    let mut h = Harness::builder()
        .with_size(size)
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| {
            ViewerApp::new(cc, model("universes.i"), ViewState::new(), None).unwrap()
        });
    step_until(&mut h, "the model to load", |a| {
        a.document().is_some() || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());

    // A wedge from the 3D view's Section menu.
    cuts_menu(&mut h);
    h.get_by_label("+ wedge").click();
    h.run_steps(2);
    cuts_menu(&mut h);
    let doc = h.state().document().unwrap();
    assert_eq!(doc.scene.cuts.len(), 1);
    let CutState::Wedge { to, .. } = doc.scene.cuts[0] else {
        panic!("a wedge")
    };
    assert!(!doc.scene.cutaway().is_empty());

    // Its `to` side's handle turns it about the axis, whatever the tool.
    h.run_steps(2);
    let rect = h.get_by_label("3D view").rect();
    let doc = h.state().document().unwrap();
    let vp_cam = doc.orbit.camera(
        rect.width() as usize,
        rect.height() as usize,
        doc.scene.cutaway(),
    );
    let r = llevant_viewer::tools::handle_size(doc);
    let CutState::Wedge { point, .. } = doc.scene.cuts[0] else {
        unreachable!()
    };
    let base = [point[0], point[1], doc.orbit.state.target[2]];
    let (s, c) = to.to_radians().sin_cos();
    let handle = [base[0] + r * c, base[1] + r * s, base[2]];
    let project = |p: [f64; 3]| {
        let [f, rt, up] = vp_cam.basis();
        let v = [0, 1, 2].map(|k| p[k] - vp_cam.eye[k]);
        let z = llevant::vec3::dot(v, f);
        let half = z * (0.5 * vp_cam.fov_deg).to_radians().tan();
        let aspect = vp_cam.width as f64 / vp_cam.height as f64;
        let x = llevant::vec3::dot(v, rt) / (half * aspect);
        let y = llevant::vec3::dot(v, up) / half;
        rect.center()
            + egui::vec2(
                (x * 0.5 * rect.width() as f64) as f32,
                (-y * 0.5 * rect.height() as f64) as f32,
            )
    };
    let from = project(handle);
    let (s2, c2) = (to + 40.0).to_radians().sin_cos();
    let target = project([base[0] + r * c2, base[1] + r * s2, base[2]]);
    h.event(egui::Event::PointerMoved(from));
    h.step();
    h.event(egui::Event::PointerButton {
        pos: from,
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: egui::Modifiers::NONE,
    });
    h.step();
    for k in 1..=10 {
        let p = from + (target - from) * (k as f32 / 10.0);
        h.event(egui::Event::PointerMoved(p));
        h.step();
    }
    h.event(egui::Event::PointerButton {
        pos: target,
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers: egui::Modifiers::NONE,
    });
    h.run_steps(2);
    let doc = h.state().document().unwrap();
    let CutState::Wedge { to: turned, .. } = doc.scene.cuts[0] else {
        unreachable!()
    };
    let (s3, c3) = turned.to_radians().sin_cos();
    let now = project([base[0] + r * c3, base[1] + r * s3, base[2]]);
    // The handle followed the pointer (to the pixel it picks through).
    assert!(
        (now - target).length() < 4.0,
        "{to} turned to {turned}: {now:?} {target:?}"
    );
    assert!(turned > to + 20.0);
    // The whole drag is one step.
    shortcut(&mut h, egui::Key::Z, egui::Modifiers::COMMAND);
    let CutState::Wedge { to: back, .. } = h.state().document().unwrap().scene.cuts[0] else {
        unreachable!()
    };
    assert_eq!(back, to);

    // A radial slice, stepped a degree with PageUp over the slice.
    shortcut(&mut h, egui::Key::Escape, egui::Modifiers::NONE);
    // A radial slice, from the slice's header.
    h.get_by_label("radial").click();
    h.run_steps(2);
    let phi = h.state().document().unwrap().slice.radial.unwrap().phi;
    let slice = h.get_by_label("slice view").rect().center();
    h.event(egui::Event::PointerMoved(slice));
    h.step();
    shortcut(&mut h, egui::Key::PageUp, egui::Modifiers::NONE);
    let r = h.state().document().unwrap().slice.radial.unwrap();
    assert_eq!(r.phi, phi + 1.0);
    assert_eq!(h.state().view_state().slice.radial, Some(r));

    // Orthographic, and back.
    h.event(egui::Event::PointerMoved(rect.center()));
    h.step();
    shortcut(&mut h, egui::Key::O, egui::Modifiers::NONE);
    assert!(h.state().document().unwrap().orbit.state.ortho);
    // The view as a `llevant render` command has its cut and projection.
    let cmd = llevant_viewer::export::cli_command(h.state().document().unwrap(), 100, 80);
    assert!(
        cmd.contains(" --wedge 0,0,") && cmd.contains(" --ortho "),
        "{cmd}"
    );
    assert!(cmd.starts_with("llevant render \"") && cmd.ends_with("-o view.png"));
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        step_until(&mut h, "tracing", |a| !a.document().unwrap().view3d.busy());
        h.run_steps(3);
        save(&mut h, &PathBuf::from(path).with_extension("cuts.png"));
    }
    shortcut(&mut h, egui::Key::O, egui::Modifiers::NONE);
    assert!(!h.state().document().unwrap().orbit.state.ortho);
}

/// The Section tool on a plane: it shows as a surface that a click selects,
/// the tip of its arrow turns it in steps, and the slice plane has handles
/// handle (Properties then shows the section).
#[test]
fn section_planes_turn_to_axes_and_the_slice_has_a_handle() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let mut h = Harness::builder()
        .with_size(egui::vec2(1400.0, 800.0))
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| {
            ViewerApp::new(cc, model("universes.i"), ViewState::new(), None).unwrap()
        });
    step_until(&mut h, "the model to load", |a| {
        a.document().is_some() || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    show(&mut h, Views::Three);

    // A plane facing the eye, from the 3D view's Section menu.
    cuts_menu(&mut h);
    h.get_by_label("+ plane").click();
    h.run_steps(3);
    cuts_menu(&mut h);
    let doc = h.state().document().unwrap();
    assert_eq!(doc.scene.cuts.len(), 1);
    assert_eq!(doc.editing.cut_selected, Some(0), "a new cut is selected");
    let CutState::Plane { point, normal } = doc.scene.cuts[0] else {
        panic!("a plane")
    };
    let r = llevant_viewer::tools::handle_size(doc);
    assert!(normal.iter().all(|x| x.abs() < 0.999), "{normal:?}");

    // Its ring about z, dragged along itself: the normal turns about z, to
    // a direction of the 15° grid.
    drag_ring(&mut h, point, 2, r, 45.0);
    let doc = h.state().document().unwrap();
    let CutState::Plane { normal: turned, .. } = doc.scene.cuts[0] else {
        panic!("a plane")
    };
    // Turned in steps of 15° of azimuth and elevation.
    let on_step = |deg: f64| ((deg / 15.0).round() * 15.0 - deg).abs() < 1e-6;
    let (az, el) = (
        turned[1].atan2(turned[0]).to_degrees(),
        turned[2].asin().to_degrees(),
    );
    assert!(
        on_step(el) && (turned[2].abs() == 1.0 || on_step(az)),
        "{turned:?}"
    );
    assert!(turned != normal, "turned");
    // One step: undone, the plane is as it was.
    shortcut(&mut h, egui::Key::Z, egui::Modifiers::COMMAND);
    let CutState::Plane { normal: back, .. } = h.state().document().unwrap().scene.cuts[0] else {
        panic!("a plane")
    };
    assert_eq!(back, normal);

    // A click on empty ground deselects; one inside the square (off its
    // outline and dot) is a click on the cells; one on its centre dot
    // selects it again.
    let far = h.get_by_label("3D view").rect().left_top() + egui::vec2(12.0, 12.0);
    click(&mut h, far);
    assert_eq!(h.state().document().unwrap().editing.cut_selected, None);
    h.run_steps(40);
    let on_plane = project_3d(&mut h, [0, 1, 2].map(|k| point[k]));
    click(&mut h, on_plane + egui::vec2(30.0, 30.0));
    assert_eq!(h.state().document().unwrap().editing.cut_selected, None);
    click(&mut h, on_plane);
    assert_eq!(h.state().document().unwrap().editing.cut_selected, Some(0));

    // The slice plane has handles whatever the tool: the centre moves the
    // slice (and selects it: its rings show), a ring turns it, and each is
    // a step of the history. (Hidden in 3D at first; the Section menu's
    // checkbox shows it.)
    assert!(!h.state().document().unwrap().slice_shown_in_3d());
    cuts_menu(&mut h);
    h.get_by_label("the slice plane in 3D, with its handles")
        .click();
    h.run_steps(3);
    cuts_menu(&mut h);
    let doc = h.state().document().unwrap();
    assert!(doc.slice_in_3d);
    let before = doc.slice.plane_state();
    let r = llevant_viewer::tools::handle_size(doc);
    let mid = llevant_viewer::tools::slice_handle_base(doc);
    let n = llevant::vec3::normalize(doc.slice.frame().n);
    let at = |h: &mut Harness<'_, ViewerApp>, t: f64| {
        project_3d(h, [0, 1, 2].map(|k| mid[k] + t * r * n[k]))
    };
    let (centre, tip) = (at(&mut h, 0.0), at(&mut h, 1.0));
    drag_between(&mut h, centre, tip);
    let doc = h.state().document().unwrap();
    let moved = doc.slice.plane_state();
    assert!(
        (moved.at - before.at).abs() > 1e-3 * doc.orbit.radius,
        "{} -> {}",
        before.at,
        moved.at
    );
    assert_eq!(doc.scene.cuts.len(), 1, "no cut was made");
    shortcut(&mut h, egui::Key::Z, egui::Modifiers::COMMAND);
    assert_eq!(h.state().document().unwrap().slice.plane_state(), before);
    shortcut(
        &mut h,
        egui::Key::Z,
        egui::Modifiers::COMMAND | egui::Modifiers::SHIFT,
    );
    assert_eq!(h.state().document().unwrap().slice.plane_state(), moved);
    shortcut(&mut h, egui::Key::Z, egui::Modifiers::COMMAND);

    // A ring, dragged along itself, turns the slice (about x: it was along z).
    drag_ring(&mut h, mid, 0, r, 45.0);
    let turned = h.state().document().unwrap().slice.plane_state();
    assert_ne!(turned, before, "turned");
    shortcut(&mut h, egui::Key::Z, egui::Modifiers::COMMAND);
    assert_eq!(h.state().document().unwrap().slice.plane_state(), before);

    // A wedge and a box beside them, for a look (from the Section menu).
    cuts_menu(&mut h);
    for name in ["+ wedge", "+ section box"] {
        h.get_by_label(name).click();
        h.run_steps(2);
    }
    cuts_menu(&mut h);

    // A cut turned off (its row's checkbox): kept, cutting nothing, not
    // drawn; one step of the history.
    cuts_menu(&mut h);
    h.get_by_label("cut 1 on").click();
    h.run_steps(2);
    let doc = h.state().document().unwrap();
    assert_eq!(
        (doc.scene.cuts.len(), doc.scene.cuts_off.clone()),
        (3, vec![0])
    );
    assert_eq!(doc.scene.cuts_on().count(), 2);
    assert_eq!(
        llevant_viewer::panels::headers::section_label(doc),
        "Section · 2 of 3"
    );
    cuts_menu(&mut h);
    shortcut(&mut h, egui::Key::Z, egui::Modifiers::COMMAND);
    assert!(h.state().document().unwrap().scene.cuts_off.is_empty());

    // Cuts not shown in 3D: their dots are no longer picked (a click there is
    // the cells'), and they still cut.
    shortcut(&mut h, egui::Key::Escape, egui::Modifiers::NONE);
    cuts_menu(&mut h);
    h.get_by_label("show cuts in 3D").click();
    cuts_menu(&mut h);
    assert!(!h.state().document().unwrap().show_cuts);
    let on_plane = project_3d(&mut h, point);
    click(&mut h, on_plane);
    let doc = h.state().document().unwrap();
    assert_eq!(doc.editing.cut_selected, None);
    assert_eq!(doc.scene.cuts_on().count(), 3);
    cuts_menu(&mut h);
    h.get_by_label("show cuts in 3D").click();
    cuts_menu(&mut h);
    step_until(&mut h, "tracing", |a| !a.document().unwrap().view3d.busy());
    h.run_steps(3);
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        save(&mut h, &PathBuf::from(path).with_extension("section.png"));
    }
}

/// The level shown, from the toolbar's menu, reaches both views; ghosts
/// from a view file reach the 3D view (records in the G-buffer).
#[test]
fn levels_from_the_toolbar_and_ghosts_from_a_view_file() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let build = |state: ViewState| {
        Harness::builder()
            .with_size(egui::vec2(1200.0, 700.0))
            .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
            .build_eframe(move |cc| {
                ViewerApp::new(cc, model("universes.i"), state.clone(), None).unwrap()
            })
    };
    let mut h = build(ViewState::new());
    step_until(&mut h, "the model to load", |a| {
        a.document().is_some() || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    assert!(h.state().document().unwrap().levels >= 3);
    h.get_by_label(&icons::with(icons::LEVELS, "All levels"))
        .click();
    h.run_steps(2);
    h.get_by_label("Level 0").click();
    h.run_steps(2);
    settle(&mut h);
    let doc = h.state().document().unwrap();
    assert_eq!(doc.scene.hiding.level, Some(0));
    assert_eq!(doc.vis.level, Some(0));
    assert_eq!(doc.slice.level, Some(0));
    step_until(&mut h, "tracing", |a| {
        let d = a.document().unwrap();
        !d.view3d.busy() && d.slice.progress() >= 1.0
    });
    h.run_steps(3);
    let png = std::env::var_os("LLEVANT_VIEWER_TEST_PNG").map(PathBuf::from);
    if let Some(p) = &png {
        save(&mut h, &p.with_extension("level0.png"));
    }

    let mut state = ViewState::new();
    state.scene.hiding.ghost = vec!["c 10,11,12,13,14".parse().unwrap()];
    let mut h = build(state);
    step_until(&mut h, "the view to load", |a| {
        a.document().is_some() || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    assert!(h.state().document().unwrap().vis.has_ghosts());
    step_until(&mut h, "tracing", |a| !a.document().unwrap().view3d.busy());
    h.run_steps(3);
    if let Some(p) = &png {
        save(&mut h, &p.with_extension("ghosts.png"));
    }
}

#[test]
fn surfaces_picked_in_the_slice_and_selected() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let mut h = Harness::builder()
        .with_size(egui::vec2(1200.0, 700.0))
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| {
            ViewerApp::new(cc, model("universes.i"), ViewState::new(), None).unwrap()
        });
    step_until(&mut h, "the model to load", |a| {
        a.document().is_some() || a.error().is_some()
    });
    step_until(&mut h, "the input's cards", |a| {
        a.document().is_some_and(|d| d.source.is_some())
    });
    show(&mut h, Views::Slice);
    h.run_steps(3);

    // A point of the slice within reach of a boundary, along the middle row.
    let rect = h.get_by_label("slice view").rect();
    let doc = h.state().document().unwrap();
    let plane = doc.slice.plane();
    let reach = llevant_viewer::pick::SURFACE_REACH * plane.cm_per_point(rect);
    let (u, v) = (plane.frame.u, plane.frame.v);
    let (pos, want) = (0..rect.width() as usize)
        .step_by(3)
        .find_map(|i| {
            let pos = egui::pos2(rect.min.x + i as f32 + 0.5, rect.center().y);
            let p = plane.from_screen(rect, pos);
            let dirs = [
                u,
                v,
                [0, 1, 2].map(|k| u[k] + v[k]),
                [0, 1, 2].map(|k| u[k] - v[k]),
            ];
            let n = doc
                .model
                .nearest_surface(p, &dirs, reach, doc.vis.level)
                .unwrap()?;
            (n.distance > 0.0).then_some((pos, n))
        })
        .expect("a boundary in the middle row");
    let id = doc.model.surface_id(want.shape).unwrap();

    // Hovering names it in the banner; a double-click puts it in the
    // inspector.
    double_click(&mut h, pos);
    let doc = h.state().document().unwrap();
    let hover = doc.pointer.hover.as_ref().expect("hovered");
    assert_eq!(hover.surface.as_ref().map(|s| s.shape), Some(want.shape));
    assert!(hover.through.is_empty(), "details only for the focus");
    let focus = doc.focus.as_ref().expect("picked");
    let s = focus.surface.as_ref().expect("a surface near the click");
    assert_eq!(s.shape, want.shape);
    assert!(focus.through.contains(&want.shape));
    let users = focus.users.clone().expect("users of the surface");
    assert!(!users.is_empty());

    let button = icons::with(icons::SELECT, &format!("Select cells on s {}", id.id));
    panel(&mut h, Panel::Inspector);
    h.run_steps(2);
    // (Scrolling is animated: let it settle.)
    h.get_by_label(&button).scroll_to_me();
    h.run_steps(30);
    h.get_by_label(&button).click();
    h.run_steps(2);
    let doc = h.state().document().unwrap();
    assert_eq!(
        doc.selection.query,
        Some(llevant_scene::query::Query::surface(id.id)),
        "{:?}",
        last_said(doc)
    );
    assert_eq!(doc.selection.cells.iter().collect::<Vec<_>>(), users);
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        step_until(&mut h, "tracing", |a| {
            a.document().unwrap().slice.progress() >= 1.0
        });
        h.run_steps(3);
        save(&mut h, &PathBuf::from(path).with_extension("surface.png"));
    }
}

#[test]
fn cell_and_surface_numbers_on_the_slice() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let mut h = Harness::builder()
        .with_size(egui::vec2(1200.0, 700.0))
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| {
            ViewerApp::new(cc, model("universes.i"), ViewState::new(), None).unwrap()
        });
    step_until(&mut h, "the model to load", |a| {
        a.document().is_some() || a.error().is_some()
    });
    show(&mut h, Views::Slice);
    h.run_steps(3);
    // Off by default: nothing found, nothing drawn.
    step_until(&mut h, "tracing", |a| {
        a.document().unwrap().slice.progress() >= 1.0
    });
    h.run_steps(30);
    assert!(h
        .state()
        .document()
        .unwrap()
        .slice
        .labels_shown()
        .is_empty());

    // Zoomed in on the first filled cells, both kinds asked for.
    {
        let d = h.state_mut().document_mut().unwrap();
        d.slice.width = 120.0;
        d.slice.centre = [60.0, 0.0];
        d.slice.labels_wanted.cells = true;
        d.slice.labels_wanted.surfaces = true;
    }
    h.run_steps(30);
    step_until(&mut h, "the labels", |a| {
        let s = &a.document().unwrap().slice;
        s.progress() >= 1.0 && s.labels_settled()
    });
    let doc = h.state().document().unwrap();
    let shown = doc.slice.labels_shown();
    let m = &doc.model;
    let cells: Vec<&str> = shown
        .iter()
        .filter(|l| !l.surface)
        .map(|l| l.text.as_str())
        .collect();
    let surfaces: Vec<&str> = shown
        .iter()
        .filter(|l| l.surface)
        .map(|l| l.text.as_str())
        .collect();
    assert!(!cells.is_empty() && !surfaces.is_empty(), "{shown:?}");
    for c in &cells {
        assert!(m.cell_index(c.parse().unwrap()).is_some(), "{c}");
    }
    for s in &surfaces {
        let id: i64 = s.split('.').next().unwrap().parse().unwrap();
        assert!(
            (0..m.shape_count() as u32).any(|k| m.surface_id(k).unwrap().id == id),
            "{s}"
        );
    }
    // The saved view keeps the choice.
    let state = h.state().view_state();
    assert!(state.slice.cell_labels && state.slice.surface_labels);

    // Outlines between materials and between universes: within those
    // between cells (each boundary of either is one between cells), and
    // fewer here (cells of one material touch); kept in the saved view.
    let outlined = |h: &mut Harness<'_, ViewerApp>, by: OutlineBy| {
        h.state_mut().document_mut().unwrap().slice.outline_by = by;
        h.run_steps(3);
        step_until(h, "tracing", |a| {
            a.document().unwrap().slice.progress() >= 1.0
        });
        h.run_steps(3);
        let (_, _, rgb) = h.state().document().unwrap().slice.image_rgb().unwrap();
        let px: Vec<bool> = rgb
            .chunks(3)
            .map(|p| p == llevant_render::colours::OUTLINE)
            .collect();
        px
    };
    let cells = outlined(&mut h, OutlineBy::Cell);
    for by in [OutlineBy::Material, OutlineBy::Universe] {
        let other = outlined(&mut h, by);
        assert_eq!(other.len(), cells.len());
        assert!(
            other.iter().zip(&cells).all(|(o, c)| !o || *c),
            "{by:?}: a line where cells have none"
        );
        let (n, nc) = (
            other.iter().filter(|&&o| o).count(),
            cells.iter().filter(|&&c| c).count(),
        );
        assert!(0 < n && n < nc, "{by:?}: {n} of {nc}");
    }
    assert_eq!(h.state().view_state().slice.outline_by, OutlineBy::Universe);
    // Their colour, for a theme or cells they would not show on: the
    // image composed again with it, and saved with the view.
    let red = [255, 0, 0];
    h.state_mut().document_mut().unwrap().slice.outline_colour = Some(red);
    h.run_steps(3);
    let (_, _, rgb) = h.state().document().unwrap().slice.image_rgb().unwrap();
    let lines = rgb.chunks(3).filter(|p| *p == red).count();
    assert!(lines > 100, "{lines} red pixels");
    assert_eq!(h.state().view_state().slice.outline_colour, Some(red));
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        h.run_steps(3);
        save(&mut h, &PathBuf::from(path).with_extension("labels.png"));
    }
}

/// A surface picked with Alt+click in the 3D view: its page opened, marked
/// in both views, the cell selection left as it was; Esc lets go of it.
#[test]
fn surface_picked_with_alt() {
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let _one = serial();
    let mut h = Harness::builder()
        .with_size(egui::vec2(1400.0, 900.0))
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| {
            ViewerApp::new(cc, model("universes.i"), ViewState::new(), None).unwrap()
        });
    step_until(&mut h, "the model and its cards", |a| {
        a.document()
            .is_some_and(|d| d.model.is_complete() && d.source.is_some())
            && !a.is_loading()
    });
    settle(&mut h);
    let m = h.state().document().unwrap().model.clone();
    let at = screen_pos_of(&mut h, m.cell_index(202).unwrap());
    let alt = egui::Modifiers::ALT;
    let alt_click = |h: &mut Harness<'_, ViewerApp>| {
        h.event(egui::Event::PointerMoved(at));
        h.input_mut()
            .events
            .push(egui::Event::ModifiersChanged(alt));
        h.step();
        for pressed in [true, false] {
            h.input_mut().events.push(egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: alt,
            });
        }
        h.input_mut()
            .events
            .push(egui::Event::ModifiersChanged(alt));
        h.run_steps(3);
    };
    alt_click(&mut h);
    let d = h.state().document().unwrap();
    let pick = d.surface_pick.clone().expect("a surface picked");
    let id = m.surface_id(pick.shape().unwrap()).unwrap().id;
    assert!(
        d.selection.is_empty(),
        "the cells selected are left as they were"
    );
    h.get_by_label_contains(&format!("surface {id} card · line"));
    h.input_mut()
        .events
        .push(egui::Event::ModifiersChanged(egui::Modifiers::NONE));
    h.run_steps(3);
    let req = h.state().document().unwrap().marks.request().cloned();
    assert_eq!(
        req.map(|r| r.surfaces),
        Some(vec![id]),
        "marked while picked"
    );
    if let Some(png) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        h.run_steps(3);
        save(
            &mut h,
            &PathBuf::from(png).with_extension("surface-pick.png"),
        );
    }
    // The same place again: the next surface through it, if there is one.
    alt_click(&mut h);
    let again = h.state().document().unwrap().surface_pick.clone().unwrap();
    assert_eq!(again.at, (pick.at + 1) % pick.picked.through.len().max(1));
    h.input_mut()
        .events
        .push(egui::Event::ModifiersChanged(egui::Modifiers::NONE));
    h.key_press(egui::Key::Escape);
    h.run_steps(3);
    let d = h.state().document().unwrap();
    assert!(d.surface_pick.is_none());
    assert!(d.marks.request().is_none());
}

/// The Ruler tool: a line dragged in the slice lists the cells it crosses
/// (contiguous, the length of the line); through the layers from a face in
/// 3D, it stops at the next void.
#[test]
fn ruler_two_points_and_through_layers() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let mut h = Harness::builder()
        .with_size(egui::vec2(1400.0, 800.0))
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| {
            ViewerApp::new(cc, model("universes.i"), ViewState::new(), None).unwrap()
        });
    step_until(&mut h, "the model to load", |a| {
        a.document().is_some() || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    // M picks the tool.
    h.event(egui::Event::PointerMoved(
        h.get_by_label("3D view").rect().center(),
    ));
    shortcut(&mut h, egui::Key::M, egui::Modifiers::NONE);
    assert_eq!(h.state().tools().active().name(), "Ruler");

    let rect = h.get_by_label("slice view").rect();
    let (from, to) = (
        rect.left_center() + egui::vec2(20.0, 0.0),
        rect.right_center() - egui::vec2(20.0, 0.0),
    );
    drag_with(&mut h, egui::PointerButton::Primary, from, to);
    let doc = h.state().document().unwrap();
    let r = doc.ruler.clone().expect("a ruler");
    assert!(r.stretches.len() > 1, "{:?}", r.stretches);
    assert!((r.stretches.last().unwrap().to - r.length()).abs() < 1e-6);
    h.get_by_label(&icons::with(icons::COPY, "Copy (CSV)"));
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        h.run_steps(3);
        save(&mut h, &PathBuf::from(path).with_extension("ruler.png"));
    }

    // Through the layers, from a face in 3D: no void inside it.
    h.get_by_label("Through the layers").click();
    h.run_steps(2);
    let c202 = h.state().document().unwrap().model.cell_index(202).unwrap();
    let at = screen_pos_of(&mut h, c202);
    click(&mut h, at);
    let doc = h.state().document().unwrap();
    let r = doc.ruler.clone().expect("a ruler through the layers");
    assert_eq!(r.mode, llevant_viewer::ruler::RulerMode::Layers);
    let m = &doc.model;
    assert!(!r.stretches.is_empty());
    for s in &r.stretches {
        let c = s.cell.expect("inside cells");
        assert!(
            m.cell(c).fill.is_some() || m.cell(c).material != 0,
            "no void"
        );
    }
    h.get_by_label(&icons::with(icons::REMOVE, "Clear")).click();
    h.run_steps(2);
    assert!(h.state().document().unwrap().ruler.is_none());
}

/// Measure on a cell placed several times: its total and mass, and each
/// instance on its own (folded), the pieces adding up to the total; no
/// mean per instance.
#[test]
fn measure_lists_each_instance() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let mut h = Harness::builder()
        .with_size(egui::vec2(1400.0, 900.0))
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| {
            ViewerApp::new(cc, model("universes.i"), ViewState::new(), None).unwrap()
        });
    step_until(&mut h, "the model to load", |a| {
        a.document().is_some() || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    let m = h.state().document().unwrap().model.clone();
    // A cell with a material at a few places.
    let c = (0..m.cells().len() as u32)
        .find(|&c| {
            let n = m.instances().ids(c).count();
            m.cell(c).fill.is_none() && m.cell(c).material != 0 && (2..=8).contains(&n)
        })
        .expect("a material cell placed a few times");
    let id = m.cell_id(c);
    let n = m.instances().ids(c).count();
    h.state_mut()
        .document_mut()
        .unwrap()
        .execute(Command::Select(Query::cell(id), SelectMode::Replace));
    h.run_steps(3);
    let measure = icons::with(icons::TARGET, "Measure");
    h.get_by_label(&measure).scroll_to_me();
    h.run_steps(30);
    h.get_by_label(&measure).click();
    step_until(&mut h, "the volumes", |a| {
        a.document()
            .unwrap()
            .results
            .volumes
            .as_ref()
            .is_some_and(|v| !v.partial)
    });
    h.run_steps(2);
    let run = h
        .state()
        .document()
        .unwrap()
        .results
        .volumes
        .clone()
        .unwrap();
    assert!(run.spec.instances);
    assert_eq!(run.instances.len(), n);
    let pieces: f64 = run.instances.iter().map(|i| i.volume).sum();
    let whole = run.cells[0].volume;
    assert!((pieces - whole).abs() < 0.03 * whole, "{pieces} vs {whole}");
    assert!(h.query_by_label_contains("per instance").is_none());
    let list = format!("{n} instances");
    h.get_all_by_label(&list).last().unwrap().scroll_to_me();
    h.run_steps(30);
    h.get_all_by_label(&list).last().unwrap().click();
    h.run_steps(3);
    h.get_by_label_contains("each on its own");
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        save(&mut h, &PathBuf::from(path).with_extension("instances.png"));
    }
}

/// The analyses: volumes as the engine gives them, the check finding the
/// zoo's one real overlap (cell 2 of `periodic.i` runs into the
/// graveyard) and going to it,
/// universes alike.
#[test]
fn volumes_problems_and_alike() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let size = egui::vec2(1400.0, 800.0);
    let mut h = Harness::builder()
        .with_size(size)
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| {
            ViewerApp::new(cc, model("periodic.i"), ViewState::new(), None).unwrap()
        });
    step_until(&mut h, "the model to load", |a| {
        a.document().is_some() || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    let m = h.state().document().unwrap().model.clone();

    panel(&mut h, Panel::Volumes);
    h.run_steps(2);
    // A new finished run, not the last one (nor volumes so far).
    let compute = |h: &mut Harness<'_, ViewerApp>| {
        let last = h.state().document().unwrap().results.volumes.clone();
        h.get_by_label(&icons::with(icons::TARGET, "Compute volumes"))
            .click();
        step_until(h, "the volumes", |a| {
            let v = &a.document().unwrap().results.volumes;
            v.as_ref().is_some_and(|v| !v.partial)
                && !last
                    .as_ref()
                    .is_some_and(|l| std::sync::Arc::ptr_eq(l, v.as_ref().unwrap()))
        });
        h.state()
            .document()
            .unwrap()
            .results
            .volumes
            .clone()
            .unwrap()
    };
    let cells = |groups: &[llevant::Group]| -> Vec<(i64, f64)> {
        groups
            .iter()
            .filter(|g| g.status == llevant::Status::Inside)
            .map(|g| (g.number.unwrap(), g.volume.estimate))
            .collect()
    };
    let got = |run: &llevant_viewer::jobs::VolumeRun| -> Vec<(i64, f64)> {
        run.cells.iter().map(|c| (c.id, c.volume)).collect()
    };
    // One pass (the default), as `Model::volumes` gives it.
    let run = compute(&mut h);
    let region = h.state().document().unwrap().region;
    let grid = llevant::VolumeGrid::new(region, run.spec.h);
    let want = m
        .volumes(&grid, None)
        .unwrap()
        .grouped(&m, llevant::GroupBy::Cell);
    assert_eq!(got(&run), cells(&want));
    assert!(run.total() > 0.0);

    // To 1 %, as `Model::volumes_to` gives them.
    h.get_by_label("1 %").click();
    h.run_steps(2);
    let run = compute(&mut h);
    let target = llevant::VolumeTarget {
        rel_sigma: 1e-2,
        h_min: run.spec.h / 64.0,
        abs_sigma: 0.0,
        max_probes: llevant::DEFAULT_PROBES,
        by: llevant::GroupBy::Cell,
    };
    let want = m
        .volumes_to(
            &grid,
            &target,
            |g| g.status == llevant::Status::Inside,
            None,
            None,
        )
        .unwrap();
    assert_eq!(got(&run), cells(&want.groups));

    // The selected cells alone, as `Model::cell_volumes` gives them.
    h.state_mut()
        .document_mut()
        .unwrap()
        .execute(Command::Select("c 2".parse().unwrap(), SelectMode::Replace));
    // The Volumes and Problems panels each have one.
    for b in h.get_all_by_label("the selection") {
        b.click();
    }
    h.run_steps(2);
    let run = compute(&mut h);
    let c2 = m.cell_index(2).unwrap();
    let target =
        llevant::BinTarget::new(m.bounds(llevant::Scope::Problem), 1e-2, run.spec.h / 64.0);
    let want = m.cell_volumes(&[c2], &target, None).unwrap();
    assert_eq!(got(&run), [(2, want.groups[0].volume.estimate)]);
    assert!(run.problems.is_empty());

    // The check over the whole problem finds the overlap; "go" selects
    // its cells.
    panel(&mut h, Panel::Problems);
    h.run_steps(2);
    h.get_all_by_label("the problem").next().unwrap().click();
    h.run_steps(2);
    h.get_by_label("Check").click();
    step_until(&mut h, "the check", |a| {
        a.document().unwrap().results.check.is_some()
    });
    h.run_steps(2);
    let doc = h.state().document().unwrap();
    let check = doc.results.check.as_ref().unwrap();
    let problems = check.volumes.problems();
    assert!(!problems.is_empty(), "the overlap is found");
    let c2 = m.cell_index(2).unwrap();
    assert!(problems
        .iter()
        .any(|&i| check.volumes.labels[i].involved.contains(&c2)));
    h.get_all_by_label(icons::GO_TO).next().unwrap().click();
    h.run_steps(2);
    let doc = h.state().document().unwrap();
    assert!(!doc.selection.is_empty());

    // Statistics: universes alike.
    shortcut(&mut h, egui::Key::Escape, egui::Modifiers::NONE);
    panel(&mut h, Panel::Statistics);
    step_until_label(
        &mut h,
        "statistics to finish counting",
        "Find universes alike",
    );
    let find = h.get_by_label("Find universes alike");
    find.scroll_to_me();
    h.run_steps(30);
    h.get_by_label("Find universes alike").click();
    step_until(&mut h, "fingerprints", |a| {
        a.document().unwrap().results.alike.is_some()
    });
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        panel(&mut h, Panel::Problems);
        h.run_steps(3);
        save(&mut h, &PathBuf::from(path).with_extension("analyses.png"));
    }
}
