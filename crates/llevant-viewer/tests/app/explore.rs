//! Exploring: picking, pins, selection, hiding and isolating, universes opened on their own, the palette, the start screen.

use super::*;

#[test]
fn pins_selection_and_undo() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let size = egui::vec2(1200.0, 700.0);
    // The app's own device, not kittest's (a CPU adapter first: WARP on
    // Windows, whose D3D12 compiler fails on the renderer's loops).
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

    // Pin here (the right-click menu), in the slice alone: anywhere in the
    // plane pins that point.
    show(&mut h, Views::Slice);
    h.run_steps(3);
    let centre = h.get_by_label("slice view").rect().center();
    pin_at(&mut h, centre);
    let doc = h.state().document().unwrap();
    assert_eq!(doc.scene.pins.len(), 1, "{:?}", last_said(doc));
    assert_eq!(doc.pin_info.len(), 1);
    let pinned = doc.scene.pins[0].point;
    // The inspector shows the pin's point.
    assert_eq!(doc.focus.as_ref().unwrap().point, pinned);
    pin_at(&mut h, centre + egui::vec2(40.0, 30.0));
    assert_eq!(h.state().document().unwrap().scene.pins.len(), 2);

    // Undo the second pin, redo it, undo again.
    let ctrl = egui::Modifiers::COMMAND;
    shortcut(&mut h, egui::Key::Z, ctrl);
    assert_eq!(h.state().document().unwrap().scene.pins.len(), 1);
    shortcut(&mut h, egui::Key::Z, ctrl | egui::Modifiers::SHIFT);
    assert_eq!(h.state().document().unwrap().scene.pins.len(), 2);
    shortcut(&mut h, egui::Key::Z, ctrl);
    assert_eq!(h.state().document().unwrap().scene.pins[0].point, pinned);

    // Esc clears the point. In 3D alone, a click on the model picks the
    // first visible cell there, as the CPU engine finds it.
    shortcut(&mut h, egui::Key::Escape, egui::Modifiers::NONE);
    show(&mut h, Views::Three);
    h.run_steps(3);
    let rect = h.get_by_label("3D view").rect();
    let doc = h.state().document().unwrap();
    let cam = doc.orbit.camera(
        rect.width() as usize,
        rect.height() as usize,
        doc.scene.cutaway(),
    );
    let (pixel, want) = (0..cam.height)
        .step_by(7)
        .flat_map(|j| (0..cam.width).step_by(7).map(move |i| (i, j)))
        .find_map(|(i, j)| pick_ray(&doc.model, &doc.vis, &cam, i, j).map(|p| ((i, j), p)))
        .expect("the model is in view");
    // A click selects, and marks no point.
    let at = rect.min + egui::vec2(pixel.0 as f32 + 0.5, pixel.1 as f32 + 0.5);
    assert!(doc.focus.is_none(), "Esc cleared the point the pin marked");
    click(&mut h, at);
    let doc = h.state().document().unwrap();
    assert!(doc.focus.is_none(), "a click marks no point");
    assert_eq!(doc.selection.cells.count(), 1);
    // A double-click marks it and the inspector shows it.
    double_click(&mut h, at);
    let doc = h.state().document().unwrap();
    assert_eq!(doc.scene.pins.len(), 1, "Select does not pin");
    let sel = doc.focus.as_ref().expect("picked");
    assert_eq!(sel.point, want.point);
    assert_eq!(sel.label, want.label);
    assert!(sel.t.is_some(), "a 3D pick");
    // Its cell is selected.
    let cell = want.cell().expect("inside a cell");
    assert!(doc.selection.cells.contains(cell));
    assert_eq!(doc.selection.cells.count(), 1);
    // Esc clears both.
    shortcut(&mut h, egui::Key::Escape, egui::Modifiers::NONE);
    let doc = h.state().document().unwrap();
    assert!(doc.focus.is_none() && doc.selection.is_empty());

    // A toolbar edit (the colour menu), undone from the toolbar.
    h.get_by_label_contains("Colour").click();
    h.run_steps(2);
    h.get_by_label("material").click();
    h.run_steps(2);
    assert_eq!(
        h.state().document().unwrap().scene.colour_by,
        Colouring::Material
    );
    h.get_by_label(&icons::with(icons::UNDO, "Undo")).click();
    h.run_steps(2);
    assert_eq!(
        h.state().document().unwrap().scene.colour_by,
        Colouring::Cell
    );

    // The saved view has the pin.
    let state = h.state().view_state();
    assert_eq!(state.scene.pins.len(), 1);

    // Both views, traced to the end, for a look.
    show(&mut h, Views::Both);
    h.run_steps(3);
    step_until(&mut h, "tracing", |a| {
        let d = a.document().unwrap();
        !d.view3d.busy() && d.slice.progress() >= 1.0
    });
    h.run_steps(3);
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        let path = PathBuf::from(path);
        save(&mut h, &path);
        // And in the light theme: the theme button (dark) turns to it.
        h.get_by_label(icons::DARK).click();
        h.run_steps(4);
        save(&mut h, &path.with_extension("light.png"));
    }
}

/// Pick a cell inside a filled cell, open its universe on its own from the
/// inspector, go back; then open the same view from a saved state.
#[test]
fn open_a_universe_and_go_back() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let size = egui::vec2(1200.0, 700.0);
    let build = |state: ViewState| {
        Harness::builder()
            .with_size(size)
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
    show(&mut h, Views::Three);
    h.run_steps(3);
    let rect = h.get_by_label("3D view").rect();
    let doc = h.state().document().unwrap();
    let cam = doc
        .orbit
        .camera(rect.width() as usize, rect.height() as usize, None);
    let (pixel, picked) = (0..cam.height)
        .step_by(5)
        .flat_map(|j| (0..cam.width).step_by(5).map(move |i| (i, j)))
        .find_map(|(i, j)| {
            pick_ray(&doc.model, &doc.vis, &cam, i, j)
                .filter(|p| p.label.path.len() >= 2)
                .map(|p| ((i, j), p))
        })
        .expect("a cell inside a filled cell in view");
    let m = &doc.model;
    let u = m.units()[m.cell(picked.label.path[1].cell).unit as usize].universe;
    let whole = m.cells().len();
    double_click(
        &mut h,
        rect.min + egui::vec2(pixel.0 as f32 + 0.5, pixel.1 as f32 + 0.5),
    );
    let open = icons::with(icons::OPEN, &format!("Open u={u}"));
    // (Scrolling is animated: let it settle.)
    h.get_by_label(&open).scroll_to_me();
    h.run_steps(30);
    h.get_by_label(&open).click();
    step_until(&mut h, "the universe to open", |a| {
        a.document().is_some_and(|d| d.universe == Some(u))
    });
    h.run_steps(2);
    let doc = h.state().document().unwrap();
    assert!(doc.model.cells().len() < whole);
    assert_eq!(doc.model.units()[0].universe, u);
    assert!(doc.model.cut_cell().is_some());
    assert_eq!(h.state().open_universes(), vec![u]);
    assert_eq!(h.state().view_state().open, vec![u]);
    h.get_by_label(&format!("model › u={u}"));
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        step_until(&mut h, "tracing", |a| !a.document().unwrap().view3d.busy());
        h.run_steps(3);
        save(&mut h, &PathBuf::from(path).with_extension("universe.png"));
    }
    h.get_by_label("⬅ Back").click();
    h.run_steps(2);
    let doc = h.state().document().unwrap();
    assert_eq!(doc.universe, None);
    assert_eq!(doc.model.cells().len(), whole);
    assert!(h.state().open_universes().is_empty());

    // A saved view of the universe opens it again, with the model behind.
    let state = ViewState {
        open: vec![u],
        ..ViewState::new()
    };
    let mut h = build(state);
    step_until(&mut h, "the saved view to load", |a| {
        a.document().is_some() || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    assert_eq!(h.state().document().unwrap().universe, Some(u));
    h.get_by_label("⬅ Back").click();
    h.run_steps(2);
    assert_eq!(h.state().document().unwrap().universe, None);
}

/// Select a cell by clicking it, add its universe from the inspector's
/// link, hide the selection, isolate, show all, undo; the context menu of
/// a right-click names what is there.
#[test]
fn select_hide_isolate_and_menus() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let size = egui::vec2(1200.0, 700.0);
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
    show(&mut h, Views::Three);
    h.run_steps(3);
    // A first click opens the inspector, which narrows the 3D view: the
    // pixels to click are found after.
    let centre = h.get_by_label("3D view").rect().center();
    click(&mut h, centre);
    h.run_steps(3);
    let rect = h.get_by_label("3D view").rect();
    let doc = h.state().document().unwrap();
    assert!(!doc.selection.is_empty(), "the first click is on the model");
    let cam = doc
        .orbit
        .camera(rect.width() as usize, rect.height() as usize, None);
    let (pixel, picked) = (0..cam.height)
        .step_by(5)
        .flat_map(|j| (0..cam.width).step_by(5).map(move |i| (i, j)))
        .find_map(|(i, j)| {
            pick_ray(&doc.model, &doc.vis, &cam, i, j)
                .filter(|p| p.label.path.len() >= 2)
                .map(|p| ((i, j), p))
        })
        .expect("a cell inside a filled cell in view");
    let m = doc.model.clone();
    let cell = picked.cell().unwrap();
    let id = m.cell_id(cell);
    let u = m.units()[m.cell(cell).unit as usize].universe;
    let at = rect.min + egui::vec2(pixel.0 as f32 + 0.5, pixel.1 as f32 + 0.5);
    // A double-click: selected, and the point (with its path) inspected.
    double_click(&mut h, at);
    let doc = h.state().document().unwrap();
    assert_eq!(
        doc.selection.query,
        Some(format!("c {id}").parse().unwrap())
    );
    // The status bar names what is under the pointer.
    assert_eq!(
        doc.pointer.hover.as_ref().and_then(|p| p.cell()),
        Some(cell)
    );

    // The inspector's link to the universe selects all its cells.
    h.get_all_by_label(&format!("u={u}"))
        .next()
        .unwrap()
        .click();
    h.run_steps(2);
    let doc = h.state().document().unwrap();
    let unit = &m.units()[m.units().iter().position(|x| x.universe == u).unwrap()];
    assert_eq!(doc.selection.cells.count(), unit.cells.len());
    // Shift+Alt+← steps the selection back (Alt+← is the camera's).
    shortcut(
        &mut h,
        egui::Key::ArrowLeft,
        egui::Modifiers::ALT | egui::Modifiers::SHIFT,
    );
    assert_eq!(h.state().document().unwrap().selection.cells.count(), 1);

    // H hides the selection (and clears it); Alt+H shows all again.
    shortcut(&mut h, egui::Key::H, egui::Modifiers::NONE);
    let doc = h.state().document().unwrap();
    assert_eq!(doc.scene.hiding.cells, vec![id]);
    assert!(!doc.vis.cells[cell as usize]);
    assert!(doc.selection.is_empty());
    shortcut(&mut h, egui::Key::H, egui::Modifiers::ALT);
    assert!(h.state().document().unwrap().vis.cells[cell as usize]);

    // Shift+H isolates: the cell and the cells holding it (wherever its
    // universe is) are shown, nothing else.
    let at = screen_pos_of(&mut h, cell);
    click(&mut h, at);
    shortcut(&mut h, egui::Key::H, egui::Modifiers::SHIFT);
    let doc = h.state().document().unwrap();
    for s in &picked.label.path {
        assert!(doc.vis.cells[s.cell as usize], "cell {}", m.cell_id(s.cell));
    }
    for c in 0..m.cells().len() as u32 {
        if doc.vis.cells[c as usize] {
            assert!(
                c == cell || m.cell(c).fill.is_some(),
                "cell {}",
                m.cell_id(c)
            );
        }
    }
    shortcut(&mut h, egui::Key::Z, egui::Modifiers::COMMAND);
    assert!(h.state().document().unwrap().scene.hiding.isolate.is_none());

    // A right-click opens the menu of what is there.
    h.run_steps(40);
    let at = screen_pos_of(&mut h, cell);
    h.event(egui::Event::PointerMoved(at));
    for pressed in [true, false] {
        h.event(egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Secondary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        });
    }
    h.run_steps(3);
    // (Submenus end with an arrow, items with their keys.)
    h.get_by_label_contains(&icons::with(icons::CELL, &format!("Cell {id}")));
    h.get_by_label_contains(&icons::with(icons::UNIVERSE, &format!("Universe {u}")));
    h.get_by_label_contains(&icons::with(icons::FRAME, "Frame selection"))
        .click();
    h.run_steps(2);

    // The axes snap the view: from above.
    h.get_by_label("View from Z").click();
    h.run_steps(2);
    let s = h.state().document().unwrap().orbit.state;
    assert_eq!((s.yaw_deg, s.pitch_deg), (270.0, 89.0));
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        step_until(&mut h, "tracing", |a| !a.document().unwrap().view3d.busy());
        h.run_steps(3);
        save(&mut h, &PathBuf::from(path).with_extension("selection.png"));
    }
}

/// Isolating closes the eyes of what it hides, in Materials and in the
/// tree; a click on one shows it too (the isolated rule widened), and the
/// line above the rows stops isolating.
#[test]
fn eyes_follow_isolating() {
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
    let m = h.state().document().unwrap().model.clone();
    let c400 = m.cell_index(400).unwrap();
    assert_eq!(m.cell(c400).material, 1);
    panel(&mut h, Panel::Materials);
    // Void is hidden; every material's cells are shown (m3, which no cell
    // is made of, has no eye).
    assert_eq!(h.get_all_by_label(icons::HIDE).count(), 1);
    assert_eq!(h.get_all_by_label(icons::SHOW).count(), 3);
    h.state_mut()
        .document_mut()
        .unwrap()
        .execute(Edit::Isolate(Some(Query::cell(400))).into());
    h.run_steps(1);
    settle(&mut h);
    h.run_steps(2);
    // m1 partly shown (cell 400 only), void and m2 hidden.
    let doc = h.state().document().unwrap();
    assert!(doc.seen.is_seen(c400));
    let m1 = doc.seen.material(1);
    assert!(m1.seen == 1 && m1.cells > 1, "{m1:?}");
    assert_eq!(h.get_all_by_label(icons::SHOW).count(), 2);
    h.get_by_label("Isolating");
    let shut: Vec<_> = h.get_all_by_label(icons::HIDE).collect();
    assert_eq!(shut.len(), 2);
    // Rows: void, m1, m2: the second shut eye is m2's.
    shut[1].click();
    h.run_steps(1);
    settle(&mut h);
    h.run_steps(2);
    let doc = h.state().document().unwrap();
    assert_eq!(
        doc.scene.hiding.isolate,
        Some(Query::cell(400).plus(Query::material(2)))
    );
    let m2 = doc.seen.material(2);
    assert_eq!(m2.seen, m2.cells);
    assert_eq!(h.get_all_by_label(icons::HIDE).count(), 1);
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        save(
            &mut h,
            &PathBuf::from(path).with_extension("eyes-isolating.png"),
        );
    }

    // The tree: cell 402 (m1) is isolated away, its eye shut; a click on
    // it shows it too.
    panel(&mut h, Panel::Envelopes);
    h.run_steps(2);
    let c402 = m.cell_index(402).unwrap();
    assert!(!h.state().document().unwrap().seen.is_seen(c402));
    let before = h.get_all_by_label(icons::HIDE).count();
    let row = h.get_by_label("cell 402").rect().center().y;
    h.get_all_by_label(icons::HIDE)
        .find(|n| (n.rect().center().y - row).abs() < 4.0)
        .expect("a shut eye on the row of cell 402")
        .click();
    h.run_steps(1);
    settle(&mut h);
    h.run_steps(2);
    let doc = h.state().document().unwrap();
    assert!(doc.seen.is_seen(c402), "{:?}", doc.scene.hiding.isolate);
    assert_eq!(h.get_all_by_label(icons::HIDE).count(), before - 1);

    // Stop: everything shown again.
    h.get_by_label(&icons::with(icons::REMOVE, "Stop")).click();
    h.run_steps(1);
    settle(&mut h);
    h.run_steps(2);
    let doc = h.state().document().unwrap();
    assert!(doc.scene.hiding.isolate.is_none());
    h.run_steps(2);
    assert!(h.query_by_label("Isolating").is_none());
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        h.run_steps(3);
        save(&mut h, &PathBuf::from(path).with_extension("eyes.png"));
    }
}

/// The panels: Find (envelopes and their eyes, a rule's matches, a rule
/// saved), the command palette, the materials' eyes, saved views, the
/// input's cards, the inspector's pages.
#[test]
fn find_palette_materials_and_views() {
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
        a.document().is_some_and(|d| d.source.is_some()) || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    let m = h.state().document().unwrap().model.clone();

    // Envelopes lists level 0; a link selects, an eye hides.
    panel(&mut h, Panel::Envelopes);
    h.run_steps(2);
    h.get_by_label("cell 400").click();
    h.run_steps(2);
    let c400 = m.cell_index(400).unwrap();
    let doc = h.state().document().unwrap();
    assert_eq!(doc.selection.query, Some("c 400".parse().unwrap()));
    // The inspector has no point yet: a click on the model gives one.
    assert!(doc.focus.is_none());
    let eyes: Vec<_> = h.get_all_by_label(icons::SHOW).collect();
    assert!(!eyes.is_empty());
    // Find takes a rule.
    panel(&mut h, Panel::Find);
    h.run_steps(2);
    type_into(&mut h, "find rule", "m 2");
    h.run_steps(2);
    let n = m
        .cells()
        .iter()
        .filter(|c| c.material == 2 && c.fill.is_none())
        .count();
    h.get_by_label(&format!("{n} cells"));
    h.get_by_label("Select all").click();
    h.run_steps(2);
    assert_eq!(h.state().document().unwrap().selection.cells.count(), n);
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        h.run_steps(3);
        save(&mut h, &PathBuf::from(path).with_extension("find.png"));
    }
    // Saved by name with the view; with the box empty, listed under Saved.
    h.get_by_label("☆ Save").click();
    h.run_steps(2);
    h.get_by_label("Save").click();
    h.run_steps(2);
    let rules = h.state().document().unwrap().scene.rules.clone();
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].rule, "m 2");
    h.get_by_label("find rule").click();
    h.run_steps(1);
    shortcut(&mut h, egui::Key::A, egui::Modifiers::COMMAND);
    h.key_press(egui::Key::Backspace);
    h.run_steps(2);
    h.get_by_label(&icons::with("☆", "m 2"));

    // The palette: a rule, Enter selects it.
    shortcut(&mut h, egui::Key::K, egui::Modifiers::COMMAND);
    type_into(&mut h, "palette text", "u 1");
    h.run_steps(2);
    shortcut(&mut h, egui::Key::Enter, egui::Modifiers::NONE);
    let doc = h.state().document().unwrap();
    assert_eq!(doc.selection.query, Some("u 1".parse().unwrap()));
    // And a slice.
    shortcut(&mut h, egui::Key::K, egui::Modifiers::COMMAND);
    type_into(&mut h, "palette text", "slice x 3.5");
    h.run_steps(2);
    shortcut(&mut h, egui::Key::Enter, egui::Modifiers::NONE);
    let s = h.state().document().unwrap().slice.state();
    assert_eq!(
        (s.axis, s.at),
        (llevant_viewer::state::AxisName::X, Some(3.5))
    );

    // Materials, a panel of Explore: the eye of m1 hides its cells in 3D.
    panel(&mut h, Panel::Materials);
    h.run_steps(2);
    let shown_before = h.state().document().unwrap().vis.cells[c400 as usize];
    assert!(shown_before && m.cell(c400).material == 1);
    let rows: Vec<_> = h.get_all_by_label(icons::SHOW).collect();
    // void is hidden (its eye is shut): the first open eye is m1's.
    rows[0].click();
    h.run_steps(2);
    let doc = h.state().document().unwrap();
    assert_eq!(doc.scene.hiding.materials, vec![1]);
    assert!(!doc.vis.cells[c400 as usize]);

    // Saved views: save, change the colouring, go back.
    panel(&mut h, Panel::Places);
    h.run_steps(2);
    h.get_by_label(&icons::with(icons::SAVE, "Save this view"))
        .click();
    h.run_steps(2);
    assert_eq!(h.state().document().unwrap().views.len(), 1);
    shortcut(&mut h, egui::Key::Z, egui::Modifiers::COMMAND);
    assert!(h
        .state()
        .document()
        .unwrap()
        .scene
        .hiding
        .materials
        .is_empty());
    h.get_by_label(icons::GO_TO).click();
    h.run_steps(2);
    assert_eq!(
        h.state().document().unwrap().scene.hiding.materials,
        vec![1]
    );
    assert_eq!(h.state().view_state().views.len(), 1);

    // A double-click on the model: the inspector shows the cell's card.
    let at = screen_pos_of(&mut h, m.cell_index(202).unwrap());
    double_click(&mut h, at);
    panel(&mut h, Panel::Inspector);
    h.run_steps(2);
    let line = h
        .state()
        .document()
        .unwrap()
        .source
        .as_ref()
        .unwrap()
        .cell(202)
        .unwrap()
        .line;
    h.get_by_label_contains(&format!("cell 202 card · line {line}"));
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        h.run_steps(3);
        save(&mut h, &PathBuf::from(path).with_extension("panels.png"));
    }

    // Its material's page, from the cell's (›), and back.
    let mat = h
        .state()
        .document()
        .unwrap()
        .model
        .cell(m.cell_index(202).unwrap())
        .material;
    h.get_by_label(&format!("Inspect m {mat}")).click();
    h.run_steps(2);
    h.get_by_label_contains(&format!("material {mat} card · line"));
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        h.run_steps(3);
        save(&mut h, &PathBuf::from(path).with_extension("material.png"));
    }
    h.get_by_label("Inspector back").click();
    h.run_steps(2);
    assert!(
        h.query_by_label_contains(&format!("material {mat} card · line"))
            .is_none(),
        "back to the cell"
    );
    h.get_by_label_contains("cell 202 card · line");
}

/// Type `text` into the field named `name`.
/// A selection of many cells, listed under Find's Recent as one long
/// `c …`, leaves the left side as wide as it was, and its divider still
/// drags it narrower.
#[test]
fn a_long_selection_keeps_the_left_side_narrow() {
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
        a.document().is_some_and(|d| d.source.is_some()) || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    h.run_steps(3);
    let width = |h: &Harness<'_, ViewerApp>| h.state().frame().left.width;
    let before = width(&h);

    // Every cell, by number, many times over: far wider than the side.
    let ids: Vec<String> = h
        .state()
        .document()
        .unwrap()
        .model
        .cells()
        .iter()
        .map(|c| c.id.to_string())
        .collect();
    let rule = format!("c {}", vec![ids.join(" "); 20].join(" "));
    assert!(rule.len() > 400);
    h.state_mut()
        .document_mut()
        .unwrap()
        .execute(Command::Select(rule.parse().unwrap(), SelectMode::Replace));
    h.run_steps(4);
    assert!(
        h.query_all_by_label_contains(&rule).next().is_some(),
        "the selection is listed under Recent"
    );
    let after = width(&h);
    assert!(
        (after - before).abs() < 1.0,
        "the left side went from {before} to {after}"
    );
    // The inspector folds the rule to a few lines; "show all"
    // unfolds it.
    let rule_rect = |h: &Harness<'_, ViewerApp>| {
        h.query_all_by_label_contains("c 10 11 12")
            .map(|n| n.rect())
            .find(|r| r.min.x > h.get_by_label("3D view").rect().max.x)
            .expect("the rule in the inspector")
    };
    let folded = rule_rect(&h);
    assert!(folded.height() < 80.0, "folded rule {folded:?}");
    h.get_by_label("show all").click();
    h.run_steps(3);
    assert!(rule_rect(&h).height() > 2.0 * folded.height());
    h.get_by_label("show less").click();
    h.run_steps(3);
    let again = rule_rect(&h);
    assert!(
        (again.height() - folded.height()).abs() < 1.0,
        "{folded:?} then {again:?}"
    );

    // Isolated, the banner's Stop stays within the side and stops it.
    h.state_mut()
        .document_mut()
        .unwrap()
        .execute(Edit::Isolate(Some(rule.parse().unwrap())).into());
    h.run_steps(4);
    let stop = h.get_by_label(&icons::with(icons::REMOVE, "Stop")).rect();
    let views = ["3D view", "slice view"].map(|v| h.get_by_label(v).rect().min.x);
    assert!(
        stop.min.x >= 0.0 && stop.max.x < views[0].min(views[1]),
        "Stop at {stop:?}, the views from x = {views:?}"
    );
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        save(&mut h, &PathBuf::from(path).with_extension("isolating.png"));
    }
    h.get_by_label(&icons::with(icons::REMOVE, "Stop")).click();
    h.run_steps(2);
    assert!(h.state().document().unwrap().scene.hiding.isolate.is_none());
    assert!((width(&h) - before).abs() < 1.0);

    // Its divider drags it narrower.
    // (The divider is the views' left edge, wherever the panel starts.)
    let views = ["3D view", "slice view"].map(|v| h.get_by_label(v).rect());
    let x = views[0].min.x.min(views[1].min.x) - 1.0;
    let y = views[0].center().y;
    drag_between(&mut h, egui::pos2(x, y), egui::pos2(x - 60.0, y));
    h.run_steps(3);
    let dragged = width(&h);
    assert!(dragged < after - 40.0, "dragged from {after} to {dragged}");
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        save(
            &mut h,
            &PathBuf::from(path).with_extension("long-selection.png"),
        );
    }
}

/// CAD-style mouse: a left drag box-selects the cells seen in the box, a
/// middle drag orbits, and Alt+← goes back to the place looked at before.
#[test]
fn box_select_middle_orbit_and_camera_history() {
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
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    show(&mut h, Views::Three);
    let rect = h.get_by_label("3D view").rect();

    // A box over the whole view selects what is seen: several cells.
    let before = h.state().document().unwrap().orbit.state;
    drag_with(
        &mut h,
        egui::PointerButton::Primary,
        rect.min + egui::vec2(5.0, 40.0),
        rect.max - egui::vec2(5.0, 5.0),
    );
    let doc = h.state().document().unwrap();
    assert!(
        doc.selection.cells.count() > 3,
        "{}",
        doc.selection.cells.count()
    );
    let rested = doc.orbit.state;
    assert_eq!(rested, before, "a left drag does not move the camera");

    // A middle drag orbits; once it has rested, Alt+← goes back.
    h.run_steps(3);
    drag_with(
        &mut h,
        egui::PointerButton::Middle,
        rect.center(),
        rect.center() + egui::vec2(120.0, 0.0),
    );
    let turned = h.state().document().unwrap().orbit.state;
    assert_ne!(turned.yaw_deg, rested.yaw_deg, "orbited");
    h.run_steps(3);
    shortcut(&mut h, egui::Key::ArrowLeft, egui::Modifiers::ALT);
    assert_eq!(h.state().document().unwrap().orbit.state, rested);
    shortcut(&mut h, egui::Key::ArrowRight, egui::Modifiers::ALT);
    assert_eq!(h.state().document().unwrap().orbit.state, turned);
}

/// The keys window (F1) fits a short window and scrolls to its last row.
#[test]
fn keys_window_scrolls_in_a_short_window() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let size = egui::vec2(1000.0, 480.0);
    let mut h = Harness::builder()
        .with_size(size)
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| {
            ViewerApp::new(cc, model("universes.i"), ViewState::new(), None).unwrap()
        });
    step_until(&mut h, "the model to load", |a| {
        a.document().is_some() || a.error().is_some()
    });
    shortcut(&mut h, egui::Key::F1, egui::Modifiers::NONE);
    h.run_steps(2);
    let first = h.get_by_label("Everywhere").rect();
    let last = h.get_by_label("this window").rect();
    assert!(first.min.y >= 0.0, "{first:?}");
    // The last row is below the screen until the list is scrolled.
    assert!(last.max.y > size.y, "{last:?}");
    let over = egui::pos2(first.center().x, 0.5 * size.y);
    h.event(egui::Event::PointerMoved(over));
    for _ in 0..40 {
        h.event(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, -200.0),
            modifiers: egui::Modifiers::NONE,
            phase: egui::TouchPhase::Move,
        });
        h.step();
    }
    h.run_steps(4);
    let last = h.get_by_label("this window").rect();
    assert!(last.max.y <= size.y, "{last:?}");
    if let Some(path) = std::env::var_os("LLEVANT_VIEWER_TEST_PNG") {
        save(&mut h, &PathBuf::from(path).with_extension("keys.png"));
    }
}

/// Rules by label: a view file isolating by label is shown once the labels
/// are read (they come with the cards, after the model may be open), and
/// the selection takes them.
#[test]
fn labels_select_and_isolate() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("viewer_labels.i");
    std::fs::write(
        &path,
        "labels
1 0 -1 fill=1 imp:n=1 $ envelope A
2 0 -2 fill=1 (20 0 0) imp:n=1 $ envelope B
8 0 1 2 -9 imp:n=1
9 0 9 imp:n=0
10 3 -7.8 -4 u=1 imp:n=1 $ NB CELL - Steel ball
11 5 -1.0 4 u=1 imp:n=1 $ NB CELL - water around

1 so 5
2 s 20 0 0 5
4 so 2
9 so 100

m3 26000 1
m5 1001 2 8016 1
",
    )
    .unwrap();
    let mut state = ViewState::new();
    state.scene.hiding.isolate = Some("$ steel ball".parse().unwrap());
    let mut h = Harness::builder()
        .with_size(egui::vec2(900.0, 600.0))
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| ViewerApp::new(cc, path.clone(), state.clone(), None).unwrap());
    step_until(&mut h, "the model and its labels", |a| {
        a.document()
            .is_some_and(|d| d.index.labels().is_some() && !d.deriving())
            || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    let doc = h.state().document().unwrap();
    let shown: Vec<i64> = (0..doc.model.cells().len() as u32)
        .filter(|&c| doc.vis.cells[c as usize])
        .map(|c| doc.model.cell_id(c))
        .collect();
    // Cell 10, with the envelopes holding it.
    assert_eq!(shown, [1, 2, 10], "{}", all_said(doc));
    h.state_mut()
        .document_mut()
        .unwrap()
        .execute(Command::Select(
            "label nb cell & m 5".parse().unwrap(),
            SelectMode::Replace,
        ));
    h.run_steps(2);
    let doc = h.state().document().unwrap();
    let selected: Vec<i64> = doc
        .selection
        .cells
        .iter()
        .map(|c| doc.model.cell_id(c))
        .collect();
    assert_eq!(selected, [11]);
}

/// `shown`: the cells the 3D view draws, behind others too, less those
/// its cuts take away wholly; frozen to its cells when it selects.
#[test]
fn shown_rule_follows_the_view() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("viewer_shown.i");
    std::fs::write(
        &path,
        "shown
1 0 -1 fill=1 imp:n=1
2 0 -2 fill=1 (20 0 0) imp:n=1
8 0 1 2 -9 imp:n=1
9 0 9 imp:n=0
10 3 -7.8 -4 u=1 imp:n=1
11 5 -1.0 4 u=1 imp:n=1

1 so 5
2 s 20 0 0 5
4 so 2
9 so 100

m3 26000 1
m5 1001 2 8016 1
",
    )
    .unwrap();
    let mut h = Harness::builder()
        .with_size(egui::vec2(900.0, 600.0))
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| ViewerApp::new(cc, path.clone(), ViewState::new(), None).unwrap());
    let settle = |h: &mut Harness<'_, ViewerApp>, what: &str| {
        h.run_steps(2);
        step_until(h, what, |a| {
            a.document().is_some_and(|d| !d.deriving()) || a.error().is_some()
        });
        assert!(h.state().error().is_none(), "{:?}", h.state().error());
    };
    settle(&mut h, "the model");
    let ids = |doc: &Document, q: &str| -> Vec<i64> {
        let set = doc.resolve(&q.parse().unwrap()).unwrap();
        set.iter().map(|c| doc.model.cell_id(c)).collect()
    };
    let edit = |h: &mut Harness<'_, ViewerApp>, e: Edit| {
        h.state_mut()
            .document_mut()
            .unwrap()
            .execute(Command::Edit(e));
    };
    // Void hidden; the envelopes are drawn as what fills them.
    assert_eq!(ids(h.state().document().unwrap(), "shown"), [10, 11]);
    // Envelope 2 cut away, and the middle of envelope 1: the ball is gone
    // everywhere, the water stays.
    edit(
        &mut h,
        Edit::AddCut(CutState::Plane {
            point: [10.0, 0.0, 0.0],
            normal: [1.0, 0.0, 0.0],
        }),
    );
    edit(
        &mut h,
        Edit::AddCut(CutState::Box {
            lo: [-3.0; 3],
            hi: [3.0; 3],
            keep: false,
        }),
    );
    settle(&mut h, "the cuts");
    let doc = h.state().document().unwrap();
    assert_eq!(ids(doc, "shown"), [11]);
    assert_eq!(ids(doc, "visible | m 3"), [10, 11]);
    // Selecting it keeps the cells it gave, whatever the view does next.
    h.state_mut()
        .document_mut()
        .unwrap()
        .execute(Command::Select(
            "shown".parse().unwrap(),
            SelectMode::Replace,
        ));
    assert_eq!(
        h.state().document().unwrap().selection.query,
        Some(Query::Cells(vec![11]))
    );
    // Saved, it keeps them too, under the name it was saved by.
    edit(
        &mut h,
        Edit::SaveRule(llevant_viewer::scene::SavedRule {
            name: "shown & m 5".into(),
            rule: "shown & m 5".into(),
        }),
    );
    let saved = &h.state().document().unwrap().scene.rules;
    assert_eq!(saved.len(), 1);
    assert_eq!(
        (saved[0].name.as_str(), saved[0].rule.as_str()),
        ("shown & m 5", "c 11")
    );
    edit(&mut h, Edit::CutOn(1, false));
    settle(&mut h, "the box off");
    let doc = h.state().document().unwrap();
    assert_eq!(ids(doc, "shown"), [10, 11]);
    assert_eq!(doc.selection.query, Some(Query::Cells(vec![11])));
    // At level 0, the envelopes, solid: the one the plane leaves.
    edit(&mut h, Edit::Level(Some(0)));
    settle(&mut h, "level 0");
    assert_eq!(ids(h.state().document().unwrap(), "shown"), [1]);
    // Hidden cells are not shown.
    edit(&mut h, Edit::Level(None));
    edit(&mut h, Edit::Hide("c 11".parse().unwrap()));
    settle(&mut h, "cell 11 hidden");
    assert_eq!(ids(h.state().document().unwrap(), "shown"), [10]);
}

/// No model: the start screen. A file opened from it loads (with its
/// progress shown), and when it changes on disk a banner offers to read it
/// again, keeping the view. Colour by density.
#[test]
fn start_screen_open_reload_and_density() {
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let _one = serial();
    // A copy of a zoo model, to change.
    let dir = std::env::temp_dir().join(format!("llevant-viewer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("universes.i");
    // Never seen before (not in the cache): loaded staged, then completed.
    let text = std::fs::read_to_string(model("universes.i")).unwrap();
    let (title, rest) = text.split_once('\n').unwrap();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::fs::write(&path, format!("{title}\nc a copy {nanos}\n{rest}")).unwrap();
    let dev = llevant_viewer::DevOptions {
        stage_always: true,
        ..Default::default()
    };
    let mut h = Harness::builder()
        .with_size(egui::vec2(1200.0, 700.0))
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| ViewerApp::empty(cc).unwrap().with_dev(&dev));
    h.run_steps(3);
    h.get_by_label(&icons::with(icons::OPEN, "Open a model…"));
    assert!(h.state().document().is_none());
    let png = std::env::var_os("LLEVANT_VIEWER_TEST_PNG").map(PathBuf::from);
    if let Some(p) = &png {
        save(&mut h, &p.with_extension("start.png"));
    }

    let ctx = h.ctx.clone();
    h.state_mut().open_file(path.clone(), &ctx);
    h.step();
    // The loading screen, then the model.
    h.get_by_label("Opening universes.i");
    step_until(&mut h, "the model to load", |a| {
        a.document().is_some() || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    // Shown before its octrees (DevOptions::stage_always).
    let doc = h.state().document().unwrap();
    assert!(
        !doc.model.is_complete() && h.state().is_completing(),
        "shown staged"
    );
    h.run_steps(2);

    // Colour by density, from the toolbar.
    h.get_by_label_contains("Colour").click();
    h.run_steps(2);
    h.get_by_label("density").click();
    h.run_steps(2);
    assert_eq!(
        h.state().document().unwrap().scene.colour_by,
        Colouring::Density
    );
    if let Some(p) = &png {
        step_until(&mut h, "tracing", |a| !a.document().unwrap().view3d.busy());
        h.run_steps(3);
        save(&mut h, &p.with_extension("density.png"));
    }

    // Turn the view, change the file (move a surface): the banner, then
    // Reload keeps the view and the undo history, shows the model until the
    // new one lands, and selects the cells that changed.
    step_until(&mut h, "the octrees", |a| {
        a.document().is_some_and(|d| d.model.is_complete())
    });
    h.state_mut().document_mut().unwrap().orbit.state.yaw_deg = 123.0;
    let text = std::fs::read_to_string(&path).unwrap();
    let mut src = migjorn::Model::parse(&text);
    let id = src.surfaces().filter_map(|s| s.id()).next().unwrap();
    let c = src.surface(id).unwrap().coeffs().unwrap();
    let k = c.len() - 1;
    src.surface_mut(id)
        .unwrap()
        .set_coeff(k, c[k] + 0.5)
        .unwrap();
    std::fs::write(&path, src.to_source()).unwrap();
    // Newer than the file loaded whatever the file system's clock step.
    let later = std::time::SystemTime::now() + Duration::from_secs(5);
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(later)
        .unwrap();
    h.state_mut().check_input_now();
    assert!(h.state().input_changed(), "the change seen");
    h.run_steps(2);
    h.get_by_label("Reload").click();
    h.step();
    assert!(h.state().document().is_some(), "the model stays shown");
    step_until(&mut h, "the model to load again", |a| !a.is_loading());
    let doc = h.state().document().unwrap();
    let msg = last_said(doc);
    assert!(doc.changed_boxes.is_some(), "where the edit changed things");
    // The check of what changed starts with the reload (and may be done).
    if msg.starts_with("reloaded") {
        assert!(msg.contains("(selected)"), "{msg}");
        assert!(msg.contains("checking what changed"), "{msg}");
    } else {
        assert!(msg.starts_with("check of what the reload changed"), "{msg}");
    }
    assert!(!doc.selection.is_empty());
    step_until(&mut h, "the check of what changed", |a| {
        a.document().is_some_and(|d| {
            d.results
                .check
                .as_ref()
                .is_some_and(|c| c.scope == "what the reload changed")
        })
    });
    let doc = h.state().document().unwrap();
    assert!(doc.history.next_undo().is_some(), "undo history kept");
    assert_eq!(doc.orbit.state.yaw_deg, 123.0);
    assert_eq!(doc.scene.colour_by, Colouring::Density);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A model that does not fit on the GPU (the upload's failure forced with
/// `DevOptions::fail_gpu_upload`) opens traced on the CPU, saying why.
#[test]
fn a_model_too_large_for_the_gpu_is_traced_on_the_cpu() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let dev = llevant_viewer::DevOptions {
        fail_gpu_upload: true,
        ..Default::default()
    };
    let mut h = Harness::builder()
        .with_size(egui::vec2(900.0, 600.0))
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| {
            let mut app = ViewerApp::empty(cc).unwrap().with_dev(&dev);
            // The GPU asked for (on a software adapter, auto would trace on
            // the CPU anyway, and rightly say nothing).
            app.send(
                Command::Backend(llevant_viewer::state::BackendChoice::Gpu),
                &cc.egui_ctx,
            );
            app.open_file(model("universes.i"), &cc.egui_ctx);
            app
        });
    step_until(&mut h, "the model to load", |a| {
        a.document().is_some() || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    // The model is shown before its GPU copy is made, which then fails.
    step_until(&mut h, "the GPU copy to fail", |a| {
        a.document().is_some_and(|d| d.gpu_preparing().is_none())
    });
    let doc = h.state().document().unwrap();
    assert_eq!(doc.view3d.backend_name(), "CPU");
    let message = all_said(doc);
    assert!(message.contains("not enough GPU memory"), "{message}");
    step_until(&mut h, "the 3D view to refine", |a| {
        a.document().is_some_and(|d| !d.view3d.busy())
    });
    assert!(h.state().document().unwrap().view3d.stats().refined >= 1.0);
}

#[test]
fn the_3d_view_moves_to_the_gpu_once_the_model_is_uploaded() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let mut h = Harness::builder()
        .with_size(egui::vec2(900.0, 600.0))
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| {
            let mut app = ViewerApp::empty(cc).unwrap();
            // The GPU asked for (on a software adapter auto traces on the
            // CPU and makes no GPU copy).
            app.send(
                Command::Backend(llevant_viewer::state::BackendChoice::Gpu),
                &cc.egui_ctx,
            );
            app.open_file(model("universes.i"), &cc.egui_ctx);
            app
        });
    step_until(&mut h, "the model to load", |a| {
        a.document().is_some() || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    // Shown before its GPU copy is made: traced on the CPU meanwhile.
    assert_eq!(h.state().document().unwrap().view3d.backend_name(), "CPU");
    step_until(&mut h, "the 3D view to move to the GPU", |a| {
        a.document()
            .is_some_and(|d| d.view3d.backend_name() == "GPU")
    });
    let doc = h.state().document().unwrap();
    assert!(doc.gpu_preparing().is_none());
    assert!(doc.gpu_model.is_ok());
    let said = all_said(doc);
    assert!(!said.contains("unavailable"), "{said}");
    step_until(&mut h, "the 3D view to refine", |a| {
        a.document().is_some_and(|d| !d.view3d.busy())
    });
    assert!(h.state().document().unwrap().view3d.stats().refined >= 1.0);
}

/// An OpenMC XML model opens directly in the viewer, resolves OpenMC cell
/// `name="..."` labels in queries, and renders both 3D and slice views.
#[test]
fn openmc_model_opens_and_searches_labels() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let openmc_path = PathBuf::from(format!(
        "{}/../../docs/examples/openmc/fusion_hts_blanket.xml",
        env!("CARGO_MANIFEST_DIR")
    ));
    let mut h = Harness::builder()
        .with_size(egui::vec2(1200.0, 700.0))
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| {
            ViewerApp::new(cc, openmc_path.clone(), ViewState::new(), None).unwrap()
        });
    step_until(&mut h, "the OpenMC model and its labels", |a| {
        a.document()
            .is_some_and(|d| d.index.labels().is_some() && !d.deriving())
            || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    let doc = h.state().document().unwrap();
    assert!(
        doc.model.diagnostics().is_empty(),
        "{:?}",
        doc.model.diagnostics()
    );

    h.state_mut()
        .document_mut()
        .unwrap()
        .execute(Command::Select(
            "label \"Tungsten\"".parse().unwrap(),
            SelectMode::Replace,
        ));
    settle(&mut h);
    let doc = h.state().document().unwrap();
    let tungsten = doc.model.cell_index(2).expect("Tungsten cell 2");
    assert_eq!(doc.selection.cells.count(), 1);
    assert!(doc.selection.cells.contains(tungsten));

    step_until(&mut h, "3D and slice views to settle", |a| {
        let d = a.document().unwrap();
        !d.view3d.busy() && d.slice.progress() >= 1.0
    });
}

/// A native TRIPOLI-4 input file opens directly in the viewer, resolves
/// volume comments as labels, and settles both 3D and slice views.
#[test]
fn tripoli4_model_opens_and_renders() {
    let _one = serial();
    if !adapter_available() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let t4_path = PathBuf::from(format!(
        "{}/../../docs/examples/tripoli4/pwr_pin_cell.t4",
        env!("CARGO_MANIFEST_DIR")
    ));
    let mut h = Harness::builder()
        .with_size(egui::vec2(1200.0, 700.0))
        .wgpu_setup(llevant_viewer::app::wgpu_options().wgpu_setup)
        .build_eframe(|cc| ViewerApp::new(cc, t4_path.clone(), ViewState::new(), None).unwrap());
    step_until(&mut h, "the TRIPOLI-4 model and its labels", |a| {
        a.document()
            .is_some_and(|d| d.index.labels().is_some() && !d.deriving())
            || a.error().is_some()
    });
    assert!(h.state().error().is_none(), "{:?}", h.state().error());
    let doc = h.state().document().unwrap();
    assert!(
        doc.model.diagnostics().is_empty(),
        "{:?}",
        doc.model.diagnostics()
    );

    h.state_mut()
        .document_mut()
        .unwrap()
        .execute(Command::Select(
            "label \"UO2 Fuel Pellet\"".parse().unwrap(),
            SelectMode::Replace,
        ));
    settle(&mut h);
    let doc = h.state().document().unwrap();
    let pellet = doc.model.cell_index(3).expect("UO2 fuel cell 3");
    assert_eq!(doc.selection.cells.count(), 1);
    assert!(doc.selection.cells.contains(pellet));

    step_until(&mut h, "3D and slice views to settle", |a| {
        let d = a.document().unwrap();
        !d.view3d.busy() && d.slice.progress() >= 1.0
    });
}
