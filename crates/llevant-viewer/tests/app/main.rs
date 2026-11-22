//! The whole application, driven headlessly (egui_kittest on wgpu): open a
//! zoo model, pin points with the Pin tool, pick in 3D, undo and redo,
//! change the scene from the panels. Skipped when no GPU adapter is
//! available (lavapipe is enough). With `LLEVANT_VIEWER_TEST_PNG=<file>`,
//! the final window is saved there.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui_kittest::kittest::{NodeT, Queryable};
use egui_kittest::Harness;
use llevant_scene::query::Query;
use llevant_viewer::app::frame::Panel;
use llevant_viewer::command::{Command, Edit};
use llevant_viewer::document::Document;
use llevant_viewer::icons;
use llevant_viewer::pick::pick_ray;
use llevant_viewer::selection::SelectMode;
use llevant_viewer::state::Colouring;
use llevant_viewer::state::CutState;
use llevant_viewer::state::OutlineBy;
use llevant_viewer::{ViewState, ViewerApp};

mod editing;
mod explore;
mod results;
mod views;

fn model(name: &str) -> PathBuf {
    PathBuf::from(format!(
        "{}/../../tools/mcnp-oracle/models/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
}

fn adapter_available() -> bool {
    llevant_gpu::check::shared_gpu().is_some()
}

/// One application at a time: GPU devices made at once on several threads
/// (each test's window and tracer) can hang the Vulkan driver.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    ONE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Step until `done`, or fail after a while.
fn step_until(h: &mut Harness<'_, ViewerApp>, what: &str, done: impl Fn(&ViewerApp) -> bool) {
    let start = Instant::now();
    while !done(h.state()) {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "timed out waiting for {what}"
        );
        h.step();
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Step until a node with `label` appears, or fail after a while.
fn step_until_label(h: &mut Harness<'_, ViewerApp>, what: &str, label: &str) {
    let start = Instant::now();
    while h.query_by_label(label).is_none() {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "timed out waiting for {what}"
        );
        h.step();
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Frames until the views have caught up with the scene (edits are
/// derived off the UI thread).
/// The newest message of `doc` but news of its search structures (built
/// when they are, after whatever was said).
fn last_said(doc: &Document) -> String {
    doc.messages
        .iter()
        .rev()
        .find(|m| !m.text.starts_with("search structures"))
        .map_or(String::new(), |m| m.text.clone())
}

/// Every message of `doc` kept, one a line.
fn all_said(doc: &Document) -> String {
    let lines: Vec<&str> = doc.messages.iter().map(|m| m.text.as_str()).collect();
    lines.join("\n")
}

fn settle(h: &mut Harness<'_, ViewerApp>) {
    step_until(h, "the views to catch up", |a| {
        a.document().is_none_or(|d| !d.deriving())
    });
}

/// A double-click at `pos`, in one frame (the harness's frames are a
/// second apart), after a pause so that a click just before does not make
/// it a triple-click.
fn double_click(h: &mut Harness<'_, ViewerApp>, pos: egui::Pos2) {
    h.step();
    h.event(egui::Event::PointerMoved(pos));
    h.step();
    for _ in 0..2 {
        for pressed in [true, false] {
            h.input_mut().events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            });
        }
    }
    h.step();
    h.step();
}

/// A right-click at `pos`: its menu.
fn right_click(h: &mut Harness<'_, ViewerApp>, pos: egui::Pos2) {
    h.event(egui::Event::PointerMoved(pos));
    for pressed in [true, false] {
        h.event(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Secondary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        });
    }
    h.run_steps(3);
}

/// Pin here, from the right-click menu at `pos`.
fn pin_at(h: &mut Harness<'_, ViewerApp>, pos: egui::Pos2) {
    right_click(h, pos);
    h.get_by_label(&icons::with(icons::PIN, "Pin here")).click();
    h.run_steps(2);
}

/// A click at `pos`, one event per frame.
fn click(h: &mut Harness<'_, ViewerApp>, pos: egui::Pos2) {
    h.event(egui::Event::PointerMoved(pos));
    for pressed in [true, false] {
        h.event(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        });
    }
    h.step();
    h.step();
}

fn shortcut(h: &mut Harness<'_, ViewerApp>, key: egui::Key, modifiers: egui::Modifiers) {
    for pressed in [true, false] {
        h.event(egui::Event::Key {
            key,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers,
        });
    }
    h.step();
    h.step();
}

/// The window as a PNG file.
fn save(h: &mut Harness<'_, ViewerApp>, path: &std::path::Path) {
    let img = h.render().expect("render");
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), img.width(), img.height());
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .unwrap()
        .write_image_data(img.as_raw())
        .unwrap();
}

/// Where in the window the 3D view shows `cell` now.
fn screen_pos_of(h: &mut Harness<'_, ViewerApp>, cell: u32) -> egui::Pos2 {
    h.run_steps(2);
    let rect = h.get_by_label("3D view").rect();
    let doc = h.state().document().unwrap();
    let cam = doc
        .orbit
        .camera(rect.width() as usize, rect.height() as usize, None);
    let (i, j) = (0..cam.height)
        .step_by(3)
        .flat_map(|j| (0..cam.width).step_by(3).map(move |i| (i, j)))
        .find(|&(i, j)| {
            pick_ray(&doc.model, &doc.vis, &cam, i, j).and_then(|p| p.cell()) == Some(cell)
        })
        .expect("the cell in view");
    rect.min + egui::vec2(i as f32 + 0.5, j as f32 + 0.5)
}

fn type_into(h: &mut Harness<'_, ViewerApp>, name: &str, text: &str) {
    h.get_by_label(name).click();
    h.run_steps(2);
    h.get_by_label(name).type_text(text);
    h.run_steps(2);
}

/// Which views are shown: one maximised, or both.
#[derive(Clone, Copy)]
enum Views {
    Three,
    Slice,
    Both,
}

/// Maximise a view, or show both, with the views' header buttons.
fn show(h: &mut Harness<'_, ViewerApp>, v: Views) {
    if h.query_by_label("Show both views").is_some() {
        h.get_by_label("Show both views").click();
        h.run_steps(3);
    }
    match v {
        Views::Both => {}
        Views::Three => h.get_by_label("Maximise the 3D view").click(),
        Views::Slice => h.get_by_label("Maximise the slice").click(),
    }
    h.run_steps(3);
}

/// Bring `p` into sight: its workspace's tab, then its own.
fn panel(h: &mut Harness<'_, ViewerApp>, p: Panel) {
    if let Some(w) = p.home() {
        h.get_by_label(&icons::with(w.icon(), w.name())).click();
        h.run_steps(2);
        h.get_by_label(&p.title()).click();
    }
    h.run_steps(2);
}

/// Open or close the 3D view's Section menu.
fn cuts_menu(h: &mut Harness<'_, ViewerApp>) {
    // (The header's button, named with the number of cuts.)
    let label = llevant_viewer::panels::headers::section_label(h.state().document().unwrap());
    h.get_by_label(&icons::with(icons::CLIP, &label)).click();
    h.run_steps(3);
}

/// Where the 3D view shows the model point `p`.
fn project_3d(h: &mut Harness<'_, ViewerApp>, p: [f64; 3]) -> egui::Pos2 {
    let rect = h.get_by_label("3D view").rect();
    let doc = h.state().document().unwrap();
    let cam = doc.orbit.camera(
        rect.width() as usize,
        rect.height() as usize,
        doc.scene.cutaway(),
    );
    let [f, rt, up] = cam.basis();
    let v = [0, 1, 2].map(|k| p[k] - cam.eye[k]);
    let z = llevant::vec3::dot(v, f);
    let half = z * (0.5 * cam.fov_deg).to_radians().tan();
    let aspect = cam.width as f64 / cam.height as f64;
    let x = llevant::vec3::dot(v, rt) / (half * aspect);
    let y = llevant::vec3::dot(v, up) / half;
    rect.center()
        + egui::vec2(
            (x * 0.5 * rect.width() as f64) as f32,
            (-y * 0.5 * rect.height() as f64) as f32,
        )
}

/// A drag of the left button from `from` to `to`.
fn drag_between(h: &mut Harness<'_, ViewerApp>, from: egui::Pos2, to: egui::Pos2) {
    drag_with(h, egui::PointerButton::Primary, from, to);
}

/// A drag with `button` from `from` to `to`, in ten steps.
fn drag_with(
    h: &mut Harness<'_, ViewerApp>,
    button: egui::PointerButton,
    from: egui::Pos2,
    to: egui::Pos2,
) {
    h.event(egui::Event::PointerMoved(from));
    h.step();
    h.event(egui::Event::PointerButton {
        pos: from,
        button,
        pressed: true,
        modifiers: egui::Modifiers::NONE,
    });
    h.step();
    for k in 1..=10 {
        h.event(egui::Event::PointerMoved(
            from + (to - from) * (k as f32 / 10.0),
        ));
        h.step();
    }
    h.event(egui::Event::PointerButton {
        pos: to,
        button,
        pressed: false,
        modifiers: egui::Modifiers::NONE,
    });
    h.run_steps(2);
}

/// Drag the ring about axis `k` through `c` (radius `rho`) 60 points along
/// itself, from its point at `t0` degrees.
fn drag_ring(h: &mut Harness<'_, ViewerApp>, c: [f64; 3], k: usize, rho: f64, t0: f64) {
    let at = |deg: f64| {
        let (s, co) = deg.to_radians().sin_cos();
        let mut p = c;
        p[(k + 1) % 3] += rho * co;
        p[(k + 2) % 3] += rho * s;
        p
    };
    let a = project_3d(h, at(t0));
    let b = project_3d(h, at(t0 + 10.0));
    let dir = (b - a).normalized();
    drag_between(h, a, a + dir * 60.0);
}
