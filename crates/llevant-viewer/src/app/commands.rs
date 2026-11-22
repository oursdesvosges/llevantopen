//! Carrying out the commands panels, tools and navigation collect each frame.

use super::*;

impl ViewerApp {
    pub(super) fn execute(&mut self, commands: Vec<Command>, ctx: &egui::Context) {
        let software = self.software();
        for c in commands {
            match c {
                Command::Backend(b) => {
                    self.backend = b;
                    if let Some(d) = &mut self.doc {
                        d.set_backend(b, software);
                    }
                }
                Command::Budget(ms) => {
                    self.budget_ms = ms;
                    if let Some(d) = &mut self.doc {
                        d.view3d.set_budget(ms / 1e3);
                    }
                }
                Command::OpenUniverse(u) => self.start_opening(u, ctx),
                Command::Inspect(s) => {
                    self.panels.inspector.open(s);
                    self.frame.show(frame::Panel::Inspector);
                }
                Command::PickSurface(p) => {
                    if let Some(d) = &mut self.doc {
                        d.execute(Command::PickSurface(p));
                        let id = d
                            .surface_pick
                            .as_ref()
                            .and_then(|s| d.model.surface_id(s.shape()?));
                        if let Some(id) = id {
                            self.panels
                                .inspector
                                .open(crate::panels::inspector::Subject::Surface(id.id));
                            self.frame.show(frame::Panel::Inspector);
                        }
                    }
                }
                Command::Workspace(w) => self.frame.workspace = w,
                Command::ShowPanel(p) => self.frame.show(p),
                Command::Maximize(v) => self.frame.maximized = v,
                Command::FoldLeft => {
                    let s = &mut self.frame.left;
                    s.open = !s.open;
                }
                Command::FoldRight => {
                    let s = &mut self.frame.right;
                    s.open = !s.open;
                }
                Command::ShowKeys => self.show_keys = true,
                Command::UseTool(name) => self.tools.activate(name),
                Command::CopyMeshes(ks) => {
                    if let Some(d) = &mut self.doc {
                        let meshes = ks.iter().filter_map(|&k| d.scene.meshes.get(k));
                        match cards_text(meshes, d.input_meshes.as_deref()) {
                            Ok(text) => {
                                ctx.copy_text(text);
                                d.messages.info(format!(
                                    "copied the cards of {} mesh{}",
                                    ks.len(),
                                    if ks.len() == 1 { "" } else { "es" }
                                ));
                            }
                            Err(e) => d.messages.warn(e),
                        }
                    }
                }
                Command::SaveMeshes(ks) => {
                    if let Some(d) = &mut self.doc {
                        let meshes = ks.iter().filter_map(|&k| d.scene.meshes.get(k));
                        let r = cards_text(meshes, d.input_meshes.as_deref())
                            .and_then(|text| save_cards(&self.path, &text));
                        d.messages.result(r);
                    }
                }
                Command::OpenFile(p) => {
                    if self.may_drop_edits("Open") {
                        self.open_file(p, ctx);
                    }
                }
                Command::OpenDialog => {
                    if !self.may_drop_edits("Open") {
                        continue;
                    }
                    let mut d = rfd::FileDialog::new()
                        .set_title("Open an MCNP, OpenMC, TRIPOLI-4, or DRAGON5 input");
                    if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
                        d = d.set_directory(dir);
                    }
                    if let Some(p) = d.pick_file() {
                        self.open_file(p, ctx);
                    }
                }
                Command::Reload => {
                    if self.may_drop_edits("Reload") {
                        self.reload(ctx);
                    }
                }
                Command::EditCard(kind, id, text) => self.edit_card(kind, id, text, ctx),
                Command::EditCells(ids, changes) => self.edit_cells(ids, changes, ctx),
                Command::Renumber(kind, pairs) => self.renumber(kind, pairs, ctx),
                Command::AddCard {
                    kind,
                    id,
                    text,
                    carve,
                } => self.add_card(kind, id, text, carve, ctx),
                Command::EditSelection => {
                    self.panels.inspector.edit_selection();
                    self.frame.show(frame::Panel::Inspector);
                }
                Command::DragSurface(id, c) => self.drag_surface(id, c, ctx),
                Command::EndDrag { commit } => self.end_drag(commit, ctx),
                Command::UndoInputEdit => self.undo_input_edit(ctx),
                Command::SaveInput => self.save_input(None, ctx),
                Command::SaveInputAs(to) => {
                    if let Some(to) = to.or_else(|| self.pick_save_as()) {
                        self.save_input(Some(to), ctx);
                    }
                }
                Command::DiscardInputEdits => self.reload(ctx),
                Command::OpenFieldDialog => {
                    let mut d = rfd::FileDialog::new()
                        .set_title(
                            "Open a field: values per cell (CSV) or an MCNP meshtal, MCTAL or cuvmsh file",
                        )
                        .add_filter(
                            "CSV, meshtal or MCTAL",
                            &["csv", "msht", "meshtal", "m", "mctal", "mctl", "txt"],
                        )
                        .add_filter("Any file", &["*"]);
                    if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
                        d = d.set_directory(dir);
                    }
                    if let (Some(p), Some(doc)) = (d.pick_file(), &mut self.doc) {
                        doc.execute(Command::LoadField(crate::state::FieldSource::new(p)));
                        self.frame.show(frame::Panel::Fields);
                    }
                }
                Command::OpenRunInputDialog => {
                    let source = self
                        .doc
                        .as_ref()
                        .and_then(|d| d.scene.field.as_ref())
                        .map(|f| f.source.clone());
                    if let Some(source) = source {
                        let mut d = rfd::FileDialog::new()
                            .set_title("The MCNP input of the run that wrote the tally file");
                        if let Some(dir) =
                            source.path.parent().filter(|d| !d.as_os_str().is_empty())
                        {
                            d = d.set_directory(dir);
                        }
                        if let (Some(p), Some(doc)) = (d.pick_file(), &mut self.doc) {
                            doc.execute(Command::LoadField(crate::state::FieldSource {
                                input: Some(p),
                                ..source
                            }));
                        }
                    }
                }
                Command::CancelLoading => {
                    if let Some(t) = &self.loading {
                        t.cancel();
                    }
                }
                Command::CopyCli => {
                    if let Some(d) = &mut self.doc {
                        let (w, h) = d.view3d.size();
                        let text = crate::export::cli_command(d, w as usize, h as usize);
                        ctx.copy_text(text);
                        d.messages
                            .info("copied the `llevant render` command of this view");
                    }
                }
                Command::Palette => self.palette.toggle(),
                Command::Back => {
                    self.back();
                    if let Some(d) = &mut self.doc {
                        d.set_backend(self.backend, software);
                        d.view3d.set_budget(self.budget_ms / 1e3);
                    }
                }
                Command::SaveView => {
                    let r = (self.view_state().save(&self.state_path))
                        .map(|()| format!("view saved to {}", self.state_path.display()));
                    if let Some(d) = &mut self.doc {
                        d.messages.result(r);
                    }
                }
                c => {
                    if let Some(d) = &mut self.doc {
                        d.execute(c);
                    }
                }
            }
        }
    }
}
