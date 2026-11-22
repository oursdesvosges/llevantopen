//! Integration tests for native TRIPOLI-4 input loading through `llevant::Model`.

use llevant::{Depth, Model, Options, Scope, Status, VolumeGrid};

#[test]
fn pwr_pin_cell_loads_classifies_and_checks_cleanly() {
    let t4 = include_str!("../../../docs/examples/tripoli4/pwr_pin_cell.t4");
    let model = Model::compile(t4.as_bytes(), &Options::default(), None).expect("compile T4 pin");
    assert!(
        model.diagnostics().is_empty(),
        "unexpected diagnostics: {:?}",
        model.diagnostics()
    );

    let pts = [
        [0.0, 0.0, 0.0],  // UO2 Fuel Pellet (VOLU 3 -> mat 1)
        [0.44, 0.0, 0.0], // Zircaloy Clad (VOLU 2 -> mat 2)
        [0.55, 0.0, 0.0], // Water Moderator (VOLU 1 -> mat 3)
        [5.0, 0.0, 0.0],  // Exterior graveyard (imp:n=0)
    ];
    let cls = model.classify(&pts, Depth::All);
    let mats: Vec<(i64, i64)> = cls
        .ids
        .iter()
        .map(|&id| {
            let lbl = &cls.labels[id as usize];
            assert_eq!(lbl.status, Status::Inside, "{lbl:?}");
            let c = model.cell(lbl.cell().unwrap());
            (c.id, c.material)
        })
        .collect();
    assert_eq!(mats[0], (3, 1));
    assert_eq!(mats[1], (2, 2));
    assert_eq!(mats[2], (1, 3));
    assert_eq!(mats[3].1, 0);

    let region = model.extent(Scope::Problem, 64);
    let vols = model
        .volumes(&VolumeGrid::new(region, 0.25), None)
        .expect("check volumes");
    assert!(
        vols.problems().is_empty(),
        "ECRASE pin cell should have zero overlaps/gaps: {:?}",
        vols.problems()
    );
}

#[test]
fn fusion_shield_box_combi_cone_and_ecrase() {
    let t4 = include_str!("../../../docs/examples/tripoli4/fusion_shield_box.t4");
    let model =
        Model::compile(t4.as_bytes(), &Options::default(), None).expect("compile T4 shield box");
    assert!(
        model.diagnostics().is_empty(),
        "unexpected diagnostics: {:?}",
        model.diagnostics()
    );

    let pts = [
        [0.0, 0.0, 0.0],  // Heavy water sphere target (VOLU 3 -> mat 3)
        [20.0, 0.0, 0.0], // Steel reflector (VOLU 2 -> mat 2)
        [40.0, 0.0, 0.0], // BPE shield (VOLU 1 -> mat 1)
        [0.0, 0.0, 30.0], // Collimator duct (VOLU 4 -> void mat 0)
    ];
    let cls = model.classify(&pts, Depth::All);
    let mats: Vec<i64> = cls
        .ids
        .iter()
        .map(|&id| {
            let lbl = &cls.labels[id as usize];
            assert_eq!(lbl.status, Status::Inside, "{lbl:?}");
            model.cell(lbl.cell().unwrap()).material
        })
        .collect();
    assert_eq!(mats, vec![3, 2, 1, 0]);

    let region = model.extent(Scope::Problem, 64);
    let vols = model
        .volumes(&VolumeGrid::new(region, 5.0), None)
        .expect("check volumes");
    assert!(
        vols.problems().is_empty(),
        "shield box should have zero overlaps/gaps: {:?}",
        vols.problems()
    );
}

#[test]
fn shorthand_combi_loads_cleanly() {
    let t4 = "GEOMETRY\n  SURF 1 SPHERE 0 0 0 10\n  TYPE 1 COMBI 1 -1\nENDG\nEND\n";
    let model = Model::from_text(t4, &Options::default());
    assert!(model.diagnostics().is_empty(), "{:?}", model.diagnostics());
    assert_eq!(model.classify_point([0.0, 0.0, 0.0]).status, Status::Inside);
    assert!(
        model
            .cell(model.classify_point([20.0, 0.0, 0.0]).cell().unwrap())
            .imp_zero
    );
}
