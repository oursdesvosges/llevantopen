//! Integration tests for native DRAGON5 `GEO:` lattice and pin-cell loading.

use llevant::{Depth, Model, Options, Scope, Status, VolumeGrid};

#[test]
fn dragon_pwr_3x3_assembly_loads_classifies_and_checks() {
    let d5 = include_str!("../../../docs/examples/dragon5/pwr_3x3_assembly.x2m");
    let model =
        Model::compile(d5.as_bytes(), &Options::default(), None).expect("compile DRAGON5 3x3");
    assert!(
        model.diagnostics().is_empty(),
        "unexpected diagnostics: {:?}",
        model.diagnostics()
    );
    assert_eq!(model.lattices().len(), 1);

    // Pin pitch = 1.26 cm; lattice covers x in [0, 3.78], y in [0, 3.78].
    let pts = [
        [0.63, 0.63, 0.0], // Element [0,0] PIN center -> fuel mix=1
        [1.07, 0.63, 0.0], // Element [0,0] clad ring r=0.44 -> clad mix=2
        [1.20, 1.20, 0.0], // Element [0,0] moderator -> mod mix=3
        [1.89, 1.89, 0.0], // Element [1,1] central GT center -> mix=2
        [3.15, 3.15, 0.0], // Element [2,2] PIN center -> fuel mix=1
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
    assert_eq!(mats, vec![1, 2, 3, 2, 1]);

    let region = model.extent(Scope::Problem, 64);
    let vols = model
        .volumes(&VolumeGrid::new(region, 0.25), None)
        .expect("check volumes");
    assert!(
        vols.problems().is_empty(),
        "DRAGON5 3x3 assembly had unexpected defects: {:?}",
        vols.problems()
    );
}

#[test]
fn dragon_hexcel_and_homogeneous_car3d_compile_cleanly() {
    let hex = r#"
HEXPIN := GEO: :: HEXCEL 2
  SIDE 1.0
  RADIUS 0.0 0.35 0.42
  MESHZ -5.0 5.0
  MIX 10 20 30 ;
"#;
    let hm = Model::compile(hex.as_bytes(), &Options::default(), None).expect("HEXCEL");
    assert!(hm.diagnostics().is_empty(), "{:?}", hm.diagnostics());
    assert_eq!(
        hm.cell(hm.classify_point([0.0, 0.0, 0.0]).cell().unwrap())
            .material,
        10
    );
    assert_eq!(
        hm.cell(hm.classify_point([0.38, 0.0, 0.0]).cell().unwrap())
            .material,
        20
    );
    assert_eq!(
        hm.cell(hm.classify_point([0.70, 0.0, 0.0]).cell().unwrap())
            .material,
        30
    );

    let grid = r#"
CORE3D := GEO: :: CAR3D 2 2 2
  MESHX 0.0 10.0 20.0
  MESHY 0.0 10.0 20.0
  MESHZ 0.0 5.0 15.0
  MIX 1 2 3 4  5 6 7 8 ;
"#;
    let gm = Model::compile(grid.as_bytes(), &Options::default(), None).expect("CAR3D");
    assert!(gm.diagnostics().is_empty(), "{:?}", gm.diagnostics());
    assert_eq!(
        gm.cell(gm.classify_point([5.0, 5.0, 2.5]).cell().unwrap())
            .material,
        1
    );
    assert_eq!(
        gm.cell(gm.classify_point([15.0, 15.0, 10.0]).cell().unwrap())
            .material,
        8
    );
}
