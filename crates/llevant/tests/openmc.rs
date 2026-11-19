use std::path::Path;

use llevant::{Depth, Model, Options, Scope, Status, VolumeGrid};

/// Verify loading, material classification, and zero-defect volume checking on the fusion HTS OpenMC benchmark.
#[test]
fn loads_complex_fusion_hts_openmc_model_with_zero_diagnostics_and_zero_defects() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/examples/openmc/fusion_hts_blanket.xml");
    let model = Model::load(&path, &Options::default(), None).expect("load fusion_hts_blanket.xml");

    assert_eq!(model.diagnostics().len(), 0, "unexpected diagnostics: {:?}", model.diagnostics());
    assert!(model.cells().len() >= 16);

    // Check plasma chamber at origin is void (material 0), tungsten at r=62 is M1,
    // PbLi breeder at r=90, y=90 is M3, HTS winding pack at r=176 is M5.
    let pts = [
        [0.0, 0.0, 0.0],     // Plasma void
        [0.0, 62.0, 0.0],    // Tungsten armor (M1)
        [0.0, 95.0, 0.0],    // Pb-17Li breeder (M3)
        [0.0, 176.0, 0.0],   // HTS winding pack (M5)
        [160.0, 0.0, 0.0],   // Fused silica diagnostic window (M7)
    ];
    let cls = model.classify(&pts, Depth::All);
    let mats: Vec<i64> = cls
        .ids
        .iter()
        .map(|&id| {
            let lbl = &cls.labels[id as usize];
            assert_eq!(lbl.status, Status::Inside);
            model.cell(lbl.cell().unwrap()).material
        })
        .collect();
    assert_eq!(mats, vec![0, 1, 3, 5, 7]);

    // Verify 0 gaps and 0 overlaps
    let region = model.extent(Scope::Problem, 64);
    let vols = model
        .volumes(&VolumeGrid::new(region, 12.0), None)
        .expect("volume check");
    assert!(
        vols.problems().is_empty(),
        "expected clean geometry, found problems: {:?}",
        vols.problems()
    );
}

/// Verify loading and point classification on the 3x3 PWR fuel pin lattice OpenMC benchmark.
#[test]
fn loads_openmc_3x3_pwr_lattice_and_classifies_central_control_rod_and_fuel_pins() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/examples/openmc/pwr_pin_lattice.xml");
    let model = Model::load(&path, &Options::default(), None).expect("load pwr_pin_lattice.xml");

    assert_eq!(model.diagnostics().len(), 0, "unexpected diagnostics: {:?}", model.diagnostics());
    assert_eq!(model.lattices().len(), 1);

    // Center pin (0, 0, 0) is B4C control rod (M5); neighbor pin (1.26, 0, 0) is UO2 fuel (M1);
    // shroud at (2.05, 0, 0) is SS304 (M6).
    let pts = [
        [0.0, 0.0, 0.0],
        [1.26, 0.0, 0.0],
        [-1.26, 1.26, 0.0],
        [2.05, 0.0, 0.0],
    ];
    let cls = model.classify(&pts, Depth::All);
    let mats: Vec<i64> = cls
        .ids
        .iter()
        .map(|&id| {
            let lbl = &cls.labels[id as usize];
            assert_eq!(lbl.status, Status::Inside);
            model.cell(lbl.cell().unwrap()).material
        })
        .collect();
    assert_eq!(mats, vec![5, 1, 1, 6]);
}
