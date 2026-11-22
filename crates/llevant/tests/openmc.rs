use std::path::Path;

use llevant::{Depth, Model, Options, Scope, Status, VolumeGrid};

/// Verify loading, material classification, and zero-defect volume checking on the fusion HTS OpenMC benchmark.
#[test]
fn loads_complex_fusion_hts_openmc_model_with_zero_diagnostics_and_zero_defects() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/examples/openmc/fusion_hts_blanket.xml");
    let model = Model::load(&path, &Options::default(), None).expect("load fusion_hts_blanket.xml");

    assert_eq!(
        model.diagnostics().len(),
        0,
        "unexpected diagnostics: {:?}",
        model.diagnostics()
    );
    assert!(model.cells().len() >= 16);

    // Check plasma chamber at origin is void (material 0), tungsten at r=62 is M1,
    // PbLi breeder at r=90, y=90 is M3, HTS winding pack at r=176 is M5.
    let pts = [
        [0.0, 0.0, 0.0],   // Plasma void
        [0.0, 62.0, 0.0],  // Tungsten armor (M1)
        [0.0, 95.0, 0.0],  // Pb-17Li breeder (M3)
        [0.0, 176.0, 0.0], // HTS winding pack (M5)
        [160.0, 0.0, 0.0], // Fused silica diagnostic window (M7)
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

    assert_eq!(
        model.diagnostics().len(),
        0,
        "unexpected diagnostics: {:?}",
        model.diagnostics()
    );
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

/// Verify loading, hex-lattice classification, and zero-defect volume checking on the 3-ring VVER OpenMC benchmark.
#[test]
fn loads_vver_hex_assembly_with_zero_defects_and_exact_pin_classification() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/examples/openmc/vver_hex_assembly.xml");
    let model = Model::load(&path, &Options::default(), None).expect("load vver_hex_assembly.xml");

    assert_eq!(
        model.diagnostics().len(),
        0,
        "unexpected diagnostics: {:?}",
        model.diagnostics()
    );
    assert_eq!(model.lattices().len(), 1);
    assert!(
        model.lattices()[0].hexagonal(),
        "expected LAT=2 hexagonal lattice"
    );

    let pitch = 1.275_f64;
    let ring1_side_x = (3.0_f64.sqrt() * 0.5) * pitch;
    let ring1_side_y = 0.5 * pitch;

    // Verify center B4C (M6), ring-1 Gd2O3 (M4) at (0, +-P), ring-1 UO2 (M1), ring-2 UO2 (M1),
    // outer coolant inside wrapper (M3), SS316 duct (M5), and water reflector (M3).
    let pts = [
        [0.0, 0.0, 0.0],                    // Center B4C rod (M6)
        [0.0, pitch, 0.0],                  // Ring 1 top Gd burnable poison (M4)
        [0.0, -pitch, 0.0],                 // Ring 1 bottom Gd burnable poison (M4)
        [ring1_side_x, ring1_side_y, 0.0],  // Ring 1 NE UO2 fuel (M1)
        [-ring1_side_x, ring1_side_y, 0.0], // Ring 1 NW UO2 fuel (M1)
        [0.0, 2.0 * pitch, 0.0],            // Ring 2 top UO2 fuel (M1)
        [3.3, 0.0, 0.0],                    // Outer coolant inside wrapper (M3)
        [3.7, 0.0, 0.0],                    // SS316 wrapper duct (M5)
        [4.2, 0.0, 0.0],                    // Outer water reflector (M3)
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
    assert_eq!(mats, vec![6, 4, 4, 1, 1, 1, 3, 5, 3]);

    // Verify 0 gaps and 0 overlaps
    let region = model.extent(Scope::Problem, 64);
    let vols = model
        .volumes(&VolumeGrid::new(region, 0.6), None)
        .expect("volume check");
    assert!(
        vols.problems().is_empty(),
        "expected clean hex geometry, found problems: {:?}",
        vols.problems()
    );
}

/// Verify 3D x-oriented OpenMC `<hex_lattice>` with two axial layers classifies each layer and ring pin accurately.
#[test]
fn loads_3d_hex_lattice_x_orientation_and_classifies_axial_slices() {
    let xml = r#"<model>
  <materials>
    <material id="1" name="Fuel A"><density units="g/cm3" value="10.4"/><nuclide ao="1.0" name="U235"/></material>
    <material id="2" name="Poison B"><density units="g/cm3" value="2.5"/><nuclide ao="1.0" name="B10"/></material>
    <material id="3" name="Reflector C"><density units="g/cm3" value="7.9"/><nuclide ao="1.0" name="Fe56"/></material>
    <material id="4" name="Water Outer"><density units="g/cm3" value="0.72"/><nuclide ao="2.0" name="H1"/></material>
  </materials>
  <geometry>
    <surface id="1" type="z-cylinder" coeffs="0 0 0.45"/>
    <surface id="2" type="z-cylinder" coeffs="0 0 3.5" boundary="vacuum"/>
    <surface id="3" type="z-plane" coeffs="-4.0" boundary="vacuum"/>
    <surface id="4" type="z-plane" coeffs="4.0" boundary="vacuum"/>

    <cell id="1" universe="1" material="1" region="-1"/>
    <cell id="2" universe="1" material="4" region="1"/>
    <cell id="3" universe="2" material="2" region="-1"/>
    <cell id="4" universe="2" material="4" region="1"/>
    <cell id="5" universe="3" material="3" region="-1"/>
    <cell id="6" universe="3" material="4" region="1"/>
    <cell id="7" universe="4" material="4"/>

    <!-- orientation=x rows (2, 3, 2): (-0.5P,+y) (+0.5P,+y) / (-P,0) (0,0) (+P,0) / (-0.5P,-y) (+0.5P,-y) -->
    <hex_lattice id="50" n_rings="2" n_axial="2" orientation="x">
      <pitch>2.0 4.0</pitch>
      <outer>4</outer>
      <center>0.0 0.0 0.0</center>
      <universes>
          1 1
        1 2 3
          1 1

          2 3
        1 1 2
          1 1
      </universes>
    </hex_lattice>
    <cell id="10" universe="0" fill="50" region="-2 3 -4"/>
  </geometry>
</model>"#;

    let model =
        Model::compile(xml.as_bytes(), &Options::default(), None).expect("load 3D x-hex model");
    assert_eq!(model.diagnostics().len(), 0, "{:?}", model.diagnostics());
    assert!(model.lattices()[0].hexagonal());

    let y_row = 3.0_f64.sqrt();
    let pts = [
        [0.0, 0.0, -2.0],    // Bottom layer center -> U2 -> M2
        [2.0, 0.0, -2.0],    // Bottom layer (+P, 0) -> U3 -> M3
        [-1.0, y_row, -2.0], // Bottom layer top-left -> U1 -> M1
        [0.0, 0.0, 2.0],     // Top layer center -> U1 -> M1
        [2.0, 0.0, 2.0],     // Top layer (+P, 0) -> U2 -> M2
        [1.0, y_row, 2.0],   // Top layer top-right -> U3 -> M3
        [3.1, 0.0, 0.0],     // Outer universe U4 -> M4
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
    assert_eq!(mats, vec![2, 3, 1, 1, 2, 3, 4]);
}
