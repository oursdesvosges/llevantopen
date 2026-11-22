"""Python API tests for native OpenMC XML loading, hex lattices, labels, and QA."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import llevant

OPENMC_EXAMPLES = Path(__file__).resolve().parents[3] / "docs" / "examples" / "openmc"


def test_openmc_models_load_and_check_cleanly():
    """Load fusion, PWR rectilinear lattice, and VVER hex lattice OpenMC XML benchmarks."""
    for name, expected_lat in [
        ("fusion_hts_blanket.xml", 0),
        ("pwr_pin_lattice.xml", 1),
        ("vver_hex_assembly.xml", 1),
    ]:
        m = llevant.load(OPENMC_EXAMPLES / name)
        assert len(m.diagnostics) == 0, f"{name}: {m.diagnostics}"
        assert m.info()["lattices"] == expected_lat
        h = 12.0 if name == "fusion_hts_blanket.xml" else 0.75
        assert m.check(h) == [], f"{name} reported geometry defects"


def test_openmc_queries_labels_slices_renders_and_fingerprints(tmp_path: Path):
    """Exercise classify, locate, trace, slice, render with label rules, volumes, and fingerprints."""
    hts = llevant.load(OPENMC_EXAMPLES / "fusion_hts_blanket.xml")

    pts = np.array([[0.0, 0.0, 0.0], [0.0, 62.0, 0.0], [0.0, 95.0, 0.0], [160.0, 0.0, 0.0]])
    cls = hts.classify(pts)
    assert list(cls.material) == [0, 1, 3, 7]
    assert hts.locate(0.0, 62.0, 0.0).material == 1

    segs = hts.trace((-250.0, 0.0, 0.0), (250.0, 0.0, 0.0))
    assert sum(s.length for s in segs) == pytest.approx(500.0)
    assert len(segs) >= 5

    sl = hts.slice(z=0.0, width=200, colour="realistic")
    assert sl.png()[:8] == b"\x89PNG\r\n\x1a\n"

    # Cell label filtering and ghost shells from OpenMC cell `name="..."` attributes.
    lo, hi = hts.bbox('label "Tungsten"')
    assert lo[0] < -60.0 and hi[0] > 60.0

    img = hts.render(
        size=(96, 96),
        colour="realistic",
        hide_rules=['label "Cryostat"'],
        ghost=['label "Tungsten"'],
    )
    assert img.rgb.shape == (96, 96, 3)
    assert (np.asarray(img.rgb) != 255).any()

    fp = hts.fingerprints()
    assert len(fp.cells) == len(hts.cells)
    diff = hts.diff(hts.reload())
    assert not diff and diff.cells.unchanged == len(hts.cells)

    vol_groups = hts.volumes(12.0).by("material")
    assert any(g.number == 3 and g.volume > 0.0 for g in vol_groups)

    out_png = tmp_path / "hts.png"
    img.save(out_png)
    assert out_png.read_bytes()[:4] == b"\x89PNG"


def test_openmc_from_text_with_label_query():
    """Compile an in-memory OpenMC XML string via `from_text` and resolve `label` rules."""
    xml = """<?xml version='1.0' encoding='utf-8'?>
<model>
  <materials>
    <material id="1" name="Tungsten Sphere">
      <density units="g/cm3" value="19.25"/>
      <nuclide ao="1.0" name="W184"/>
    </material>
  </materials>
  <geometry>
    <surface id="1" type="sphere" coeffs="0 0 0 4.0"/>
    <surface id="2" type="sphere" coeffs="0 0 0 10.0" boundary="vacuum"/>
    <cell id="5" material="1" region="-1" name="Tungsten Core"/>
    <cell id="6" material="void" region="1 -2" name="Outer Cavity"/>
  </geometry>
</model>
"""
    m = llevant.from_text(xml)
    assert m.locate(0.0, 0.0, 0.0).material == 1
    assert {c for c, _, _ in m.instance_boxes('label "tungsten"')} == {5}
    assert {c for c, _, _ in m.instance_boxes('label "outer cavity"')} == {6}


def test_vver_hex_lattice_matches_openmc_element_centers():
    """Cross-check every hexagonal ring position in `vver_hex_assembly.xml` against OpenMC element geometry."""
    import xml.etree.ElementTree as ET

    openmc = pytest.importorskip("openmc")
    xml_path = OPENMC_EXAMPLES / "vver_hex_assembly.xml"
    m = llevant.load(xml_path)

    root = ET.parse(xml_path).getroot()
    mats = openmc.Materials.from_xml_element(root.find("materials"))
    geom = openmc.Geometry.from_xml_element(root.find("geometry"), mats)
    hex_lats = [lat for lat in geom.get_all_lattices().values() if isinstance(lat, openmc.HexLattice)]
    assert len(hex_lats) == 1
    lat = hex_lats[0]
    p = float(lat.pitch[0])

    # Test axial coordinate centers for y-oriented 3-ring lattice: a=(sqrt(3)/2*P, 0.5*P), b=(0, P)
    a = np.array([math.sqrt(3.0) * 0.5 * p, 0.5 * p, 0.0])
    b = np.array([0.0, p, 0.0])
    sample_pts = []
    expected_mats = []
    for i1 in range(-2, 3):
        for i2 in range(-2, 3):
            if max(abs(i1), abs(i2), abs(-i1 - i2)) < lat.num_rings:
                pt = i1 * a + i2 * b
                sample_pts.append(pt)
                idx, _ = lat.find_element(pt)
                univ = lat.get_universe(idx)
                # Center pin (univ 3) is B4C (mat 6); burnable poison (univ 2) is Gd2O3 (mat 4); fuel (univ 1) is UO2 (mat 1).
                expected_mats.append({1: 1, 2: 4, 3: 6}[univ.id])

    res = m.classify(np.array(sample_pts))
    assert list(res.material) == expected_mats


def test_openmc_rotated_and_translated_cell_matches_oracle(tmp_path: Path):
    """Cross-check Euler-angle and 3x3-matrix rotated/translated OpenMC cells against OpenMC."""
    openmc = pytest.importorskip("openmc")
    openmc.reset_auto_ids()

    m1 = openmc.Material(1, name="Fuel")
    m1.add_nuclide("U235", 1.0)
    m1.set_density("g/cm3", 10.0)
    m2 = openmc.Material(2, name="Water")
    m2.add_nuclide("H1", 2.0)
    m2.add_nuclide("O16", 1.0)
    m2.set_density("g/cm3", 1.0)
    mats = openmc.Materials([m1, m2])

    s_pin = openmc.Sphere(surface_id=1, x0=3.0, y0=0.0, z0=0.0, r=1.5)
    c1 = openmc.Cell(cell_id=1, fill=m1, region=-s_pin)
    c2 = openmc.Cell(cell_id=2, fill=m2, region=+s_pin)
    u_child = openmc.Universe(universe_id=2, cells=[c1, c2])

    s_out = openmc.Sphere(surface_id=10, r=20.0, boundary_type="vacuum")
    c_root = openmc.Cell(cell_id=10, fill=u_child, region=-s_out)
    c_root.translation = (1.0, 2.0, 3.0)
    c_root.rotation = (30.0, 45.0, 60.0)
    geom = openmc.Geometry([c_root])
    xml_path = tmp_path / "rotated.xml"
    openmc.Model(geom, mats).export_to_model_xml(str(xml_path))

    lm = llevant.load(xml_path)
    assert lm.diagnostics == []

    rng = np.random.default_rng(42)
    pts = rng.uniform(-12.0, 12.0, size=(400, 3))
    r_mat = c_root.rotation_matrix
    t_vec = np.array(c_root.translation)
    cls = lm.classify(pts)
    for k, p in enumerate(pts):
        if np.linalg.norm(p) > 19.8:
            continue
        r_loc = r_mat @ (p - t_vec)
        expected = 1 if np.linalg.norm(r_loc - np.array([3.0, 0.0, 0.0])) < 1.5 else 2
        assert cls.material[k] == expected, f"point {p} mismatch"

    # Explicit 9-entry rotation matrix in XML also matches OpenMC
    xml_mat = f"""<model>
  <materials>
    <material id="1"><density units="g/cm3" value="10.0"/><nuclide ao="1.0" name="U235"/></material>
    <material id="2"><density units="g/cm3" value="1.0"/><nuclide ao="1.0" name="H1"/></material>
  </materials>
  <geometry>
    <surface id="1" type="sphere" coeffs="3 0 0 1.5"/>
    <surface id="10" type="sphere" coeffs="0 0 0 20" boundary="vacuum"/>
    <cell id="1" universe="2" material="1" region="-1"/>
    <cell id="2" universe="2" material="2" region="1"/>
    <cell id="10" fill="2" region="-10" translation="1 2 3" rotation="0 1 0 -1 0 0 0 0 1"/>
  </geometry>
</model>"""
    m_mat = llevant.from_text(xml_mat)
    assert m_mat.locate(1.0, 5.0, 3.0).material == 1
    assert m_mat.locate(1.0, -1.0, 3.0).material == 2


def test_openmc_rotated_translated_lattices_match_oracle(tmp_path: Path):
    """Cross-check rotated & translated OpenMC RectLattice and HexLattice against OpenMC."""
    openmc = pytest.importorskip("openmc")
    openmc.reset_auto_ids()

    m1 = openmc.Material(1, name="Fuel")
    m1.add_nuclide("U235", 1.0)
    m1.set_density("g/cm3", 10.0)
    m2 = openmc.Material(2, name="Water")
    m2.add_nuclide("H1", 2.0)
    m2.add_nuclide("O16", 1.0)
    m2.set_density("g/cm3", 1.0)
    m3 = openmc.Material(3, name="Absorber")
    m3.add_nuclide("B10", 1.0)
    m3.set_density("g/cm3", 2.5)
    mats = openmc.Materials([m1, m2, m3])

    cyl = openmc.ZCylinder(r=0.4)
    u1 = openmc.Universe(cells=[openmc.Cell(fill=m1, region=-cyl), openmc.Cell(fill=m2, region=+cyl)])
    u3 = openmc.Universe(cells=[openmc.Cell(fill=m3, region=-cyl), openmc.Cell(fill=m2, region=+cyl)])
    u_out = openmc.Universe(cells=[openmc.Cell(fill=m2)])

    rlat = openmc.RectLattice()
    rlat.lower_left = (-2.0, -2.0)
    rlat.pitch = (2.0, 2.0)
    rlat.outer = u_out
    rlat.universes = [[u1, u3], [u3, u1]]

    hlat = openmc.HexLattice()
    hlat.center = (0.0, 0.0)
    hlat.pitch = (2.0,)
    hlat.outer = u_out
    hlat.orientation = "y"
    hlat.universes = [[u1] * 6, [u3]]

    s_rect = openmc.Sphere(r=4.0)
    s_hex = openmc.Sphere(r=4.0)
    s_world = openmc.Sphere(r=20.0, boundary_type="vacuum")
    x_mid = openmc.XPlane(0.0)

    c_rect = openmc.Cell(
        fill=openmc.Universe(
            cells=[openmc.Cell(fill=rlat, region=-s_rect), openmc.Cell(fill=m2, region=+s_rect)]
        ),
        region=-s_world & -x_mid,
    )
    c_rect.translation = (-6.0, 0.0, 0.0)
    c_rect.rotation = (0.0, 0.0, 25.0)

    c_hex = openmc.Cell(
        fill=openmc.Universe(
            cells=[openmc.Cell(fill=hlat, region=-s_hex), openmc.Cell(fill=m2, region=+s_hex)]
        ),
        region=-s_world & +x_mid,
    )
    c_hex.translation = (6.0, 0.0, 0.0)
    c_hex.rotation = (0.0, 0.0, 35.0)

    geom = openmc.Geometry([c_rect, c_hex])
    xml_path = tmp_path / "rot_lattices.xml"
    openmc.Model(geom, mats).export_to_model_xml(str(xml_path))

    lm = llevant.load(xml_path)
    assert lm.diagnostics == []

    # Check transformed element centers: world = t + R^T * local
    rot_r = c_rect.rotation_matrix
    assert lm.locate(*((-6.0, 0.0, 0.0) + rot_r.T @ np.array([-1.0, -1.0, 0.0]))).material == 3
    assert lm.locate(*((-6.0, 0.0, 0.0) + rot_r.T @ np.array([1.0, -1.0, 0.0]))).material == 1

    rot_h = c_hex.rotation_matrix
    assert lm.locate(*((6.0, 0.0, 0.0) + rot_h.T @ np.array([0.0, 0.0, 0.0]))).material == 3
    assert lm.locate(*((6.0, 0.0, 0.0) + rot_h.T @ np.array([0.0, 2.0, 0.0]))).material == 1


def test_openmc_cones_quadrics_and_tori_match_oracle(tmp_path: Path):
    """Cross-check ZCone, ZTorus, Quadric, and mixed reflective/vacuum boundary cells against OpenMC."""
    openmc = pytest.importorskip("openmc")
    openmc.reset_auto_ids()

    m1 = openmc.Material(1)
    m1.add_nuclide("Fe56", 1.0)
    m1.set_density("g/cm3", 7.8)
    m2 = openmc.Material(2)
    m2.add_nuclide("H1", 1.0)
    m2.set_density("g/cm3", 1.0)
    mats = openmc.Materials([m1, m2])

    s_cone = openmc.ZCone(surface_id=1, x0=1.0, y0=-2.0, z0=0.5, r2=0.25)
    s_tor = openmc.ZTorus(surface_id=2, x0=0.0, y0=0.0, z0=0.0, a=8.0, b=1.5, c=2.0)
    s_quad = openmc.Quadric(
        surface_id=3, a=1.0, b=2.0, c=0.5, d=0.2, e=-0.1, f=0.3, g=-1.0, h=0.5, j=-0.2, k=-9.0
    )
    px0 = openmc.XPlane(-15.0, boundary_type="reflective")
    px1 = openmc.XPlane(15.0, boundary_type="reflective")
    py0 = openmc.YPlane(-15.0, boundary_type="reflective")
    py1 = openmc.YPlane(15.0, boundary_type="reflective")
    pz0 = openmc.ZPlane(-15.0, boundary_type="vacuum")
    pz1 = openmc.ZPlane(15.0, boundary_type="vacuum")
    box = +px0 & -px1 & +py0 & -py1 & +pz0 & -pz1

    c1 = openmc.Cell(cell_id=1, fill=m1, region=-s_tor & box)
    c2 = openmc.Cell(cell_id=2, fill=m1, region=+s_tor & -s_quad & box)
    c3 = openmc.Cell(cell_id=3, fill=m1, region=+s_tor & +s_quad & -s_cone & box)
    c4 = openmc.Cell(cell_id=4, fill=m2, region=+s_tor & +s_quad & +s_cone & box)

    xml_path = tmp_path / "surfaces.xml"
    openmc.Model(openmc.Geometry([c1, c2, c3, c4]), mats).export_to_model_xml(str(xml_path))
    lm = llevant.load(xml_path)
    assert lm.diagnostics == []
    assert lm.check(3.0) == []

    rng = np.random.default_rng(7)
    pts = rng.uniform(-11.0, 11.0, size=(500, 3))
    cls = lm.classify(pts)
    for k, p in enumerate(pts):
        if s_tor.evaluate(p) < 0:
            exp = 1
        elif s_quad.evaluate(p) < 0:
            exp = 2
        elif s_cone.evaluate(p) < 0:
            exp = 3
        else:
            exp = 4
        assert cls.cell[k] == exp


def test_openmc_split_geometry_and_materials_directory(tmp_path: Path):
    """Load an OpenMC model directory containing separate `geometry.xml` and `materials.xml`."""
    openmc = pytest.importorskip("openmc")
    openmc.reset_auto_ids()

    m1 = openmc.Material(5, name="Copper")
    m1.add_element("Cu", 1.0)
    m1.set_density("g/cm3", 8.96)
    mats = openmc.Materials([m1])

    s = openmc.Sphere(r=6.0, boundary_type="vacuum")
    geom = openmc.Geometry([openmc.Cell(cell_id=12, fill=m1, region=-s, name="Copper Sphere")])

    mats.export_to_xml(str(tmp_path / "materials.xml"))
    geom.export_to_xml(str(tmp_path / "geometry.xml"))

    lm = llevant.load(tmp_path)
    assert lm.diagnostics == []
    loc = lm.locate(0.0, 0.0, 0.0)
    assert loc.material == 5
    assert loc.density == pytest.approx(-8.96)


def test_read_openmc_tally_cell_material_and_meshes(tmp_path: Path):
    """Load OpenMC HDF5 statepoint cell, material, regular, rectilinear, and cylindrical mesh tallies."""
    h5py = pytest.importorskip("h5py")

    xml = """<model>
  <materials>
    <material id="1" name="Fuel"><density units="g/cm3" value="10.0"/><nuclide ao="1.0" name="U235"/></material>
    <material id="2" name="Moderator"><density units="g/cm3" value="1.0"/><nuclide ao="2.0" name="H1"/></material>
  </materials>
  <geometry>
    <surface id="1" type="sphere" coeffs="0 0 0 4.0"/>
    <surface id="2" type="sphere" coeffs="0 0 0 10.0" boundary="vacuum"/>
    <cell id="10" material="1" region="-1" name="Core"/>
    <cell id="20" material="2" region="1 -2" name="Reflector"/>
  </geometry>
</model>"""
    model = llevant.from_text(xml)

    sp_path = tmp_path / "statepoint.10.h5"
    with h5py.File(sp_path, "w") as f:
        tgrp = f.create_group("tallies")
        tgrp.attrs["ids"] = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        fgrp = tgrp.create_group("filters")
        mgrp = tgrp.create_group("meshes")

        # Mesh 1: Regular 2x3x4
        m1 = mgrp.create_group("mesh 1")
        m1.create_dataset("type", data=b"regular")
        m1.create_dataset("dimension", data=np.array([2, 3, 4], dtype=np.int32))
        m1.create_dataset("lower_left", data=np.array([0.0, 0.0, 0.0]))
        m1.create_dataset("upper_right", data=np.array([2.0, 6.0, 12.0]))

        # Mesh 2: Rectilinear 2x2x1
        m2 = mgrp.create_group("mesh 2")
        m2.create_dataset("type", data=b"rectilinear")
        m2.create_dataset("x_grid", data=np.array([-4.0, 0.0, 4.0]))
        m2.create_dataset("y_grid", data=np.array([-4.0, 0.0, 4.0]))
        m2.create_dataset("z_grid", data=np.array([-5.0, 5.0]))

        # Mesh 3: Cylindrical nr=2, nphi=2, nz=2 with origin (1, 2, 3)
        m3 = mgrp.create_group("mesh 3")
        m3.create_dataset("type", data=b"cylindrical")
        m3.create_dataset("r_grid", data=np.array([0.0, 1.0, 2.0]))
        m3.create_dataset("phi_grid", data=np.array([0.0, math.pi, 2.0 * math.pi]))
        m3.create_dataset("z_grid", data=np.array([-5.0, 0.0, 5.0]))
        m3.create_dataset("origin", data=np.array([1.0, 2.0, 3.0]))

        # Filters
        f1 = fgrp.create_group("filter 1")
        f1.create_dataset("type", data=b"cell")
        f1.create_dataset("n_bins", data=2)
        f1.create_dataset("bins", data=np.array([10, 20], dtype=np.int32))

        f2 = fgrp.create_group("filter 2")
        f2.create_dataset("type", data=b"material")
        f2.create_dataset("n_bins", data=2)
        f2.create_dataset("bins", data=np.array([1, 2], dtype=np.int32))

        f3 = fgrp.create_group("filter 3")
        f3.create_dataset("type", data=b"mesh")
        f3.create_dataset("bins", data=1)

        f4 = fgrp.create_group("filter 4")
        f4.create_dataset("type", data=b"mesh")
        f4.create_dataset("bins", data=2)

        f5 = fgrp.create_group("filter 5")
        f5.create_dataset("type", data=b"mesh")
        f5.create_dataset("bins", data=3)

        # Tally 1: cell filter, 2 scores, n_realizations = 4
        t1 = tgrp.create_group("tally 1")
        t1.create_dataset("name", data=b"cell_flux")
        t1.create_dataset("n_realizations", data=4)
        t1.create_dataset("filters", data=np.array([1], dtype=np.int32))
        t1.create_dataset("nuclides", data=np.array([b"total"]))
        t1.create_dataset("score_bins", data=np.array([b"flux", b"heating"]))
        # shape (2 filter bins, 1*2 scores, 3): sum, sum_sq, unused
        res1 = np.zeros((2, 2, 3))
        res1[0, 0, :2] = [40.0, 412.0]   # bin 10 flux: mean=10, var=(103-100)/3=1, std=1, rel=0.1
        res1[1, 0, :2] = [80.0, 1648.0]  # bin 20 flux: mean=20, std=2, rel=0.1
        res1[0, 1, :2] = [12.0, 36.0]
        res1[1, 1, :2] = [24.0, 144.0]
        t1.create_dataset("results", data=res1)

        # Tally 2: material filter
        t2 = tgrp.create_group("tally 2")
        t2.create_dataset("n_realizations", data=2)
        t2.create_dataset("filters", data=np.array([2], dtype=np.int32))
        t2.create_dataset("score_bins", data=np.array([b"fission"]))
        res2 = np.array([[[10.0, 50.0, 0.0]], [[30.0, 450.0, 0.0]]])
        t2.create_dataset("results", data=res2)

        # Tally 3: Regular mesh (2, 3, 4) in Fortran order -> value(ix, iy, iz) = ix*100 + iy*10 + iz
        t3 = tgrp.create_group("tally 3")
        t3.create_dataset("n_realizations", data=1)
        t3.create_dataset("filters", data=np.array([3], dtype=np.int32))
        t3.create_dataset("score_bins", data=np.array([b"flux"]))
        reg_vals = np.array(
            [ix * 100.0 + iy * 10.0 + iz for ix in range(2) for iy in range(3) for iz in range(4)]
        ).reshape((2, 3, 4))
        reg_flat = reg_vals.ravel(order="F")
        res3 = np.stack([reg_flat, reg_flat * reg_flat, np.zeros_like(reg_flat)], axis=1)[:, None, :]
        t3.create_dataset("results", data=res3)

        # Tally 4: Rectilinear mesh (2, 2, 1) in Fortran order
        t4 = tgrp.create_group("tally 4")
        t4.create_dataset("n_realizations", data=1)
        t4.create_dataset("filters", data=np.array([4], dtype=np.int32))
        t4.create_dataset("score_bins", data=np.array([b"flux"]))
        rect_flat = np.array([1.0, 2.0, 3.0, 4.0])  # (0,0,0), (1,0,0), (0,1,0), (1,1,0)
        res4 = np.stack([rect_flat, rect_flat, np.zeros_like(rect_flat)], axis=1)[:, None, :]
        t4.create_dataset("results", data=res4)

        # Tally 5: Cylindrical mesh (nr=2, nphi=2, nz=2) in Fortran order
        t5 = tgrp.create_group("tally 5")
        t5.create_dataset("n_realizations", data=1)
        t5.create_dataset("filters", data=np.array([5], dtype=np.int32))
        t5.create_dataset("score_bins", data=np.array([b"flux"]))
        cyl_vals = np.array(
            [ir * 100.0 + iphi * 10.0 + iz for ir in range(2) for iphi in range(2) for iz in range(2)]
        ).reshape((2, 2, 2))
        cyl_flat = cyl_vals.ravel(order="F")
        res5 = np.stack([cyl_flat, cyl_flat, np.zeros_like(cyl_flat)], axis=1)[:, None, :]
        t5.create_dataset("results", data=res5)

    # 1. Cell tally (default + named score + rel error + slice render)
    f_cell = llevant.read_openmc_tally(model, sp_path, tally=1, score="flux", unit="n/cm2-s")
    assert f_cell.kind == "cells"
    assert f_cell.unit == "n/cm2-s"
    assert f_cell.value(0.0, 0.0, 0.0) == pytest.approx(10.0)
    assert f_cell.value(6.0, 0.0, 0.0) == pytest.approx(20.0)

    f_heat = llevant.read_openmc_tally(model, sp_path, tally=1, score="heating")
    assert f_heat.value(0.0, 0.0, 0.0) == pytest.approx(3.0)

    f_err = llevant.read_openmc_tally(model, sp_path, tally=1, score=0, error=True)
    assert f_err.value(0.0, 0.0, 0.0) == pytest.approx(0.1)
    assert f_err.value(6.0, 0.0, 0.0) == pytest.approx(0.1)

    img = model.slice(z=0.0, width=64, field=f_cell)
    assert img.png()[:4] == b"\x89PNG"

    # 2. Material tally
    f_mat = llevant.read_openmc_tally(model, sp_path, tally=2)
    assert f_mat.value(0.0, 0.0, 0.0) == pytest.approx(5.0)
    assert f_mat.value(6.0, 0.0, 0.0) == pytest.approx(15.0)

    # 3. Regular mesh (Fortran order -> C order)
    f_reg = llevant.read_openmc_tally(model, sp_path, tally=3)
    assert f_reg.kind == "mesh"
    assert f_reg.value(1.5, 5.0, 10.5) == pytest.approx(123.0)  # ix=1, iy=2, iz=3
    assert f_reg.value(1.5, 1.0, 1.5) == pytest.approx(100.0)   # ix=1, iy=0, iz=0

    # 4. Rectilinear mesh
    f_rect = llevant.read_openmc_tally(model, sp_path, tally=4)
    assert f_rect.value(-2.0, -2.0, 0.0) == pytest.approx(1.0)
    assert f_rect.value(2.0, -2.0, 0.0) == pytest.approx(2.0)
    assert f_rect.value(-2.0, 2.0, 0.0) == pytest.approx(3.0)
    assert f_rect.value(2.0, 2.0, 0.0) == pytest.approx(4.0)

    # 5. Cylindrical mesh (r, phi, z Fortran -> r, z, theta C, with origin (1, 2, 3))
    f_cyl = llevant.read_openmc_tally(model, sp_path, tally=5)
    # origin + (0, +1.5, -2.5): ir=1, iphi=0 (phi=90 deg), iz=0 -> 100
    assert f_cyl.value(1.0, 3.5, 0.5) == pytest.approx(100.0)
    # origin + (0, -1.5, +2.5): ir=1, iphi=1 (phi=270 deg), iz=1 -> 111
    assert f_cyl.value(1.0, 0.5, 5.5) == pytest.approx(111.0)

