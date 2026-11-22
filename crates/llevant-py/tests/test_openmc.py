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
