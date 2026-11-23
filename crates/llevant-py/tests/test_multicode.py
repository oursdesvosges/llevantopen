"""Python API tests for TRIPOLI-4, DRAGON5, to_mcnp translation, masses, and lost particles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import llevant

DOCS_EXAMPLES = Path(__file__).resolve().parents[3] / "docs" / "examples"


def test_tripoli4_and_dragon5_load_and_check_cleanly():
    """Load native TRIPOLI-4 and DRAGON5 decks directly through llevant.load."""
    t4_pin = llevant.load(DOCS_EXAMPLES / "tripoli4" / "pwr_pin_cell.t4")
    assert t4_pin.diagnostics == []
    assert t4_pin.locate(0.0, 0.0, 0.0).material == 1
    assert t4_pin.locate(0.44, 0.0, 0.0).material == 2
    assert t4_pin.locate(0.55, 0.0, 0.0).material == 3
    assert t4_pin.check(0.25) == []

    t4_shield = llevant.load(DOCS_EXAMPLES / "tripoli4" / "fusion_shield_box.t4")
    assert t4_shield.diagnostics == []
    assert t4_shield.locate(0.0, 0.0, 0.0).material == 3
    assert t4_shield.locate(0.0, 0.0, 30.0).material == 0
    assert t4_shield.check(5.0) == []

    d5 = llevant.load(DOCS_EXAMPLES / "dragon5" / "pwr_3x3_assembly.x2m")
    assert d5.diagnostics == []
    assert d5.info()["lattices"] == 1
    # Pin pitch 1.26 cm in [0, 3.78]^2: (0.63, 0.63) is pin [0,0] (fuel mix 1), (1.89, 1.89) is guide tube (mix 2).
    assert d5.locate(0.63, 0.63, 0.0).material == 1
    assert d5.locate(1.89, 1.89, 0.0).material == 2
    assert d5.check(0.25) == []

    d5_split = llevant.load(DOCS_EXAMPLES / "dragon5" / "depletion_pin_splitr.x2m")
    assert d5_split.diagnostics == []
    assert d5_split.locate(0.63, 0.63, 2.5).material == 1
    assert d5_split.locate(0.63 + 0.44, 0.63, 2.5).material == 4
    assert d5_split.check(0.25) == []

    omc_dist = llevant.load(DOCS_EXAMPLES / "openmc" / "distributed_pin_plots.xml")
    assert omc_dist.diagnostics == []
    assert len(llevant.openmc_plots(DOCS_EXAMPLES / "openmc" / "distributed_pin_plots.xml")) == 2
    assert omc_dist.check(0.25) == []


def test_to_mcnp_translation_path_text_and_output(tmp_path: Path):
    """to_mcnp converts OpenMC XML, TRIPOLI-4, and DRAGON5 files or raw text to valid MCNP."""
    t4_path = DOCS_EXAMPLES / "tripoli4" / "pwr_pin_cell.t4"
    out_path = tmp_path / "translated.i"
    mcnp_from_path = llevant.to_mcnp(t4_path, output=out_path)
    assert out_path.is_file()
    assert out_path.read_text(encoding="utf-8") == mcnp_from_path
    assert "imp:n=1" in mcnp_from_path and "imp:n=0" in mcnp_from_path

    reloaded = llevant.load(out_path)
    assert reloaded.diagnostics == []
    assert reloaded.locate(0.0, 0.0, 0.0).material == 1

    d5_text = (DOCS_EXAMPLES / "dragon5" / "pwr_3x3_assembly.x2m").read_text(encoding="utf-8")
    mcnp_d5 = llevant.to_mcnp(d5_text)
    assert "lat=1" in mcnp_d5 and llevant.from_text(mcnp_d5).diagnostics == []

    t4_inline = "GEOMETRY\n  SURF 1 SPHERE 0 0 0 8\n  TYPE 1 COMBI 1 -1\nENDG\nEND\n"
    m_inline = llevant.from_text(t4_inline)
    assert m_inline.locate(0.0, 0.0, 0.0).is_inside
    assert m_inline.cells[m_inline.locate(20.0, 0.0, 0.0).cell].imp_zero


def test_group_mass_on_volumes_and_volumes_to():
    """Group.mass and Group.mass_sigma are populated on volumes(), volumes_to(), and cell_volumes()."""
    m = llevant.load(DOCS_EXAMPLES / "tripoli4" / "pwr_pin_cell.t4")
    by_mat = {g.number: g for g in m.volumes(0.2).by("material")}
    assert by_mat[1].mass is not None and by_mat[1].mass > 0.0
    assert by_mat[1].mass_sigma is not None and by_mat[1].mass_sigma >= 0.0

    refined = {g.number: g for g in m.volumes_to(0.2, 0.05, by="material")}
    assert refined[1].mass is not None and refined[1].mass > 0.0

    by_cells = {g.number: g for g in m.cell_volumes([3], 0.2, 0.05)}
    assert by_cells[3].mass is not None
    expected = (np.pi * (0.41**2) * 20.0) * 10.4
    assert by_cells[3].mass == pytest.approx(expected, rel=0.05)


def test_diagnose_lost_particles_from_points_and_logs(tmp_path: Path):
    """diagnose_lost_particles explains overlaps, gaps, and log coordinates."""
    mcnp = """lost test
1 1 -7.8 -1 imp:n=1
2 2 -1.0 -2 imp:n=1
3 0 3 imp:n=0

1 rpp -2 2 -2 2 -2 2
2 rpp 1 5 -2 2 -2 2
3 rpp -10 10 -10 10 -10 10

mode n
m1 26000 1.0
m2 1001 2.0 8016 1.0
"""
    m = llevant.from_text(mcnp)

    single = llevant.diagnose_lost_particles(m, (1.5, 0.0, 0.0), radius=1.0)
    assert len(single) == 1
    assert single[0]["status"] == "overlap"
    assert single[0]["involved"] == [1, 2]
    assert "overlap" in single[0]["diagnosis"]

    log_file = tmp_path / "mcnp_openmc.log"
    log_file.write_text(
        " 1  x, y, z =  -4.0  0.0  0.0\nParticle 2 lost at (1.5, 0.0, 0.0)\n0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    lost_plots = tmp_path / "lost_plots"
    from_log = llevant.diagnose_lost_particles(m, log_file, radius=1.0, plots=lost_plots, px=64)
    assert [r["status"] for r in from_log] == ["gap", "overlap", "inside"]
    assert len(from_log[0]["plots"]) == 3
    for row in from_log:
        for fname in row["plots"]:
            assert (lost_plots / fname).read_bytes()[:4] == b"\x89PNG"


def test_dedicated_tripoli_and_dragon_wrappers(tmp_path: Path):
    """Test tripoli_to_mcnp, from_tripoli, dragon_to_mcnp, and from_dragon."""
    t4_path = DOCS_EXAMPLES / "tripoli4" / "pwr_pin_cell.t4"
    t4_out = tmp_path / "t4.mcnp"
    t4_mcnp = llevant.tripoli_to_mcnp(t4_path, output=t4_out)
    assert t4_out.is_file() and "imp:n=1" in t4_mcnp
    m_t4 = llevant.from_tripoli(t4_path)
    assert m_t4.locate(0.0, 0.0, 0.0).material == 1
    with pytest.raises(ValueError):
        llevant.tripoli_to_mcnp("invalid not a tripoli deck")

    d5_path = DOCS_EXAMPLES / "dragon5" / "pwr_3x3_assembly.x2m"
    d5_out = tmp_path / "d5.mcnp"
    d5_mcnp = llevant.dragon_to_mcnp(d5_path, output=d5_out)
    assert d5_out.is_file() and "lat=1" in d5_mcnp
    m_d5 = llevant.from_dragon(d5_path)
    assert m_d5.locate(0.63, 0.63, 0.0).material == 1
    assert m_d5.locate(1.89, 1.89, 0.0).material == 2
    with pytest.raises(ValueError, match="MIX"):
        llevant.dragon_to_mcnp("PIN := GEO: :: CARCEL 1 MESHX -0.5 0.5 MESHY -0.5 0.5 RADIUS 0.0 0.4 MIX 1 2 99 ;")


def test_openmc_plots_check_report_and_mass_balance(tmp_path: Path):
    """Test openmc_plots, render_openmc_plots, check_report, and mass_balance."""
    hts_path = DOCS_EXAMPLES / "openmc" / "fusion_hts_blanket.xml"
    plots_xml = tmp_path / "plots.xml"
    plots_xml.write_text(
        """<?xml version="1.0"?>
<plots>
  <plot id="1" type="slice" basis="xy" color_by="material" filename="py_hts_xy">
    <origin>0 0 0</origin>
    <width>400 400</width>
    <pixels>64 64</pixels>
  </plot>
  <plot id="2" type="slice" basis="xz" color_by="cell" filename="py_hts_xz.png">
    <origin>0 0 0</origin>
    <width>400 300</width>
    <pixels>64 48</pixels>
  </plot>
</plots>
""",
        encoding="utf-8",
    )
    parsed = llevant.openmc_plots(hts_path, plots_xml=plots_xml)
    assert [p["id"] for p in parsed] == [1, 2]
    assert parsed[0]["basis"] == "xy"

    plot_out = tmp_path / "rendered_plots"
    written = llevant.render_openmc_plots(
        hts_path, plot_out, plots_xml=plots_xml, outlines=True
    )
    assert [p.name for p in written] == ["py_hts_xy.png", "py_hts_xz.png"]
    for p in written:
        assert p.read_bytes()[:4] == b"\x89PNG"

    # Also test rendering directly from a compiled Model with ids filter.
    hts_model = llevant.load(hts_path)
    written_one = llevant.render_openmc_plots(
        hts_model, tmp_path / "one_plot", plots_xml=plots_xml, ids=[2], px=48
    )
    assert [p.name for p in written_one] == ["py_hts_xz.png"]

    # check_report on a geometry with an overlap + on a clean model.
    overlap_model = llevant.from_text(
        """overlap qa
1 1 -7.8 -1 imp:n=1
2 2 -1.0 -2 imp:n=1
3 0 3 imp:n=0

1 rpp -2 2 -2 2 -2 2
2 rpp 1 5 -2 2 -2 2
3 rpp -10 10 -10 10 -10 10

mode n
m1 26000 1.0
m2 1001 2.0 8016 1.0
"""
    )
    qa_dir = tmp_path / "qa_overlap"
    report = llevant.check_report(overlap_model, 0.5, qa_dir, px=64)
    assert len(report["defects"]) >= 1
    assert (qa_dir / "report.json").is_file()
    md = (qa_dir / "report.md").read_text(encoding="utf-8")
    assert "defect_001_xy.png" in md
    assert len(report["defects"][0]["plots"]) == 3
    for fname in report["defects"][0]["plots"]:
        assert (qa_dir / fname).read_bytes()[:4] == b"\x89PNG"

    clean_model = llevant.from_tripoli(DOCS_EXAMPLES / "tripoli4" / "pwr_pin_cell.t4")
    clean_qa = llevant.check_report(clean_model, 0.25, tmp_path / "qa_clean", px=48)
    assert clean_qa["defects"] == []
    assert "No gaps or overlaps found." in Path(clean_qa["report_md"]).read_text(encoding="utf-8")

    # mass_balance summary
    mb = llevant.mass_balance(clean_model, 0.2, by="material")
    assert mb["total_mass_g"] > 0.0
    assert mb["total_mass_kg"] == pytest.approx(mb["total_mass_g"] / 1000.0)
    assert mb["total_mass_sigma_g"] >= 0.0
    assert any(g["number"] == 1 and (g["mass_g"] or 0.0) > 0.0 for g in mb["groups"])
