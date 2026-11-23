# %% [markdown]
# # Llevant Use-Case Cookbook
#
# Concise, self-contained examples covering every major neutronics workflow:
# multi-code loading & translation (OpenMC XML, TRIPOLI-4, DRAGON5 GEO:),
# geometry QA & defect plots, lost-particle triage, stochastic volumes & masses,
# OpenMC statepoint filter slicing, distributed materials & plots, ray tracing,
# structural fingerprinting/diffing, PBR cutaways & ghost shells, and the Web Studio API.
#
# Run all use cases or select one:
#   python crates/llevant-py/examples/use_cases.py --list
#   python crates/llevant-py/examples/use_cases.py --case all --out /tmp/llevant_examples
#   python crates/llevant-py/examples/use_cases.py --case openmc_statepoint_tallies

# %%
from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import llevant
import numpy as np

REPO = Path(__file__).resolve().parents[3]
EXAMPLES = REPO / "docs" / "examples"
WORKSPACE = REPO.parent
NOTEBOOK = "ipykernel" in sys.modules


def save_image(img: llevant.Image, out: Path, name: str) -> Path:
    """Display inline in a notebook or write `<out>/<name>.png` in script mode."""
    if NOTEBOOK:
        from IPython.display import display

        display(img)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{name}.png"
    img.save(target)
    print(f"  wrote {target}")
    return target


# %% [markdown]
# ## 1. Multi-code loading & MCNP conversion


# %%
def case_multicode_load_and_convert(out: Path) -> None:
    """Load OpenMC XML, TRIPOLI-4, and DRAGON5 SPLITR/SPLITZ inputs natively and export MCNP."""
    models = {
        "openmc": EXAMPLES / "openmc" / "pwr_pin_lattice.xml",
        "tripoli4": EXAMPLES / "tripoli4" / "pwr_pin_cell.t4",
        "dragon5_splitr": EXAMPLES / "dragon5" / "depletion_pin_splitr.x2m",
        "dragon5_car2d": EXAMPLES / "dragon5" / "pwr_3x3_assembly.x2m",
    }
    for kind, path in models.items():
        m = llevant.load(path)
        assert not m.diagnostics, f"{kind}: {m.diagnostics}"
        info = m.info()
        print(f"  [{kind}] cells={info['cells']} surfaces={info['surfaces']} lattices={info['lattices']}")

    d5_mcnp = llevant.to_mcnp(models["dragon5_splitr"])
    assert "c DRAGON5 boundaries:" in d5_mcnp
    t4_mcnp = llevant.to_mcnp(models["tripoli4"])
    assert "c TRIPOLI-4 boundary conditions:" in t4_mcnp and "boundary=REFLECTION" in t4_mcnp
    out.mkdir(parents=True, exist_ok=True)
    (out / "depletion_pin_splitr.i").write_text(d5_mcnp, encoding="utf-8")


# %% [markdown]
# ## 2. Geometry QA check & zoomed defect plots


# %%
def case_qa_check_and_plots(out: Path) -> None:
    """Verify a clean fusion blanket and generate defect slice plots for an overlap test."""
    hts = llevant.load(EXAMPLES / "openmc" / "fusion_hts_blanket.xml")
    assert hts.check(h=5.0) == []
    print("  fusion_hts_blanket: 0 defects at h=5.0 cm")

    overlapped = llevant.from_text(
        """qa overlap demo
1 1 -7.8 -1 imp:n=1
2 2 -1.0 -2 imp:n=1
3 0 3 imp:n=0

1 rpp -3 3 -3 3 -3 3
2 rpp 1 6 -3 3 -3 3
3 rpp -20 20 -20 20 -20 20

mode n
m1 26000 1.0
m2 1001 2.0 8016 1.0
"""
    )
    report = llevant.check_report(overlapped, h=0.4, out_dir=out / "qa_overlap", px=240)
    print(f"  overlap defects detected: {len(report['defects'])} -> {report['report_md']}")
    assert len(report["defects"]) >= 1


# %% [markdown]
# ## 3. Lost-particle coordinate & log triage


# %%
def case_lost_particle_triage(out: Path) -> None:
    """Diagnose lost-particle coordinates and pasted log lines with zoomed XY/XZ/YZ slices."""
    m = llevant.from_text(
        """lost particle triage demo
1 1 -7.8 -1 imp:n=1
2 2 -1.0 -2 1 imp:n=1
3 0 2 -3 imp:n=1
4 0 4 imp:n=0

1 so 4
2 so 6
3 so 12
4 so 15

mode n
m1 26000 1.0
m2 1001 2.0 8016 1.0
"""
    )
    log_text = "1 x, y, z = 13.0 0.0 0.0\nParticle 2 lost at (0.0, 0.0, 4.0)\n"
    diags = llevant.diagnose_lost_particles(m, log_text, radius=2.0, plots=out / "lost_triage", px=200)
    for row in diags:
        print(f"  point={row['point']} status={row['status']} -> {row['diagnosis']}")
    assert diags[0]["status"] == "gap"


# %% [markdown]
# ## 4. Stochastic volumes & mass inventory


# %%
def case_volumes_and_mass(out: Path) -> None:
    """Compute stochastic volumes, uncertainties, and masses by material on VVER-1000."""
    _ = out
    m = llevant.load(EXAMPLES / "openmc" / "vver_hex_assembly.xml")
    mb = llevant.mass_balance(m, h=0.8, by="material")
    print(
        f"  total_mass={mb['total_mass_g']:.3e} +/- {mb['total_mass_sigma_g']:.2e} g "
        f"({len(mb['groups'])} groups)"
    )
    for g in mb["groups"][:5]:
        print(f"    m{g['number']:>2}: vol={g['volume_cm3']:.3f} cm3 mass={g['mass_g']} g")
    assert mb["total_mass_g"] > 0.0


# %% [markdown]
# ## 5. OpenMC HDF5 statepoint inspection, filter slicing & CSV export


# %%
def case_openmc_statepoint_tallies(out: Path) -> None:
    """Create a multi-filter OpenMC statepoint, slice by score/energy/error, and overlay on geometry."""
    import h5py

    out.mkdir(parents=True, exist_ok=True)
    sp = out / "demo_statepoint.h5"
    with h5py.File(sp, "w") as f:
        t_grp = f.create_group("tallies")
        t_grp.attrs["ids"] = np.array([101], dtype=np.int32)
        f_cell = t_grp.create_group("filters/filter 1")
        f_cell.create_dataset("type", data=np.bytes_("cell"))
        f_cell.create_dataset("bins", data=np.array([2, 3, 4, 8, 9, 11, 12], dtype=np.int32))
        f_erg = t_grp.create_group("filters/filter 2")
        f_erg.create_dataset("type", data=np.bytes_("energy"))
        f_erg.create_dataset("bins", data=np.array([0.0, 0.625, 2.0e7], dtype=np.float64))

        t101 = t_grp.create_group("tally 101")
        t101.create_dataset("n_realizations", data=np.int32(20))
        t101.create_dataset("filters", data=np.array([1, 2], dtype=np.int32))
        t101.create_dataset("score_bins", data=np.array([b"flux", b"heating"]))
        results = np.zeros((14, 2, 3), dtype=np.float64)
        for ci, val in enumerate([1e14, 5e13, 2e13, 8e12, 2e12, 4e11, 5e10]):
            results[2 * ci + 0, 0, 0] = 0.3 * val * 20.0
            results[2 * ci + 1, 0, 0] = 0.7 * val * 20.0
            results[2 * ci + 1, 1, 0] = 1e-12 * val * 20.0
        t101.create_dataset("results", data=results)

    meta = llevant.openmc_tallies(sp)
    print(f"  statepoint tallies={len(meta)} scores={meta[0]['scores']}")
    hts = llevant.load(EXAMPLES / "openmc" / "fusion_hts_blanket.xml")
    fld = llevant.read_openmc_tally(hts, sp, tally=101, score="heating", energy=1)
    save_image(hts.slice(z=0.0, width=480, field=fld), out, "hts_heating_fast")
    csv_out = llevant.export_openmc_tally_csv(
        hts, sp, out / "hts_heating_fast.csv", tally=101, score="heating", energy="total"
    )
    print(f"  exported tally CSV: {csv_out}")


# %% [markdown]
# ## 6. Distributed material warnings & OpenMC `<plots>` automation


# %%
def case_distributed_materials_and_plots(out: Path) -> None:
    """Inspect OpenMC distributed-material warnings and render XML <plots> specifications."""
    xml_path = EXAMPLES / "openmc" / "distributed_pin_plots.xml"
    mcnp = llevant.to_mcnp(xml_path)
    warnings = [line for line in mcnp.splitlines() if line.startswith("c WARNING:")]
    print("  translated warnings:", warnings)
    assert warnings
    written = llevant.render_openmc_plots(xml_path, out / "openmc_plots", outlines=True)
    print("  rendered plot files:", [p.name for p in written])
    assert len(written) == 2


# %% [markdown]
# ## 7. Analytical ray tracing & material chord-length breakdown


# %%
def case_ray_trace_and_chord_lengths(out: Path) -> None:
    """Trace single and batch rays and accumulate analytical material chord lengths."""
    _ = out
    hts = llevant.load(EXAMPLES / "openmc" / "fusion_hts_blanket.xml")
    segs = hts.trace((-250.0, 0.0, 0.0), (250.0, 0.0, 0.0))
    print(f"  single equatorial ray segments: {len(segs)}")

    lo, hi = hts.extent()
    rng = np.random.default_rng(7)
    starts = rng.uniform(lo, hi, size=(2000, 3))
    ends = rng.uniform(lo, hi, size=(2000, 3))
    tr = hts.trace_lines(starts, ends)
    lengths = tr.lengths("material")
    print("  batch chord lengths by material:", {k: round(v, 1) for k, v in sorted(lengths.items(), key=str)})


# %% [markdown]
# ## 8. Structural fingerprinting & revision diffing


# %%
def case_model_diff_and_fingerprints(out: Path) -> None:
    """Compare two cask revisions independent of cell/surface renumbering."""
    _ = out
    v1 = llevant.load(EXAMPLES / "cask.i")
    v2 = llevant.load(EXAMPLES / "cask-v2.i")
    diff = v1.diff(v2)
    print(
        f"  cask v1 vs v2 differs={bool(diff)}, changed_cells={len(diff.cells)}, "
        f"unchanged={diff.cells.unchanged}"
    )
    assert bool(diff)


# %% [markdown]
# ## 9. 3D realistic PBR cutaway with label ghosting


# %%
def case_render_cutaway_ghost_labels(out: Path) -> None:
    """Render a clipped 3D view hiding outer cryostat and showing Tungsten as a ghost shell."""
    hts = llevant.load(EXAMPLES / "openmc" / "fusion_hts_blanket.xml")
    img = hts.render(
        direction=(0.6, 0.7, -0.4),
        size=(480, 360),
        colour="realistic",
        clip=((0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        hide_rules=['label "Cryostat"'],
        ghost=['label "Tungsten"'],
    )
    save_image(img, out, "hts_cutaway_ghost")


# %% [markdown]
# ## 10. Web Studio REST API integration


# %%
def case_web_studio_api(out: Path) -> None:
    """Exercise Web Studio REST endpoints (/api/locate, /api/trace, /api/lost, /api/convert, /api/volume)."""
    _ = out
    if not (WORKSPACE / "web_viewer.py").is_file():
        print("  skipped: web_viewer.py is outside standalone repo checkout")
        return
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    import web_viewer

    server = ThreadingHTTPServer(("127.0.0.1", 0), web_viewer.LlevantWebHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        for route in (
            "/api/locate?model=fusion_hts_blanket&x=0&y=62&z=0",
            "/api/trace?model=fusion_hts_blanket&from=-250,0,0&to=250,0,0",
            "/api/lost?model=fusion_hts_blanket&x=0&y=62&z=0&radius=1.0",
            "/api/convert?model=pwr_pin_lattice",
            "/api/volume?model=vver_hex_assembly&by=material&h=1.2&mass=1",
        ):
            with urllib.request.urlopen(f"{base}{urllib.parse.quote(route, safe='/?=&,')}", timeout=30) as resp:
                assert resp.status == 200
                payload = json.loads(resp.read().decode("utf-8"))
                print(f"  GET {route.split('?')[0]} -> keys={sorted(payload.keys())}")
    finally:
        server.shutdown()
        server.server_close()


CASES: dict[str, Callable[[Path], None]] = {
    "multicode_load_and_convert": case_multicode_load_and_convert,
    "qa_check_and_plots": case_qa_check_and_plots,
    "lost_particle_triage": case_lost_particle_triage,
    "volumes_and_mass": case_volumes_and_mass,
    "openmc_statepoint_tallies": case_openmc_statepoint_tallies,
    "distributed_materials_and_plots": case_distributed_materials_and_plots,
    "ray_trace_and_chord_lengths": case_ray_trace_and_chord_lengths,
    "model_diff_and_fingerprints": case_model_diff_and_fingerprints,
    "render_cutaway_ghost_labels": case_render_cutaway_ghost_labels,
    "web_studio_api": case_web_studio_api,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run curated Llevant use-case examples.")
    parser.add_argument("--case", default="all", help="Case name or 'all' (default: all)")
    parser.add_argument("--out", default="use_cases_out", help="Output directory for artifacts")
    parser.add_argument("--list", action="store_true", help="List available use-case names")
    args = parser.parse_args()

    if args.list:
        for name, fn in CASES.items():
            print(f"{name:34s} {fn.__doc__}")
        return

    out = Path(args.out)
    selected = list(CASES.items()) if args.case == "all" else [(args.case, CASES[args.case])]
    for name, fn in selected:
        print(f"==> {name}")
        fn(out)


if __name__ == "__main__":
    main()
