"""Generate an animated 1280x720 showcase MP4 highlighting Llevant's core neutronics workflows.

Usage:
    python crates/llevant-py/examples/make_showcase_video.py
    python crates/llevant-py/examples/make_showcase_video.py --out docs/img/llevant_use_cases.mp4
"""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import llevant
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
BG = (11, 17, 32)
CARD_BG = (17, 24, 39)
CARD_BORDER = (34, 48, 72)
CYAN = (56, 189, 248)
EMERALD = (52, 211, 153)
AMBER = (251, 191, 36)
ROSE = (251, 113, 133)
TEXT = (241, 245, 249)
MUTED = (148, 163, 184)

REPO = Path(__file__).resolve().parents[3]
EXAMPLES = REPO / "docs" / "examples"


def _load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load DejaVu fonts cleanly with safe fallback to Pillow's default font."""
    if mono:
        name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    else:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for root in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
    ):
        p = root / name
        if p.is_file():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


F_TITLE = _load_font(26, bold=True)
F_SUB = _load_font(15, bold=False)
F_CARD = _load_font(16, bold=True)
F_BODY = _load_font(14, bold=False)
F_MONO = _load_font(13, bold=False, mono=True)
F_MONO_B = _load_font(14, bold=True, mono=True)
F_BADGE = _load_font(12, bold=True, mono=True)


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    """Convert `#rrggbb` string to an RGB tuple."""
    s = h.lstrip("#")
    if len(s) != 6:
        return (148, 163, 184)
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def fit_image(rgb: np.ndarray, max_w: int, max_h: int) -> Image.Image:
    """Aspect-preserving high-quality resize for fitting llevant images inside cards."""
    src = Image.fromarray(rgb, mode="RGB")
    scale = min(max_w / src.width, max_h / src.height)
    nw = max(1, int(round(src.width * scale)))
    nh = max(1, int(round(src.height * scale)))
    return src.resize((nw, nh), Image.Resampling.LANCZOS)


def new_frame(title: str, subtitle: str, scene_num: int, total_scenes: int, progress: float) -> Image.Image:
    """Create the dark studio frame shell with top header bar and scene progress bar."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    # Top header bar
    draw.rectangle((0, 0, W, 68), fill=(15, 23, 42))
    draw.line((0, 68, W, 68), fill=CARD_BORDER, width=2)
    draw.rounded_rectangle((24, 14, 152, 54), radius=8, fill=(14, 116, 144))
    draw.text((38, 23), "LLEVANT", fill=TEXT, font=F_CARD)
    draw.text((170, 14), title, fill=TEXT, font=F_TITLE)
    draw.text((170, 44), subtitle, fill=MUTED, font=F_SUB)
    pill = f"USE CASE {scene_num}/{total_scenes}"
    draw.rounded_rectangle((W - 186, 18, W - 24, 48), radius=14, fill=(30, 41, 59), outline=CYAN, width=1)
    draw.text((W - 170, 26), pill, fill=CYAN, font=F_BADGE)
    # Thin bottom progress indicator
    draw.rectangle((0, H - 5, W, H), fill=(15, 23, 42))
    draw.rectangle((0, H - 5, int(W * max(0.0, min(1.0, progress))), H), fill=CYAN)
    return img


def draw_card(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    heading: str,
    badge: str | None = None,
    badge_color: tuple[int, int, int] = EMERALD,
) -> tuple[int, int, int, int]:
    """Draw a rounded studio panel card and return the inner content bounding box."""
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(box, radius=12, fill=CARD_BG, outline=CARD_BORDER, width=2)
    draw.text((x0 + 16, y0 + 12), heading, fill=CYAN, font=F_CARD)
    if badge:
        bbox = draw.textbbox((0, 0), badge, font=F_BADGE)
        bw = (bbox[2] - bbox[0]) + 22
        bx0 = x1 - bw - 14
        draw.rounded_rectangle((bx0, y0 + 10, x1 - 14, y0 + 32), radius=10, fill=(20, 34, 48), outline=badge_color, width=1)
        draw.text((bx0 + 11, y0 + 14), badge, fill=badge_color, font=F_BADGE)
    draw.line((x0 + 12, y0 + 40, x1 - 12, y0 + 40), fill=CARD_BORDER, width=1)
    return (x0 + 14, y0 + 48, x1 - 14, y1 - 14)


def paste_centered(canvas: Image.Image, inner: tuple[int, int, int, int], rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Scale and center an RGB numpy array inside `inner`, returning pasted pixel bounds."""
    ix0, iy0, ix1, iy1 = inner
    fitted = fit_image(rgb, ix1 - ix0, iy1 - iy0)
    px = ix0 + (ix1 - ix0 - fitted.width) // 2
    py = iy0 + (iy1 - iy0 - fitted.height) // 2
    canvas.paste(fitted, (px, py))
    return (px, py, px + fitted.width, py + fitted.height)


def render_scene_1_multicode() -> list[Image.Image]:
    """Scene 1: Native OpenMC XML, TRIPOLI-4, and DRAGON5 SPLITR/SPLITZ geometry gallery."""
    print("Rendering Scene 1: Multi-Code Geometry...")
    hts = llevant.load(EXAMPLES / "openmc" / "fusion_hts_blanket.xml")
    vver = llevant.load(EXAMPLES / "openmc" / "vver_hex_assembly.xml")
    t4 = llevant.load(EXAMPLES / "tripoli4" / "pwr_pin_cell.t4")
    d5 = llevant.load(EXAMPLES / "dragon5" / "depletion_pin_splitr.x2m")

    img_hts = hts.render(direction=(0.6, 0.7, -0.4), size=(540, 235), colour="realistic", clip=((0, 0, 0), (0, -1, 0)))
    img_vver = vver.render(direction=(0.55, 0.65, -0.45), size=(540, 235), colour="material")
    img_t4 = t4.slice(z=0.0, width=420, colour="material", outline=True)
    img_d5 = d5.slice(z=2.5, width=420, colour="cell", outline=True)

    n_frames = 78
    base = new_frame(
        "Multi-Code CSG Without Conversion Hassle",
        "Native OpenMC XML, TRIPOLI-4 (ECRASE + boundaries), and DRAGON5 GEO: (SPLITR equal-area rings)",
        1,
        6,
        1 / 6,
    )
    cards = [
        ((24, 84, 628, 382), "OpenMC XML: Tokamak Blanket", f"{hts.info()['cells']} cells | 0 diags", CYAN, img_hts.rgb),
        ((652, 84, 1256, 382), "OpenMC XML: VVER-1000 Hex Lattice", "LAT=2 3-ring", CYAN, img_vver.rgb),
        ((24, 398, 628, 696), "TRIPOLI-4: PWR Pin + Boundary", "ECRASE + REFL", EMERALD, img_t4.rgb),
        ((652, 398, 1256, 696), "DRAGON5 GEO: SPLITR 3 1 Burnup Pin", "Equal-area rings", AMBER, img_d5.rgb),
    ]
    for box, heading, badge, bcol, rgb in cards:
        inner = draw_card(base, box, heading, badge, bcol)
        paste_centered(base, inner, rgb)

    return [base.copy() for _ in range(n_frames)]


def render_scene_2_orbit() -> list[Image.Image]:
    """Scene 2: 3D PBR cutaway orbit with label rules & ghost shell."""
    print("Rendering Scene 2: 3D PBR Cutaway & Label Ghosting...")
    hts = llevant.load(EXAMPLES / "openmc" / "fusion_hts_blanket.xml")
    mats = hts.materials()[:6]

    n_keyframes = 14
    keyframes: list[tuple[float, Image.Image, float]] = []
    for k in range(n_keyframes):
        frac = k / (n_keyframes - 1)
        az = -72.0 + frac * 84.0
        el = 22.0 + 6.0 * math.sin(frac * math.pi)
        az_r, el_r = math.radians(az), math.radians(el)
        d = (math.cos(el_r) * math.cos(az_r), math.cos(el_r) * math.sin(az_r), -math.sin(el_r))
        t0 = time.perf_counter()
        im = hts.render(
            direction=d,
            size=(720, 520),
            colour="realistic",
            clip=((0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
            hide_rules=['label "Cryostat"'],
            ghost=['label "Tungsten"'],
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        keyframes.append((az, Image.fromarray(im.rgb, mode="RGB"), dt_ms))

    n_frames = 96
    frames: list[Image.Image] = []
    for f in range(n_frames):
        u = f / max(1, n_frames - 1)
        pos = u * (n_keyframes - 1)
        i0 = min(n_keyframes - 2, int(pos))
        alpha = pos - i0
        az = keyframes[i0][0] * (1 - alpha) + keyframes[i0 + 1][0] * alpha
        dt_ms = keyframes[i0][2]
        blended = Image.blend(keyframes[i0][1], keyframes[i0 + 1][1], alpha)

        frame = new_frame(
            "3D Analytical PBR Cutaway & Rule Ghosting",
            "Live Cook-Torrance shading + DSL isolation (`label \"Cryostat\"`, ghost `label \"Tungsten\"`)",
            2,
            6,
            (1 + u) / 6,
        )
        inner_l = draw_card(frame, (24, 84, 804, 696), "Fusion Blanket Cutaway (Live Render)", f"az={az:+.1f} deg", CYAN)
        paste_centered(frame, inner_l, np.asarray(blended))

        inner_r = draw_card(frame, (824, 84, 1256, 696), "Scene Controls & Materials", f"{dt_ms:.0f} ms/frame", EMERALD)
        d_ctx = ImageDraw.Draw(frame)
        rx, ry = inner_r[0] + 6, inner_r[1] + 6
        lines = [
            ("Query DSL", CYAN, F_CARD),
            ("  clip:   y < 0 midplane", TEXT, F_MONO),
            ('  hide:   label "Cryostat"', TEXT, F_MONO),
            ('  ghost:  label "Tungsten"', AMBER, F_MONO),
            ("", TEXT, F_MONO),
            ("Detected Material Looks", CYAN, F_CARD),
        ]
        for txt, col, fnt in lines:
            d_ctx.text((rx, ry), txt, fill=col, font=fnt)
            ry += 26
        ry += 6
        for mat in mats:
            col = hex_to_rgb(mat["colour"])
            d_ctx.rounded_rectangle((rx, ry, rx + 22, ry + 22), radius=4, fill=col, outline=(203, 213, 225), width=1)
            d_ctx.text((rx + 32, ry + 3), f"m{mat['material']:<2}  {mat['look']:<10}  rho={mat['density'][0]:.2f}", fill=TEXT, font=F_MONO)
            ry += 34
        frames.append(frame)
    return frames


def render_scene_3_slicer_trace() -> list[Image.Image]:
    """Scene 3: Sweeping Z-slice with animated ray scan & batch material chord lengths."""
    print("Rendering Scene 3: Sweeping Slicer & Analytical Ray Trace...")
    hts = llevant.load(EXAMPLES / "openmc" / "fusion_hts_blanket.xml")
    lo, hi = hts.extent()
    rng = np.random.default_rng(7)
    tr = hts.trace_lines(rng.uniform(lo, hi, size=(3000, 3)), rng.uniform(lo, hi, size=(3000, 3)))
    lengths = sorted(((k, v) for k, v in tr.lengths("material").items() if k > 0), key=lambda x: -x[1])[:7]
    max_len = max(v for _, v in lengths)
    mat_colors = {m["material"]: hex_to_rgb(m["colour"]) for m in hts.materials()}
    mat_looks = {m["material"]: m["look"] for m in hts.materials()}

    n_slices = 16
    z_vals = np.linspace(-95.0, 95.0, n_slices)
    slice_imgs = [Image.fromarray(hts.slice(z=float(z), width=600, colour="realistic", outline=True).rgb, mode="RGB") for z in z_vals]

    n_frames = 96
    frames: list[Image.Image] = []
    for f in range(n_frames):
        u = f / max(1, n_frames - 1)
        pos = u * (n_slices - 1)
        i0 = min(n_slices - 2, int(pos))
        alpha = pos - i0
        z_cur = z_vals[i0] * (1 - alpha) + z_vals[i0 + 1] * alpha
        s_img = Image.blend(slice_imgs[i0], slice_imgs[i0 + 1], alpha)

        frame = new_frame(
            "Exact 2D Slicer & Batch Ray Chord Lengths",
            "Row-exact cross sections paired with parallel `trace_lines` material inventory",
            3,
            6,
            (2 + u) / 6,
        )
        inner_l = draw_card(frame, (24, 84, 710, 696), "2D Analytical XY Slice Sweep", f"z = {z_cur:+.1f} cm", CYAN)
        px0, py0, px1, py1 = paste_centered(frame, inner_l, np.asarray(s_img))
        d_ctx = ImageDraw.Draw(frame)
        # Animated equatorial trace overlay across slice middle
        mid_y = (py0 + py1) // 2
        ray_x = px0 + int((px1 - px0) * (0.1 + 0.8 * ((u * 2.0) % 1.0)))
        d_ctx.line((px0 + 12, mid_y, px1 - 12, mid_y), fill=AMBER, width=2)
        d_ctx.ellipse((ray_x - 6, mid_y - 6, ray_x + 6, mid_y + 6), fill=AMBER, outline=(11, 17, 32), width=2)

        inner_r = draw_card(frame, (730, 84, 1256, 696), "3,000-Ray Chord Lengths by Material", "trace_lines", EMERALD)
        rx, ry = inner_r[0] + 10, inner_r[1] + 12
        d_ctx.text((rx, ry), "Cumulative optical path length [cm]:", fill=MUTED, font=F_BODY)
        ry += 34
        bar_max_w = inner_r[2] - inner_r[0] - 215
        for mid, val in lengths:
            col = mat_colors.get(mid, CYAN)
            look = mat_looks.get(mid, f"m{mid}")
            d_ctx.text((rx, ry + 4), f"m{mid:<2} {look:<8}", fill=TEXT, font=F_MONO)
            bw = max(4, int(bar_max_w * (val / max_len)))
            d_ctx.rounded_rectangle((rx + 120, ry + 3, rx + 120 + bw, ry + 23), radius=4, fill=col)
            d_ctx.text((rx + 128 + bw, ry + 4), f"{val:,.0f}", fill=MUTED, font=F_MONO)
            ry += 44
        frames.append(frame)
    return frames


def render_scene_4_statepoint() -> list[Image.Image]:
    """Scene 4: OpenMC HDF5 statepoint Thermal Flux vs Fast Heating split-wipe reveal."""
    print("Rendering Scene 4: OpenMC Statepoint Filter Overlays...")
    hts = llevant.load(EXAMPLES / "openmc" / "fusion_hts_blanket.xml")
    with tempfile.TemporaryDirectory() as tmpdir:
        sp = Path(tmpdir) / "statepoint.50.h5"
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
            t101.create_dataset("n_realizations", data=np.int32(50))
            t101.create_dataset("filters", data=np.array([1, 2], dtype=np.int32))
            t101.create_dataset("score_bins", data=np.array([b"flux", b"heating"]))
            results = np.zeros((14, 2, 3), dtype=np.float64)
            for ci, val in enumerate([1e14, 6e13, 3e13, 8e12, 2e12, 5e11, 8e10]):
                results[2 * ci + 0, 0, 0] = 0.8 * val * 50.0  # thermal flux higher in blanket/water
                results[2 * ci + 1, 0, 0] = 0.2 * val * 50.0
                results[2 * ci + 0, 1, 0] = 1e-13 * val * 50.0
                results[2 * ci + 1, 1, 0] = 3e-12 * val * 50.0  # fast heating peaked in first wall
            t101.create_dataset("results", data=results)

        fld_flux = llevant.read_openmc_tally(hts, sp, tally=101, score="flux", energy=0)
        fld_heat = llevant.read_openmc_tally(hts, sp, tally=101, score="heating", energy=1)
        img_flux = fit_image(hts.slice(z=0.0, width=700, field=fld_flux).rgb, 730, 560)
        img_heat = fit_image(hts.slice(z=0.0, width=700, field=fld_heat).rgb, 730, 560)

    n_frames = 84
    frames: list[Image.Image] = []
    for f in range(n_frames):
        u = f / max(1, n_frames - 1)
        wipe = 0.15 + 0.70 * 0.5 * (1.0 - math.cos(u * 2.0 * math.pi))
        split_px = max(1, min(img_flux.width - 1, int(img_flux.width * wipe)))
        comp = img_heat.copy()
        comp.paste(img_flux.crop((0, 0, split_px, img_flux.height)), (0, 0))

        frame = new_frame(
            "OpenMC HDF5 Statepoint Multi-Filter Slicing",
            "Filter natively by `tally`, `score`, `nuclide`, `energy`, `particle`, `distribcell`, and `mesh`",
            4,
            6,
            (3 + u) / 6,
        )
        inner_l = draw_card(frame, (24, 84, 804, 696), "Split Comparison: Thermal Flux vs Fast Heating", "statepoint.50.h5", AMBER)
        px0, py0, _, py1 = paste_centered(frame, inner_l, np.asarray(comp))
        d_ctx = ImageDraw.Draw(frame)
        wx = px0 + split_px
        d_ctx.line((wx, py0, wx, py1), fill=TEXT, width=3)
        d_ctx.rounded_rectangle((px0 + 8, py0 + 8, px0 + 188, py0 + 32), radius=6, fill=(14, 116, 144))
        d_ctx.text((px0 + 16, py0 + 13), "score='flux' E=0", fill=TEXT, font=F_BADGE)
        d_ctx.rounded_rectangle((px0 + img_flux.width - 204, py0 + 8, px0 + img_flux.width - 8, py0 + 32), radius=6, fill=(180, 83, 9))
        d_ctx.text((px0 + img_flux.width - 194, py0 + 13), "score='heating' E=1", fill=TEXT, font=F_BADGE)

        inner_r = draw_card(frame, (824, 84, 1256, 696), "Statepoint Pipeline", "h5py zero-copy", EMERALD)
        rx, ry = inner_r[0] + 10, inner_r[1] + 14
        snippets = [
            ("tallies = openmc_tallies(sp)", CYAN),
            ("  tally 101: [flux, heating]", TEXT),
            ("  filters:   CellFilter (7)", TEXT),
            ("             EnergyFilter (2)", TEXT),
            ("", TEXT),
            ("read_openmc_tally(", AMBER),
            ("  model, sp,", TEXT),
            ("  score='heating',", TEXT),
            ("  energy=1,  # 0.625..20 MeV", MUTED),
            (")", AMBER),
            ("", TEXT),
            ("Also supported in Web Studio:", CYAN),
            ("  GET /api/tallies?path=...", EMERALD),
            ("  GET /api/slice?score=...&energy=1", EMERALD),
        ]
        for line, col in snippets:
            d_ctx.text((rx, ry), line, fill=col, font=F_MONO)
            ry += 28
        frames.append(frame)
    return frames


def render_scene_5_qa_lost() -> list[Image.Image]:
    """Scene 5: Stochastic overlap QA report & lost-particle triage with pulsing target."""
    print("Rendering Scene 5: Geometry QA & Lost-Particle Triage...")
    m_ov = llevant.from_text(
        """overlap qa
1 1 -7.8 -1 imp:n=1
2 2 -1.0 -2 imp:n=1
3 0 3 imp:n=0

1 rpp -3 3 -3 3 -3 3
2 rpp 1 6 -3 3 -3 3
3 rpp -15 15 -15 15 -15 15

mode n
m1 26000 1.0
m2 1001 2.0 8016 1.0
"""
    )
    m_gap = llevant.from_text(
        """lost triage
1 1 -7.8 -1 imp:n=1
2 2 -1.0 -2 1 imp:n=1
3 0 2 -3 imp:n=1
4 0 4 imp:n=0

1 so 3
2 so 5
3 so 11
4 so 15

mode n
m1 26000 1.0
m2 1001 2.0 8016 1.0
"""
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        rep = llevant.check_report(m_ov, h=0.4, out_dir=Path(tmpdir) / "qa", px=460)
        ov_png = Image.open(Path(tmpdir) / "qa" / rep["defects"][0]["plots"][0]).convert("RGB")
        ov_vol = rep["defects"][0]["volume"]
        ov_sig = rep["defects"][0]["sigma"]

        lost = llevant.diagnose_lost_particles(m_gap, "1 x, y, z = 12.5 0.0 0.0", radius=4.0, plots=Path(tmpdir) / "lost", px=460)
        lost_png = Image.open(Path(tmpdir) / "lost" / lost[0]["plots"][0]).convert("RGB")
        diag_str = lost[0]["diagnosis"]

    n_frames = 84
    frames: list[Image.Image] = []
    for f in range(n_frames):
        u = f / max(1, n_frames - 1)
        pulse = int(12 + 7 * math.sin(u * 4.0 * math.pi))
        frame = new_frame(
            "Automated QA Defect Plots & Lost-Particle Triage",
            "Auto-framed defect bounding boxes + MCNP/OpenMC log coordinate diagnosis",
            5,
            6,
            (4 + u) / 6,
        )
        inner_l = draw_card(frame, (24, 84, 628, 696), "QA Overlap Defect Slice", f"vol={ov_vol:.2f} +/- {ov_sig:.2f} cm3", ROSE)
        paste_centered(frame, (inner_l[0], inner_l[1], inner_l[2], inner_l[3] - 64), np.asarray(ov_png))
        d_ctx = ImageDraw.Draw(frame)
        d_ctx.text((inner_l[0] + 8, inner_l[3] - 46), "llevant check model.i --h 0.4 --plots qa/", fill=CYAN, font=F_MONO)
        d_ctx.text((inner_l[0] + 8, inner_l[3] - 24), "Outputs: report.json, report.md + defect_001_{xy,xz,yz}.png", fill=MUTED, font=F_MONO)

        inner_r = draw_card(frame, (652, 84, 1256, 696), "Lost-Particle Triage (12.5, 0, 0)", "status = gap", AMBER)
        px0, py0, px1, py1 = paste_centered(frame, (inner_r[0], inner_r[1], inner_r[2], inner_r[3] - 64), np.asarray(lost_png))
        cx, cy = (px0 + px1) // 2, (py0 + py1) // 2
        d_ctx.ellipse((cx - pulse, cy - pulse, cx + pulse, cy + pulse), outline=AMBER, width=2)
        d_ctx.line((cx - 18, cy, cx + 18, cy), fill=AMBER, width=2)
        d_ctx.line((cx, cy - 18, cx, cy + 18), fill=AMBER, width=2)
        d_ctx.text((inner_r[0] + 8, inner_r[3] - 46), f"Diagnosis: {diag_str[:56]}", fill=TEXT, font=F_MONO)
        d_ctx.text((inner_r[0] + 8, inner_r[3] - 24), "Viewer: Check -> Problems -> Lost particle -> Go", fill=MUTED, font=F_MONO)
        frames.append(frame)
    return frames


def render_scene_6_mass_diff() -> list[Image.Image]:
    """Scene 6: Mass balance inventory & structural revision diffing."""
    print("Rendering Scene 6: Mass Inventory & Revision Diff...")
    vver = llevant.load(EXAMPLES / "openmc" / "vver_hex_assembly.xml")
    mb = llevant.mass_balance(vver, h=0.8, by="material")
    known = sorted((g for g in mb["groups"] if g["mass_g"] is not None), key=lambda g: -g["mass_g"])[:6]
    max_mass = max(g["mass_g"] for g in known)

    c1 = llevant.load(EXAMPLES / "cask.i")
    c2 = llevant.load(EXAMPLES / "cask-v2.i")
    diff = c1.diff(c2)
    s1 = c1.slice(y=0.0, width=320, colour="material", outline=True).rgb
    s2 = c2.slice(y=0.0, width=320, colour="material", outline=True).rgb

    n_frames = 78
    frames: list[Image.Image] = []
    for f in range(n_frames):
        u = f / max(1, n_frames - 1)
        grow = min(1.0, u * 1.8)
        frame = new_frame(
            "Mass Inventories & Numbering-Agnostic Revision Diff",
            "Stochastic mass totals with propagated uncertainty + geometric fingerprint diffing",
            6,
            6,
            (5 + u) / 6,
        )
        inner_l = draw_card(
            frame,
            (24, 84, 628, 696),
            "VVER-1000 Stochastic Mass Inventory",
            f"Total {mb['total_mass_kg']:.2f} kg",
            EMERALD,
        )
        d_ctx = ImageDraw.Draw(frame)
        rx, ry = inner_l[0] + 8, inner_l[1] + 14
        bar_max = inner_l[2] - inner_l[0] - 210
        for g in known:
            mval = g["mass_g"]
            bw = max(4, int(bar_max * (mval / max_mass) * grow))
            d_ctx.text((rx, ry + 5), f"m{g['number']:<2}", fill=CYAN, font=F_MONO_B)
            d_ctx.rounded_rectangle((rx + 56, ry + 4, rx + 56 + bw, ry + 26), radius=5, fill=EMERALD)
            d_ctx.text((rx + 66 + bw, ry + 5), f"{mval:7.1f} +/- {g['mass_sigma_g']:.1f} g", fill=TEXT, font=F_MONO)
            ry += 48
        d_ctx.text((rx, inner_l[3] - 26), "llevant volume assembly.xml --h 0.8 --by material --mass", fill=MUTED, font=F_MONO)

        inner_r = draw_card(
            frame,
            (652, 84, 1256, 696),
            "Structural Revision Diff (cask v1 vs v2)",
            f"{len(diff.cells)} changed",
            AMBER,
        )
        half_w = (inner_r[2] - inner_r[0] - 16) // 2
        left_box = (inner_r[0], inner_r[1] + 10, inner_r[0] + half_w, inner_r[3] - 96)
        right_box = (inner_r[0] + half_w + 16, inner_r[1] + 10, inner_r[2], inner_r[3] - 96)
        paste_centered(frame, left_box, s1)
        paste_centered(frame, right_box, s2)
        d_ctx.text((left_box[0] + 8, left_box[1] - 4), "cask.i (rev 1)", fill=CYAN, font=F_MONO_B)
        d_ctx.text((right_box[0] + 8, right_box[1] - 4), "cask-v2.i (rev 2)", fill=AMBER, font=F_MONO_B)
        d_ctx.text(
            (inner_r[0] + 8, inner_r[3] - 64),
            f"changed={len(diff.cells)}  unchanged={diff.cells.unchanged}  differs={bool(diff)}",
            fill=TEXT,
            font=F_MONO_B,
        )
        d_ctx.text(
            (inner_r[0] + 8, inner_r[3] - 32),
            "CLI: llevant ... | Python: llevant.load(...) | Web: python web_viewer.py",
            fill=EMERALD,
            font=F_MONO,
        )
        frames.append(frame)
    return frames


def concat_with_crossfades(scenes: list[list[Image.Image]], fade: int = 8) -> list[Image.Image]:
    """Assemble scene frame lists with smooth linear cross-fades."""
    out: list[Image.Image] = []
    for idx, frames in enumerate(scenes):
        if idx == 0:
            out.extend(frames)
            continue
        prev_tail = out[-fade:]
        del out[-fade:]
        for k in range(fade):
            alpha = (k + 1) / (fade + 1)
            out.append(Image.blend(prev_tail[k], frames[k], alpha))
        out.extend(frames[fade:])
    return out


def encode_mp4(frames: list[Image.Image], out_path: Path, fps: int) -> None:
    """Pipe RGB24 frames directly to ffmpeg H.264 without writing temporary PNGs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "22",
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.flush()
    finally:
        proc.stdin.close()
        proc.stdin = None
    err = proc.stderr.read() if proc.stderr is not None else b""
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err.decode('utf-8', errors='replace')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Llevant showcase MP4 video.")
    parser.add_argument(
        "--out",
        default=str(REPO / "docs" / "img" / "llevant_use_cases.mp4"),
        help="Output MP4 path",
    )
    parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24)")
    parser.add_argument("--preview-dir", default="", help="Optional directory to save 1 keyframe per scene")
    args = parser.parse_args()

    scenes = [
        render_scene_1_multicode(),
        render_scene_2_orbit(),
        render_scene_3_slicer_trace(),
        render_scene_4_statepoint(),
        render_scene_5_qa_lost(),
        render_scene_6_mass_diff(),
    ]
    if args.preview_dir:
        pdir = Path(args.preview_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        for idx, s in enumerate(scenes, start=1):
            s[len(s) // 2].save(pdir / f"scene_{idx:02d}.png")

    timeline = concat_with_crossfades(scenes, fade=8)
    out_path = Path(args.out)
    encode_mp4(timeline, out_path, args.fps)
    mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path} ({len(timeline)} frames, {len(timeline) / args.fps:.1f}s, {mb:.2f} MB)")


if __name__ == "__main__":
    main()
