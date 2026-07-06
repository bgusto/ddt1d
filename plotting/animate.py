#!/usr/bin/env python3
"""Closeup movie of a hotspot's temperature profile, annotated with the
detonation-radius (coupling) criterion from routines/radius.py.

Three segments: an intro title card (paused on frame 0), a radius-definition
card (also paused on frame 0, text + arrows + shaded coupled region), then the
simulation plays forward. The camera starts fixed on +-CAMERA_HALF_WIDTH, then
pans/zooms toward the right-side front at a constant guessed rate -- evolution
here is symmetric, so only one side is tracked, and it's a linear approximation
(not real tracking) meant to be tuned by eye.

Usage:  python3 plotting/animate.py   (run from the plotting/ directory)
"""
import os
import sys
import glob
import numpy as np
import h5py
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from routines.readflash import readflash
from routines.dcj import dcj
from routines.detrad import detrad

# ============================== user parameters ==============================
CASE_DIR = "../ddt1d-n123e04-mr-tol100e08-gauss-dens100e07-temp178e09-ampl0130-stdv1250"
OUTFILE  = "te-closeup.mp4"
FPS      = 24
STRIDE   = 1             # take every Nth plotfile; 1 = full temporal resolution
                         # (things move fast once burning takes off -- don't skip frames)
SIM_HOLD = 3             # render each simulation frame this many times (slows the pace
                         # without re-reading more files; tune this + FPS for runtime)

# camera: symmetric evolution, so we only bother following the right-side front.
# Fixed at [-CAMERA_HALF_WIDTH, +CAMERA_HALF_WIDTH] until PAN_START_FRAME, then
# pans + zooms LINEARLY (constant guessed speed, constant zoom rate) toward the
# right front -- not real tracking, just an approximation to tune by eye later.
CAMERA_HALF_WIDTH     = 25_000.0   # cm; fixed view before the pan starts
CAMERA_HALF_WIDTH_END = 5_000.0    # cm; zoomed-in half-width by the last frame (guess)
PAN_START_FRAME       = 120        # sim-frame index (post-STRIDE) where panning begins
PAN_SPEED             = 150.0      # cm per sim-frame, +x = follow the right front (guess)
STOP_FRAME            = None       # last sim-frame index (post-STRIDE) to play; None = all

CM_TO_M = 0.01   # all data/geometry above is in cm; displayed in meters

# --- intro title card: sets the scene before any physics animation starts ---
# a real pause -- the sim is frozen at frame 0 for the whole segment, camera fixed.
INTRO_FRAMES = 300
INTRO_FADE   = (15, 45, 240, 285)     # frame indices: fade-in start/done, fade-out start/done
INTRO_TEXT = (
    "We're looking at a one-dimensional slice of a tiny region of warm fuel\n"
    "in the interior of a star. The scale is only centimeters wide.\n\n"
    "A small hot spot sits at the center of an otherwise uniform,\n"
    "unburned medium. As it burns, the critical question is whether the\n"
    "reaction stays a subsonic deflagration, or transitions into a\n"
    "self-sustained detonation."
)

# --- radius-definition card: a second pause, right after the intro, still frozen
# at frame 0 (r_det is an initial-condition quantity anyway) ---
RADIUS_FRAMES = 350
RADIUS_FADE   = (15, 45, 300, 335)   # frame indices *within this segment*
RADIUS_TEXT = (
    r"$r_{det}$: where the spontaneous ignition wave keeps pace with the"
    "\n"
    r"Chapman$-$Jouguet speed, $|u_{sp}| \geq D_{CJ}$."
    "\n"
)

# --- coupling-succeeded card: a mid-playback pause where the front visibly
# takes off (a guess at the frame index -- tune COUPLING_PAUSE_FRAME by eye) ---
COUPLING_PAUSE_FRAME = 99            # sim-frame index (post-STRIDE) to freeze on
COUPLING_FRAMES      = 250
COUPLING_FADE        = (15, 45, 200, 235)   # frame indices *within this pause*
COUPLING_TEXT = (
    "The spontaneous wave velocity has reach the Chapman-Jouguet\n"
    "velocity: coupling has been achieved.\n\n"
    "The reaction is now in lock-step with the driven shock."
)

LINE_COLOR    = "#d1495b"   # temperature curve
COUPLED_COLOR = "#2e6f6e"   # shaded region where |u_sp| >= D_CJ
TEXT_COLOR    = "#1b1b1b"
# ===============================================================================


def read_time(path):
    with h5py.File(path, "r") as f:
        return next(float(v) for n, v in f["real scalars"][:] if n.decode().strip() == "time")


def fade_alpha(i, t0, t1, t2, t3):
    """Piecewise-linear alpha: 0 -> 1 over [t0,t1], holds 1 over [t1,t2], 1 -> 0 over [t2,t3]."""
    if i <= t0 or i >= t3:
        return 0.0
    if i < t1:
        return (i - t0) / (t1 - t0)
    if i <= t2:
        return 1.0
    return 1.0 - (i - t2) / (t3 - t2)


case_dir = os.path.join(os.path.dirname(__file__), CASE_DIR)
files = sorted(glob.glob(os.path.join(case_dir, "ddt1d_hdf5_plt_cnt_[0-9]*")))[::STRIDE]
if not files:
    sys.exit(f"no plot files in {case_dir!r}")
print(f"{len(files)} plot files ({STRIDE=})")

print("reading frames...")
frames = []
for k, path in enumerate(files):
    d = readflash(path)
    d["time"] = read_time(path)
    frames.append(d)
    if k % 50 == 0:
        print(f"  {k}/{len(files)}")

d0 = frames[0]
r_det, xc, xL, xR = detrad(d0["x"], d0["igtm"], d0["uspx"], dcj(d0["gamc"]))
print(f"initial r_det = {r_det:.4g} cm, center = {xc:.4g} cm")

# y-range across every frame's full profile (camera pans, so don't pre-filter by window)
Tlo = min(d["temp"].min() for d in frames)
Thi = max(d["temp"].max() for d in frames)
ypad = 0.08 * (Thi - Tlo)

n_sim = len(frames)
n_play = min(STOP_FRAME + 1, n_sim) if STOP_FRAME is not None else n_sim
print(f"T range [{Tlo:.3e}, {Thi:.3e}] K, playing {n_play}/{n_sim} sim frames")


def camera_xlim(fr):
    """Fixed at +-CAMERA_HALF_WIDTH until PAN_START_FRAME, then a linear pan+zoom
    toward the right front (reaching CAMERA_HALF_WIDTH_END by the last played
    frame). A guess, not real tracking -- tune the constants above. Returns
    (lo, hi) in cm; convert to meters at the call site."""
    t_pan = max(0, fr - PAN_START_FRAME)
    center = xc + PAN_SPEED * t_pan
    frac = min(t_pan / max(n_play - 1 - PAN_START_FRAME, 1), 1.0)
    half_now = CAMERA_HALF_WIDTH + frac * (CAMERA_HALF_WIDTH_END - CAMERA_HALF_WIDTH)
    return center - half_now, center + half_now


# ------------------------------ figure setup ------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.set_xlim((xc - CAMERA_HALF_WIDTH) * CM_TO_M, (xc + CAMERA_HALF_WIDTH) * CM_TO_M)
ax.set_ylim(Tlo - ypad, Thi + ypad)
ax.set_xlabel("(m)")
ax.set_ylabel("temperature (K)")
# fixed plain-number formatting so the axis offset/scale text can't pop in or
# out as the camera pans and the visible range changes from frame to frame
ax.ticklabel_format(axis="x", style="plain", useOffset=False)

line, = ax.plot([], [], color=LINE_COLOR, lw=2.0)
title = ax.set_title("")

coupled_span = ax.axvspan(xL * CM_TO_M, xR * CM_TO_M, color=COUPLED_COLOR, alpha=0.0, zorder=0)
edge_L = ax.axvline(xL * CM_TO_M, color=COUPLED_COLOR, lw=1.2, ls="--", alpha=0.0)
edge_R = ax.axvline(xR * CM_TO_M, color=COUPLED_COLOR, lw=1.2, ls="--", alpha=0.0)

arrow_y = Thi + 0.5 * ypad
radius_arrow = ax.annotate(
    "", xy=(xR * CM_TO_M, arrow_y), xytext=(xL * CM_TO_M, arrow_y),
    arrowprops=dict(arrowstyle="<->", color=TEXT_COLOR, lw=1.3), alpha=0.0,
)
radius_label = ax.text(xc * CM_TO_M, arrow_y, f"$r_{{det}}$ = {r_det * CM_TO_M:.2f} m",
                        ha="center", va="bottom", fontsize=10, color=TEXT_COLOR, alpha=0.0)
radius_text = ax.text(
    0.02, 0.97, RADIUS_TEXT, transform=ax.transAxes, ha="left", va="top",
    fontsize=9.5, color=TEXT_COLOR, alpha=0.0,
    bbox=dict(boxstyle="round", fc="white", ec=COUPLED_COLOR, alpha=0.0),
)

intro_text = fig.text(0.16, 0.5, INTRO_TEXT, ha="left", va="center",
                       fontsize=10, color=TEXT_COLOR, alpha=0.0)

coupling_text = ax.text(
    0.02, 0.97, COUPLING_TEXT, transform=ax.transAxes, ha="left", va="top",
    fontsize=9.5, color=TEXT_COLOR, alpha=0.0,
    bbox=dict(boxstyle="round", fc="white", ec=LINE_COLOR, alpha=0.0),
)

# -------------------------------- shot plan --------------------------------
# each entry: (fr, show_title, intro_alpha, radius_alpha, coupling_alpha).
# INTRO and RADIUS are pauses frozen on frame 0; MAIN plays every sim frame
# forward with a second pause inserted at COUPLING_PAUSE_FRAME.
plan = []
for i in range(INTRO_FRAMES):
    plan.append((0, False, fade_alpha(i, *INTRO_FADE), 0.0, 0.0))
for i in range(RADIUS_FRAMES):
    plan.append((0, False, 0.0, fade_alpha(i, *RADIUS_FADE), 0.0))
for fr in range(n_play):
    if fr == COUPLING_PAUSE_FRAME:
        for i in range(COUPLING_FRAMES):
            plan.append((fr, True, 0.0, 0.0, fade_alpha(i, *COUPLING_FADE)))
    else:
        for _ in range(SIM_HOLD):
            plan.append((fr, True, 0.0, 0.0, 0.0))

total_frames = len(plan)
print(f"{total_frames} total frames ({total_frames / FPS:.1f} s)")

# ------------------------------- render loop -------------------------------
writer = FFMpegWriter(fps=FPS, bitrate=4000)
with writer.saving(fig, OUTFILE, dpi=110):
    for i, (fr, show_title, ia, ra, ca) in enumerate(plan):
        d = frames[fr]
        line.set_data(d["x"] * CM_TO_M, d["temp"])
        lo, hi = camera_xlim(fr)
        ax.set_xlim(lo * CM_TO_M, hi * CM_TO_M)
        title.set_text(f"frame {fr + 1}/{n_play}   t = {d['time']:.3e} s" if show_title else "")

        intro_text.set_alpha(ia)

        coupled_span.set_alpha(0.22 * ra)
        edge_L.set_alpha(ra)
        edge_R.set_alpha(ra)
        radius_arrow.set_alpha(ra)
        radius_label.set_alpha(ra)
        radius_text.set_alpha(ra)
        radius_text.get_bbox_patch().set_alpha(0.85 * ra)

        coupling_text.set_alpha(ca)
        coupling_text.get_bbox_patch().set_alpha(0.85 * ca)

        writer.grab_frame()

        if i % 100 == 0:
            print(f"  wrote frame {i}/{total_frames}")

print(f"wrote {OUTFILE}")
