"""Sim-vs-real audit of a logged run: every signal we log, real vs a matched sim run of the same checkpoint.

    python scripts/analyze_log.py <real run dir> [--sim <resim run dir>] [--checkpoint path.pt]

<real run dir> is a deploy.py --log run (obs/ + optionally rosbag/). --sim defaults to the newest resim_* inside it
(make one with `replay_log.py <run> --mode resim --checkpoint <the robot's checkpoint>`). Writes
<run>/analysis/summary.md (flagged table + numbers per category) and plots.

Categories: A timing, B IMU, C joint encoders, D actuation/torque, E policy outputs, F observations (shift +
counterfactual inference), G whole body (feet, centre of mass, tracking), H configuration.
Flags: RED >2x real/sim (or failed check), AMBER 1.3-2x.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import sys

sys.path.append(".")

import numpy as np

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run")
parser.add_argument("--sim", default=None)
parser.add_argument("--checkpoint", default=None, help="Checkpoint for counterfactual inference (default: task's).")
parser.add_argument("--out", default=None)
args = parser.parse_args()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from booster_deploy.utils.obs_logger import load_obs_log  # noqa: E402
from booster_deploy.utils.lowstate_bag import read_low_state, read_joint_ctrl  # noqa: E402

# ------------------------------------------------------------------ loading

RUN = args.run.rstrip("/")
OBS_DIR = RUN + "/obs" if os.path.isdir(RUN + "/obs") else RUN
SIM_DIR = args.sim or (sorted(glob.glob(RUN + "/resim_*"))[-1] if glob.glob(RUN + "/resim_*") else None)
OUT = args.out or RUN + "/analysis"
os.makedirs(OUT, exist_ok=True)

real = load_obs_log(OBS_DIR)
sim = load_obs_log(SIM_DIR + "/obs") if SIM_DIR else None
meta = real["meta"]
cfg = meta["cfg"]
DT = meta["policy_dt"]
JN = meta["joint_names"]
SJN = meta["sim_joint_names"]
S2R = [SJN.index(n) for n in JN]           # sim-order array -> real order
N = min(len(real["step"]), len(sim["step"])) if sim else len(real["step"])
bag = None
if os.path.isdir(RUN + "/rosbag") and glob.glob(RUN + "/rosbag/*.db3"):
    bag = read_low_state(RUN + "/rosbag")
    ctrl = read_joint_ctrl(RUN + "/rosbag")

LEG = [i for i, n in enumerate(JN) if any(k in n for k in ("Hip", "Knee", "Ankle"))]
KP = np.array(cfg["robot"]["joint_stiffness"])
KD = np.array(cfg["robot"]["joint_damping"])
EFFORT = np.array(cfg["robot"]["effort_limit"])

rows: list[tuple[str, str, str, str, str]] = []   # (category, check, real, sim/ref, flag)
notes: dict[str, list[str]] = {}


def flag_ratio(r, s, lo=1.3, hi=2.0):
    if s is None or not np.isfinite(r) or not np.isfinite(s):
        return ""
    a, b = abs(r), abs(s)
    if b < 1e-9 and a < 1e-9:
        return "ok"
    ratio = max(a, b) / max(min(a, b), 1e-9)
    return "RED" if ratio > hi else ("AMBER" if ratio > lo else "ok")


def add(cat, check, r, s="", flag=""):
    rows.append((cat, check, r, s, flag))


def fmt(x, p=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}g}"


def note(cat, text):
    notes.setdefault(cat, []).append(text)


# ------------------------------------------------------------------ phases

motion_path = meta["files"]["motion_path"]["path"]
local_motion = os.path.join("tasks/hoi_track", cfg["policy"]["motion_path"]) if meta["policy_class"] == "HoiTrackPolicy" else None
REF = np.load(local_motion) if local_motion and os.path.isfile(local_motion) else None


def phases(log, n):
    """Per-step phase label. Hoi: transition / hold / clip phases from the reference clip (trunk height + hand
    contact); others: one 'run' phase."""
    holding = log["holding"][:n].astype(bool)
    hs = log["hold_step"][:n]
    fr = np.maximum(log["motion_frame"][:n] - 1, 0)
    lab = np.array(["run"] * n, dtype=object)
    if REF is None or not holding.any():
        return lab, fr
    z = REF["ref_anchor_pos_w"][:, 2]
    contact = REF["ref_contact_flag"].any(1)
    on = np.nonzero(contact)[0]
    c0, c1 = (on.min(), on.max()) if len(on) else (0, 0)
    zmin1 = int(np.argmin(z[: (c0 + c1) // 2])) if c1 > c0 else int(np.argmin(z))
    zmin2 = (c0 + c1) // 2 + int(np.argmin(z[(c0 + c1) // 2: c1 + 1])) if c1 > c0 else zmin1
    obj = REF["ref_object_pos_w"][:, 2]
    top = zmin1 + int(np.argmax(obj[zmin1:zmin2])) if zmin2 > zmin1 else zmin1
    settle = c1 + int(np.argmax(z[c1:] >= z[-1] - 0.01)) if c1 < len(z) - 1 else len(z) - 1
    bounds = [(0, zmin1, "squat_grasp"), (zmin1, top, "rise_lift"), (top, zmin2 - (zmin2 - top) // 2, "carry"),
              (zmin2 - (zmin2 - top) // 2, zmin2, "squat_place"), (zmin2, settle, "rise_release"),
              (settle, 10 ** 6, "stand")]
    for i in range(n):
        if holding[i]:
            lab[i] = "transition" if hs[i] * DT < 2.5 else "hold"
        else:
            for a, b, name in bounds:
                if a <= fr[i] < b:
                    lab[i] = name
    return lab, fr


PH_R, FR_R = phases(real, N)
PH_S, FR_S = phases(sim, N) if sim else (None, None)
PHASE_ORDER = [p for p in ["transition", "hold", "squat_grasp", "rise_lift", "carry", "squat_place", "rise_release",
                           "stand", "run"] if p in set(PH_R)]
HOLD = PH_R == "hold"
t_real = np.arange(N) * DT


def shade_phases(ax):
    colors = {"transition": "#eee", "hold": "#ddd", "squat_grasp": "#fde", "rise_lift": "#fcc", "carry": "#eef",
              "squat_place": "#efe", "rise_release": "#cfc", "stand": "#fff"}
    start = 0
    for i in range(1, N + 1):
        if i == N or PH_R[i] != PH_R[start]:
            ax.axvspan(start * DT, i * DT, color=colors.get(PH_R[start], "#fff"), lw=0)
            ax.text((start + i) / 2 * DT, ax.get_ylim()[1], PH_R[start], fontsize=6, ha="center", va="top", rotation=90)
            start = i


# ------------------------------------------------------------------ A. timing

dtm = np.diff(real["t_mono"][:N]) * 1e3
stalls = dtm > 30
add("A timing", "control step period median / p99 / max (ms)", f"{np.median(dtm):.1f} / {np.percentile(dtm, 99):.1f} / {dtm.max():.0f}",
    f"{DT*1e3:.0f} nominal", "RED" if dtm.max() > 40 else ("AMBER" if np.percentile(dtm, 99) > 25 else "ok"))
add("A timing", "steps late by >10 ms (count, % of run)", f"{stalls.sum()} ({100*stalls.mean():.1f}%)", "0", "RED" if stalls.sum() else "ok")
STALL_STEPS = np.nonzero(stalls)[0] + 1
big = np.argsort(-dtm)[:5]
note("A timing", "largest control stalls: " + ", ".join(f"{dtm[i]:.0f} ms at step {i+1} ({PH_R[i+1]})" for i in sorted(big) if dtm[i] > 30))
for ph in PHASE_ORDER:
    mm = PH_R[1:N] == ph
    if mm.any() and (dtm[mm[:len(dtm)]] > 30).any():
        note("A timing", f"{ph}: {(dtm[mm[:len(dtm)]] > 30).sum()} late steps, worst {dtm[mm[:len(dtm)]].max():.0f} ms")
if bag is not None:
    g = np.diff(bag["t"]) * 1e3
    add("A timing", "/low_state period median / max gap (ms)", f"{np.median(g):.2f} / {g.max():.1f}", "2.0 nominal",
        "RED" if g.max() > 20 else ("AMBER" if g.max() > 6 else "ok"))
    if len(ctrl["t"]):
        jt = ctrl["t"]
        idx = np.searchsorted(jt, real["t_wall"][:N])
        ok = idx < len(jt)
        lat = (jt[idx[ok]] - real["t_wall"][:N][ok]) * 1e3
        add("A timing", "policy step -> command on the bus, median / p99 (ms)", f"{np.median(lat):.1f} / {np.percentile(lat, 99):.1f}", "", "")
    add("A timing", "motor temperature max (C) / joint", f"{bag['par_temperature'].max():.0f} / {JN[int(bag['par_temperature'].max(0).argmax())]}", "", "AMBER" if bag["par_temperature"].max() > 60 else "ok")
    add("A timing", "motor 'lost' counter nonzero", str(int((bag["par_lost"] > 0).sum())), "0", "RED" if (bag["par_lost"] > 0).any() else "ok")

fig, ax = plt.subplots(2, 1, figsize=(10, 5))
ax[0].plot(t_real[1:], dtm); ax[0].axhline(DT * 1e3, color="k", ls=":"); ax[0].set_ylabel("step period ms")
ax[0].set_title("A. control-loop timing (real)")
if bag is not None:
    ax[1].plot(bag["t"][1:] - real["t_wall"][0], np.diff(bag["t"]) * 1e3, lw=0.5); ax[1].set_ylabel("/low_state gap ms")
ax[1].set_xlabel("s since policy start")
fig.tight_layout(); fig.savefig(f"{OUT}/A_timing.png", dpi=110); plt.close(fig)

# ------------------------------------------------------------------ B. IMU

TRAIN_NOISE_STD = {"base_ang_vel": 0.4 / np.sqrt(12), "joint_pos": 0.02 / np.sqrt(12), "joint_vel": 1.0 / np.sqrt(12)}


def detrend_std(x, w=25):
    """Std of x minus its moving average (w samples): noise around the slow motion."""
    k = np.ones(w) / w
    sm = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, x)
    return (x - sm)[w:-w].std(0)


def window_mask_bag(phase_mask):
    if bag is None:
        return None
    tt = bag["t"]
    ts = real["t_wall"][:N]
    sel = np.zeros(len(tt), bool)
    for i in np.nonzero(phase_mask)[0]:
        a = np.searchsorted(tt, ts[i]); b = np.searchsorted(tt, ts[i] + DT)
        sel[a:b] = True
    return sel


hold_bag = window_mask_bag(HOLD)
gyro_obs_r = real["root_ang_vel_b"][:N]
gyro_obs_s = sim["root_ang_vel_b"][:N] if sim else None
if HOLD.any():
    nr = detrend_std(gyro_obs_r[HOLD], 5); ns = detrend_std(gyro_obs_s[HOLD], 5) if sim else None
    add("B IMU", "gyro noise in obs during hold, x/y/z (rad/s)", " / ".join(fmt(v) for v in nr),
        f"train noise std {TRAIN_NOISE_STD['base_ang_vel']:.3f} (sim has none)",
        "RED" if nr.max() > 2 * TRAIN_NOISE_STD["base_ang_vel"] else ("AMBER" if nr.max() > TRAIN_NOISE_STD["base_ang_vel"] else "ok"))
    add("B IMU", "gyro mean during hold, x/y/z (rad/s)", " / ".join(fmt(v) for v in gyro_obs_r[HOLD].mean(0)),
        " / ".join(fmt(v) for v in gyro_obs_s[HOLD].mean(0)) if sim else "", "AMBER" if np.abs(gyro_obs_r[HOLD].mean(0)).max() > 0.02 else "ok")
if bag is not None and hold_bag.any():
    gy = bag["gyro"][hold_bag]
    f = np.fft.rfftfreq(len(gy), 1 / 500.0); P = np.abs(np.fft.rfft(gy - gy.mean(0), axis=0)) ** 2
    hf = P[f > 25].sum(0) / P[f > 0.5].sum(0)
    add("B IMU", "gyro power above 25 Hz during hold (share, x/y/z), 500 Hz bag", " / ".join(f"{v:.0%}" for v in hf), "", "AMBER" if hf.max() > 0.5 else "ok")
    acc = bag["acc"][hold_bag]
    gmag = np.linalg.norm(acc, axis=1)
    add("B IMU", "accelerometer |g| during hold mean / std (m/s^2)", f"{gmag.mean():.3f} / {gmag.std():.3f}", "9.81", "AMBER" if abs(gmag.mean() - 9.81) > 0.2 else "ok")
    # tilt from accelerometer vs from the orientation estimate
    rpy = bag["rpy"][hold_bag]
    g_acc = acc / gmag[:, None]
    cr, sr, cp, sp = np.cos(rpy[:, 0]), np.sin(rpy[:, 0]), np.cos(rpy[:, 1]), np.sin(rpy[:, 1])
    g_rpy = np.stack([-sp, sr * cp, cr * cp], 1)          # gravity-up direction in body frame from roll/pitch
    ang = np.degrees(np.arccos(np.clip((g_acc * g_rpy).sum(1), -1, 1)))
    add("B IMU", "accelerometer vs orientation estimate tilt disagreement, hold (deg)", f"{ang.mean():.2f}", "0", "AMBER" if ang.mean() > 1.0 else "ok")
    note("B IMU", f"hold roll/pitch from IMU estimate: {np.degrees(rpy[:,0].mean()):.2f} / {np.degrees(rpy[:,1].mean()):.2f} deg; "
                  f"from accelerometer: {np.degrees(np.arctan2(g_acc[:,1], g_acc[:,2]).mean()):.2f} / {np.degrees(np.arcsin(-g_acc[:,0]).mean()):.2f} deg")
    yaw = np.unwrap(bag["rpy"][hold_bag, 2])
    drift = (yaw[-1] - yaw[0]) / max(bag["t"][hold_bag][-1] - bag["t"][hold_bag][0], 1e-3)
    add("B IMU", "yaw drift during hold (deg/min)", f"{np.degrees(drift) * 60:.2f}", "", "AMBER" if abs(np.degrees(drift) * 60) > 2 else "ok")
def smooth_bag(x, w=10):
    k = np.ones(w) / w
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, x)


def compare_rates(x, y, max_lag):
    """corr, scale (std ratio y/x, sign from corr: no noise bias), lag in samples maximizing corr; x,y smoothed."""
    best = (-2, 0)
    for L in range(-max_lag, max_lag + 1):
        c = np.corrcoef(np.roll(x, L)[50:-50], y[50:-50])[0, 1]
        if c > best[0]:
            best = (c, L)
    return best[0], np.sign(best[0]) * y[50:-50].std() / max(x[50:-50].std(), 1e-12), best[1]


if bag is not None:
    # gyro vs body rate from the orientation estimate (whole run): axis/sign/scale/lag
    # Differentiate on the firmware's uniform 2 ms grid: bag receive times bunch up (1% of gaps < 0.5 ms), and
    # dividing by them turns the derivative into spikes.
    rpy = smooth_bag(np.unwrap(bag["rpy"], axis=0)); d = np.gradient(rpy, 0.002, axis=0)
    r, p = rpy[:, 0], rpy[:, 1]
    wx = d[:, 0] - np.sin(p) * d[:, 2]
    wy = np.cos(r) * d[:, 1] + np.sin(r) * np.cos(p) * d[:, 2]
    wz = -np.sin(r) * d[:, 1] + np.cos(r) * np.cos(p) * d[:, 2]
    wb = np.stack([wx, wy, wz], 1)
    gy = smooth_bag(bag["gyro"])
    res = []
    for a in range(3):
        c, sc, L = compare_rates(wb[:, a], gy[:, a], 25)
        res.append((c, sc, L * 2.0))
    add("B IMU", "gyro vs rate of orientation estimate: corr / scale / lag(ms), x y z",
        "; ".join(f"{c:.2f}/{s:.2f}/{l:+.0f}" for c, s, l in res), "1.00/1.00/0",
        "RED" if min(c for c, _, _ in res) < 0.7 or any(abs(s - 1) > 0.3 for _, s, _ in res) else
        ("AMBER" if any(abs(l) > 10 for _, _, l in res) else "ok"))
if sim and HOLD.any():
    dr = np.degrees(real["root_rpy_w"][:N][HOLD, :2].mean(0)); ds = np.degrees(sim["root_rpy_w"][:N][HOLD, :2].mean(0))
    add("B IMU", "trunk roll / pitch during hold (deg)", f"{dr[0]:.2f} / {dr[1]:.2f}", f"{ds[0]:.2f} / {ds[1]:.2f}",
        "RED" if np.abs(dr - ds).max() > 3 else ("AMBER" if np.abs(dr - ds).max() > 1.5 else "ok"))

if bag is not None:
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    if hold_bag.any():
        gy = bag["gyro"][hold_bag]; f = np.fft.rfftfreq(len(gy), 1 / 500.0)
        for a, lbl in enumerate("xyz"):
            ax[0].semilogy(f[1:], np.abs(np.fft.rfft(gy[:, a] - gy[:, a].mean()))[1:] ** 2, lw=0.6, label=lbl)
        ax[0].set_xlabel("Hz"); ax[0].set_title("B. gyro spectrum during hold (real, 500 Hz)"); ax[0].legend()
    ax[1].plot(t_real, np.degrees(real["root_rpy_w"][:N, 1]), label="real pitch")
    if sim: ax[1].plot(t_real, np.degrees(sim["root_rpy_w"][:N, 1]), label="sim pitch")
    ax[1].plot(t_real, np.degrees(real["root_rpy_w"][:N, 0]), label="real roll")
    if sim: ax[1].plot(t_real, np.degrees(sim["root_rpy_w"][:N, 0]), label="sim roll")
    ax[1].set_ylabel("deg"); ax[1].legend(fontsize=7); shade_phases(ax[1])
    fig.tight_layout(); fig.savefig(f"{OUT}/B_imu.png", dpi=110); plt.close(fig)

# ------------------------------------------------------------------ C. joint encoders

qr, qs = real["joint_pos"][:N], sim["joint_pos"][:N] if sim else None
vr, vs = real["joint_vel"][:N], sim["joint_vel"][:N] if sim else None
if HOLD.any():
    nqr = detrend_std(qr[HOLD], 5); nvr = detrend_std(vr[HOLD], 5)
    nqs = detrend_std(qs[HOLD], 5) if sim else None; nvs = detrend_std(vs[HOLD], 5) if sim else None
    add("C joints", "joint position noise in obs, hold, max over legs (rad)", fmt(nqr[LEG].max()),
        f"train noise std {TRAIN_NOISE_STD['joint_pos']:.4f}", "AMBER" if nqr[LEG].max() > TRAIN_NOISE_STD["joint_pos"] else "ok")
    add("C joints", "joint velocity noise in obs, hold, max over legs (rad/s) / joint", f"{nvr[LEG].max():.3f} / {JN[LEG[int(nvr[LEG].argmax())]]}",
        f"train noise std {TRAIN_NOISE_STD['joint_vel']:.2f}", "AMBER" if nvr[LEG].max() > TRAIN_NOISE_STD["joint_vel"] else "ok")
    if sim:
        off = qr[HOLD].mean(0) - qs[HOLD].mean(0)
        worst = np.argsort(-np.abs(off))[:4]
        add("C joints", "hold pose real - sim, worst joints (rad)", ", ".join(f"{JN[j]} {off[j]:+.3f}" for j in worst), "0",
            "RED" if np.abs(off[LEG]).max() > 0.05 else ("AMBER" if np.abs(off[LEG]).max() > 0.02 else "ok"))
if bag is not None:
    q, dq, t = bag["q"], bag["dq"], bag["t"]
    dpos = np.diff(q, axis=0)
    qstep = np.array([np.min(np.abs(c[np.abs(c) > 1e-7])) if (np.abs(c) > 1e-7).any() else np.nan for c in dpos.T])
    add("C joints", "encoder smallest step, legs, max (rad)", fmt(np.nanmax(qstep[LEG])), "", "AMBER" if np.nanmax(qstep[LEG]) > 1e-3 else "ok")
    fd = np.gradient(smooth_bag(q), 0.002, axis=0)   # uniform grid, see the gyro check above
    dqs = smooth_bag(dq)
    out = np.array([(lambda c, sc, L: (c, sc, L * 2.0))(*compare_rates(fd[:, j], dqs[:, j], 15)) for j in LEG])
    note("C joints", "motor dq vs d(q)/dt per leg joint (corr/scale/lag ms): " + ", ".join(
        f"{JN[j]} {c:.2f}/{sc:.2f}/{L:+.0f}" for j, (c, sc, L) in zip(LEG, out)))
    add("C joints", "motor dq vs d(q)/dt, legs: min corr / scale range / lag range (ms)",
        f"{out[:,0].min():.2f} / {out[:,1].min():.2f}-{out[:,1].max():.2f} / {out[:,2].min():+.0f}..{out[:,2].max():+.0f}", "1 / 1 / 0",
        "RED" if out[:, 0].min() < 0.8 or np.abs(out[:, 1] - 1).max() > 0.3 else ("AMBER" if np.abs(out[:, 2]).max() > 6 else "ok"))
# command -> response sign/order sanity
def follow_corr(log):
    """Per joint: best correlation of position with its own command over lags 0-10 steps (absolute signals)."""
    tg, q = log["dof_targets"][:N], log["joint_pos"][:N]
    out = np.full(len(JN), np.nan)
    for j in range(len(JN)):
        if tg[:, j].std() > 1e-3:
            out[j] = max(np.corrcoef(tg[: N - k, j], q[k:, j])[0, 1] for k in range(0, 11))
    return out


fcr = follow_corr(real); fcs = follow_corr(sim) if sim else None
bad = [f"{JN[j]} {fcr[j]:.2f} (sim {fcs[j]:.2f})" if sim else f"{JN[j]} {fcr[j]:.2f}" for j in range(len(JN))
       if np.isfinite(fcr[j]) and (fcr[j] < 0.5 or (sim and fcs[j] - fcr[j] > 0.2))]
add("C joints", "joints whose position follows their command much worse than in sim", ", ".join(bad) or "none", "none",
    "RED" if bad else "ok")

# ------------------------------------------------------------------ D. actuation and torque

taur, taus = real["feedback_torque"][:N], sim["feedback_torque"][:N] if sim else None


def pd_fit(tg, q, v, tau, idx):
    exp = KP * (tg - q) - KD * v
    out = {}
    for j in idx:
        x, y = exp[1:, j], tau[1:, j]
        out[j] = ((x * y).sum() / max((x * x).sum(), 1e-12), np.corrcoef(x, y)[0, 1] if x.std() > 1e-6 else np.nan)
    return out


fr_ = pd_fit(real["dof_targets"][:N], qr, vr, taur, LEG)
fs_ = pd_fit(sim["dof_targets"][:N], qs, vs, taus, LEG) if sim else None
slr = np.array([fr_[j][0] for j in LEG]); sls = np.array([fs_[j][0] for j in LEG]) if sim else None
add("D actuation", "measured torque / ideal PD torque, legs: min-max slope",
    f"{slr.min():.2f}-{slr.max():.2f}", f"{sls.min():.2f}-{sls.max():.2f}" if sim else "",
    "RED" if (sim and np.abs(slr - sls).max() > 0.5) else ("AMBER" if (sim and np.abs(slr - sls).max() > 0.25) else "ok"))
note("D actuation", "torque/PD slope per leg joint (real vs sim): " + ", ".join(
    f"{JN[j]} {fr_[j][0]:.2f}/{fs_[j][0]:.2f}" if sim else f"{JN[j]} {fr_[j][0]:.2f}" for j in LEG))
satr = (np.abs(taur) >= 0.9 * EFFORT).mean(0); sats = (np.abs(taus) >= 0.9 * EFFORT).mean(0) if sim else None
add("D actuation", "time at >=90% torque limit, worst leg joint", f"{100*satr[LEG].max():.1f}% {JN[LEG[int(satr[LEG].argmax())]]}",
    f"{100*sats[LEG].max():.1f}% {JN[LEG[int(sats[LEG].argmax())]]}" if sim else "", flag_ratio(satr[LEG].max() + 1e-3, sats[LEG].max() + 1e-3) if sim else "")
for name, s_ in (("real", None),):
    pass
if bag is not None:
    ank = [i for i, n in enumerate(JN) if "Ankle" in n]
    pt = np.abs(bag["par_tau"][:, ank]).max(0)
    note("D actuation", "ankle drive-motor |tau| max (parallel side, Nm): " + ", ".join(f"{JN[j]} {v:.1f}" for j, v in zip(ank, pt)))


def lag_ms(tg, q, j):
    a, b = np.diff(tg[:, j]), np.diff(q[:, j])
    if a.std() < 1e-4:
        return np.nan
    cc = [np.corrcoef(a[: len(a) - k], b[k:])[0, 1] for k in range(0, 8)]
    return int(np.nanargmax(cc)) * DT * 1e3


lr = np.array([lag_ms(real["dof_targets"][:N], qr, j) for j in LEG])
ls = np.array([lag_ms(sim["dof_targets"][:N], qs, j) for j in LEG]) if sim else None
add("D actuation", "command->position response lag, legs median / max (ms)", f"{np.nanmedian(lr):.0f} / {np.nanmax(lr):.0f}",
    f"{np.nanmedian(ls):.0f} / {np.nanmax(ls):.0f}" if sim else "", flag_ratio(np.nanmedian(lr) + 1, np.nanmedian(ls) + 1) if sim else "")
ter = real["dof_targets"][:N] - qr; tes = (sim["dof_targets"][:N] - qs) if sim else None
track_rows = []
for ph in PHASE_ORDER:
    m = PH_R == ph
    if m.sum() < 5:
        continue
    a = np.sqrt((ter[m][:, LEG] ** 2).mean(0)); b = np.sqrt((tes[m][:, LEG] ** 2).mean(0)) if sim else None
    j = int(np.argmax(a / (b + 1e-3))) if sim else int(np.argmax(a))
    track_rows.append(f"{ph}: worst {JN[LEG[j]]} {a[j]:.3f} vs sim {b[j]:.3f}" if sim else f"{ph}: {JN[LEG[j]]} {a[j]:.3f}")
note("D actuation", "command-vs-position error (rad RMS) per phase: " + "; ".join(track_rows))
if sim:
    m = np.isin(PH_R, ["squat_grasp", "rise_lift", "squat_place", "rise_release"])
    if m.any():
        a = np.sqrt((ter[m][:, LEG] ** 2).mean()); b = np.sqrt((tes[m][:, LEG] ** 2).mean())
        add("D actuation", "command-vs-position error during squats/rises, legs RMS (rad)", fmt(a), fmt(b), flag_ratio(a, b))

fig, axs = plt.subplots(3, 4, figsize=(15, 8)); axs = axs.ravel()
for k, j in enumerate(LEG[:12]):
    axs[k].plot(t_real, taur[:, j], lw=0.6, label="real tau")
    if sim: axs[k].plot(t_real, taus[:, j], lw=0.6, label="sim tau")
    axs[k].axhline(EFFORT[j], color="r", lw=0.5); axs[k].axhline(-EFFORT[j], color="r", lw=0.5)
    axs[k].set_title(JN[j], fontsize=8)
axs[0].legend(fontsize=6); fig.suptitle("D. leg joint torques (red = effort limit)")
fig.tight_layout(); fig.savefig(f"{OUT}/D_torques.png", dpi=110); plt.close(fig)

# ------------------------------------------------------------------ E. policy outputs

ar, as_ = real["action"][:N], sim["action"][:N] if sim else None


def hf_share(a, fc=5.0):
    f = np.fft.rfftfreq(len(a), DT); P = np.abs(np.fft.rfft(a - a.mean(0), axis=0)) ** 2
    return P[f > fc].sum(0) / max(P[f > 0].sum(), 1e-12) * P.shape[1] if False else P[f > fc].sum(0) / np.maximum(P[f > 0].sum(0), 1e-12)


m = ~np.isin(PH_R, ["transition"])
hr = hf_share(ar[m][:, :len(JN)]); hs = hf_share(as_[m][:, :len(JN)]) if sim else None
add("E actions", "action power above 5 Hz (share), legs mean", f"{hr[[min(j, ar.shape[1]-1) for j in LEG]].mean():.0%}",
    f"{hs[[min(j, as_.shape[1]-1) for j in LEG]].mean():.0%}" if sim else "", flag_ratio(hr.mean(), hs.mean()) if sim else "")
jr = np.sqrt((np.diff(ar[m], 2, axis=0) ** 2).mean()); js = np.sqrt((np.diff(as_[m], 2, axis=0) ** 2).mean()) if sim else None
add("E actions", "action jerk (RMS 2nd difference)", fmt(jr), fmt(js) if sim else "", flag_ratio(jr, js) if sim else "")
sr_, ss_ = ar[m].std(0).mean(), as_[m].std(0).mean() if sim else None
add("E actions", "action std, mean over outputs", fmt(sr_), fmt(ss_) if sim else "", flag_ratio(sr_, ss_) if sim else "")

# ------------------------------------------------------------------ F. observations

layout = meta.get("obs_layout") or []
groups: dict[str, list[int]] = {}
pos = 0
for name, w in layout:
    base = name.split("/", 1)[1] if "/" in name else name
    groups.setdefault(base, []).extend(range(pos, pos + w)); pos += w
obr, obs_s = real["obs"][:N], sim["obs"][:N] if sim else None
shift_rows = []
if sim and groups:
    for g, idx in groups.items():
        a, b = obr[:, idx], obs_s[:, idx]
        diff = np.abs(a - b).mean()
        scale = max(b.std(), 1e-6)
        stdr = a.std(0).mean() / max(b.std(0).mean(), 1e-9)
        hfr = hf_share(a).mean(); hfs = hf_share(b).mean()
        shift_rows.append((g, diff / scale, stdr, hfr, hfs))
    shift_rows.sort(key=lambda r: -r[1])
    for g, d, s, hfr, hfs in shift_rows[:6]:
        add("F observations", f"term '{g}': |real-sim| / sim std; std ratio; >5 Hz share real/sim",
            f"{d:.2f}; {s:.2f}x; {hfr:.0%}", f"{hfs:.0%}", "RED" if d > 1.0 or s > 2 or s < 0.5 else ("AMBER" if d > 0.5 or s > 1.3 else "ok"))

# counterfactual inference
ckpt = args.checkpoint or meta["files"]["checkpoint_path"]["path"]
if not os.path.isfile(ckpt):
    alt = os.path.join("tasks/hoi_track", cfg["policy"]["checkpoint_path"])
    ckpt = alt if os.path.isfile(alt) else None
cf_rows = []
if ckpt and groups:
    import torch
    from booster_deploy.utils.policy_model import load_policy_model
    model = load_policy_model(ckpt)

    def act(o):
        with torch.no_grad():
            return np.stack([model(torch.from_numpy(x.astype(np.float32))[None]).reshape(-1).numpy() for x in o])

    base = act(obr)
    ident = np.abs(base - real["action"][:N]).max()
    focus = np.isin(PH_R, ["squat_grasp", "rise_lift", "squat_place", "rise_release"])
    leg_out = [j for j in range(base.shape[1]) if SJN[j] in [JN[i] for i in LEG]] if base.shape[1] == len(SJN) else list(range(base.shape[1]))

    def smooth(x, w=5):
        k = np.ones(w) / w
        return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, x)

    variants = {}
    for g, idx in groups.items():
        if sim:
            o = obr.copy(); o[:, idx] = obs_s[:, idx]; variants[f"swap '{g}' with sim"] = o
        if g in ("base_ang_vel", "joint_vel"):
            o = obr.copy(); o[:, idx] = smooth(obr[:, idx]); variants[f"low-pass '{g}' (5-step mean)"] = o
    scale = base[focus][:, leg_out].std() if focus.any() else base.std()
    for name, o in variants.items():
        a = act(o)
        d = np.sqrt(((a - base)[focus][:, leg_out] ** 2).mean()) / max(scale, 1e-9) if focus.any() else np.nan
        cf_rows.append((name, d))
    cf_rows.sort(key=lambda r: -r[1])
    add("F observations", "counterfactual check: recomputed vs logged actions, max diff", f"{ident:.1e}", "~0", "ok" if ident < 1e-3 else "RED")
    for name, d in cf_rows[:6]:
        add("F observations", f"counterfactual: {name} -> leg action change in squats/rises (x action std)", f"{d:.2f}", "",
            "RED" if d > 0.5 else ("AMBER" if d > 0.2 else "ok"))

# ------------------------------------------------------------------ G. whole body (FK)

import mujoco  # noqa: E402
from booster_assets import BOOSTER_ASSETS_DIR  # noqa: E402

# The robot alone is enough for FK and its CoM; object scenes may not exist on this machine.
scene = cfg["robot"]["mjcf_path"].replace("{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR))
M = mujoco.MjModel.from_xml_path(scene)
D_ = mujoco.MjData(M)
# The robot's hinge joints in model order are the controller's real joint order (see
# MujocoController._robot_joint_addresses); MJCF joint names differ from the config's.
jq = [int(M.jnt_qposadr[j]) for j in range(M.njnt) if M.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
      and "_drive_" not in (mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_JOINT, j) or "")
      and "_rod_" not in (mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_JOINT, j) or "")]
assert len(jq) == len(JN), f"{len(jq)} hinge joints in {scene}, expected {len(JN)}"
feet = {s: mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("left", "right")}
sole = {s: [g for g in range(M.ngeom) if M.geom_bodyid[g] == b and M.geom_contype[g]][0] for s, b in feet.items()}
root = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, M.body(1).name)


def sole_corners(g):
    hs = M.geom_size[g]
    c = np.array([[sx * hs[0], sy * hs[1], -hs[2]] for sx in (-1, 1) for sy in (-1, 1)])
    return D_.geom_xpos[g] + c @ D_.geom_xmat[g].reshape(3, 3).T     # heel, heel, toe, toe (x -/+)


def fk_metrics(joint_pos, quat):
    """Per step: per foot toe-up angle (deg), heel lift relative to toe (cm); CoM margin to the front/back edge
    of the support area (cm) in the heading frame."""
    T = len(joint_pos)
    out = {"toe_up_l": np.zeros(T), "toe_up_r": np.zeros(T), "heel_lift_l": np.zeros(T), "heel_lift_r": np.zeros(T),
           "margin_front": np.zeros(T), "margin_back": np.zeros(T), "com_x": np.zeros(T)}
    for i in range(T):
        D_.qpos[:] = 0; D_.qpos[3:7] = quat[i]; D_.qpos[2] = 1.0
        D_.qpos[jq] = joint_pos[i]
        mujoco.mj_kinematics(M, D_); mujoco.mj_comPos(M, D_)
        w, x, y, z = quat[i]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        Rh = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])   # world -> heading
        corners = []
        for s, tag in (("left", "l"), ("right", "r")):
            g = sole[s]
            fx = D_.geom_xmat[g].reshape(3, 3)[:, 0]
            out[f"toe_up_{tag}"][i] = np.degrees(np.arcsin(np.clip(fx[2], -1, 1)))
            c = sole_corners(g)
            out[f"heel_lift_{tag}"][i] = (c[:2, 2].mean() - c[2:, 2].mean()) * 100
            corners.append(c)
        c = np.concatenate(corners) @ Rh.T
        low = c[:, 2].min()
        support = c[c[:, 2] < low + 0.01]                 # corners within 1 cm of the lowest = on the floor
        com = D_.subtree_com[root] @ Rh.T
        out["com_x"][i] = com[0] - c[:, 0].mean()
        out["margin_front"][i] = (support[:, 0].max() - com[0]) * 100
        out["margin_back"][i] = (com[0] - support[:, 0].min()) * 100
    return out


Gr = fk_metrics(qr, real["root_quat_w"][:N])
Gs = fk_metrics(qs, sim["root_quat_w"][:N]) if sim else None
Gref = None
if REF is not None:
    fr = np.clip(FR_R, 0, len(REF["ref_joint_pos"]) - 1)
    Gref = fk_metrics(REF["ref_joint_pos"][fr][:, S2R], REF["ref_anchor_quat_w"][fr])
np.savez(f"{OUT}/fk_metrics.npz", **{f"real_{k}": v for k, v in Gr.items()},
         **({f"sim_{k}": v for k, v in Gs.items()} if sim else {}), **({f"ref_{k}": v for k, v in Gref.items()} if Gref else {}),
         phase=PH_R.astype(str))
focus = np.isin(PH_R, ["squat_grasp", "rise_lift", "carry", "squat_place", "rise_release"])
for key, label in (("heel_lift_l", "left heel lift vs toe (cm), min/max in motion"), ("heel_lift_r", "right heel lift vs toe (cm), min/max in motion"),
                   ("margin_back", "CoM margin to back edge of support (cm), min in motion"), ("margin_front", "CoM margin to front edge of support (cm), min in motion")):
    if not focus.any():
        break
    r = Gr[key][focus]; s = Gs[key][focus] if sim else None; rf = Gref[key][focus] if Gref else None
    if "margin" in key:
        add("G whole body", label, f"{r.min():.1f}", f"sim {s.min():.1f} | ref {rf.min():.1f}" if sim else "",
            "RED" if r.min() < 1.0 else ("AMBER" if r.min() < 3.0 else "ok"))
    else:
        add("G whole body", label, f"{r.min():+.1f} / {r.max():+.1f}", f"sim {s.min():+.1f} / {s.max():+.1f} | ref {rf.min():+.1f} / {rf.max():+.1f}" if sim else "",
            "RED" if np.abs(r).max() > 2.0 + np.abs(rf).max() else ("AMBER" if np.abs(r).max() > 1.0 + np.abs(rf).max() else "ok"))
if REF is not None:
    refj = REF["ref_joint_pos"][np.clip(FR_R, 0, len(REF["ref_joint_pos"]) - 1)][:, S2R]
    er = np.sqrt(((qr - refj)[focus][:, LEG] ** 2).mean(0)); es = np.sqrt(((qs - refj)[focus][:, LEG] ** 2).mean(0)) if sim else None
    w = np.argsort(-(er - (es if sim else 0)))[:4]
    add("G whole body", "joint tracking vs reference in motion, worst legs real (rad RMS)", ", ".join(f"{JN[LEG[k]]} {er[k]:.3f}" for k in w),
        ", ".join(f"{es[k]:.3f}" for k in w) if sim else "", flag_ratio(er[w[0]], es[w[0]]) if sim else "")
    rq = REF["ref_anchor_quat_w"][np.clip(FR_R, 0, len(REF["ref_joint_pos"]) - 1)]
    def pitch(q):
        w_, x_, y_, z_ = q.T
        return np.degrees(np.arcsin(np.clip(2 * (w_ * y_ - z_ * x_), -1, 1)))
    pr, ps, pref = pitch(real["root_quat_w"][:N]), pitch(sim["root_quat_w"][:N]) if sim else None, pitch(rq)
    a = np.sqrt(((pr - pref)[focus] ** 2).mean()); b = np.sqrt(((ps - pref)[focus] ** 2).mean()) if sim else None
    add("G whole body", "trunk pitch error vs reference in motion (deg RMS)", f"{a:.1f}", f"{b:.1f}" if sim else "", flag_ratio(a, b) if sim else "")
# payload: change in arm/leg torque once carrying vs during hold
carry = PH_R == "carry"
if carry.any() and HOLD.any() and sim:
    arm = [i for i, n in enumerate(JN) if "Shoulder" in n or "Elbow" in n]
    dr_ = np.abs(taur[carry][:, arm].mean(0) - taur[HOLD][:, arm].mean(0)).sum()
    ds_ = np.abs(taus[carry][:, arm].mean(0) - taus[HOLD][:, arm].mean(0)).sum()
    add("G whole body", "arm torque change carrying vs holding (sum |dNm|), a payload proxy", fmt(dr_), fmt(ds_) + " (sim box 1 kg)", flag_ratio(dr_, ds_))

fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
for a_, key, lbl in ((ax[0], "heel_lift_l", "left heel - toe height (cm)"), (ax[1], "heel_lift_r", "right heel - toe height (cm)"),
                     (ax[2], "margin_back", "CoM margin to back edge (cm)"), (ax[3], "margin_front", "CoM margin to front edge (cm)")):
    a_.plot(t_real, Gr[key], label="real", lw=1)
    if sim: a_.plot(t_real, Gs[key], label="sim", lw=1)
    if Gref: a_.plot(t_real, Gref[key], label="reference", lw=1, ls="--")
    a_.set_ylabel(lbl, fontsize=8); a_.axhline(0, color="k", lw=0.5)
    for st in STALL_STEPS: a_.axvline(st * DT, color="r", lw=0.3, alpha=0.5)
    shade_phases(a_)
ax[0].legend(fontsize=7); ax[-1].set_xlabel("s since policy start (red ticks = control stalls)")
fig.suptitle("G. feet and balance: real vs sim vs reference"); fig.tight_layout(); fig.savefig(f"{OUT}/G_feet_balance.png", dpi=110); plt.close(fig)

fig, axs = plt.subplots(3, 4, figsize=(15, 8)); axs = axs.ravel()
for k, j in enumerate(LEG[:12]):
    axs[k].plot(t_real, qr[:, j], lw=0.8, label="real")
    if sim: axs[k].plot(t_real, qs[:, j], lw=0.8, label="sim")
    if REF is not None: axs[k].plot(t_real, refj[:, j], lw=0.8, ls="--", label="ref")
    axs[k].set_title(JN[j], fontsize=8)
axs[0].legend(fontsize=6); fig.suptitle("G. leg joint angles: real vs sim vs reference")
fig.tight_layout(); fig.savefig(f"{OUT}/G_leg_joints.png", dpi=110); plt.close(fig)

# ------------------------------------------------------------------ H. configuration

if REF is not None:
    asc = REF["action_scale"]
    dep = 0.25 * EFFORT / KP
    mism = np.abs(asc[S2R] - dep) / asc[S2R]
    add("H config", "action scale in reference vs 0.25*effort/kp of deployed gains (max rel diff)", f"{mism.max():.1%}", "0",
        "AMBER" if mism.max() > 0.05 else "ok")
    dj = np.abs(REF["default_joint_pos"][S2R] - np.array(cfg["robot"]["default_joint_pos"])).max()
    add("H config", "default pose: reference vs deploy config (max rad)", fmt(dj), "0", "AMBER" if dj > 1e-3 else "ok")
    names_ok = list(REF["joint_names"]) == SJN if "joint_names" in REF else None
    add("H config", "joint order: reference vs deploy sim_joint_names", str(names_ok), "True", "ok" if names_ok else "RED")
if bag is not None and len(ctrl["t"]):
    kp_sent = ctrl["kp"][-1]; kd_sent = ctrl["kd"][-1]
    add("H config", "gains on the bus == config (max |diff| kp / kd)", f"{np.abs(kp_sent - KP).max():.3g} / {np.abs(kd_sent - KD).max():.3g}", "0",
        "RED" if np.abs(kp_sent - KP).max() > 1e-3 else "ok")
    add("H config", "MotorCmd.weight sent", ", ".join(map(str, np.unique(ctrl["weight"]).round(3))), "?", "AMBER")
add("H config", "sim scene ankle model", "parallel (linkage)" if "parallel" in scene else "serial pitch/roll joints",
    "training: parallel (BoosterK1AnkleParaWrapper)", "AMBER" if "parallel" not in scene else "ok")
d_ = cfg["mujoco"].get("actuator_delay_range_s")
add("H config", "sim actuator delay range (s)", str(d_), "training 0.010-0.040", "ok" if d_ else "AMBER")

# ------------------------------------------------------------------ write summary

with open(f"{OUT}/summary.md", "w") as f:
    f.write(f"# Sim-vs-real audit: {meta.get('task')}\n\nreal: `{OBS_DIR}`  \nsim: `{SIM_DIR}`  \nsteps compared: {N} "
            f"({N*DT:.1f}s); phases: {', '.join(f'{p} {int((PH_R==p).sum()*DT*10)/10}s' for p in PHASE_ORDER)}\n\n")
    f.write("| category | check | real | sim / reference | flag |\n|---|---|---|---|---|\n")
    for r in rows:
        f.write("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |\n")
    f.write("\n## Notes\n")
    for k, v in notes.items():
        for t in v:
            f.write(f"- **{k}**: {t}\n")
    f.write("\nPlots: " + ", ".join(sorted(os.path.basename(p) for p in glob.glob(f"{OUT}/*.png"))) + "\n")
print(open(f"{OUT}/summary.md").read())
