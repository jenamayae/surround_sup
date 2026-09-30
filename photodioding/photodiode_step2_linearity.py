"""Step 2: sensor linearity (NI USB-6211 + eemagine XS-355).

Wiring as in step 1, final version: white -> AI0, black -> AI GND, shield -> AI GND,
red -> +5V, AI8 jumpered to +5V (so ai0 reads output relative to the supply).

Shows k-of-16 dither patterns (4x4 Bayer order, only full-black and full-white
pixels, so gamma plays no role) plus a 50% pattern at 8 px scale as the
additivity check. Every test pattern is bracketed by black and white reference
patches; each test is normalized as (V - black) / (white - black) using the
references measured just before and after it, which cancels offset drift and
display/sensor warm-up. Order is randomized within each repeat.

If the sensor is linear here, the k/16 patterns read k/16 and the 1 px and 8 px
50% patterns agree. Output: printed table, CSV, raw samples (.npz), plot (.png).
"""
import sys
import csv
import datetime
import numpy as np
import nidaqmx
import nidaqmx.system
from nidaqmx.constants import TerminalConfiguration as TC, AcquisitionType, READ_ALL_AVAILABLE
from psychopy import visual, event, core

# ---- settings -------------------------------------------------------------
SCREEN = 0                   # PsychoPy screen index of the ASUS
EXPECTED_SIZE = (2560, 1440)  # native resolution; anything else means scaling
REFRESH_HZ = 120
FS = 10000                   # DAQ sample rate, Hz
V_RANGE = 1.0                # +/- volts
PATCH_PX = 256               # power of two keeps the texture exact and pixel-aligned
BG = 0.0                     # background, PsychoPy rgb (-1..1); 0 = mid gray
SETTLE_FRAMES = 30           # frames shown before measuring each state (0.25 s)
MEASURE_FRAMES = 60          # frames averaged per state (0.5 s, integer frames)
N_REPS = 5                   # repeats of the full randomized set
K_LEVELS = list(range(0, 17, 2))  # k of 16 pixels white
SEED = None                  # set an int for a reproducible order

BAYER4 = np.array([[0, 8, 2, 10],
                   [12, 4, 14, 6],
                   [3, 11, 1, 9],
                   [15, 7, 13, 5]])


def dither(k, scale=1):
    """PATCH_PX x PATCH_PX pattern with k of every 16 pixels white, in Bayer order.
    k=8 gives a checkerboard; scale>1 enlarges each pixel into a scale x scale block."""
    tile = np.where(BAYER4 < k, 1.0, -1.0)
    tile = np.kron(tile, np.ones((scale, scale)))
    reps = PATCH_PX // tile.shape[0]
    return np.tile(tile, (reps, reps))


# ---- DAQ ---------------------------------------------------------------------
devices = list(nidaqmx.system.System.local().devices)
if not devices:
    sys.exit("No NI device found. Check the USB cable and NI MAX.")
dev = next((d for d in devices if "6211" in d.product_type), devices[0])
print(f"Using {dev.name} ({dev.product_type})")

# ---- display -------------------------------------------------------------------
win = visual.Window(screen=SCREEN, fullscr=True, color=BG, units="pix")
if tuple(win.size) != EXPECTED_SIZE:
    win.close()
    sys.exit(f"Window is {tuple(win.size)}, expected {EXPECTED_SIZE}. "
             "Check Windows display scaling; 1 px patterns need 1:1 pixel mapping.")
fr = win.getActualFrameRate(nIdentical=20, nMaxFrames=240)
print(f"Measured refresh: {fr}")
if fr is None or abs(fr - REFRESH_HZ) > 2:
    print(f"WARNING: refresh is not {REFRESH_HZ} Hz; frame-locked averaging will be off.")


def make_stim(arr):
    return visual.ImageStim(win, image=arr, size=(PATCH_PX, PATCH_PX), units="pix",
                            pos=(0, 0), interpolate=False, colorSpace="rgb")


stim_black = make_stim(dither(0))
stim_white = make_stim(dither(16))
tests = [(f"k{k:02d}_1px", k / 16, make_stim(dither(k, 1))) for k in K_LEVELS]
tests.append(("k08_8px", 0.5, make_stim(dither(8, 8))))

msg = visual.TextStim(win, pos=(0, -win.size[1] / 2 + 100), height=24, color=0.6,
                      wrapWidth=win.size[0] * 0.8)
n_meas = int(round(MEASURE_FRAMES * FS / REFRESH_HZ))


def pull(task):
    return np.asarray(task.read(number_of_samples_per_channel=READ_ALL_AVAILABLE), dtype=float)


def frame(stim, text):
    stim.draw()
    msg.text = text
    msg.draw()
    win.flip()
    if "escape" in event.getKeys(["escape"]):
        win.close()
        core.quit()


def run_state(task, stim, text):
    """Show stim for SETTLE_FRAMES, then collect MEASURE_FRAMES worth of samples."""
    for _ in range(SETTLE_FRAMES):
        frame(stim, text)
    pull(task)
    chunks, n = [], 0
    while n < n_meas:
        frame(stim, text)
        c = pull(task)
        chunks.append(c)
        n += c.size
    return np.concatenate(chunks)[:n_meas]


rng = np.random.default_rng(SEED)
records, raw = [], []

with nidaqmx.Task() as t:
    t.ai_channels.add_ai_voltage_chan(f"{dev.name}/ai0", terminal_config=TC.DIFF,
                                      min_val=-V_RANGE, max_val=V_RANGE)
    t.timing.cfg_samp_clk_timing(FS, sample_mode=AcquisitionType.CONTINUOUS,
                                 samps_per_chan=FS * 60)
    t.start()

    # positioning with live readout on a white patch
    live = np.zeros(0)
    event.clearEvents()
    while True:
        live = np.concatenate([live, pull(t)])[-FS // 2:]
        reading = f"{live.mean() * 1e3:8.3f} mV  sd {live.std() * 1e3:6.3f}" if live.size else "waiting"
        stim_white.draw()
        msg.text = (f"Tape the sensor over the center patch.\nlive: {reading}\n"
                    f"SPACE = start ({N_REPS} repeats)     ESC = quit")
        msg.draw()
        win.flip()
        keys = event.getKeys(["space", "escape"])
        if "escape" in keys:
            win.close()
            core.quit()
        if "space" in keys:
            break

    core.wait(1.0)  # hand away from the sensor
    running = "Running. Do not touch the sensor.     ESC = abort"
    for rep in range(N_REPS):
        order = rng.permutation(len(tests))
        seq = [("ref_black", None, stim_black)]
        for i in order:
            name, nominal, stim = tests[i]
            seq += [(name, nominal, stim), ("ref_white", None, stim_white),
                    ("ref_black", None, stim_black)]
        for pos, (name, nominal, stim) in enumerate(seq):
            x = run_state(t, stim, f"{running}\nrepeat {rep + 1}/{N_REPS}")
            raw.append(x)
            h = x.size // 2
            records.append(dict(rep=rep, pos=pos, name=name, nominal=nominal,
                                mean_v=x.mean(), sd_v=x.std(),
                                first_half_v=x[:h].mean(), second_half_v=x[h:].mean(),
                                clipped=float(np.mean(np.abs(x) >= 0.99 * V_RANGE))))
    t.stop()

# ---- normalize each test by its neighbouring references -----------------------
by_rep = {}
for r in records:
    by_rep.setdefault(r["rep"], []).append(r)

results = {}
for rep, recs in by_rep.items():
    for p, r in enumerate(recs):
        if r["name"].startswith("ref_"):
            continue
        blacks = [recs[p - 1]["mean_v"], recs[p + 2]["mean_v"]]     # before and after
        whites = [recs[p + 1]["mean_v"]]                             # after
        if p - 2 >= 0 and recs[p - 2]["name"] == "ref_white":
            whites.append(recs[p - 2]["mean_v"])                     # before, if any
        b, w = np.mean(blacks), np.mean(whites)
        r["normalized"] = (r["mean_v"] - b) / (w - b)
        results.setdefault(r["name"], (r["nominal"], []))[1].append(r["normalized"])

# ---- report ----------------------------------------------------------------------
swing = (np.mean([r["mean_v"] for r in records if r["name"] == "ref_white"])
         - np.mean([r["mean_v"] for r in records if r["name"] == "ref_black"]))
settle = max(abs(r["second_half_v"] - r["first_half_v"]) for r in records) / abs(swing)
clipped = max(r["clipped"] for r in records)

lines = [f"{'pattern':10s} {'nominal':>8s} {'measured':>9s} {'sd':>7s} {'error':>8s}"]
for name, (nominal, vals) in results.items():
    m, s = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else float("nan")
    lines.append(f"{name:10s} {nominal:8.4f} {m:9.4f} {s:7.4f} {m - nominal:+8.4f}")
lines.append(f"white-minus-black swing {swing * 1e3:.2f} mV")
lines.append(f"largest within-state settling (2nd half minus 1st) {settle:.2%} of swing")
if clipped > 0:
    lines.append(f"CLIPPING in some states: up to {clipped:.1%} of samples")
print("\n".join(lines))

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base = f"photodiode_step2_{stamp}"
with open(base + ".csv", "w", newline="") as f:
    fields = ["rep", "pos", "name", "nominal", "mean_v", "sd_v", "first_half_v",
              "second_half_v", "clipped", "normalized"]
    wr = csv.DictWriter(f, fieldnames=fields)
    wr.writeheader()
    for r in records:
        wr.writerow({k: r.get(k, "") for k in fields})
np.savez(base + ".npz", raw=np.array(raw), fs=FS, v_range=V_RANGE,
         names=np.array([r["name"] for r in records]), reps=np.array([r["rep"] for r in records]))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for name, (nominal, vals) in results.items():
        mk = "s" if name.endswith("8px") else "o"
        ax.errorbar(nominal, np.mean(vals), yerr=np.std(vals, ddof=1), fmt=mk, ms=5,
                    label="50% at 8 px" if mk == "s" else None, color="C1" if mk == "s" else "C0")
    ax.set_xlabel("fraction of pixels white (k/16)")
    ax.set_ylabel("normalized sensor reading")
    ax.set_title("Sensor linearity")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=150)
except Exception as e:
    print(f"(plot skipped: {e})")

print(f"Saved {base}.csv, {base}.npz" + (f", {base}.png" if "fig" in dir() else ""))

msg.pos = (0, 0)
msg.text = "\n".join(lines) + "\n\nAny key to close."
msg.draw()
win.flip()
event.waitKeys()
win.close()
core.quit()
