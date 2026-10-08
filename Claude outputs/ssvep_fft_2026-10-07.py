"""Epoch sub-jena07 EEG at condition onsets, FFT each trial, plot amplitude spectra per condition."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from antio.libeep import read_cnt
from antio.parser import read_info, read_triggers, read_data

cnt_path = sys.argv[1]
out_png = sys.argv[2]

# --- analysis parameters ---
skip_frames = 80  # excluded from each trial start; first frame marker (code 64) is sent at this frame
epoch_duration_s = 10.0  # (1280 - 80) frames / 120 Hz; 12 and 7.5 Hz harmonics land on exact FFT bins
roi_channels = ["O2"]
center_flicker_hz = 12
surround_flicker_hz = 7.5
plot_freq_range_hz = (1, 40)
condition_labels = {
    "1": "Condition 1: center only (45°, contrast 0.10)",
    "2": "Condition 2: center 45° + surround 45° (iso)",
    "4": "Condition 3: center 45° + surround 135° (cross)",
}

cnt = read_cnt(cnt_path)
fs = cnt.get_sample_frequency()
ch_names = read_info(cnt)[0]
data_uv = read_data(cnt)
onsets, _, descriptions, _, _ = read_triggers(cnt)

n_samples = int(round(epoch_duration_s * fs))
roi_idx = [ch_names.index(ch) for ch in roi_channels]
freqs = np.fft.rfftfreq(n_samples, 1 / fs)


def epoch_amplitude_spectrum(onset_sample):
    """Single-sided amplitude spectrum (µV) per ROI channel for one epoch."""
    epoch = data_uv[roi_idx, onset_sample:onset_sample + n_samples]
    # remove DC offset and linear drift per channel
    t = np.arange(n_samples)
    coefs = np.polyfit(t, epoch.T, 1)
    epoch = epoch - (np.outer(coefs[0], t) + coefs[1][:, None])
    spectrum = np.abs(np.fft.rfft(epoch, axis=1)) / n_samples
    spectrum[:, 1:] *= 2  # single-sided
    return spectrum


spectra_by_code = {code: [] for code in condition_labels}
epoch_start_offsets = []
for i, (onset, code) in enumerate(zip(onsets, descriptions)):
    if code in spectra_by_code:
        # epoch starts at the first frame marker after the condition onset (= frame 80)
        marker = next(o for o, d in zip(onsets[i + 1:], descriptions[i + 1:]) if d == "64")
        epoch_start_offsets.append(marker - onset)
        spectra_by_code[code].append(epoch_amplitude_spectrum(marker))
print("onset->first marker (samples):", sorted(set(epoch_start_offsets)), "expected", 80 / 120 * fs)

fig, axes = plt.subplots(len(condition_labels), 1, figsize=(9, 9), sharex=True, sharey=True)
in_range = (freqs >= plot_freq_range_hz[0]) & (freqs <= plot_freq_range_hz[1])
for ax, (code, label) in zip(axes, condition_labels.items()):
    trials = np.array(spectra_by_code[code])  # trials x channels x freqs
    mean_amp = trials.mean(axis=(0, 1))       # average over trials, then ROI channels
    # mark multiples of each tag frequency within the plotted range
    harmonic_lines = [(n * center_flicker_hz, "--") for n in range(1, 4)] + [(n * surround_flicker_hz, ":") for n in range(1, 6)]
    for f, ls in [(f, ls) for f, ls in harmonic_lines if f <= plot_freq_range_hz[1]]:
        ax.axvline(f, color="0.6", lw=1, ls=ls, zorder=0)
    ax.plot(freqs[in_range], mean_amp[in_range], color="#2a5d9f", lw=1.5)
    ax.set_title(f"{label}  (n = {len(trials)} trials)", loc="left", fontsize=10)
    ax.set_ylabel("Amplitude (µV)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92")
    i6 = np.argmin(abs(freqs - center_flicker_hz)); i75 = np.argmin(abs(freqs - surround_flicker_hz))
    ib=lambda f: np.argmin(abs(freqs - f))
    print(f"{label}: n={len(trials)}  " + "  ".join(f"{f:g} Hz={mean_amp[ib(f)]:.3f}" for f in (center_flicker_hz, 2*center_flicker_hz, surround_flicker_hz, 2*surround_flicker_hz, 3*surround_flicker_hz, 4*surround_flicker_hz)) + " µV")
axes[-1].set_xlabel("Frequency (Hz)")
axes[-1].set_xlim(plot_freq_range_hz)
fig.suptitle("2026-10-07 19:45 recording — SSVEP amplitude spectra, channel O2, frames 80–1280 of each trial (10 s)\n"
             "dashed = 12 Hz center & multiples (24, 36) · dotted = 7.5 Hz surround & multiples (15, 22.5, 30, 37.5)", fontsize=10)
fig.tight_layout()
fig.savefig(out_png, dpi=150)
print("resolution", freqs[1], "Hz; epoch samples", n_samples)
