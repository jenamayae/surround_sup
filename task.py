from __future__ import annotations

import random

import numpy as np

from config import (
    CENTERS,
    angle_to_quadrant,
    center_change_count_per_trial,
    center_change_duration_sign_reversals,
    center_flicker_hz,
    contrast_change_interval_s,
    contrast_change_start_s,
    contrast_response_keys,
    contrast_task_eligible_eccentricities,
    fixation_red_duration_s,
    frames_per_reversal,
)

QUEST_INTENSITY_VALS = np.geomspace(0.01, 0.1, num=13).tolist()
QUEST_THRESHOLD_VALS = np.geomspace(0.005, 0.15, num=25).tolist()

QUEST_CONFIG = {
    "intensity_vals": QUEST_INTENSITY_VALS,
    "threshold_vals": QUEST_THRESHOLD_VALS,
    "slope_vals": [2],
    "lower_asymptote_vals": [0.25],
    "lapse_rate_vals": [0.02],
    "response_vals": ["Correct", "Incorrect"],
    "start_intensity": QUEST_INTENSITY_VALS[-1],
    "psychometric_func": "weibull",
    "stim_scale": "linear",
    "stim_selection_method": "minNEntropy",
    "stim_selection_options": {"N": 3, "maxConsecutiveReps": 2},
    "param_estimation_method": "mean",
}


def build_keypad_map(mapping):
    inverted = {value: key for key, value in mapping.items()}
    inverted.update({f"num_{value}": key for key, value in mapping.items()})
    return inverted


RESPONSE_KEY_TO_QUADRANT = build_keypad_map(contrast_response_keys)


def eligible_centers(eligible_eccentricities=contrast_task_eligible_eccentricities):
    return [
        {**center, "center_index": index}
        for index, center in enumerate(CENTERS)
        if center["eccentricity"] in eligible_eccentricities
        and center["quadrant"] is not None
    ]


def eligible_center_indices(eligible_eccentricities=contrast_task_eligible_eccentricities):
    return [
        center["center_index"]
        for center in eligible_centers(eligible_eccentricities)
    ]


def center_change_duration_aligned_s(
    flicker_hz=center_flicker_hz,
    duration_sign_reversals=center_change_duration_sign_reversals,
):
    return float(duration_sign_reversals) / float(flicker_hz)


def is_sign_reversal_frame(frame_num, refresh_hz, flicker_hz=center_flicker_hz):
    return frame_num > 0 and frame_num % frames_per_reversal(refresh_hz, flicker_hz) == 0


def next_sign_reversal_frame(frame_num, refresh_hz, flicker_hz=center_flicker_hz):
    candidate_frame = max(1, int(frame_num))
    while not is_sign_reversal_frame(candidate_frame, refresh_hz, flicker_hz):
        candidate_frame += 1
    return candidate_frame


def advance_by_sign_reversals(
    start_frame,
    refresh_hz,
    duration_sign_reversals=center_change_duration_sign_reversals,
    flicker_hz=center_flicker_hz,
):
    if duration_sign_reversals < 1:
        raise ValueError("duration_sign_reversals must be >= 1.")

    reversal_count = 0
    candidate_frame = int(start_frame) + 1
    while True:
        if is_sign_reversal_frame(candidate_frame, refresh_hz, flicker_hz):
            reversal_count += 1
            if reversal_count == int(duration_sign_reversals):
                return candidate_frame
        candidate_frame += 1


def build_event_frame_lists(
    refresh_hz,
    trial_duration_s,
    change_start_s=contrast_change_start_s,
    change_interval_s=contrast_change_interval_s,
    change_count=center_change_count_per_trial,
    duration_sign_reversals=center_change_duration_sign_reversals,
    flicker_hz=center_flicker_hz,
):
    onset_frames = []
    offset_frames = []
    trial_frame_count = round(trial_duration_s * refresh_hz)

    for event_index in range(change_count):
        requested_onset_frame = round(
            (change_start_s + event_index * change_interval_s) * refresh_hz
        )
        onset_frame = next_sign_reversal_frame(
            requested_onset_frame,
            refresh_hz,
            flicker_hz,
        )
        offset_frame = advance_by_sign_reversals(
            onset_frame,
            refresh_hz,
            duration_sign_reversals,
            flicker_hz,
        )

        if onset_frames and onset_frame < offset_frames[-1]:
            raise ValueError("Contrast events overlap within a trial.")
        if offset_frame > trial_frame_count:
            raise ValueError(
                "Contrast events do not fit within trial_duration_s. "
                f"Got start={change_start_s}, interval={change_interval_s}, count={change_count}, "
                f"duration_sign_reversals={duration_sign_reversals}, trial_duration_s={trial_duration_s}."
            )

        onset_frames.append(onset_frame)
        offset_frames.append(offset_frame)

    return onset_frames, offset_frames


def sample_trial_task(refresh_hz, trial_duration_s):
    onset_frames, offset_frames = build_event_frame_lists(refresh_hz, trial_duration_s)
    eligible = eligible_centers()
    targets = [random.choice(eligible) for _ in onset_frames]

    return {
        "contrast_event_onset_frames": onset_frames,
        "contrast_event_offset_frames": offset_frames,
        "contrast_event_target_indices": [target["center_index"] for target in targets],
        "contrast_event_target_quadrants": [target["quadrant"] for target in targets],
    }


def response_key_to_quadrant(key):
    return RESPONSE_KEY_TO_QUADRANT.get(key)


def score_quadrant_response(target_quadrant, response_quadrant):
    if response_quadrant is None:
        return "no_response"
    if response_quadrant == target_quadrant:
        return "hit"
    return "miss"


def red_fixation_frames(refresh_hz, duration_s=fixation_red_duration_s):
    return max(1, round(duration_s * refresh_hz))


def format_log_list(values):
    serialized_values = ["N/A" if value is None else str(value) for value in values]
    return "[" + "; ".join(serialized_values) + "]"


class QuestPlusController:
    """QUEST+ (Watson 2017) in numpy, matching PsychoPy's QuestPlusHandler / questplus.

    Parameter grid axes: threshold, slope, lower asymptote, lapse rate.
    """

    def __init__(
        self,
        condition_id=None,
        quest_config=QUEST_CONFIG,
        seed=None,
    ):
        # only the options used in QUEST_CONFIG are implemented
        if (
            quest_config["psychometric_func"] != "weibull"
            or quest_config["stim_scale"] != "linear"
            or quest_config["stim_selection_method"] != "minNEntropy"
            or quest_config["param_estimation_method"] != "mean"
        ):
            raise ValueError("QuestPlusController supports weibull / linear / minNEntropy / mean only.")

        self.condition_id = condition_id
        self.quest_config = quest_config
        self.intensity_vals = np.array(quest_config["intensity_vals"], dtype=float)
        self.start_intensity = float(quest_config["start_intensity"])
        self.n_best = int(quest_config["stim_selection_options"]["N"])
        self.max_consecutive_reps = int(quest_config["stim_selection_options"]["maxConsecutiveReps"])
        # recorded in questplus_run-XXX.json so the contrast choices can be replayed
        self.seed = int(np.random.SeedSequence().entropy) if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)

        threshold, slope, lower_asymptote, lapse_rate = np.meshgrid(
            np.array(quest_config["threshold_vals"], dtype=float),
            np.array(quest_config["slope_vals"], dtype=float),
            np.array(quest_config["lower_asymptote_vals"], dtype=float),
            np.array(quest_config["lapse_rate_vals"], dtype=float),
            indexing="ij",
        )
        self.threshold_grid = threshold
        # p(correct) for every intensity x parameter combination
        self.p_correct = weibull_prob_correct(
            self.intensity_vals.reshape((-1, 1, 1, 1, 1)),
            threshold,
            slope,
            lower_asymptote,
            lapse_rate,
        )
        # uniform prior
        self.posterior = np.full(threshold.shape, 1.0 / threshold.size)

        self.selection_count = 0
        self.history = []  # [{"contrast": c, "response": "Correct" | "Incorrect"}]
        self.observation_count = 0

    def select_next_target_contrast(self):
        self.selection_count += 1
        if self.selection_count == 1:
            return self.start_intensity

        expected_entropy = expected_posterior_entropy(self.p_correct, self.posterior)
        candidate_indices = np.argsort(expected_entropy)[: self.n_best]
        recent_contrasts = [entry["contrast"] for entry in self.history[-self.max_consecutive_reps:]]
        while True:
            contrast = float(self.intensity_vals[self.rng.choice(candidate_indices)])
            # redraw if this would exceed max_consecutive_reps of the same contrast
            if len(recent_contrasts) < self.max_consecutive_reps or any(
                previous != contrast for previous in recent_contrasts
            ):
                return contrast

    def update(self, target_contrast, correct):
        intensity_index = self.intensity_index(target_contrast)
        likelihood = self.p_correct[intensity_index]
        self.posterior = self.posterior * (likelihood if correct else 1 - likelihood)
        self.posterior /= self.posterior.sum()
        self.history.append(
            {
                "contrast": float(self.intensity_vals[intensity_index]),
                "response": self.quest_config["response_vals"][0 if correct else 1],
            }
        )
        self.observation_count += 1

    def estimated_threshold(self):
        if self.observation_count == 0:
            return self.start_intensity
        return float((self.posterior * self.threshold_grid).sum())

    def threshold_posterior(self):
        # marginal posterior over threshold_vals
        return self.posterior.sum(axis=(1, 2, 3))

    def intensity_index(self, contrast):
        matches = np.flatnonzero(np.isclose(self.intensity_vals, contrast, rtol=0, atol=1e-12))
        if len(matches) != 1:
            raise ValueError(f"Contrast {contrast} is not one of the QUEST+ intensity_vals.")
        return int(matches[0])


def weibull_prob_correct(intensity, threshold, slope, lower_asymptote, lapse_rate):
    # questplus weibull, linear scale: 1 - lapse - (1 - guess - lapse) * exp(-(x / t) ** slope)
    return 1 - lapse_rate - (1 - lower_asymptote - lapse_rate) * np.exp(
        -((intensity / threshold) ** slope)
    )


def expected_posterior_entropy(p_correct, posterior):
    # p_correct: (n_intensities, *grid); posterior: grid. Returns one value per intensity.
    param_axes = tuple(range(1, p_correct.ndim))
    expected = np.zeros(p_correct.shape[0])
    for likelihood in (p_correct, 1 - p_correct):
        joint = likelihood * posterior
        outcome_prob = joint.sum(axis=param_axes)
        new_posterior = joint / outcome_prob.reshape((-1,) + (1,) * len(param_axes))
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(new_posterior > 0, new_posterior * np.log(new_posterior), 0.0)
        expected += outcome_prob * -terms.sum(axis=param_axes)
    return expected
