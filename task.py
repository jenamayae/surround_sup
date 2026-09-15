from __future__ import annotations

import random

import numpy as np
from psychopy.data.staircase import QuestPlusHandler

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
)

QUEST_INTENSITY_VALS = np.geomspace(0.02, 0.2, num=13).tolist()
# ~ [0.20, 0.17, 0.14, 0.11, 0.093, 0.077, 0.063, 0.052, 0.043, 0.036, 0.029, 0.024, 0.020]
QUEST_THRESHOLD_VALS = np.geomspace(0.01, 0.5, num=25).tolist()

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
    "stim_selection_method": "minEntropy",
    "stim_selection_options": {},
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
    if frame_num <= 0:
        return False

    previous_cycle_index = int((frame_num - 1) * flicker_hz / refresh_hz)
    current_cycle_index = int(frame_num * flicker_hz / refresh_hz)
    return current_cycle_index != previous_cycle_index


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
    def __init__(
        self,
        condition_id=None,
        n_trials=1000,
        quest_config=QUEST_CONFIG,
    ):
        self.condition_id = condition_id
        self.start_intensity = float(quest_config["start_intensity"])
        self.observation_count = 0

        self.handler = QuestPlusHandler(
            nTrials=n_trials,
            intensityVals=[float(value) for value in quest_config["intensity_vals"]],
            thresholdVals=[float(value) for value in quest_config["threshold_vals"]],
            slopeVals=[float(value) for value in quest_config["slope_vals"]],
            lowerAsymptoteVals=[
                float(value) for value in quest_config["lower_asymptote_vals"]
            ],
            lapseRateVals=[float(value) for value in quest_config["lapse_rate_vals"]],
            responseVals=list(quest_config["response_vals"]),
            startIntensity=self.start_intensity,
            psychometricFunc=quest_config["psychometric_func"],
            stimScale=quest_config["stim_scale"],
            stimSelectionMethod=quest_config["stim_selection_method"],
            stimSelectionOptions=dict(quest_config["stim_selection_options"]),
            paramEstimationMethod=quest_config["param_estimation_method"],
        )
        # StairHandler defaults this unused snapshot label to "2AFC".
        self.handler.method = "4AFC"

    def select_next_target_contrast(self):
        return float(next(self.handler))

    def update(self, target_contrast, correct):
        response = "Correct" if correct else "Incorrect"
        self.handler.addResponse(response, intensity=float(target_contrast))
        self.observation_count += 1

    def estimated_threshold(self):
        if self.observation_count == 0:
            return self.start_intensity
        estimate = self.handler.paramEstimate
        threshold = estimate["threshold"]
        return float(threshold)
