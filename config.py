import numpy as np
from psychopy.tools.coordinatetools import pol2cart

# --- display ---
refresh_hz = 120
bg_color = (0, 0, 0)
units = "deg"

# --- per machine settings ---
# config_local.py not tracked by git;

from config_local import *  # noqa: F401,F403

_required_local_settings = (
    "simulate_labjack",
    "calibration_mode",
    "monitor_name",
    "screen_index",
    "monitor_pixels",
    "monitor_width_cm",
    "viewing_distance_cm",
    "fullscreen",
    "display_gamma",
    "simulate_eyetracker",
    "eyetracker_serial",
    "simulate_audio",
    "audio_device_name",
    "audio_sample_rate_hz",
)
_missing_local_settings = [
    name for name in _required_local_settings if name not in globals()
]
if _missing_local_settings:
    raise ImportError(
        "config_local.py is missing required settings: "
        + ", ".join(_missing_local_settings)
        + ". Set them in config_local.py."
    )

# --- stimulation frequencies ---
center_flicker_hz = 6
surround_flicker_hz = 7.5

def frames_per_reversal(refresh_hz, flicker_hz):
    frames = refresh_hz / flicker_hz
    if not float(frames).is_integer():
        raise ValueError(
            f"refresh_hz / flicker_hz must be an integer, got {refresh_hz} / {flicker_hz} = {frames}"
        )
    return int(frames)

# --- centers ---
center_diameter_deg = 2
center_holes_diameter_deg = center_diameter_deg + 1
center_sf = 1.0

center_layout_deg = {  # eccentricity: [angles]
     3.6: [0, 90, 180, 270],
     7.1: [45, 135, 225, 315],
    8.5: [0, 90, 180, 270],
    10.7: [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5],
}

def angle_to_quadrant(angle_deg):
    if 0 < angle_deg < 90:
        return "upper_right"
    if 90 < angle_deg < 180:
        return "upper_left"
    if 180 < angle_deg < 270:
        return "lower_left"
    if 270 < angle_deg < 360:
        return "lower_right"
    return None

def build_centers(layout_deg):
    centers = []
    for eccentricity, angles in layout_deg.items():
        for angle in angles:
            x, y = pol2cart(angle, eccentricity, units="deg")
            centers.append(
                {
                    "eccentricity": eccentricity,
                    "angle_deg": angle,
                    "quadrant": angle_to_quadrant(angle),
                    "pos": (x, y),
                }
            )
    return centers


CENTERS = build_centers(center_layout_deg)
center_positions_deg = [center["pos"] for center in CENTERS]

# --- surround ---
surround_diameter_deg = 28
surround_sf = 1.0

# --- mask ---
raised_cos_fringe, raised_cos_fringe_params = "raisedCos", {"fringeWidth": 0.1}

# -- photocell box ---
photocell_box_size_deg = (0.5, 0.5)
photocell_box_pos_deg = (27, -15.0)

# --- trials ---
trial_duration_s = 10.6667

# --- phases ---
practice_iti_duration_s = 2
training_iti_duration_s = 3  # orientation report prompt appears 3 s after stimulus offset
testing_iti_duration_s = 7
orientation_feedback_duration_s = 1
training_post_feedback_fixation_s = 3

practice_instruction_pages = [
    (
        "Maintain fixation on the white cross in the center of the screen.\n\n"
        "Four locations in the stimulus will remain empty. When a circle briefly "
        "appears in one of them, press the matching keypad key:\n"
        "7 = upper left, 9 = upper right, 1 = lower left, 3 = lower right.\n\n"
        "If you are unsure, simply guess.\n\n"
        "Press SPACE to continue."
    ),
    (
        "After each trial you will be asked which way the center circles were tilted:\n"
        "7 = /     9 = \\\n\n"
        "Press SPACE to begin practice."
    ),
]
# placeholder until the wording about the noise is decided
training_instruction_pages = [
    "Practice is over. The experiment will now begin.\n\nPress SPACE to begin.",
]

practice_continue_text = "Press SPACE to keep practicing.\nPress ENTER when you are ready to begin."
practice_continue_keys = {"space": "keep_practicing", "return": "ready", "num_enter": "ready"}

# quest_set: practice gets its own QUEST+ states, discarded after the phase; training and testing share "main"
phase_definitions = [
    {
        "phase_name": "practice",
        "trials_per_condition": 1,  # one shuffled batch at a time until the participant is ready
        "repeat_until_ready": True,
        "reinforced_per_condition": 0,
        "iti_duration_s": practice_iti_duration_s,
        "orientation_report": True,
        "post_feedback_fixation_s": 0,
        "record": False,  # no triggers or eye-tracker events
        "show_probe_outlines": True,
        "quest_set": "practice",
        "instruction_pages": practice_instruction_pages,
    },
    {
        "phase_name": "training",
        "trials_per_condition": 3,
        "repeat_until_ready": False,
        "reinforced_per_condition": 3,
        "iti_duration_s": training_iti_duration_s,
        "orientation_report": True,
        "post_feedback_fixation_s": training_post_feedback_fixation_s,
        "record": True,
        "show_probe_outlines": False,
        "quest_set": "main",
        "instruction_pages": training_instruction_pages,
    },
    {
        "phase_name": "testing",
        "trials_per_condition": 10,
        "repeat_until_ready": False,
        "reinforced_per_condition": 7,
        "iti_duration_s": testing_iti_duration_s,
        "orientation_report": False,
        "post_feedback_fixation_s": 0,
        "record": True,
        "show_probe_outlines": False,
        "quest_set": "main",
        "instruction_pages": [],  # continues straight on from training
    },
]

# --- task ---
contrast_task_eligible_eccentricities = [7.1]
contrast_change_start_s = 1
contrast_change_interval_s = 1.7
center_change_count_per_trial = 6
center_change_duration_sign_reversals = 2
fixation_red_duration_s = 0.333
contrast_response_keys = {
    "upper_left": "7",
    "upper_right": "9",
    "lower_left": "1",
    "lower_right": "3",
}
orientation_report_keys = {"7": 45, "9": 135}  # 7 = "/", 9 = "\"
orientation_report_prompt_text = "Which way were the center circles tilted?\n\n7 = /     9 = \\"

# --- practice probe outlines ---
probe_outline_diameter_deg = 2.5  # inside the gray hole around each task center
probe_outline_luminance = 0.6  # fraction of white; background is 0.5
probe_outline_line_width_px = 2

# --- QUEST+ ---
# intensity = absolute target contrast; each group starts at the top of its own list
center_only_quest_intensity_vals = np.geomspace(0.01, 0.10, num=20).tolist()
surround_quest_intensity_vals = np.geomspace(0.01, 0.20, num=20).tolist()
quest_threshold_vals = np.geomspace(0.005, 0.25, num=25).tolist()

def build_quest_config(intensity_vals):
    return {
        "intensity_vals": intensity_vals,
        "threshold_vals": quest_threshold_vals,
        "slope_vals": [2],
        "lower_asymptote_vals": [0.25],
        "lapse_rate_vals": [0.02],
        "response_vals": ["Correct", "Incorrect"],
        "start_intensity": intensity_vals[-1],
        "psychometric_func": "weibull",
        "stim_scale": "linear",
        "stim_selection_method": "minNEntropy",
        "stim_selection_options": {"N": 3, "maxConsecutiveReps": 2},
        "param_estimation_method": "mean",
    }

quest_configs = {
    "center_only": build_quest_config(center_only_quest_intensity_vals),
    "surround": build_quest_config(surround_quest_intensity_vals),
}

def quest_group_for_condition(condition_row):
    # center only = no surround grating
    if condition_row["surround_orientation"] is None:
        return "center_only"
    return "surround"

# --- hardware ---
labjack_device_type = "T7"
labjack_connection_type = "USB"
labjack_identifier = "ANY"
labjack_fio_lines = [f"FIO{i}" for i in range(8)]
labjack_fio_mask = 0xFF

trigger_pulse_width_s = 0.005
trigger_min_gap_s = 0.005
frame_marker_interval = 80
assert frame_marker_interval % frames_per_reversal(refresh_hz, center_flicker_hz) == 0
assert frame_marker_interval % frames_per_reversal(refresh_hz, surround_flicker_hz) == 0

# --- eye tracker ---
eyetracker_gaze_output_hz = 60  # must match Frequency set in Eye Tracker Manager

# --- fear conditioning ---
cs_orientations = [45, 135]  # CS+ drawn at random from these at the start of each session
us_max_consecutive_nonreinforced = 2  # counted along CS+ trials only, skipping CS- trials
us_duration_s = 0.1
us_volume = 1.0  # PsychoPy volume (0-1); fix this and the OS volume when calibrating to 90 dB
us_noise_seed = 0  # same white-noise samples every session

# --- event codes ---
# conditions use values 1-6 on FIO0-FIO2; the codes below take one bit each so they stay decodable when combined
frame_marker_code = 8
stimulus_offset_code = 16
us_onset_code = 32  # sent with stimulus_offset_code on the same flip (16 + 32 = 48)
orientation_report_onset_code = 64
orientation_report_response_code = 128  # sent on the feedback onset flip, within one frame of the keypress
condition_event_code_rows = [
    {"condition_id": 1, "event_code": 1,  "center_orientation": 45,  "surround_orientation": None, "center_contrast": 0.1, "surround_contrast": 0.0},
    {"condition_id": 2, "event_code": 2,  "center_orientation": 45,  "surround_orientation": 45,   "center_contrast": 0.1, "surround_contrast": 1.0},
    {"condition_id": 3, "event_code": 3,  "center_orientation": 45,  "surround_orientation": 135,  "center_contrast": 0.1, "surround_contrast": 1.0},
    {"condition_id": 4, "event_code": 4,  "center_orientation": 135, "surround_orientation": None, "center_contrast": 0.1, "surround_contrast": 0.0},
    {"condition_id": 5, "event_code": 5, "center_orientation": 135, "surround_orientation": 45,   "center_contrast": 0.1, "surround_contrast": 1.0},
    {"condition_id": 6, "event_code": 6, "center_orientation": 135, "surround_orientation": 135,  "center_contrast": 0.1, "surround_contrast": 1.0},
]
