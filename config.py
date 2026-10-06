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

# --- contrasts ---
center_contrast = 0.05
surround_contrast = 1.0

# --- mask ---
raised_cos_fringe, raised_cos_fringe_params = "raisedCos", {"fringeWidth": 0.1}

# -- photocell box ---
photocell_box_size_deg = (0.5, 0.5)
photocell_box_pos_deg = (27, -15.0)

# --- trials ---
trial_duration_s = 10.6667
iti_duration_s = 2

task_instruction_text = (
    "Maintain fixation on the white cross in the center of the screen.\n\n"
    "Four locations in the stimulus will remain empty. When a circle briefly "
    "appears in one of them, press the matching keypad key:\n"
    "7 = upper left, 9 = upper right, 1 = lower left, 3 = lower right.\n\n"
    "If you are unsure, simply guess.\n\n"
    "Press SPACE to begin."
)

block_definitions = [
    {
        "block_name": "ssvep_task",
        "trials_per_condition": 10,
    },
]

# --- task ---
contrast_task_eligible_eccentricities = [7.1]
contrast_change_start_s = 1
contrast_change_interval_s = 1.7
center_change_count_per_trial = 6
center_change_duration_sign_reversals = 1
fixation_red_duration_s = 0.333
contrast_response_keys = {
    "upper_left": "7",
    "upper_right": "9",
    "lower_left": "1",
    "lower_right": "3",
}
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

# --- event codes ---
# one bit per code so overlapping events stay decodable on FIO0-FIO7
frame_marker_code = 64
stimulus_offset_code = 128
condition_event_code_rows = [
    {"condition_id": 1, "event_code": 1,  "center_orientation": 45,  "surround_orientation": None},
    {"condition_id": 2, "event_code": 2,  "center_orientation": 45,  "surround_orientation": 45},
    {"condition_id": 3, "event_code": 4,  "center_orientation": 45,  "surround_orientation": 135},
#    {"condition_id": 4, "event_code": 8,  "center_orientation": 135, "surround_orientation": None},
#    {"condition_id": 5, "event_code": 16, "center_orientation": 135, "surround_orientation": 45},
#    {"condition_id": 6, "event_code": 32, "center_orientation": 135, "surround_orientation": 135},
]
