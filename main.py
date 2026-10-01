import ctypes
from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
from psychopy import core, data, event, gui, visual

from config import (
    calibration_mode,
    center_change_count_per_trial,
    center_change_duration_sign_reversals,
    center_contrast,
    center_diameter_deg,
    center_flicker_hz,
    center_sf,
    condition_event_code_rows,
    contrast_change_interval_s,
    contrast_change_start_s,
    contrast_response_keys,
    contrast_task_eligible_eccentricities,
    display_gamma,
    fixation_red_duration_s,
    frame_marker_code,
    frame_marker_interval,
    fullscreen,
    iti_duration_s,
    monitor_name,
    monitor_pixels,
    monitor_width_cm,
    refresh_hz,
    screen_index,
    simulate_labjack,
    stimulus_offset_code,
    surround_contrast,
    surround_diameter_deg,
    surround_flicker_hz,
    surround_sf,
    task_instruction_text,
    trial_duration_s,
    viewing_distance_cm,
)
from hardware import (
    LabjackFio8BitTrigger,
    SimulatedLabjackFio8BitTrigger,
)
from stimuli import (
    draw_stimulus,
    grating_texture_precision,
    make_stimuli,
    make_window,
    prepare_stimuli_for_trial,
    stim_for_contrast,
)
from task import (
    QUEST_CONFIG,
    QuestPlusController,
    center_change_duration_aligned_s,
    eligible_center_indices,
    format_log_list,
    red_fixation_frames,
    response_key_to_quadrant,
    score_quadrant_response,
)
from trials import generate_trials


def hardware_gamma_ramp_max_deviation(ramp):
    # ramp: 3 x 256 uint16 from GetDeviceGammaRamp; identity is i / 255 * 65535
    ramp = np.asarray(ramp, dtype=float) / 65535
    return float(np.max(np.abs(ramp - np.linspace(0, 1, ramp.shape[1]))))


def check_hardware_gamma_ramp_is_identity(win, tolerance=0.002):
    ramp = np.zeros((3, 256), dtype=np.uint16)
    # _dc is a ctypes pointer (GetDC's restype is reassigned at runtime), so cast instead of wrapping
    dc = ctypes.cast(win.winHandle._dc, ctypes.c_void_p)
    if not ctypes.windll.gdi32.GetDeviceGammaRamp(dc, ramp.ctypes.data_as(ctypes.c_void_p)):
        raise RuntimeError("GetDeviceGammaRamp failed; cannot verify the hardware gamma ramp.")
    deviation = hardware_gamma_ramp_max_deviation(ramp)
    if deviation > tolerance:
        raise RuntimeError(
            f"Hardware gamma ramp deviates from identity by {deviation:.4f} "
            f"(tolerance {tolerance}). Gamma is corrected in software; the ramp must be identity."
        )


def setup_display():
    win = make_window(
        size=monitor_pixels,
        fullscr=fullscreen,
        screen=screen_index,
        monitor_name=monitor_name,
        monitor_width_cm=monitor_width_cm,
        viewing_distance_cm=viewing_distance_cm,
    )
    check_hardware_gamma_ramp_is_identity(win)
    measured_refresh_hz = win.getActualFrameRate()
    if measured_refresh_hz is None or abs(measured_refresh_hz - refresh_hz) > 1:
        raise RuntimeError(
            f"Measured refresh rate {measured_refresh_hz} Hz does not match "
            f"configured refresh_hz {refresh_hz} Hz."
        )
    win.recordFrameIntervals = True
    stims = make_stimuli(win)
    return win, stims, refresh_hz


def prepare_run_paths(exp_info):
    run_root = Path("data") / f"sub-{exp_info['subject']}" / f"ses-{exp_info['session']}"
    rawdata_dir = run_root / "rawdata"
    metadata_dir = run_root / "metadata"
    derivatives_dir = run_root / "derivatives"

    rawdata_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    run_id = exp_info["run"]
    return {
        "run_root": run_root,
        "rawdata_dir": rawdata_dir,
        "metadata_dir": metadata_dir,
        "derivatives_dir": derivatives_dir,
        "session_info_path": metadata_dir / f"session_info_run-{run_id}.json",
        "trial_conditions_path": metadata_dir / f"trial_conditions_run-{run_id}",
    }


def get_next_run_id(sub_id, ses_id):
    metadata_dir = Path("data") / f"sub-{sub_id}" / f"ses-{ses_id}" / "metadata"
    if not metadata_dir.exists():
        return "001"

    run_ids = []
    for path in metadata_dir.glob("trial_conditions_run-*.csv"):
        run_str = path.stem.removeprefix("trial_conditions_run-")
        if run_str.isdigit():
            run_ids.append(int(run_str))

    if not run_ids:
        return "001"

    return f"{max(run_ids) + 1:03d}"


def collect_exp_info():
    participant_info: dict[str, object] = {
        "subject": "",
        "session": "001",
    }

    if not gui.DlgFromDict(
        participant_info,
        title="Surround Suppression Stimuli",
        sortKeys=False,
    ).OK:
        return None

    sub_id = str(participant_info["subject"])
    ses_id = str(participant_info["session"])
    return {
        "subject": sub_id,
        "session": ses_id,
        "run": get_next_run_id(sub_id, ses_id),
    }


def write_session_info_json(path, session_info):
    path.write_text(json.dumps(session_info, indent=2) + "\n")


def make_session_info(exp_info, datetime_start, refresh_hz, code_version, gl_vendor):
    return {
        "participant": {
            "sub_id": exp_info["subject"],
            "ses_id": exp_info["session"],
            "run_id": exp_info["run"],
        },
        "session": {
            "datetime_start": datetime_start,
        },
        "code_version": code_version,
        "stim configs": {
            "center_flicker_hz": center_flicker_hz,
            "surround_flicker_hz": surround_flicker_hz,
            "center_diameter_deg": center_diameter_deg,
            "center_sf": center_sf,
            "surround_diameter_deg": surround_diameter_deg,
            "surround_sf": surround_sf,
        },
        "trial configs": {
            "trial_duration_s": trial_duration_s,
            "iti_duration_s": iti_duration_s,
            "center_contrast": center_contrast,
            "surround_contrast": surround_contrast,
            "contrast_task_eligible_eccentricities": contrast_task_eligible_eccentricities,
            "contrast_change_start_s": contrast_change_start_s,
            "contrast_change_interval_s": contrast_change_interval_s,
            "center_change_count_per_trial": center_change_count_per_trial,
            "center_change_duration_sign_reversals": center_change_duration_sign_reversals,
            "center_change_duration_aligned_s": center_change_duration_aligned_s(),
            "fixation_red_duration_s": fixation_red_duration_s,
        },
        "quest configs": {
            "quest_intensity_definition": "absolute_target_contrast",
            "quest_intensity_vals": QUEST_CONFIG["intensity_vals"],
            "quest_threshold_vals": QUEST_CONFIG["threshold_vals"],
            "quest_slope_vals": QUEST_CONFIG["slope_vals"],
            "quest_lower_asymptote_vals": QUEST_CONFIG["lower_asymptote_vals"],
            "quest_lapse_rate_vals": QUEST_CONFIG["lapse_rate_vals"],
            "quest_response_vals": QUEST_CONFIG["response_vals"],
            "quest_start_intensity": QUEST_CONFIG["start_intensity"],
            "quest_psychometric_func": QUEST_CONFIG["psychometric_func"],
            "quest_stim_scale": QUEST_CONFIG["stim_scale"],
            "quest_stim_selection_method": QUEST_CONFIG["stim_selection_method"],
            "quest_stim_selection_options": QUEST_CONFIG["stim_selection_options"],
            "quest_param_estimation_method": QUEST_CONFIG["param_estimation_method"],
        },
        "hardware": {
            "monitor_name": monitor_name,
            "monitor_pixels": monitor_pixels,
            "monitor_width_cm": monitor_width_cm,
            "viewing_distance_cm": viewing_distance_cm,
            "refresh_hz": refresh_hz,
            "gamma_correction": "software gamma correction, hardware ramp identity",
            "gl_vendor": gl_vendor,
            "grating_texture_precision": grating_texture_precision(gl_vendor, sys.platform),
            "config_local": {
                "simulate_labjack": simulate_labjack,
                "calibration_mode": calibration_mode,
                "monitor_name": monitor_name,
                "screen_index": screen_index,
                "monitor_pixels": monitor_pixels,
                "monitor_width_cm": monitor_width_cm,
                "viewing_distance_cm": viewing_distance_cm,
                "fullscreen": fullscreen,
                "display_gamma": display_gamma,
            },
        },
        "events": {
            "calibration_mode": calibration_mode,
            "stimulus_offset_code": stimulus_offset_code,
            "frame_marker_code": frame_marker_code,
            "frame_marker_interval": frame_marker_interval,
            "trigger_pulse": "set on flip, cleared on next flip (1 frame)",
        },
        "condition_event_codes": {
            f"condition_{row['condition_id']:02d}": row
            for row in condition_event_code_rows
        },
    }


def make_experiment_handler(exp_info, run_paths):
    exp = data.ExperimentHandler(
        name="surround_suppression_stimuli",
        extraInfo=dict(exp_info),
        dataFileName=str(run_paths["trial_conditions_path"]),
        saveWideText=False,
        savePickle=False,
    )

    for name in ("thisRow.t", "notes"):
        if name in exp.dataNames:
            exp.dataNames.remove(name)

    return exp


class ExperimentQuit(Exception):
    pass


def make_quest_controllers():
    return {
        row["condition_id"]: QuestPlusController(condition_id=row["condition_id"])
        for row in condition_event_code_rows
    }


def check_quest_contrasts_have_stims(stims, quest_controllers):
    for quest_controller in quest_controllers.values():
        for intensity in quest_controller.handler.intensityVals:
            for center_index in eligible_center_indices():
                stim_for_contrast(stims["centers"][center_index], intensity)


def save_questplus_handlers(quest_controllers, run_paths, run_id):
    condition_snapshots = {}
    for condition_id, quest_controller in sorted(quest_controllers.items()):
        serialized_handler = quest_controller.handler.saveAsJson(fileName=None)
        condition_snapshots[f"condition_{condition_id:02d}"] = json.loads(
            serialized_handler.lstrip("\ufeff")
        )

    path = run_paths["metadata_dir"] / f"questplus_run-{run_id}.json"
    snapshot = {
        "run_id": str(run_id),
        "conditions": condition_snapshots,
    }
    path.write_text(json.dumps(snapshot, indent=2) + "\n")


def show_message(win, text, key_list=None):
    event.clearEvents(eventType="keyboard")
    msg = visual.TextStim(win=win, text=text, alignText="left")
    msg.draw()
    win.flip()
    key_list = ["space", "escape"] if key_list is None else key_list

    while True:
        keys = event.waitKeys(keyList=key_list)
        if keys is None:
            continue
        if "escape" in keys:
            quit_experiment(win)
        for key in keys:
            if key != "escape":
                return key


def quit_experiment(win):
    raise ExperimentQuit()


def pause_experiment(win):
    event.clearEvents(eventType="keyboard")
    msg = visual.TextStim(
        win=win,
        text="Experiment paused.\nPress P or SPACE to resume.\nPress ESC to quit.",
        alignText="left",
    )

    while True:
        msg.draw()
        win.flip()
        keys = event.waitKeys(keyList=["p", "space", "escape"])
        if keys is None:
            continue
        if "escape" in keys:
            quit_experiment(win)
        if "p" in keys or "space" in keys:
            event.clearEvents(eventType="keyboard")
            return


def handle_runtime_keys(win):
    keys = event.getKeys(keyList=["p", "escape"])
    if "escape" in keys:
        quit_experiment(win)
    if "p" in keys:
        pause_experiment(win)


def get_task_response():
    key_list = list(contrast_response_keys.values())
    key_list.extend(f"num_{value}" for value in contrast_response_keys.values())
    keys = event.getKeys(keyList=key_list)
    if not keys:
        return None
    return keys[0]


def build_trial_events(trial, n_frames, refresh_hz):
    fixation_red_duration_frame_count = red_fixation_frames(refresh_hz)
    onset_frames = trial["contrast_event_onset_frames"]
    offset_frames = trial["contrast_event_offset_frames"]
    events = []

    if len(onset_frames) != len(offset_frames):
        raise ValueError(
            "contrast_event_onset_frames and contrast_event_offset_frames must match."
        )

    for event_index, (onset_frame, offset_frame) in enumerate(
        zip(onset_frames, offset_frames)
    ):
        if offset_frame > n_frames:
            raise ValueError("Contrast event offset exceeds the trial frame count.")
        if event_index < len(onset_frames) - 1:
            response_deadline = onset_frames[event_index + 1]
        else:
            response_deadline = n_frames

        if events and onset_frame < events[-1]["offset_frame"]:
            raise ValueError("Contrast events overlap within a trial.")

        events.append(
            {
                "onset_frame": onset_frame,
                "offset_frame": offset_frame,
                "fixation_offset_frame": onset_frame + fixation_red_duration_frame_count,
                "response_deadline": response_deadline,
                "target_index": trial["contrast_event_target_indices"][event_index],
                "target_quadrant": trial["contrast_event_target_quadrants"][event_index],
                "target_contrast": None,
                "outcome": None,
                "response_quadrant": None,
                "response_frame": None,
            }
        )

    return events


def prepare_trial_event(event_info, quest_controller):
    event_info["target_contrast"] = quest_controller.select_next_target_contrast()


def start_trial_event(event_info, event_index, fixation_red_until_frame):
    if event_info["target_contrast"] is None:
        raise ValueError(
            f"Contrast event {event_index} started before QUEST selection was prepared."
        )
    fixation_red_until_frame = max(
        fixation_red_until_frame,
        event_info["fixation_offset_frame"],
    )
    return event_index, fixation_red_until_frame


def finalize_elapsed_events(frame_num, events, event_index, quest_controller):
    while event_index < len(events):
        event_info = events[event_index]
        if event_info["outcome"] is not None:
            event_index += 1
            continue
        if frame_num < event_info["response_deadline"]:
            break

        if event_info["response_quadrant"] is None:
            event_info["outcome"] = "no_response"
        else:
            event_info["outcome"] = score_quadrant_response(
                target_quadrant=event_info["target_quadrant"],
                response_quadrant=event_info["response_quadrant"],
            )

        if (
            event_info["target_contrast"] is not None
            and event_info["outcome"] != "no_response"
        ):
            quest_controller.update(
                event_info["target_contrast"],
                correct=(event_info["outcome"] == "hit"),
            )
        event_index += 1

    return event_index


def record_response(frame_num, response_key, events, event_index, quest_controller):
    response_quadrant = response_key_to_quadrant(response_key)

    while event_index < len(events) and events[event_index]["outcome"] is not None:
        event_index += 1

    if event_index >= len(events):
        return event_index

    event_info = events[event_index]
    if not (event_info["onset_frame"] <= frame_num < event_info["response_deadline"]):
        return event_index

    event_info["response_quadrant"] = response_quadrant
    event_info["response_frame"] = frame_num
    return event_index


def run_trial(
    win,
    stims,
    trial,
    trial_duration_s,
    refresh_hz,
    quest_controller,
    trigger=None,
    trial_num=None,
):
    n_frames = round(trial_duration_s * refresh_hz)
    trial_events = build_trial_events(trial, n_frames, refresh_hz)
    response_event_index = 0
    next_event_index = 0
    active_event_index = None
    fixation_red_until_frame = -1
    task_center_indices = eligible_center_indices()

    prepare_stimuli_for_trial(stims, trial)
    next_prepared_index = 0
    if trial_events:
        prepare_trial_event(
            trial_events[0],
            quest_controller,
        )
        next_prepared_index = 1

    trigger_is_high = False

    for frame_num in range(n_frames):
        response_event_index = finalize_elapsed_events(
            frame_num,
            trial_events,
            response_event_index,
            quest_controller,
        )
        while (
            next_prepared_index < len(trial_events)
            and next_prepared_index <= response_event_index
        ):
            prepare_trial_event(
                trial_events[next_prepared_index],
                quest_controller,
            )
            next_prepared_index += 1

        if (
            active_event_index is not None
            and frame_num >= trial_events[active_event_index]["offset_frame"]
        ):
            active_event_index = None

        while (
            next_event_index < len(trial_events)
            and frame_num == trial_events[next_event_index]["onset_frame"]
        ):
            event_info = trial_events[next_event_index]
            active_event_index, fixation_red_until_frame = start_trial_event(
                event_info,
                next_event_index,
                fixation_red_until_frame,
            )
            next_event_index += 1

        fixation_color = "red" if frame_num < fixation_red_until_frame else "white"
        active_event = None
        if active_event_index is not None:
            candidate_event = trial_events[active_event_index]
            if frame_num < candidate_event["offset_frame"]:
                active_event = candidate_event

        draw_stimulus(
            stims=stims,
            frame_num=frame_num,
            refresh_hz=refresh_hz,
            surround_orientation=trial["surround_orientation"],
            center_contrast=trial["center_contrast"],
            fixation_color=fixation_color,
            surround_contrast=trial["surround_contrast"],
            center_flicker_hz=center_flicker_hz,
            surround_flicker_hz=surround_flicker_hz,
            task_center_indices=task_center_indices,
            target_index=None if active_event is None else active_event["target_index"],
            target_contrast=(
                None if active_event is None else active_event["target_contrast"]
            ),
        )

        code = None
        if frame_num == 0:
            code = trial["condition_id"]
        elif frame_num % frame_marker_interval == 0:
            code = frame_marker_code
        if trigger is not None:
            if code is not None:
                win.callOnFlip(trigger.send_code, code)
                trigger_is_high = True
            elif trigger_is_high:
                win.callOnFlip(trigger.send_code, 0)
                trigger_is_high = False
        win.flip()

        response_key = get_task_response()
        if response_key is not None:
            response_event_index = record_response(
                frame_num,
                response_key,
                trial_events,
                response_event_index,
                quest_controller,
            )

        handle_runtime_keys(win)

    finalize_elapsed_events(n_frames, trial_events, response_event_index, quest_controller)
    if trigger is not None and trigger_is_high:
        trigger.send_code(0)

    return trial_events


def run_iti(win, stims, iti_duration_s, refresh_hz, trigger=None):
    n_frames = round(iti_duration_s * refresh_hz)

    event.clearEvents(eventType="keyboard")

    for frame_index in range(n_frames):
        stims["fixation"].color = "white"
        stims["fixation"].draw()
        if calibration_mode and frame_index < 12:
            stims["photocell_box"].contrast = -1.0
            stims["photocell_box"].draw()

        if trigger is not None and frame_index == 0:
            win.callOnFlip(trigger.send_code, stimulus_offset_code)
        elif trigger is not None and frame_index == 1:
            win.callOnFlip(trigger.send_code, 0)
        win.flip()

        handle_runtime_keys(win)


def log_trial(
    exp,
    trial_num,
    trial,
    events,
    quest_estimated_threshold,
    dropped_frame_count,
):
    surround_present = trial["surround_orientation"] is not None

    row = {
        "trial_num": trial_num,
        "block_name": trial["block_name"],
        "condition_id": trial["condition_id"],
        "center_orientation": trial["center_orientation"],
        "center_contrast": trial["center_contrast"],
        "surround_present": surround_present,
        "surround_orientation": trial["surround_orientation"] if surround_present else "N/A",
        "surround_contrast": trial["surround_contrast"] if surround_present else "N/A",
        "contrast_task_outcomes": format_log_list(
            [event_info["outcome"] for event_info in events]
        ),
        "contrast_task_no_response_count": sum(
            event_info["outcome"] == "no_response" for event_info in events
        ),
        "presented_target_contrasts": format_log_list(
            [event_info["target_contrast"] for event_info in events]
        ),
        "quest_estimated_threshold": quest_estimated_threshold,
        "dropped_frame_count": dropped_frame_count,
    }

    for key, value in row.items():
        exp.addData(key, value)
    exp.nextEntry()


def show_block_transition(win, block_name):
    if block_name == "ssvep_task":
        show_message(win, task_instruction_text)


def run_experiment_trials(
    win,
    stims,
    exp,
    refresh_hz,
    trigger,
    quest_controllers,
    frame_interval_log,
):
    current_block_name = None
    trials = generate_trials(refresh_hz=refresh_hz)

    for trial_num, trial in enumerate(trials, start=1):
        if trial["block_name"] != current_block_name:
            show_block_transition(win, trial["block_name"])
            current_block_name = trial["block_name"]

        quest_controller = quest_controllers[trial["condition_id"]]
        win.frameIntervals = []
        trial_events = run_trial(
            win=win,
            stims=stims,
            trial=trial,
            trial_duration_s=trial_duration_s,
            refresh_hz=refresh_hz,
            quest_controller=quest_controller,
            trigger=trigger,
            trial_num=trial_num,
        )
        intervals = list(win.frameIntervals)
        # index 0 spans the pre-trial gap (previous flip to frame 0), so it is not counted
        dropped = [
            i for i, dt in enumerate(intervals) if i >= 1 and dt > 1.5 / refresh_hz
        ]
        frame_interval_log[str(trial_num)] = {
            "intervals_s": intervals,
            "dropped_frame_indices": dropped,
        }
        run_iti(
            win=win,
            stims=stims,
            iti_duration_s=iti_duration_s,
            refresh_hz=refresh_hz,
            trigger=trigger,
        )

        log_trial(
            exp=exp,
            trial_num=trial_num,
            trial=trial,
            events=trial_events,
            quest_estimated_threshold=quest_controller.estimated_threshold(),
            dropped_frame_count=len(dropped),
        )


def get_git_state():
    repo_dir = Path(__file__).resolve().parent

    def run_git(*args):
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_dir,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("git is not installed or not on PATH.") from exc
        except subprocess.CalledProcessError as exc:
            if "not a git repository" in exc.stderr:
                raise RuntimeError(f"{repo_dir} is not a git repository.") from exc
            raise RuntimeError(
                f"git {' '.join(args)} failed in {repo_dir}: {exc.stderr.strip()}"
            ) from exc
        return result.stdout

    return {
        "commit": run_git("rev-parse", "HEAD").strip(),
        "dirty": bool(run_git("status", "--porcelain").strip()),
        "diff": run_git("diff", "HEAD"),
        "untracked": run_git("ls-files", "--others", "--exclude-standard").splitlines(),
    }


def main():
    logging.basicConfig(level=logging.INFO)
    datetime_start = datetime.now().isoformat(timespec="seconds")
    code_version = get_git_state()
    exp_info = collect_exp_info()
    if exp_info is None:
        core.quit()

    run_paths = prepare_run_paths(exp_info)
    exp = make_experiment_handler(exp_info, run_paths)
    win, stims, refresh_hz = setup_display()
    quest_controllers = make_quest_controllers()
    check_quest_contrasts_have_stims(stims, quest_controllers)

    session_info = make_session_info(
        exp_info,
        datetime_start,
        refresh_hz,
        code_version=code_version,
        gl_vendor=win.glVendor,
    )
    write_session_info_json(run_paths["session_info_path"], session_info)

    trigger_backend = None
    frame_interval_log = {}

    try:
        trigger_class = SimulatedLabjackFio8BitTrigger if simulate_labjack else LabjackFio8BitTrigger
        trigger_backend = trigger_class()
        trigger_backend.open()

        run_experiment_trials(
            win=win,
            stims=stims,
            exp=exp,
            refresh_hz=refresh_hz,
            trigger=trigger_backend,
            quest_controllers=quest_controllers,
            frame_interval_log=frame_interval_log,
        )
        show_message(win, "End of experiment. Thank you!\nPress SPACE to exit.")
    except ExperimentQuit:
        pass
    finally:
        if trigger_backend is not None:
            trigger_backend.close()
        exp.saveAsWideText(str(run_paths["trial_conditions_path"]) + ".csv")
        save_questplus_handlers(
            quest_controllers=quest_controllers,
            run_paths=run_paths,
            run_id=exp_info["run"],
        )
        frame_intervals_path = run_paths["metadata_dir"] / f"frame_intervals_run-{exp_info['run']}.json"
        frame_intervals_path.write_text(json.dumps(frame_interval_log, indent=2) + "\n")
        win.close()

    core.quit()


if __name__ == "__main__":
    main()

#gittest