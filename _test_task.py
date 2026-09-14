from pathlib import Path
from collections import Counter
import math
import sys
import types
import unittest
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def install_psychopy_stubs():
    required_modules = [
        "psychopy",
        "psychopy.core",
        "psychopy.data",
        "psychopy.event",
        "psychopy.gui",
        "psychopy.visual",
        "psychopy.monitors",
        "psychopy.tools",
        "psychopy.tools.coordinatetools",
        "psychopy.data.staircase",
        "psychopy.visual.grating",
    ]
    if all(module_name in sys.modules for module_name in required_modules):
        return

    psychopy = types.ModuleType("psychopy")
    core = types.ModuleType("psychopy.core")
    data = types.ModuleType("psychopy.data")
    event = types.ModuleType("psychopy.event")
    gui = types.ModuleType("psychopy.gui")
    visual = types.ModuleType("psychopy.visual")
    monitors = types.ModuleType("psychopy.monitors")
    tools = types.ModuleType("psychopy.tools")
    coordinatetools = types.ModuleType("psychopy.tools.coordinatetools")
    staircase = types.ModuleType("psychopy.data.staircase")
    grating = types.ModuleType("psychopy.visual.grating")

    class DummyQuestPlusHandler:
        def __init__(self, *args, **kwargs):
            self.paramEstimate = {"threshold": 0.1}

        def __iter__(self):
            return self

        def __next__(self):
            return 0.1

        def addResponse(self, *args, **kwargs):
            return None

    class DummyMonitor:
        def __init__(self, *args, **kwargs):
            return

        def setSizePix(self, *args, **kwargs):
            return

        def setWidth(self, *args, **kwargs):
            return

        def setDistance(self, *args, **kwargs):
            return

    class DummyStim:
        def __init__(self, *args, **kwargs):
            return

    def pol2cart(angle, radius, units="deg"):
        angle_rad = math.radians(angle) if units == "deg" else angle
        return radius * math.cos(angle_rad), radius * math.sin(angle_rad)

    staircase.QuestPlusHandler = DummyQuestPlusHandler
    grating.GratingStim = DummyStim
    visual.TextStim = DummyStim
    visual.Window = DummyStim
    monitors.Monitor = DummyMonitor
    coordinatetools.pol2cart = pol2cart

    data.staircase = staircase
    tools.coordinatetools = coordinatetools
    psychopy.core = core
    psychopy.data = data
    psychopy.event = event
    psychopy.gui = gui
    psychopy.visual = visual
    psychopy.monitors = monitors
    psychopy.tools = tools

    sys.modules["psychopy"] = psychopy
    sys.modules["psychopy.core"] = core
    sys.modules["psychopy.data"] = data
    sys.modules["psychopy.data.staircase"] = staircase
    sys.modules["psychopy.event"] = event
    sys.modules["psychopy.gui"] = gui
    sys.modules["psychopy.visual"] = visual
    sys.modules["psychopy.visual.grating"] = grating
    sys.modules["psychopy.monitors"] = monitors
    sys.modules["psychopy.tools"] = tools
    sys.modules["psychopy.tools.coordinatetools"] = coordinatetools


install_psychopy_stubs()

import main
import stimuli
from hardware import BaseTrigger, TriggerDispatcher
from task import build_event_frame_lists
from trials import generate_trials


class ContrastEventSchedulerTests(unittest.TestCase):
    def test_aligned_events_snap_onset_and_offset_to_sign_reversals(self):
        onset_frames, offset_frames = build_event_frame_lists(
            refresh_hz=60,
            trial_duration_s=2.0,
            change_start_s=1.05,
            change_interval_s=1.0,
            change_count=1,
            duration_sign_reversals=2,
            flicker_hz=6,
        )

        self.assertEqual(onset_frames, [70])
        self.assertEqual(offset_frames, [90])
        self.assertEqual(offset_frames[0] - onset_frames[0], 20)

    def test_aligned_events_validate_fit_using_aligned_offset(self):
        with self.assertRaises(ValueError):
            build_event_frame_lists(
                refresh_hz=60,
                trial_duration_s=1.4,
                change_start_s=1.05,
                change_interval_s=1.0,
                change_count=1,
                duration_sign_reversals=2,
                flicker_hz=6,
            )

    def test_aligned_events_reject_overlap_after_snapping(self):
        with self.assertRaises(ValueError):
            build_event_frame_lists(
                refresh_hz=60,
                trial_duration_s=3.0,
                change_start_s=1.05,
                change_interval_s=0.25,
                change_count=2,
                duration_sign_reversals=2,
                flicker_hz=6,
            )


class TrialEventConstructionTests(unittest.TestCase):
    def test_build_trial_events_uses_trial_offset_frames(self):
        trial = {
            "contrast_event_onset_frames": [30],
            "contrast_event_offset_frames": [35],
            "contrast_event_target_indices": [2],
            "contrast_event_target_quadrants": ["upper_left"],
        }

        events = main.build_trial_events(trial, n_frames=100, refresh_hz=60)

        self.assertEqual(events[0]["onset_frame"], 30)
        self.assertEqual(events[0]["offset_frame"], 35)


class FlickerIndicatorTests(unittest.TestCase):
    def test_indicator_contrast_tracks_center_phase_state(self):
        self.assertEqual(stimuli.flicker_indicator_contrast_from_phase(0.25), 1.0)
        self.assertEqual(stimuli.flicker_indicator_contrast_from_phase(0.75), 0.0)

    def test_indicator_config_uses_degree_units(self):
        self.assertEqual(stimuli.photocell_box_size_deg, (0.5, 0.5))
        self.assertEqual(stimuli.photocell_box_pos_deg, (27, -15.0))


class RecordingStim:
    def __init__(self):
        self.color = None
        self.contrast = None
        self.ori = None
        self.phase = None
        self.draw_count = 0

    def draw(self):
        self.draw_count += 1


def make_recording_stims(center_count=8):
    return {
        "fixation_hole": RecordingStim(),
        "fixation": RecordingStim(),
        "centers": [RecordingStim() for _ in range(center_count)],
        "center_holes": [RecordingStim() for _ in range(center_count)],
        "surround": RecordingStim(),
        "photocell_box": RecordingStim(),
    }


class ContrastDetectionRenderingTests(unittest.TestCase):
    def test_task_has_four_eligible_center_indices(self):
        self.assertEqual(main.eligible_center_indices(), [4, 5, 6, 7])

    def test_task_centers_are_empty_without_an_active_target(self):
        task_center_indices = [4, 5, 6, 7]

        for surround_orientation in (None, 45):
            with self.subTest(surround_orientation=surround_orientation):
                stims = make_recording_stims()
                stimuli.draw_stimulus(
                    stims=stims,
                    frame_num=0,
                    refresh_hz=60,
                    surround_orientation=surround_orientation,
                    center_contrast=0.2,
                    task_center_indices=task_center_indices,
                )

                self.assertEqual(
                    [center.draw_count for center in stims["centers"]],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                )

    def test_active_target_uses_absolute_contrast_and_current_flicker_phase(self):
        stims = make_recording_stims()
        stimuli.prepare_stimuli_for_trial(
            stims,
            {"center_orientation": 135, "surround_orientation": None},
        )

        stimuli.draw_stimulus(
            stims=stims,
            frame_num=10,
            refresh_hz=60,
            surround_orientation=None,
            center_contrast=0.2,
            task_center_indices=[4, 5, 6, 7],
            target_index=5,
            target_contrast=0.08,
        )

        self.assertEqual(
            [center.draw_count for center in stims["centers"]],
            [1, 1, 1, 1, 0, 1, 0, 0],
        )
        self.assertEqual(stims["centers"][5].contrast, 0.08)
        self.assertEqual(stims["centers"][5].phase, 0.75)
        self.assertEqual(stims["centers"][5].ori, 135)


class DummyQuestController:
    def __init__(self):
        self.updates = []

    def update(self, target_contrast, correct):
        self.updates.append((target_contrast, correct))


class SelectingQuestController(DummyQuestController):
    def select_next_target_contrast(self):
        return 0.08


class ContrastResponseScoringTests(unittest.TestCase):
    def test_most_recent_response_before_deadline_is_scored(self):
        quest_controller = DummyQuestController()
        events = [
            {
                "onset_frame": 10,
                "offset_frame": 12,
                "fixation_offset_frame": 15,
                "response_deadline": 20,
                "target_index": 0,
                "target_quadrant": "upper_left",
                "target_contrast": 0.2,
                "outcome": None,
                "response_quadrant": None,
                "response_frame": None,
            }
        ]

        event_index = main.record_response(12, "9", events, 0, quest_controller)
        self.assertEqual(event_index, 0)
        self.assertEqual(events[0]["response_quadrant"], "upper_right")
        self.assertEqual(events[0]["response_frame"], 12)
        self.assertEqual(quest_controller.updates, [])

        event_index = main.record_response(18, "7", events, event_index, quest_controller)
        self.assertEqual(event_index, 0)
        self.assertEqual(events[0]["response_quadrant"], "upper_left")
        self.assertEqual(events[0]["response_frame"], 18)
        self.assertIsNone(events[0]["outcome"])

        event_index = main.finalize_elapsed_events(20, events, event_index, quest_controller)
        self.assertEqual(event_index, 1)
        self.assertEqual(events[0]["outcome"], "hit")
        self.assertEqual(quest_controller.updates, [(0.2, True)])

    def test_prepare_event_uses_quest_intensity_as_absolute_target_contrast(self):
        event_info = {"target_contrast": None}

        main.prepare_trial_event(event_info, SelectingQuestController())

        self.assertEqual(event_info["target_contrast"], 0.08)

    def test_wrong_quadrant_and_no_response_are_incorrect_for_quest(self):
        quest_controller = DummyQuestController()
        events = [
            {
                "onset_frame": 10,
                "response_deadline": 20,
                "target_quadrant": "upper_left",
                "target_contrast": 0.08,
                "outcome": None,
                "response_quadrant": "lower_right",
            },
            {
                "onset_frame": 20,
                "response_deadline": 30,
                "target_quadrant": "upper_right",
                "target_contrast": 0.1,
                "outcome": None,
                "response_quadrant": None,
            },
        ]

        event_index = main.finalize_elapsed_events(
            30,
            events,
            event_index=0,
            quest_controller=quest_controller,
        )

        self.assertEqual(event_index, 2)
        self.assertEqual(
            [event["outcome"] for event in events],
            ["miss", "no_response"],
        )
        self.assertEqual(quest_controller.updates, [(0.08, False), (0.1, False)])


class TrialLoggingTests(unittest.TestCase):
    def test_finalize_trial_logs_records_presented_target_contrasts(self):
        trial = {}
        events = [
            {
                "onset_frame": 10,
                "target_quadrant": "upper_left",
                "outcome": "hit",
                "target_contrast": 0.2,
            },
            {
                "onset_frame": 30,
                "target_quadrant": "lower_right",
                "outcome": "miss",
                "target_contrast": 0.05,
            },
        ]

        main.finalize_trial_logs(trial, events)

        self.assertEqual(trial["contrast_event_1_onset_frame"], 10)
        self.assertEqual(
            trial["contrast_event_target_quadrants_log"],
            "[upper_left; lower_right]",
        )
        self.assertEqual(trial["contrast_task_outcomes"], "[hit; miss]")
        self.assertEqual(trial["presented_target_contrasts"], "[0.2; 0.05]")
        self.assertNotIn("presented_contrast_decrements", trial)

    def test_trial_row_preserves_task_and_marker_fields_without_e_stim(self):
        class RecordingExperiment:
            def __init__(self):
                self.row = {}
                self.entry_count = 0

            def addData(self, key, value):
                self.row[key] = value

            def nextEntry(self):
                self.entry_count += 1

        trial = generate_trials(refresh_hz=120, shuffle=False)[0]
        trial.update({
            "contrast_event_1_onset_frame": trial["contrast_event_onset_frames"][0],
            "contrast_event_target_quadrants_log": "[upper_left]",
            "contrast_task_outcomes": "[hit]",
            "presented_target_contrasts": "[0.1]",
            "quest_estimated_threshold": 0.1,
        })
        exp = RecordingExperiment()

        main.log_trial(exp, 1, trial, 0, 1280, 120)

        self.assertEqual(exp.entry_count, 1)
        self.assertEqual(exp.row["block_name"], "ssvep_task")
        self.assertEqual(exp.row["condition_id"], trial["condition_id"])
        self.assertEqual(exp.row["presented_target_contrasts"], "[0.1]")
        self.assertEqual(exp.row["trial_end_frame_global"], 1280)
        self.assertFalse(any("e_stim" in key or "cs_plus" in key for key in exp.row))
        self.assertFalse(any(key.startswith("iti_tilt_") for key in exp.row))


class SessionMetadataTests(unittest.TestCase):
    def test_collect_exp_info_uses_one_dialog_without_voltage(self):
        with patch.object(main.gui, "DlgFromDict", return_value=types.SimpleNamespace(OK=True), create=True) as dialog, patch.object(
            main, "get_next_run_id", return_value="002"
        ):
            exp_info = main.collect_exp_info()

        dialog.assert_called_once()
        self.assertEqual(exp_info, {"subject": "", "session": "001", "run": "002"})

    def test_make_session_info_records_one_task_block_without_conditioning(self):
        exp_info = {
            "subject": "subj",
            "session": "001",
            "run": "001",
        }

        session_info = main.make_session_info(
            exp_info=exp_info,
            datetime_start="2026-08-04T12:00:00",
            refresh_hz=120.0,
        )

        self.assertEqual(
            session_info["trial configs"]["block_definitions"],
            {"ssvep_task": {"trials_per_condition": 20}},
        )
        self.assertEqual(
            session_info["trial configs"]["quest_intensity_definition"],
            "absolute_target_contrast",
        )
        self.assertNotIn("e_stim", session_info)
        self.assertNotIn("e_stim_trigger_code", session_info["events"])
        self.assertNotIn("cs_plus_center_orientation", session_info["trial configs"])
        self.assertFalse(any("iti_tilt" in key or "show_iti" in key for key in session_info["trial configs"]))


class TrialGenerationTests(unittest.TestCase):
    def test_single_task_block_has_20_trials_per_condition(self):
        trials = generate_trials(refresh_hz=120)

        self.assertEqual(len(trials), 120)
        self.assertEqual(Counter(trial["condition_id"] for trial in trials), {i: 20 for i in range(1, 7)})
        self.assertEqual({trial["block_name"] for trial in trials}, {"ssvep_task"})
        self.assertEqual([trial["trial_index_within_block"] for trial in trials], list(range(1, 121)))
        self.assertTrue(all("e_stim_scheduled" not in trial for trial in trials))
        self.assertTrue(all(
            len({trial["condition_id"] for trial in trials[index:index + 3]}) > 1
            for index in range(len(trials) - 2)
        ))


class FakeWindow:
    def __init__(self, event_log=None):
        self.pending = []
        self.event_log = [] if event_log is None else event_log

    def callOnFlip(self, function, *args, **kwargs):
        self.event_log.append(("scheduled", function.__name__))
        self.pending.append((function, args, kwargs))

    def flip(self):
        self.event_log.append(("flip", len(self.pending)))
        pending = list(self.pending)
        self.pending.clear()
        for function, args, kwargs in pending:
            function(*args, **kwargs)


class InterTrialIntervalTests(unittest.TestCase):
    def test_iti_draws_only_fixation_for_configured_frames(self):
        win = FakeWindow()
        stims = make_recording_stims()

        with patch.object(main.event, "clearEvents", create=True), patch.object(
            main, "handle_runtime_keys"
        ):
            next_frame = main.run_iti(
                win=win,
                stims=stims,
                iti_duration_s=0.5,
                refresh_hz=8,
                global_frame_num=10,
            )

        self.assertEqual(next_frame, 14)
        self.assertEqual(stims["fixation"].draw_count, 4)
        self.assertEqual(stims["fixation"].color, "white")
        self.assertTrue(all(
            stim.draw_count == 0 for name, stim in stims.items()
            if name != "fixation" and not isinstance(stim, list)
        ))


class FakeDispatcher:
    def __init__(self, event_log=None):
        self.submissions = []
        self.event_log = event_log

    def submit(self, code, *, label, frame_num=None, trial_num=None):
        self.submissions.append(
            {
                "code": code,
                "label": label,
                "frame_num": frame_num,
                "trial_num": trial_num,
            }
        )
        if self.event_log is not None:
            self.event_log.append(("trigger_submit", label, frame_num))


class CallOnFlipHardwareSchedulingTests(unittest.TestCase):
    def make_trial(self, **overrides):
        trial = {
            "condition_id": 7,
            "surround_orientation": None,
            "center_contrast": 0.4,
            "surround_contrast": 0.0,
            "contrast_event_onset_frames": [],
            "contrast_event_offset_frames": [],
            "contrast_event_target_indices": [],
            "contrast_event_target_quadrants": [],
        }
        trial.update(overrides)
        return trial

    def test_run_trial_uses_call_on_flip_for_trial_start_and_frame_marker(self):
        event_log = []
        win = FakeWindow(event_log=event_log)
        dispatcher = FakeDispatcher(event_log=event_log)
        trial = self.make_trial()

        with patch.object(main, "build_trial_events", return_value=[]), patch.object(
            main, "prepare_stimuli_for_trial"
        ), patch.object(main, "draw_stimulus"), patch.object(
            main, "finalize_elapsed_events", side_effect=lambda *args: args[2]
        ), patch.object(
            main, "get_task_response", return_value=None
        ), patch.object(
            main, "handle_runtime_keys"
        ), patch.object(
            main, "finalize_trial_logs"
        ):
            main.run_trial(
                win=win,
                stims={},
                trial=trial,
                trial_duration_s=81,
                refresh_hz=1,
                global_frame_num=0,
                quest_controller=DummyQuestController(),
                trigger_dispatcher=dispatcher,
                trial_num=4,
            )

        self.assertEqual(
            [entry[0] for entry in event_log[:6]],
            ["scheduled", "flip", "trigger_submit", "flip", "flip", "flip"],
        )
        self.assertEqual(
            dispatcher.submissions,
            [
                {"code": 7, "label": "trial_start", "frame_num": 0, "trial_num": 4},
                {
                    "code": main.frame_marker_code,
                    "label": "frame_marker",
                    "frame_num": 80,
                    "trial_num": 4,
                },
                {
                    "code": main.stimulus_offset_code,
                    "label": "stimulus_offset",
                    "frame_num": 81,
                    "trial_num": 4,
                },
            ],
        )



class FakeTriggerBackend(BaseTrigger):
    def __init__(self):
        super().__init__()
        self.codes = []

    def open(self):
        return

    def require_open(self):
        return

    def pulse_event(self, code: int, pulse_width_s=None, min_gap_s=None):
        self.codes.append(code)


class TriggerDispatcherTests(unittest.TestCase):
    def test_trigger_dispatcher_processes_requests_in_fifo_order(self):
        backend = FakeTriggerBackend()
        dispatcher = TriggerDispatcher(backend)
        try:
            dispatcher.submit(11, label="first")
            dispatcher.submit(12, label="second")
            dispatcher.submit(13, label="third")
            dispatcher.drain()
            self.assertEqual(backend.codes, [11, 12, 13])
        finally:
            dispatcher.close()


if __name__ == "__main__":
    unittest.main()
