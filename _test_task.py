from collections import Counter
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import time
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

import numpy as np

import main
import stimuli
from config import block_definitions, condition_event_code_rows
from hardware import BaseTrigger, TriggerDispatcher
from task import (
    QUEST_CONFIG,
    QUEST_INTENSITY_VALS,
    QuestPlusController,
    build_event_frame_lists,
    weibull_prob_correct,
)
from trials import generate_trials


class DisplaySetupTests(unittest.TestCase):
    def test_setup_display_uses_configured_120_hz(self):
        win = types.SimpleNamespace(getActualFrameRate=lambda: 120.0)
        stims = object()

        with patch.object(main, "make_window", return_value=win) as make_window, patch.object(
            main, "make_stimuli", return_value=stims
        ), patch.object(main, "check_hardware_gamma_ramp_is_identity"):
            result = main.setup_display()

        self.assertEqual(result, (win, stims, 120.0))
        self.assertEqual(make_window.call_args.kwargs["screen"], 1)


class SoftwareGammaCorrectionTests(unittest.TestCase):
    def grating_contrasts(self):
        return QUEST_INTENSITY_VALS + [stimuli.center_contrast, stimuli.surround_contrast]

    def texture_luminance(self, texture):
        return ((texture.astype(float) + 1) / 2) ** stimuli.display_gamma

    def test_mean_gray_rgb(self):
        self.assertAlmostEqual(
            stimuli.luminance_to_rgb(0.5),
            2 * 0.5 ** (1 / stimuli.display_gamma) - 1,
            places=15,
        )

    def test_luminance_to_rgb_on_arrays(self):
        rgb = stimuli.luminance_to_rgb(np.array([0.0, 0.5, 1.0]))

        self.assertEqual(rgb[0], -1.0)
        self.assertAlmostEqual(rgb[1], stimuli.luminance_to_rgb(0.5), places=15)
        self.assertEqual(rgb[2], 1.0)

    def test_textures_are_one_cycle_square_power_of_two(self):
        texture = stimuli.grating_texture(0.05)

        self.assertEqual(texture.shape, (256, 256))
        self.assertEqual(texture.dtype, np.float32)
        self.assertTrue(np.all(texture == texture[0]))

    def test_textures_have_mean_gray_and_requested_michelson_contrast(self):
        x = (np.arange(stimuli.grating_texture_res) + 0.5) / stimuli.grating_texture_res
        for contrast in self.grating_contrasts():
            with self.subTest(contrast=contrast):
                luminance = self.texture_luminance(stimuli.grating_texture(contrast))[0]
                mean_luminance = luminance.mean()
                # Michelson contrast of a sinusoid = amplitude / mean; the amplitude is
                # taken from the first Fourier component because texel centres miss the peaks
                amplitude = 2 * abs(np.mean(luminance * np.exp(-2j * np.pi * x)))

                self.assertAlmostEqual(mean_luminance, 0.5, places=6)
                self.assertAlmostEqual(amplitude / mean_luminance, contrast, places=6)

    def test_textures_match_psychopy_sin_phase(self):
        x = (np.arange(stimuli.grating_texture_res) + 0.5) / stimuli.grating_texture_res
        for contrast in self.grating_contrasts():
            with self.subTest(contrast=contrast):
                luminance = self.texture_luminance(stimuli.grating_texture(contrast))[0]
                expected = 0.5 * (1 + contrast * np.sin(2 * np.pi * x - np.pi / 2))

                self.assertLess(np.max(np.abs(luminance - expected)), 1e-6)

    def test_zero_contrast_texture_is_flat_mean_gray(self):
        texture = stimuli.grating_texture(0.0)

        self.assertTrue(np.all(texture == np.float32(stimuli.luminance_to_rgb(0.5))))

    def test_stim_lookup_uses_nearest_contrast_within_tolerance(self):
        stims_by_contrast = {0.05: "low", 0.1: "high"}

        self.assertEqual(stimuli.stim_for_contrast(stims_by_contrast, 0.05 + 1e-12), "low")
        with self.assertRaises(ValueError):
            stimuli.stim_for_contrast(stims_by_contrast, 0.05 + 1e-6)

    def test_every_quest_contrast_has_a_task_center_stim(self):
        stims = {"centers": [
            {contrast: object() for contrast in QUEST_INTENSITY_VALS}
            for _ in range(8)
        ]}
        controllers = {1: types.SimpleNamespace(intensity_vals=QUEST_INTENSITY_VALS)}
        main.check_quest_contrasts_have_stims(stims, controllers)

        controllers[1].intensity_vals = QUEST_INTENSITY_VALS + [0.2]
        with self.assertRaises(ValueError):
            main.check_quest_contrasts_have_stims(stims, controllers)

    def test_gamma_ramp_deviation(self):
        identity = np.tile(np.arange(256) * 257, (3, 1)).astype(np.uint16)
        corrected = np.tile(
            np.round((np.arange(256) / 255) ** (1 / 2.28) * 65535), (3, 1)
        ).astype(np.uint16)

        self.assertAlmostEqual(main.hardware_gamma_ramp_max_deviation(identity), 0.0, places=12)
        self.assertGreater(main.hardware_gamma_ramp_max_deviation(corrected), 0.002)

    def test_texture_precision_follows_gl_vendor(self):
        self.assertEqual(
            stimuli.grating_texture_precision("nvidia corporation", "win32"), "16-bit float"
        )
        self.assertEqual(stimuli.grating_texture_precision("ati technologies inc.", "win32"), "32-bit float")
        self.assertEqual(stimuli.grating_texture_precision("nvidia corporation", "darwin"), "32-bit float")

    def test_session_info_logs_gamma_correction(self):
        session_info = main.make_session_info(
            exp_info={"subject": "subj", "session": "001", "run": "001"},
            datetime_start="2026-09-30T12:00:00",
            refresh_hz=120,
            code_version={"commit": "abc", "dirty": False, "diff": ""},
            gl_vendor="nvidia corporation",
        )
        hardware = session_info["hardware"]

        self.assertEqual(hardware["config_local"]["display_gamma"], stimuli.display_gamma)
        self.assertEqual(
            hardware["gamma_correction"],
            "software gamma correction, hardware ramp identity",
        )
        self.assertEqual(hardware["gl_vendor"], "nvidia corporation")
        self.assertIn(hardware["grating_texture_precision"], ("16-bit float", "32-bit float"))


class QuestPlusConfigurationTests(unittest.TestCase):
    def test_guessing_rate_matches_four_choice_task(self):
        self.assertEqual(QUEST_CONFIG["lower_asymptote_vals"], [0.25])
        self.assertEqual(len(main.contrast_response_keys), 4)


def small_quest_config(**overrides):
    config = dict(QUEST_CONFIG)
    config.update(overrides)
    return config


def hand_weibull(x, threshold, slope=2, guess=0.25, lapse=0.02):
    return 1 - lapse - (1 - guess - lapse) * math.exp(-((x / threshold) ** slope))


def hand_expected_entropy(intensity_vals, threshold_vals, posterior):
    # plain-python QUEST+ expected entropy, one value per intensity
    expected = []
    for x in intensity_vals:
        total = 0.0
        for correct in (True, False):
            joint = []
            for t, prior in zip(threshold_vals, posterior):
                p = hand_weibull(x, t)
                joint.append(prior * (p if correct else 1 - p))
            outcome_prob = sum(joint)
            entropy = -sum((j / outcome_prob) * math.log(j / outcome_prob) for j in joint if j > 0)
            total += outcome_prob * entropy
        expected.append(total)
    return expected


class QuestPlusComputationTests(unittest.TestCase):
    def test_weibull_matches_hand_computed_values(self):
        # at x = threshold: 1 - 0.02 - 0.73 * e^-1
        self.assertAlmostEqual(weibull_prob_correct(0.02, 0.02, 2, 0.25, 0.02), 0.711448, places=6)
        # x = threshold / 2: exp(-0.25); x = 2 * threshold: exp(-4)
        self.assertAlmostEqual(weibull_prob_correct(0.01, 0.02, 2, 0.25, 0.02), 0.411475, places=6)
        self.assertAlmostEqual(weibull_prob_correct(0.04, 0.02, 2, 0.25, 0.02), 0.966630, places=6)
        # asymptotes: guessing at zero contrast, 1 - lapse at high contrast
        self.assertAlmostEqual(weibull_prob_correct(1e-9, 0.02, 2, 0.25, 0.02), 0.25, places=6)
        self.assertAlmostEqual(weibull_prob_correct(1.0, 0.02, 2, 0.25, 0.02), 0.98, places=6)

    def test_one_update_matches_hand_computed_posterior(self):
        controller = QuestPlusController(
            quest_config=small_quest_config(intensity_vals=[0.02], threshold_vals=[0.01, 0.04]),
            seed=0,
        )
        controller.update(0.02, correct=True)
        # p(correct | t=0.01) = 0.966630, p(correct | t=0.04) = 0.411475; flat prior
        self.assertAlmostEqual(controller.threshold_posterior()[0], 0.966630 / 1.378105, places=5)
        self.assertAlmostEqual(controller.threshold_posterior()[1], 0.411475 / 1.378105, places=5)
        self.assertAlmostEqual(
            controller.estimated_threshold(),
            (0.01 * 0.966630 + 0.04 * 0.411475) / 1.378105,
            places=6,
        )

        controller.update(0.02, correct=False)
        p_low, p_high = 0.966630 * 0.033370, 0.411475 * 0.588525
        self.assertAlmostEqual(controller.threshold_posterior()[0], p_low / (p_low + p_high), places=5)

    def test_first_selection_is_start_intensity_and_no_update_leaves_estimate(self):
        controller = QuestPlusController(seed=0)
        self.assertEqual(controller.select_next_target_contrast(), QUEST_CONFIG["start_intensity"])
        self.assertEqual(controller.estimated_threshold(), QUEST_CONFIG["start_intensity"])
        self.assertTrue(np.allclose(controller.threshold_posterior(), 1 / 25))

    def test_selection_draws_from_three_lowest_expected_entropies(self):
        intensity_vals = [0.005, 0.01, 0.02, 0.04, 0.08]
        threshold_vals = [0.005, 0.01, 0.02, 0.04, 0.08]
        controller = QuestPlusController(
            quest_config=small_quest_config(
                intensity_vals=intensity_vals,
                threshold_vals=threshold_vals,
                start_intensity=0.08,
            ),
            seed=1,
        )
        for contrast, correct in ((0.08, True), (0.04, True), (0.02, False)):
            controller.update(contrast, correct)
        controller.select_next_target_contrast()  # start intensity

        posterior = controller.threshold_posterior()
        entropies = hand_expected_entropy(intensity_vals, threshold_vals, posterior)
        best_three = {intensity_vals[i] for i in sorted(range(5), key=entropies.__getitem__)[:3]}

        picks = {controller.select_next_target_contrast() for _ in range(300)}
        self.assertEqual(picks, best_three)

    def test_selection_never_exceeds_two_consecutive_repeats(self):
        intensity_vals = [0.005, 0.01, 0.02, 0.04, 0.08]
        controller = QuestPlusController(
            quest_config=small_quest_config(
                intensity_vals=intensity_vals,
                threshold_vals=intensity_vals,
                start_intensity=0.08,
            ),
            seed=2,
        )
        controller.select_next_target_contrast()  # start intensity
        for _ in range(200):
            contrast = controller.select_next_target_contrast()
            last_two = [entry["contrast"] for entry in controller.history[-2:]]
            if len(last_two) == 2:
                self.assertFalse(last_two[0] == last_two[1] == contrast)
            controller.update(contrast, correct=random.random() < 0.6)

    def test_simulated_observer_threshold_is_recovered(self):
        # 300 staircases of 60 events (one condition, one run) with a known threshold
        true_threshold = 0.02
        estimates = []
        for run in range(300):
            observer_rng = np.random.default_rng(1000 + run)
            controller = QuestPlusController(seed=run)
            for _ in range(60):
                contrast = controller.select_next_target_contrast()
                p = weibull_prob_correct(contrast, true_threshold, 2, 0.25, 0.02)
                controller.update(contrast, correct=observer_rng.random() < p)
            estimates.append(controller.estimated_threshold())
        estimates = np.array(estimates)

        self.assertAlmostEqual(np.median(estimates), true_threshold, delta=0.1 * true_threshold)
        within_25_percent = np.mean(np.abs(estimates / true_threshold - 1) < 0.25)
        self.assertGreater(within_25_percent, 0.9)

    def test_update_plus_selection_is_well_under_one_frame(self):
        controller = QuestPlusController(seed=0)
        controller.select_next_target_contrast()
        durations_ms = []
        for _ in range(60):
            start_s = time.perf_counter()
            contrast = controller.select_next_target_contrast()
            controller.update(contrast, correct=random.random() < 0.6)
            durations_ms.append((time.perf_counter() - start_s) * 1000)
        print(
            f"\nQUEST+ update + selection: median {np.median(durations_ms):.3f} ms, "
            f"max {max(durations_ms):.3f} ms"
        )
        self.assertLess(np.median(durations_ms), 1.0)


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


def make_recording_stims(center_count=8, center_contrasts=(0.2, 0.08)):
    return {
        "fixation_hole": RecordingStim(),
        "fixation": RecordingStim(),
        "centers": [
            {contrast: RecordingStim() for contrast in center_contrasts}
            for _ in range(center_count)
        ],
        "center_holes": [RecordingStim() for _ in range(center_count)],
        "surround": {stimuli.surround_contrast: RecordingStim()},
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
                    [
                        sum(center.draw_count for center in center_by_contrast.values())
                        for center_by_contrast in stims["centers"]
                    ],
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
            [
                sum(center.draw_count for center in center_by_contrast.values())
                for center_by_contrast in stims["centers"]
            ],
            [1, 1, 1, 1, 0, 1, 0, 0],
        )
        self.assertEqual(stims["centers"][5][0.08].draw_count, 1)
        self.assertEqual(stims["centers"][5][0.08].phase, 0.75)
        self.assertEqual(stims["centers"][5][0.08].ori, 135)


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

    def test_wrong_quadrant_updates_quest_but_no_response_does_not(self):
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
        self.assertEqual(quest_controller.updates, [(0.08, False)])


class TrialLoggingTests(unittest.TestCase):
    def test_log_trial_builds_complete_row_from_events_without_mutating_trial(self):
        class RecordingExperiment:
            def __init__(self):
                self.row = {}
                self.entry_count = 0

            def addData(self, key, value):
                self.row[key] = value

            def nextEntry(self):
                self.entry_count += 1

        trial = generate_trials(refresh_hz=120, shuffle=False)[0]
        original_trial = dict(trial)
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
                "outcome": "no_response",
                "target_contrast": 0.05,
            },
        ]
        exp = RecordingExperiment()

        main.log_trial(
            exp=exp,
            trial_num=1,
            trial=trial,
            events=events,
            quest_estimated_threshold=0.1,
            dropped_frame_count=0,
        )

        self.assertEqual(exp.entry_count, 1)
        self.assertEqual(exp.row["block_name"], "ssvep_task")
        self.assertEqual(exp.row["condition_id"], trial["condition_id"])
        self.assertEqual(exp.row["contrast_task_outcomes"], "[hit; no_response]")
        self.assertEqual(exp.row["contrast_task_no_response_count"], 1)
        self.assertEqual(exp.row["presented_target_contrasts"], "[0.2; 0.05]")
        self.assertEqual(exp.row["quest_estimated_threshold"], 0.1)
        self.assertEqual(trial, original_trial)
        removed_fields = {
            "block_index",
            "trial_index_within_block",
            "refresh_hz",
            "trial_start_frame_global",
            "trial_end_frame_global",
            "contrast_event_1_onset_frame",
            "contrast_event_target_quadrants",
        }
        self.assertTrue(removed_fields.isdisjoint(exp.row))
        self.assertFalse(any("e_stim" in key or "cs_plus" in key for key in exp.row))
        self.assertFalse(any(key.startswith("iti_tilt_") for key in exp.row))


class SessionMetadataTests(unittest.TestCase):
    def test_quest_controllers_are_saved_in_one_run_file(self):
        controllers = {
            condition_id: QuestPlusController(condition_id=condition_id, seed=condition_id)
            for condition_id in (1, 2)
        }
        controllers[2].update(0.1, correct=True)
        controllers[2].update(0.1, correct=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_dir = Path(temp_dir)
            main.save_questplus_handlers(
                quest_controllers=controllers,
                run_paths={"metadata_dir": metadata_dir},
                run_id="003",
            )
            saved_paths = list(metadata_dir.iterdir())
            snapshot = json.loads(saved_paths[0].read_text())

        self.assertEqual([path.name for path in saved_paths], ["questplus_run-003.json"])
        self.assertEqual(snapshot["run_id"], "003")
        self.assertEqual(list(snapshot["conditions"]), ["condition_01", "condition_02"])
        condition = snapshot["conditions"]["condition_02"]
        self.assertEqual(condition["settings"], json.loads(json.dumps(QUEST_CONFIG)))
        self.assertEqual(condition["seed"], 2)
        self.assertEqual(
            condition["trials"],
            [{"contrast": 0.1, "response": "Correct"}, {"contrast": 0.1, "response": "Incorrect"}],
        )
        self.assertEqual(condition["threshold_vals"], QUEST_CONFIG["threshold_vals"])
        self.assertAlmostEqual(sum(condition["threshold_posterior"]), 1.0)
        self.assertAlmostEqual(
            condition["estimated_threshold"], controllers[2].estimated_threshold()
        )
        self.assertEqual(snapshot["conditions"]["condition_01"]["trials"], [])

    def test_experiment_handler_disables_automatic_csv_save(self):
        handler = types.SimpleNamespace(dataNames=[])
        run_paths = {"trial_conditions_path": Path("trial_conditions_run-001")}

        with patch.object(
            main.data,
            "ExperimentHandler",
            return_value=handler,
            create=True,
        ) as experiment_handler:
            result = main.make_experiment_handler(
                {"subject": "subj", "session": "001", "run": "001"},
                run_paths,
            )

        self.assertIs(result, handler)
        self.assertFalse(experiment_handler.call_args.kwargs["saveWideText"])

    def test_collect_exp_info_uses_one_dialog_without_voltage(self):
        with patch.object(main.gui, "DlgFromDict", return_value=types.SimpleNamespace(OK=True), create=True) as dialog, patch.object(
            main, "get_next_run_id", return_value="002"
        ):
            exp_info = main.collect_exp_info()

        dialog.assert_called_once()
        self.assertEqual(exp_info, {"subject": "", "session": "001", "run": "002"})

    def test_make_session_info_groups_quest_config_separately(self):
        exp_info = {
            "subject": "subj",
            "session": "001",
            "run": "001",
        }

        session_info = main.make_session_info(
            exp_info=exp_info,
            datetime_start="2026-08-04T12:00:00",
            refresh_hz=120.0,
            code_version={"commit": "abc", "dirty": False, "diff": ""},
            gl_vendor="nvidia corporation",
        )

        self.assertEqual(
            session_info["quest configs"]["quest_intensity_definition"],
            "absolute_target_contrast",
        )
        self.assertTrue(session_info["quest configs"])
        self.assertTrue(
            all(key.startswith("quest_") for key in session_info["quest configs"])
        )
        self.assertFalse(
            any(key.startswith("quest_") for key in session_info["trial configs"])
        )
        self.assertNotIn("photocell_box_size_deg", session_info["stim configs"])
        self.assertNotIn("photocell_box_pos_deg", session_info["stim configs"])
        self.assertNotIn("center_positions_deg", session_info["stim configs"])
        self.assertNotIn("block_definitions", session_info["trial configs"])
        self.assertNotIn("contrast_response_keys", session_info["trial configs"])
        self.assertNotIn("e_stim", session_info)
        self.assertNotIn("e_stim_trigger_code", session_info["events"])
        self.assertNotIn("cs_plus_center_orientation", session_info["trial configs"])
        self.assertFalse(any("iti_tilt" in key or "show_iti" in key for key in session_info["trial configs"]))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session_info_run-001.json"
            main.write_session_info_json(path, session_info)
            saved_info = json.loads(path.read_text())

        self.assertEqual(saved_info["participant"], session_info["participant"])
        self.assertEqual(saved_info["quest configs"], session_info["quest configs"])


class TrialGenerationTests(unittest.TestCase):
    def test_single_task_block_has_configured_trials_per_condition(self):
        trials = generate_trials(refresh_hz=120)
        trials_per_condition = block_definitions[0]["trials_per_condition"]
        condition_ids = [row["condition_id"] for row in condition_event_code_rows]

        self.assertEqual(len(trials), trials_per_condition * len(condition_ids))
        self.assertEqual(
            Counter(trial["condition_id"] for trial in trials),
            {condition_id: trials_per_condition for condition_id in condition_ids},
        )
        self.assertEqual({trial["block_name"] for trial in trials}, {"ssvep_task"})
        self.assertTrue(all("block_index" not in trial for trial in trials))
        self.assertTrue(all("trial_index_within_block" not in trial for trial in trials))
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
            result = main.run_iti(
                win=win,
                stims=stims,
                iti_duration_s=0.5,
                refresh_hz=8,
            )

        self.assertIsNone(result)
        self.assertEqual(stims["fixation"].draw_count, 4)
        self.assertEqual(stims["fixation"].color, "white")
        self.assertTrue(all(
            stim.draw_count == 0 for name, stim in stims.items()
            if name != "fixation" and not isinstance(stim, (list, dict))
        ))
        self.assertTrue(all(stim.draw_count == 0 for stim in stims["surround"].values()))

    def test_iti_sends_stimulus_offset_on_first_flip_and_clears_on_second(self):
        event_log = []
        win = FakeWindow(event_log=event_log)
        trigger = FakeTrigger(event_log=event_log)

        with patch.object(main.event, "clearEvents", create=True), patch.object(
            main, "handle_runtime_keys"
        ):
            main.run_iti(
                win=win,
                stims=make_recording_stims(),
                iti_duration_s=0.5,
                refresh_hz=8,
                trigger=trigger,
            )

        self.assertEqual(trigger.codes, [main.stimulus_offset_code, 0])
        self.assertEqual(
            event_log,
            [
                ("scheduled", "send_code"),
                ("flip", 1),
                ("send_code", main.stimulus_offset_code),
                ("scheduled", "send_code"),
                ("flip", 1),
                ("send_code", 0),
                ("flip", 0),
                ("flip", 0),
            ],
        )


class FakeTrigger:
    def __init__(self, event_log=None):
        self.codes = []
        self.event_log = event_log

    def send_code(self, code):
        self.codes.append(code)
        if self.event_log is not None:
            self.event_log.append(("send_code", code))


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
        trigger = FakeTrigger(event_log=event_log)
        trial = self.make_trial()

        with patch.object(main, "build_trial_events", return_value=[]), patch.object(
            main, "prepare_stimuli_for_trial"
        ), patch.object(main, "draw_stimulus"), patch.object(
            main, "finalize_elapsed_events", side_effect=lambda *args: args[2]
        ), patch.object(
            main, "get_task_response", return_value=None
        ), patch.object(
            main, "handle_runtime_keys"
        ):
            result = main.run_trial(
                win=win,
                stims={},
                trial=trial,
                trial_duration_s=81,
                refresh_hz=1,
                quest_controller=DummyQuestController(),
                trigger=trigger,
                trial_num=4,
            )

        self.assertEqual(result, [])
        self.assertEqual(
            event_log[:7],
            [
                ("scheduled", "send_code"),
                ("flip", 1),
                ("send_code", 7),
                ("scheduled", "send_code"),
                ("flip", 1),
                ("send_code", 0),
                ("flip", 0),
            ],
        )
        self.assertEqual(
            event_log[-4:],
            [
                ("scheduled", "send_code"),
                ("flip", 1),
                ("send_code", main.frame_marker_code),
                ("send_code", 0),
            ],
        )
        self.assertEqual(trigger.codes, [7, 0, main.frame_marker_code, 0])



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
