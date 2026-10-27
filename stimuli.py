import numpy as np
from psychopy import monitors, visual
from psychopy.visual.grating import GratingStim

from config import (
    calibration_mode,
    center_diameter_deg,
    center_flicker_hz,
    center_holes_diameter_deg,
    center_positions_deg,
    center_sf,
    condition_event_code_rows,
    display_gamma,
    frames_per_reversal,
    orientation_report_prompt_text,
    photocell_box_pos_deg,
    photocell_box_size_deg,
    probe_outline_diameter_deg,
    probe_outline_line_width_px,
    probe_outline_luminance,
    quest_configs,
    raised_cos_fringe,
    raised_cos_fringe_params,
    surround_diameter_deg,
    surround_flicker_hz,
    surround_sf,
    units,
)
from task import eligible_center_indices

grating_texture_res = 256
contrast_match_tolerance = 1e-9


def luminance_to_rgb(L):
    # L is luminance as a fraction of white (0..1); returns PsychoPy rgb (-1..1)
    return 2 * L ** (1 / display_gamma) - 1


def grating_texture(contrast, res=grating_texture_res):
    # one cycle, sampled at texel centres; sin(2*pi*x - pi/2) matches PsychoPy's "sin"
    x = (np.arange(res) + 0.5) / res
    row = luminance_to_rgb(0.5 * (1 + contrast * np.sin(2 * np.pi * x - np.pi / 2)))
    return np.tile(row, (res, 1)).astype(np.float32)


def grating_texture_precision(gl_vendor, platform):
    # mirrors PsychoPy's _createTexture: numpy tex is RGB16F on NVIDIA (non-Mac), else RGB32F
    if platform != "darwin" and gl_vendor.startswith("nvidia"):
        return "16-bit float"
    return "32-bit float"


def stim_for_contrast(stims_by_contrast, contrast):
    nearest = min(stims_by_contrast, key=lambda value: abs(value - contrast))
    if abs(nearest - contrast) > contrast_match_tolerance:
        raise ValueError(f"No precomputed grating for contrast {contrast}.")
    return stims_by_contrast[nearest]


def phase_reversal(frame_num, refresh_hz, flicker_hz):
    cycle_index = frame_num // frames_per_reversal(refresh_hz, flicker_hz)
    return 0.25 if cycle_index % 2 == 0 else 0.75


def flicker_indicator_contrast_from_phase(center_phase):
    return 1.0 if center_phase == 0.25 else 0.0


def make_window(
    size,
    fullscr,
    screen,
    monitor_name,
    monitor_width_cm,
    viewing_distance_cm,
):
    mon = monitors.Monitor(monitor_name)
    mon.setSizePix(size)
    mon.setWidth(monitor_width_cm)
    mon.setDistance(viewing_distance_cm)
    # in memory only: a saved gamma makes PsychoPy load it into the hardware ramp
    gamma_grid = mon.getGammaGrid()
    if gamma_grid is not None:
        gamma_grid = gamma_grid.copy()
        gamma_grid[1:, 2] = 1
        mon.setGammaGrid(gamma_grid)
    mon.setGamma(None)

    mean_gray = luminance_to_rgb(0.5)
    return visual.Window(
        size=size,
        fullscr=fullscr,
        screen=screen,
        monitor=mon,
        units=units,
        color=(mean_gray, mean_gray, mean_gray),
        bpc=(10, 10, 10),
    )


def make_stimuli(win):
    mean_gray = luminance_to_rgb(0.5)
    mean_gray_rgb = (mean_gray, mean_gray, mean_gray)
    task_center_indices = eligible_center_indices()

    fixation_hole = GratingStim(
        win=win,
        tex=None,  # pyright: ignore
        color=mean_gray_rgb,
        mask=raised_cos_fringe,
        maskParams=raised_cos_fringe_params,
        size=center_diameter_deg,
        sf=0,
        pos=(0, 0),
        autoLog=False,
    )

    fixation = visual.TextStim(
        win=win,
        text="+",
        pos=(0, 0),
        height=1.5,
        color="white",
        autoLog=False,
    )

    center_contrasts = {row["center_contrast"] for row in condition_event_code_rows}
    surround_contrasts = {
        row["surround_contrast"]
        for row in condition_event_code_rows
        if row["surround_orientation"] is not None
    }

    # every contrast either QUEST+ group can request
    quest_intensity_vals = sorted(
        {value for quest_config in quest_configs.values() for value in quest_config["intensity_vals"]}
    )

    centers = []
    center_holes = []
    for center_index, pos in enumerate(center_positions_deg):
        # one stim per contrast this center can show, textures uploaded once here
        contrasts = list(center_contrasts)
        if center_index in task_center_indices:
            contrasts += quest_intensity_vals
        centers.append(
            {
                contrast: GratingStim(
                    win=win,
                    tex=grating_texture(contrast),
                    mask=raised_cos_fringe,
                    maskParams=raised_cos_fringe_params,
                    size=center_diameter_deg,
                    sf=center_sf,
                    contrast=1.0,
                    ori=0,
                    phase=0.0,
                    pos=pos,
                    autoLog=False,
                )
                for contrast in contrasts
            }
        )
        center_holes.append(
            GratingStim(
                win=win,
                tex=None,  # pyright: ignore
                color=mean_gray_rgb,
                mask=raised_cos_fringe,
                maskParams=raised_cos_fringe_params,
                size=center_holes_diameter_deg,
                pos=pos,
                sf=0,
                autoLog=False,
            )
        )

    surround = {
        contrast: GratingStim(
            win=win,
            tex=grating_texture(contrast),
            mask=raised_cos_fringe,
            maskParams=raised_cos_fringe_params,
            size=surround_diameter_deg,
            sf=surround_sf,
            contrast=1.0,
            ori=0,
            phase=0.0,
            pos=(0, 0),
            autoLog=False,
        )
        for contrast in surround_contrasts
    }

    # practice only: faint rings marking where targets can appear
    outline_rgb = luminance_to_rgb(probe_outline_luminance)
    probe_outlines = [
        visual.Circle(
            win=win,
            radius=probe_outline_diameter_deg / 2,
            edges=128,
            lineColor=(outline_rgb, outline_rgb, outline_rgb),
            lineWidth=probe_outline_line_width_px,
            fillColor=None,
            pos=center_positions_deg[center_index],
            autoLog=False,
        )
        for center_index in task_center_indices
    ]

    orientation_prompt = visual.TextStim(
        win=win,
        text=orientation_report_prompt_text,
        pos=(0, 0),
        height=1.0,
        color="white",
        autoLog=False,
    )
    orientation_feedback = visual.TextStim(
        win=win,
        text="",
        pos=(0, 0),
        height=1.0,
        color="white",
        autoLog=False,
    )

    photocell_box = GratingStim(
        win=win,
        tex=None,  # pyright: ignore
        color="white",
        size=photocell_box_size_deg,
        sf=0,
        contrast=0.0,
        pos=photocell_box_pos_deg,
        autoLog=False,
    )

    return {
        "fixation_hole": fixation_hole,
        "fixation": fixation,
        "centers": centers,
        "center_holes": center_holes,
        "surround": surround,
        "probe_outlines": probe_outlines,
        "orientation_prompt": orientation_prompt,
        "orientation_feedback": orientation_feedback,
        "photocell_box": photocell_box,
    }


def prepare_stimuli_for_trial(stims, trial):
    for center_by_contrast in stims["centers"]:
        for center in center_by_contrast.values():
            center.ori = trial["center_orientation"]

    if trial["surround_orientation"] is not None:
        for surround in stims["surround"].values():
            surround.ori = trial["surround_orientation"]


def draw_stimulus(
    stims,
    frame_num,
    refresh_hz,
    surround_orientation,
    center_contrast,
    surround_contrast,
    fixation_color="white",
    center_flicker_hz=center_flicker_hz,
    surround_flicker_hz=surround_flicker_hz,
    task_center_indices=(),
    target_index=None,
    target_contrast=None,
    show_probe_outlines=False,
):
    center_phase = phase_reversal(frame_num, refresh_hz, center_flicker_hz)
    stims["photocell_box"].contrast = flicker_indicator_contrast_from_phase(center_phase)

    centers_to_draw = []
    for center_index, center_by_contrast in enumerate(stims["centers"]):
        if center_index in task_center_indices:
            if center_index != target_index or target_contrast is None:
                continue
            center = stim_for_contrast(center_by_contrast, target_contrast)
        else:
            center = stim_for_contrast(center_by_contrast, center_contrast)
        center.phase = center_phase
        centers_to_draw.append(center)

    if surround_orientation is not None:
        surround = stim_for_contrast(stims["surround"], surround_contrast)
        surround.phase = phase_reversal(frame_num, refresh_hz, surround_flicker_hz)
        surround.draw()
        for hole in stims["center_holes"]:
            hole.draw()

    for center in centers_to_draw:
        center.draw()

    if show_probe_outlines:
        for outline in stims["probe_outlines"]:
            outline.draw()

    if calibration_mode:
        stims["photocell_box"].draw()
    stims["fixation"].color = fixation_color
    stims["fixation_hole"].draw()
    stims["fixation"].draw()
