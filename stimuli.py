from psychopy import monitors, visual
from psychopy.visual.grating import GratingStim

from config import (
    bg_color,
    center_diameter_deg,
    center_flicker_hz,
    center_holes_diameter_deg,
    center_positions_deg,
    center_sf,
    frames_per_reversal,
    photocell_box_pos_deg,
    photocell_box_size_deg,
    raised_cos_fringe,
    raised_cos_fringe_params,
    surround_contrast,
    surround_diameter_deg,
    surround_flicker_hz,
    surround_sf,
    units,
)


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

    return visual.Window(
        size=size,
        fullscr=fullscr,
        screen=screen,
        monitor=mon,
        units=units,
        color=bg_color,
        bpc=(10, 10, 10),
    )


def make_stimuli(win):
    fixation_hole = GratingStim(
        win=win,
        tex=None,  # pyright: ignore
        color=bg_color,
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

    centers = []
    center_holes = []
    for pos in center_positions_deg:
        centers.append(
            GratingStim(
                win=win,
                tex="sin",
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
        )
        center_holes.append(
            GratingStim(
                win=win,
                tex=None,  # pyright: ignore
                color=bg_color,
                mask=raised_cos_fringe,
                maskParams=raised_cos_fringe_params,
                size=center_holes_diameter_deg,
                pos=pos,
                sf=0,
                autoLog=False,
            )
        )

    surround = GratingStim(
        win=win,
        tex="sin",
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
        "photocell_box": photocell_box,
    }


def prepare_stimuli_for_trial(stims, trial):
    for center in stims["centers"]:
        center.ori = trial["center_orientation"]

    if trial["surround_orientation"] is not None:
        stims["surround"].ori = trial["surround_orientation"]


def draw_stimulus(
    stims,
    frame_num,
    refresh_hz,
    surround_orientation,
    center_contrast,
    fixation_color="white",
    surround_contrast=surround_contrast,
    center_flicker_hz=center_flicker_hz,
    surround_flicker_hz=surround_flicker_hz,
    task_center_indices=(),
    target_index=None,
    target_contrast=None,
):
    center_phase = phase_reversal(frame_num, refresh_hz, center_flicker_hz)
    stims["photocell_box"].contrast = flicker_indicator_contrast_from_phase(center_phase)

    centers_to_draw = []
    for center_index, center in enumerate(stims["centers"]):
        center.phase = center_phase
        if center_index in task_center_indices:
            if center_index != target_index or target_contrast is None:
                continue
            center.contrast = target_contrast
        else:
            center.contrast = center_contrast
        centers_to_draw.append(center)

    if surround_orientation is not None:
        stims["surround"].phase = phase_reversal(frame_num, refresh_hz, surround_flicker_hz)
        stims["surround"].contrast = surround_contrast
        stims["surround"].draw()
        for hole in stims["center_holes"]:
            hole.draw()

    for center in centers_to_draw:
        center.draw()

    stims["photocell_box"].draw()
    stims["fixation"].color = fixation_color
    stims["fixation_hole"].draw()
    stims["fixation"].draw()
