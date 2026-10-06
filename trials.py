import json
import random

from config import (
    block_definitions,
    center_contrast,
    condition_event_code_rows,
    cs_orientations_deg,
    max_consecutive_unreinforced_cs_plus,
    surround_contrast,
    trial_duration_s,
    us_count_per_cs_plus_condition,
)
from task import sample_trial_task

reinforcement_max_attempts = 10000


def build_base_trial(condition_row, refresh_hz, cs_plus_orientation):
    task_info = sample_trial_task(
        refresh_hz=refresh_hz,
        trial_duration_s=trial_duration_s,
    )
    return {
        "condition_id": condition_row["condition_id"],
        "event_code": condition_row["event_code"],
        "center_orientation": condition_row["center_orientation"],
        "surround_orientation": condition_row["surround_orientation"],
        "center_contrast": center_contrast,
        "surround_contrast": surround_contrast,
        "cs_type": cs_type_for_orientation(
            condition_row["center_orientation"], cs_plus_orientation
        ),
        "reinforced": False,
        **task_info,
    }


def shuffle_with_condition_run_limit(trials, max_consecutive=2, first_condition_ids=None):
    trials_by_condition = {}
    for trial in trials:
        trials_by_condition.setdefault(trial["condition_id"], []).append(trial)

    for group in trials_by_condition.values():
        random.shuffle(group)

    condition_counts = {
        condition_id: len(group)
        for condition_id, group in trials_by_condition.items()
    }
    ordered_condition_ids = build_condition_order(
        condition_counts, max_consecutive, first_condition_ids
    )

    ordered_trials = []
    for condition_id in ordered_condition_ids:
        ordered_trials.append(trials_by_condition[condition_id].pop())

    return ordered_trials


def build_condition_order(condition_counts, max_consecutive, first_condition_ids=None):
    total_trials = sum(condition_counts.values())
    ordered_condition_ids = []

    def backtrack(last_condition_ids):
        if len(ordered_condition_ids) == total_trials:
            return True

        candidate_ids = [
            condition_id
            for condition_id, remaining_count in condition_counts.items()
            if remaining_count > 0
        ]
        # first trial restricted to the given conditions (CS+)
        if not ordered_condition_ids and first_condition_ids is not None:
            candidate_ids = [
                condition_id for condition_id in candidate_ids
                if condition_id in first_condition_ids
            ]
        random.shuffle(candidate_ids)
        candidate_ids.sort(key=lambda condition_id: condition_counts[condition_id], reverse=True)

        for condition_id in candidate_ids:
            if len(last_condition_ids) >= max_consecutive and all(
                recent_condition_id == condition_id
                for recent_condition_id in last_condition_ids[-max_consecutive:]
            ):
                continue

            condition_counts[condition_id] -= 1
            ordered_condition_ids.append(condition_id)

            if backtrack((last_condition_ids + [condition_id])[-max_consecutive:]):
                return True

            ordered_condition_ids.pop()
            condition_counts[condition_id] += 1

        return False

    if not backtrack([]):
        raise ValueError(
            "Unable to generate a trial order without exceeding the condition run limit."
        )

    return ordered_condition_ids


def generate_block_trials(
    block_definition,
    refresh_hz,
    cs_plus_orientation,
    shuffle=True,
    first_trial_cs_plus=False,
):
    block_trials = []

    for condition_row in condition_event_code_rows:
        for _ in range(block_definition["trials_per_condition"]):
            trial = build_base_trial(condition_row, refresh_hz, cs_plus_orientation)
            trial["block_name"] = block_definition["block_name"]
            block_trials.append(trial)

    if shuffle:
        first_condition_ids = None
        if first_trial_cs_plus:
            first_condition_ids = cs_plus_condition_ids(cs_plus_orientation)
        block_trials = shuffle_with_condition_run_limit(
            block_trials, first_condition_ids=first_condition_ids
        )

    return block_trials


def generate_trials(
    refresh_hz,
    cs_plus_orientation,
    shuffle=True,
):
    trials = []

    for block_index, block_definition in enumerate(block_definitions):
        trials.extend(
            generate_block_trials(
                block_definition=block_definition,
                refresh_hz=refresh_hz,
                cs_plus_orientation=cs_plus_orientation,
                shuffle=shuffle,
                first_trial_cs_plus=(block_index == 0),
            )
        )

    assign_reinforcement(trials)
    return trials


def assign_reinforcement(
    trials,
    us_count_per_condition=us_count_per_cs_plus_condition,
    max_consecutive_unreinforced=max_consecutive_unreinforced_cs_plus,
    max_attempts=reinforcement_max_attempts,
):
    """Mark CS+ trials as reinforced in place.

    First appearance of each CS+ condition is always reinforced; the rest are
    drawn at random, rejecting draws that leave too many unreinforced CS+ in a row.
    """
    cs_plus_positions = [
        index for index, trial in enumerate(trials) if trial["cs_type"] == "cs_plus"
    ]
    positions_by_condition = {}
    for index in cs_plus_positions:
        positions_by_condition.setdefault(trials[index]["condition_id"], []).append(index)

    for condition_id, positions in positions_by_condition.items():
        if not 1 <= us_count_per_condition <= len(positions):
            raise ValueError(
                f"us_count_per_condition={us_count_per_condition} must be between 1 and "
                f"the {len(positions)} CS+ trials of condition {condition_id}."
            )

    for _ in range(max_attempts):
        reinforced_positions = set()
        for positions in positions_by_condition.values():
            reinforced_positions.add(positions[0])
            reinforced_positions.update(
                random.sample(positions[1:], us_count_per_condition - 1)
            )
        run_length = longest_unreinforced_run(cs_plus_positions, reinforced_positions)
        if run_length <= max_consecutive_unreinforced:
            for index in cs_plus_positions:
                trials[index]["reinforced"] = index in reinforced_positions
            return trials

    raise ValueError(
        "Unable to place reinforced trials without exceeding "
        f"{max_consecutive_unreinforced} unreinforced CS+ trials in a row."
    )


def generate_trial_schedule(refresh_hz, seed=None):
    """Pre-generate one participant's CS+ assignment and full trial list."""
    seed = random.SystemRandom().randrange(2**32) if seed is None else int(seed)
    # task targets in sample_trial_task use the module-level random, so seed it too
    random.seed(seed)
    cs_plus_orientation = random.choice(cs_orientations_deg)
    trials = generate_trials(refresh_hz=refresh_hz, cs_plus_orientation=cs_plus_orientation)
    return {
        "seed": seed,
        "cs_plus_orientation": cs_plus_orientation,
        "cs_minus_orientation": cs_minus_orientation_for(cs_plus_orientation),
        "generation_settings": schedule_generation_settings(refresh_hz),
        "trials": trials,
    }


def save_trial_schedule(path, schedule):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schedule, indent=2) + "\n")


def load_trial_schedule(path, refresh_hz):
    schedule = json.loads(path.read_text())
    # a schedule made under different settings would no longer match the config
    if schedule["generation_settings"] != schedule_generation_settings(refresh_hz):
        raise ValueError(
            f"{path} was generated with settings that differ from the current config. "
            "Use a new subject ID or move the old schedule file."
        )
    return schedule


def load_or_generate_trial_schedule(path, refresh_hz):
    if path.exists():
        return load_trial_schedule(path, refresh_hz)
    schedule = generate_trial_schedule(refresh_hz)
    save_trial_schedule(path, schedule)
    return schedule


def schedule_generation_settings(refresh_hz):
    # json round-trip so tuples and lists compare equal after loading
    return json.loads(json.dumps({
        "refresh_hz": refresh_hz,
        "trial_duration_s": trial_duration_s,
        "block_definitions": block_definitions,
        "condition_event_code_rows": condition_event_code_rows,
        "cs_orientations_deg": cs_orientations_deg,
        "us_count_per_cs_plus_condition": us_count_per_cs_plus_condition,
        "max_consecutive_unreinforced_cs_plus": max_consecutive_unreinforced_cs_plus,
    }))


def cs_type_for_orientation(center_orientation, cs_plus_orientation):
    return "cs_plus" if center_orientation == cs_plus_orientation else "cs_minus"


def cs_minus_orientation_for(cs_plus_orientation):
    return next(
        orientation for orientation in cs_orientations_deg
        if orientation != cs_plus_orientation
    )


def cs_plus_condition_ids(cs_plus_orientation):
    return {
        row["condition_id"] for row in condition_event_code_rows
        if row["center_orientation"] == cs_plus_orientation
    }


def longest_unreinforced_run(cs_plus_positions, reinforced_positions):
    longest_run = 0
    current_run = 0
    for index in cs_plus_positions:
        current_run = 0 if index in reinforced_positions else current_run + 1
        longest_run = max(longest_run, current_run)
    return longest_run
