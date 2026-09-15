import random

from config import (
    block_definitions,
    center_contrast,
    condition_event_code_rows,
    surround_contrast,
    trial_duration_s,
)
from task import sample_trial_task


def build_base_trial(condition_row, refresh_hz):
    task_info = sample_trial_task(
        refresh_hz=refresh_hz,
        trial_duration_s=trial_duration_s,
    )
    return {
        "condition_id": condition_row["condition_id"],
        "center_orientation": condition_row["center_orientation"],
        "surround_orientation": condition_row["surround_orientation"],
        "center_contrast": center_contrast,
        "surround_contrast": surround_contrast,
        **task_info,
    }


def shuffle_with_condition_run_limit(trials, max_consecutive=2):
    trials_by_condition = {}
    for trial in trials:
        trials_by_condition.setdefault(trial["condition_id"], []).append(trial)

    for group in trials_by_condition.values():
        random.shuffle(group)

    condition_counts = {
        condition_id: len(group)
        for condition_id, group in trials_by_condition.items()
    }
    ordered_condition_ids = build_condition_order(condition_counts, max_consecutive)

    ordered_trials = []
    for condition_id in ordered_condition_ids:
        ordered_trials.append(trials_by_condition[condition_id].pop())

    return ordered_trials


def build_condition_order(condition_counts, max_consecutive):
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
    shuffle=True,
):
    block_trials = []

    for condition_row in condition_event_code_rows:
        for _ in range(block_definition["trials_per_condition"]):
            trial = build_base_trial(condition_row, refresh_hz)
            trial["block_name"] = block_definition["block_name"]
            block_trials.append(trial)

    if shuffle:
        block_trials = shuffle_with_condition_run_limit(block_trials)

    return block_trials


def generate_trials(
    refresh_hz,
    shuffle=True,
):
    trials = []

    for block_definition in block_definitions:
        trials.extend(
            generate_block_trials(
                block_definition=block_definition,
                refresh_hz=refresh_hz,
                shuffle=shuffle,
            )
        )

    return trials
