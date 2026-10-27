from collections import Counter
import math
import random

from config import (
    block_definitions,
    condition_event_code_rows,
    trial_duration_s,
    us_max_consecutive_nonreinforced,
    us_reinforced_trials_per_condition,
)
from task import sample_trial_task


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
        "center_contrast": condition_row["center_contrast"],
        "surround_contrast": condition_row["surround_contrast"],
        "cs_type": "cs_plus" if condition_row["center_orientation"] == cs_plus_orientation else "cs_minus",
        "reinforced": False,  # set by assign_reinforcement
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
    cs_plus_orientation,
    shuffle=True,
):
    block_trials = []

    for condition_row in condition_event_code_rows:
        for _ in range(block_definition["trials_per_condition"]):
            trial = build_base_trial(condition_row, refresh_hz, cs_plus_orientation)
            trial["block_name"] = block_definition["block_name"]
            block_trials.append(trial)

    if shuffle:
        block_trials = shuffle_with_condition_run_limit(block_trials)

    assign_reinforcement(
        block_trials,
        reinforced_per_condition=us_reinforced_trials_per_condition,
        max_consecutive_nonreinforced=us_max_consecutive_nonreinforced,
    )
    return block_trials


def generate_trials(
    refresh_hz,
    cs_plus_orientation,
    shuffle=True,
):
    trials = []

    for block_definition in block_definitions:
        trials.extend(
            generate_block_trials(
                block_definition=block_definition,
                refresh_hz=refresh_hz,
                cs_plus_orientation=cs_plus_orientation,
                shuffle=shuffle,
            )
        )

    return trials


def assign_reinforcement(trials, reinforced_per_condition, max_consecutive_nonreinforced):
    """Set "reinforced" in place on exactly reinforced_per_condition trials of each CS+ condition.

    Along the CS+ trial sequence (CS- trials skipped) the first trial is reinforced and
    no more than max_consecutive_nonreinforced trials in a row are unreinforced.
    """
    cs_plus_trials = [trial for trial in trials if trial["cs_type"] == "cs_plus"]
    trial_counts = Counter(trial["condition_id"] for trial in cs_plus_trials)
    check_reinforcement_is_feasible(trial_counts, reinforced_per_condition, max_consecutive_nonreinforced)

    trials_left = dict(trial_counts)
    reinforced_left = {condition_id: reinforced_per_condition for condition_id in trial_counts}
    decisions = []

    def backtrack(nonreinforced_run):
        if len(decisions) == len(cs_plus_trials):
            return True

        condition_id = cs_plus_trials[len(decisions)]["condition_id"]
        options = [True, False]
        # try reinforcing first with this condition's remaining reinforcement rate
        if random.random() >= reinforced_left[condition_id] / trials_left[condition_id]:
            options.reverse()

        for reinforced in options:
            if reinforced and reinforced_left[condition_id] == 0:
                continue
            if not reinforced and (
                not decisions
                or nonreinforced_run == max_consecutive_nonreinforced
                or reinforced_left[condition_id] == trials_left[condition_id]
            ):
                continue

            trials_left[condition_id] -= 1
            reinforced_left[condition_id] -= int(reinforced)
            decisions.append(reinforced)
            next_run = 0 if reinforced else nonreinforced_run + 1
            # prune: every remaining unreinforced trial must fit in a run after a reinforced one
            unreinforced_total_left = sum(trials_left.values()) - sum(reinforced_left.values())
            room_left = (max_consecutive_nonreinforced - next_run) + (
                max_consecutive_nonreinforced * sum(reinforced_left.values())
            )
            if unreinforced_total_left <= room_left and backtrack(next_run):
                return True

            decisions.pop()
            reinforced_left[condition_id] += int(reinforced)
            trials_left[condition_id] += 1

        return False

    if not backtrack(0):
        raise ValueError("Unable to generate a reinforcement schedule within the CS+ run limit.")

    for trial, reinforced in zip(cs_plus_trials, decisions):
        trial["reinforced"] = reinforced


def check_reinforcement_is_feasible(trial_counts, reinforced_per_condition, max_consecutive_nonreinforced):
    reinforced_total = reinforced_per_condition * len(trial_counts)
    unreinforced_total = sum(trial_counts.values()) - reinforced_total
    # first CS+ trial is reinforced, so each unreinforced run follows a reinforced trial
    if (
        reinforced_per_condition < 1
        or any(reinforced_per_condition > count for count in trial_counts.values())
        or unreinforced_total > max_consecutive_nonreinforced * reinforced_total
    ):
        trials_per_condition = max(trial_counts.values())
        minimum = math.ceil(trials_per_condition / (max_consecutive_nonreinforced + 1))
        raise ValueError(
            f"us_reinforced_trials_per_condition = {reinforced_per_condition} is infeasible with "
            f"{trials_per_condition} trials per condition and us_max_consecutive_nonreinforced = "
            f"{max_consecutive_nonreinforced}; it must be between {minimum} and {trials_per_condition}."
        )
