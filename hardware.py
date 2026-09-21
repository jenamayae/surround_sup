"""LabJack trigger helpers for 8-bit event codes on FIO0-FIO7.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from config import (
    frame_marker_code,
    frame_marker_interval,
    labjack_connection_type,
    labjack_device_type,
    labjack_fio_lines,
    labjack_fio_mask,
    labjack_identifier,
    stimulus_offset_code,
    trigger_min_gap_s,
    trigger_pulse_width_s,
)


class BaseTrigger:
    """Synchronous trigger backend interface."""

    def __init__(self, pulse_width_s=trigger_pulse_width_s, min_gap_s=trigger_min_gap_s):
        self.pulse_width_s = pulse_width_s
        self.min_gap_s = min_gap_s
        self.io_lock = threading.Lock()
        self.last_clear_time = None

    def open(self):
        raise NotImplementedError

    def require_open(self):
        raise NotImplementedError

    def pulse_event(self, code: int, pulse_width_s=None, min_gap_s=None):
        raise NotImplementedError

    def close(self):
        return


class TriggerDispatcher:
    """Owns async trigger dispatch for a synchronous trigger backend."""

    def __init__(self, trigger: BaseTrigger):
        self.trigger = trigger
        self._queue: queue.Queue[dict[str, object] | None] = queue.Queue()
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="TriggerDispatcher",
            daemon=True,
        )
        self._worker.start()

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return

            code = int(item["code"])
            label = item["label"]
            frame_num = item["frame_num"]
            trial_num = item["trial_num"]
            try:
                self.trigger.pulse_event(code=code)
            except Exception as exc:
                logging.error(
                    "Trigger error for label=%s code=%s frame_num=%s trial_num=%s: %s",
                    label,
                    code,
                    frame_num,
                    trial_num,
                    exc,
                )
                print(
                    "WARNING: Trigger error for "
                    f"label={label} code={code} frame_num={frame_num} "
                    f"trial_num={trial_num}: {exc}"
                )
            finally:
                self._queue.task_done()

    def submit(self, code, *, label, frame_num=None, trial_num=None):
        if self._closed:
            raise RuntimeError("TriggerDispatcher is closed.")
        self._queue.put(
            {
                "code": int(code),
                "label": str(label),
                "frame_num": frame_num,
                "trial_num": trial_num,
            }
        )

    def drain(self):
        self._queue.join()

    def close(self):
        if self._closed:
            return
        self.drain()
        self._closed = True
        self._queue.put(None)
        self._worker.join()


class LabjackFio8BitTrigger(BaseTrigger):
    """Send 8-bit event codes on LabJack FIO0-FIO7."""

    def __init__(
        self,
        device_type=labjack_device_type,
        connection_type=labjack_connection_type,
        identifier=labjack_identifier,
        fio_lines=labjack_fio_lines,
        fio_mask=labjack_fio_mask,
        pulse_width_s=trigger_pulse_width_s,
        min_gap_s=trigger_min_gap_s,
    ):
        super().__init__(pulse_width_s=pulse_width_s, min_gap_s=min_gap_s)
        self.device_type = device_type
        self.connection_type = connection_type
        self.identifier = identifier
        self.fio_lines = list(fio_lines)
        self.fio_mask = fio_mask
        self.ljm = None
        self.handle = None

    def open(self):
        """Open the LabJack and configure FIO0-FIO7 as digital outputs."""
        if self.handle is not None:
            return

        try:
            from labjack import ljm
        except ImportError as exc:
            raise RuntimeError("Could not import labjack-ljm. Install with: pip install labjack-ljm") from exc

        self.ljm = ljm
        self.handle = self.ljm.openS(self.device_type, self.connection_type, self.identifier)
        self.configure_fio_outputs()
        logging.info(
            "LabJack trigger opened: device_type=%s connection_type=%s identifier=%s",
            self.device_type,
            self.connection_type,
            self.identifier,
        )

    def require_open(self):
        if self.handle is None or self.ljm is None:
            raise RuntimeError("LabJack is not open. Call open() first.")
        return self.ljm, self.handle

    def configure_fio_outputs(self):
        """Set FIO0-FIO7 as digital outputs and clear the bus."""
        ljm, handle = self.require_open()
        try:
            analog_enable = int(ljm.eReadName(handle, "DIO_ANALOG_ENABLE"))
            ljm.eWriteName(handle, "DIO_ANALOG_ENABLE", float(analog_enable & ~self.fio_mask))
        except Exception as exc:
            logging.warning("Could not set DIO_ANALOG_ENABLE: %s", exc)
            print(f"WARNING: Could not set DIO_ANALOG_ENABLE: {exc}")

        ljm.eWriteName(handle, "FIO_DIRECTION", float(self.fio_mask))
        self.clear()

    def send_code(self, code: int):
        """Drive FIO0-FIO7 with the given 8-bit code."""
        ljm, handle = self.require_open()
        if not 0 <= int(code) <= 255:
            raise ValueError(f"Trigger code must be in [0, 255], got {code}")

        values = [float((int(code) >> bit) & 1) for bit in range(8)]
        ljm.eWriteNames(handle, len(self.fio_lines), self.fio_lines, values)

    def clear(self):
        self.send_code(0)
        self.last_clear_time = time.perf_counter()

    def wait_for_low_gap(self, min_gap_s: float):
        if min_gap_s <= 0 or self.last_clear_time is None:
            return

        elapsed_s = time.perf_counter() - self.last_clear_time
        remaining_s = min_gap_s - elapsed_s
        if remaining_s > 0:
            time.sleep(remaining_s)

    def pulse_event(self, code: int, pulse_width_s=None, min_gap_s=None):
        """Set code, hold briefly, then clear to zero."""
        pulse_width_s = self.pulse_width_s if pulse_width_s is None else pulse_width_s
        min_gap_s = self.min_gap_s if min_gap_s is None else min_gap_s

        if not 1 <= int(code) <= 255:
            raise ValueError(f"Trigger code must be in [1, 255], got {code}")

        self.require_open()
        with self.io_lock:
            self.wait_for_low_gap(min_gap_s)
            self.send_code(code)
            if pulse_width_s > 0:
                time.sleep(pulse_width_s)
            self.clear()

    def close(self):
        if self.handle is not None and self.ljm is not None:
            try:
                self.clear()
            except Exception as exc:
                logging.error("LabJack trigger clear error during close: %s", exc)
            try:
                self.ljm.close(self.handle)
            except Exception as exc:
                logging.error("LabJack trigger close error: %s", exc)

        self.handle = None
        self.ljm = None
        self.last_clear_time = None


class SimulatedLabjackFio8BitTrigger(BaseTrigger):
    """Drop-in trigger object for testing without LabJack hardware."""

    def open(self):
        logging.info("Simulated LabJack trigger opened")

    def require_open(self):
        return

    def send_code(self, code):
        if not 0 <= int(code) <= 255:
            raise ValueError(f"Trigger code must be in [0, 255], got {code}")
        logging.info("Simulated trigger code=%s t=%s", code, time.perf_counter())

    def pulse_event(self, code: int, pulse_width_s=None, min_gap_s=None):
        pulse_width_s = self.pulse_width_s if pulse_width_s is None else pulse_width_s
        min_gap_s = self.min_gap_s if min_gap_s is None else min_gap_s

        if not 1 <= int(code) <= 255:
            raise ValueError(f"Trigger code must be in [1, 255], got {code}")

        with self.io_lock:
            if min_gap_s > 0 and self.last_clear_time is not None:
                elapsed_s = time.perf_counter() - self.last_clear_time
                remaining_s = min_gap_s - elapsed_s
                if remaining_s > 0:
                    time.sleep(remaining_s)

            logging.info("Simulated trigger code=%s", code)
            if pulse_width_s > 0:
                time.sleep(pulse_width_s)
            self.last_clear_time = time.perf_counter()


def log_trigger_settings(
    pulse_width_s=trigger_pulse_width_s,
    min_gap_s=trigger_min_gap_s,
    offset_code=stimulus_offset_code,
    frame_code=frame_marker_code,
    frame_interval=frame_marker_interval,
):
        logging.info(
        "Trigger settings: pulse_width_s=%s min_gap_s=%s offset_code=%s "
        "frame_code=%s frame_interval=%s",
        pulse_width_s,
        min_gap_s,
        offset_code,
        frame_code,
        frame_interval,
    )


def send_stimulus_offset(
    dispatcher: TriggerDispatcher,
    *,
    frame_num=None,
    trial_num=None,
    code: int = stimulus_offset_code,
):
    dispatcher.submit(
        code,
        label="stimulus_offset",
        frame_num=frame_num,
        trial_num=trial_num,
    )


def send_trial_start_codes(
    dispatcher: TriggerDispatcher,
    trial,
    *,
    frame_num=None,
    trial_num=None,
):
    """Send the condition-specific trial-start marker."""
    dispatcher.submit(
        trial["condition_id"],
        label="trial_start",
        frame_num=frame_num,
        trial_num=trial_num,
    )


def maybe_send_frame_marker(
    dispatcher: TriggerDispatcher,
    frame_num: int,
    *,
    trial_num=None,
    interval: int = frame_marker_interval,
    code: int = frame_marker_code,
) -> bool:
    """Pulse code on every Nth frame. Returns True if sent."""
    if interval <= 0:
        return False
    if frame_num > 0 and frame_num % interval == 0:
        dispatcher.submit(
            code,
            label="frame_marker",
            frame_num=frame_num,
            trial_num=trial_num,
        )
        return True
    return False
