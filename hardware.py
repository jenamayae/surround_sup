"""LabJack trigger helpers for 8-bit event codes on FIO0-FIO7, eye tracker, and US playback.
"""

from __future__ import annotations

import csv
import logging
import queue
import threading
import time

import numpy as np

from config import (
    audio_device_name,
    audio_sample_rate_hz,
    eyetracker_gaze_output_hz,
    eyetracker_serial,
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
    us_duration_s,
    us_noise_seed,
    us_volume,
)

# gaze x/y: Tobii display-area coordinates (0-1, origin top-left)
# gaze origin: eye position in Tobii user coordinates (mm, origin at tracker center, z toward the participant)
eyetracking_sample_columns = [
    "device_time_stamp_us",
    "system_time_stamp_us",
    "left_gaze_x_display",
    "left_gaze_y_display",
    "left_gaze_validity",
    "right_gaze_x_display",
    "right_gaze_y_display",
    "right_gaze_validity",
    "left_pupil_diameter_mm",
    "left_pupil_validity",
    "right_pupil_diameter_mm",
    "right_pupil_validity",
    "left_gaze_origin_x_mm",
    "left_gaze_origin_y_mm",
    "left_gaze_origin_z_mm",
    "left_gaze_origin_validity",
    "right_gaze_origin_x_mm",
    "right_gaze_origin_y_mm",
    "right_gaze_origin_z_mm",
    "right_gaze_origin_validity",
]
eyetracking_event_columns = [
    "system_time_stamp_us",
    "label",
    "code",
    "trial_num",
    "frame_num",
]


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


class TobiiEyeTracker:
    """Record Tobii gaze, pupil and eye-position samples plus flip-locked event times."""

    def __init__(
        self,
        serial=eyetracker_serial,
        expected_gaze_output_hz=eyetracker_gaze_output_hz,
    ):
        self.serial = serial
        self.expected_gaze_output_hz = expected_gaze_output_hz
        self.tr = None
        self.eyetracker = None
        self.info = None
        self.buffer_lock = threading.Lock()
        self.samples = []
        self.events = []

    def open(self):
        """Find the tracker by serial, check its frequency, and start gaze recording."""
        if self.eyetracker is not None:
            return

        try:
            import tobii_research as tr
        except ImportError as exc:
            raise RuntimeError("Could not import tobii_research. Install with: pip install tobii-research") from exc

        self.tr = tr
        matches = [et for et in tr.find_all_eyetrackers() if et.serial_number == self.serial]
        if not matches:
            raise RuntimeError(f"Eye tracker with serial {self.serial} not found.")
        eyetracker = matches[0]

        gaze_output_hz = eyetracker.get_gaze_output_frequency()
        if abs(gaze_output_hz - self.expected_gaze_output_hz) > 0.5:
            raise RuntimeError(
                f"Eye tracker frequency is {gaze_output_hz} Hz but eyetracker_gaze_output_hz is "
                f"{self.expected_gaze_output_hz} Hz. Set Frequency in Eye Tracker Manager."
            )

        self.info = {
            "model": eyetracker.model,
            "serial_number": eyetracker.serial_number,
            "firmware_version": eyetracker.firmware_version,
            "device_name": eyetracker.device_name,
            "address": eyetracker.address,
            "gaze_output_hz": gaze_output_hz,
            "sdk_version": str(getattr(tr, "__version__", "unknown")),
        }
        self.eyetracker = eyetracker
        eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.on_gaze_sample, as_dictionary=True)
        logging.info("Tobii eye tracker opened: %s", self.info)

    def on_gaze_sample(self, sample):
        # runs on the Tobii SDK thread
        row = gaze_sample_to_row(sample)
        with self.buffer_lock:
            self.samples.append(row)

    def current_time_us(self):
        # same clock as each sample's system_time_stamp
        return self.tr.get_system_time_stamp()

    def mark_event(self, label, code, trial_num=None, frame_num=None):
        """Record the current eye-tracker clock time; call via win.callOnFlip."""
        row = [self.current_time_us(), label, code, trial_num, frame_num]
        with self.buffer_lock:
            self.events.append(row)

    def write_buffered_data(self, samples_path, events_path):
        """Append buffered samples and events to their TSV files and clear the buffers."""
        with self.buffer_lock:
            samples, self.samples = self.samples, []
            events, self.events = self.events, []
        append_rows_tsv(samples_path, eyetracking_sample_columns, samples)
        append_rows_tsv(events_path, eyetracking_event_columns, events)

    def close(self):
        if self.eyetracker is not None:
            try:
                self.eyetracker.unsubscribe_from(self.tr.EYETRACKER_GAZE_DATA, self.on_gaze_sample)
            except Exception as exc:
                logging.error("Eye tracker unsubscribe error during close: %s", exc)
        self.eyetracker = None


class SimulatedTobiiEyeTracker(TobiiEyeTracker):
    """Drop-in eye tracker for testing without hardware; records events, no samples."""

    def open(self):
        self.info = {"model": "simulated", "gaze_output_hz": self.expected_gaze_output_hz}
        logging.info("Simulated eye tracker opened")

    def current_time_us(self):
        return time.perf_counter_ns() // 1000

    def close(self):
        return


class UnconditionedStimulus:
    """White-noise burst US with abrupt onset, played through PsychoPy's PTB backend."""

    def __init__(
        self,
        duration_s=us_duration_s,
        sample_rate_hz=audio_sample_rate_hz,
        volume=us_volume,
        device_name=audio_device_name,
        noise_seed=us_noise_seed,
    ):
        self.duration_s = duration_s
        self.sample_rate_hz = sample_rate_hz
        self.volume = volume
        self.device_name = device_name
        self.noise_seed = noise_seed
        self.sound = None
        self.info = None

    def open(self):
        if self.sound is not None:
            return

        # prefs only take effect if psychopy.sound has not been imported yet
        from psychopy import prefs
        prefs.hardware["audioLib"] = ["ptb"]
        prefs.hardware["audioLatencyMode"] = 3  # aggressive low latency
        if self.device_name:
            prefs.hardware["audioDevice"] = self.device_name
        from psychopy import sound

        if sound.audioLib != "ptb":
            raise RuntimeError(f"US needs the PTB audio backend for flip-scheduled onsets, got {sound.audioLib}.")

        self.sound = sound.Sound(
            value=white_noise_buffer(self.duration_s, self.sample_rate_hz, self.noise_seed),
            sampleRate=self.sample_rate_hz,
            stereo=True,
            hamming=False,  # PsychoPy ramps on/off by default; the US starts abruptly
            volume=self.volume,
        )
        self.info = {
            "audio_lib": sound.audioLib,
            "audio_device_name": self.device_name or "system default",
            "audio_sample_rate_hz": self.sample_rate_hz,
            "audio_latency_mode": 3,
        }
        logging.info("US opened: %s", self.info)

    def play_on_next_flip(self, win):
        """Schedule the burst for the next flip; call just before win.flip()."""
        self.sound.play(when=win.getFutureFlipTime(clock="ptb"))

    def close(self):
        if self.sound is not None:
            self.sound.stop()
        self.sound = None


class SimulatedUnconditionedStimulus(UnconditionedStimulus):
    """Drop-in US for testing without audio hardware; logs instead of playing."""

    def open(self):
        self.info = {"audio_lib": "simulated", "audio_sample_rate_hz": self.sample_rate_hz}
        logging.info("Simulated US opened")

    def play_on_next_flip(self, win):
        logging.info("Simulated US onset scheduled t=%s", time.perf_counter())

    def close(self):
        return


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
        trial["event_code"],
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


def gaze_sample_to_row(sample):
    """Flatten one tobii_research gaze-data dictionary into eyetracking_sample_columns order."""
    row = [sample["device_time_stamp"], sample["system_time_stamp"]]
    for eye in ("left", "right"):
        row.extend(sample[f"{eye}_gaze_point_on_display_area"])
        row.append(sample[f"{eye}_gaze_point_validity"])
    for eye in ("left", "right"):
        row.append(sample[f"{eye}_pupil_diameter"])
        row.append(sample[f"{eye}_pupil_validity"])
    for eye in ("left", "right"):
        row.extend(sample[f"{eye}_gaze_origin_in_user_coordinate_system"])
        row.append(sample[f"{eye}_gaze_origin_validity"])
    return row


def white_noise_buffer(duration_s, sample_rate_hz, seed):
    """Uniform white noise in [-1, 1], identical on both channels, with no onset or offset ramp."""
    rng = np.random.default_rng(seed)
    samples = rng.uniform(-1.0, 1.0, round(duration_s * sample_rate_hz))
    return np.column_stack([samples, samples])


def append_rows_tsv(path, columns, rows):
    """Append rows to a TSV file, writing the header if the file is new."""
    write_header = not path.exists()
    with open(path, "a", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        if write_header:
            writer.writerow(columns)
        writer.writerows(["N/A" if value is None else value for value in row] for row in rows)
