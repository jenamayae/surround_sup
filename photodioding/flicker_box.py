from psychopy import core, event, visual

from config import (
    fullscreen,
    frames_per_reversal,
    monitor_name,
    monitor_pixels,
    monitor_width_cm,
    refresh_hz,
    screen_index,
    viewing_distance_cm,
)
from stimuli import make_window

# Reversal rate (black<->white switches per second), same convention as config.py.
box_flicker_hz = 15
box_size_cm = 5.0


def main():
    win = make_window(
        size=monitor_pixels,
        fullscr=fullscreen,
        screen=screen_index,
        monitor_name=monitor_name,
        monitor_width_cm=monitor_width_cm,
        viewing_distance_cm=viewing_distance_cm,
    )
    box = visual.Rect(
        win,
        width=box_size_cm,
        height=box_size_cm,
        units="cm",
        pos=(0, 0),
        fillColor="white",
        lineColor=None,
    )
    frames_per_state = frames_per_reversal(refresh_hz, box_flicker_hz)

    frame_num = 0
    while not event.getKeys(keyList=["escape"]):
        is_white = (frame_num // frames_per_state) % 2 == 0
        box.fillColor = "white" if is_white else "black"
        box.draw()
        win.flip()
        frame_num += 1

    win.close()
    core.quit()


if __name__ == "__main__":
    main()
