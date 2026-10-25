import codey
import rocky
import time
import event


TESTS = [
    ("fwd", 50, 0.2), ("fwd", 50, 0.5), ("fwd", 50, 1.0), ("fwd", 50, 2.0),
    ("fwd", 30, 2.0), ("fwd", 70, 2.0),
    ("left", 50, 0.2), ("left", 50, 0.5), ("left", 50, 1.0),
    ("right", 50, 0.2), ("right", 50, 0.5), ("right", 50, 1.0),
    ("left_deg", 40, 90), ("right_deg", 40, 90),
]
cur = 0


def run(kind, speed, val):
    if kind == "fwd":
        rocky.forward(speed)
        time.sleep(val)
        rocky.stop()
    elif kind == "left":
        rocky.turn_left(speed)
        time.sleep(val)
        rocky.stop()
    elif kind == "right":
        rocky.turn_right(speed)
        time.sleep(val)
        rocky.stop()
    elif kind == "left_deg":
        rocky.turn_left_by_degree(val, speed)
    elif kind == "right_deg":
        rocky.turn_right_by_degree(val, speed)


@event.start
def start_cb():
    print("TEST,i,kind,speed,value,battery_pct")
    codey.display.show(cur)


@event.button_b_pressed
def b_cb():
    global cur
    cur = (cur + 1) % len(TESTS)
    codey.display.show(cur)


@event.button_a_pressed
def a_cb():
    kind, speed, val = TESTS[cur]
    run(kind, speed, val)
    print("TEST," + str(cur) + "," + kind + "," + str(speed) + "," + str(val) + ","
          + str(codey.battery.get_percentage()))
