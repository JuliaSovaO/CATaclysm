import codey
import rocky
import time
import event
import gc
from array import array

MODES = 6
N = 20
DT = 0.1
mode = 0
idx = 0
test = 0

TESTS = [
    ("fwd", 50, 0.2), ("fwd", 50, 0.5), ("fwd", 50, 1.0), ("fwd", 50, 2.0),
    ("fwd", 30, 2.0), ("fwd", 70, 2.0),
    ("left", 50, 0.2), ("left", 50, 0.5), ("left", 50, 1.0),
    ("right", 50, 0.2), ("right", 50, 0.5), ("right", 50, 1.0),
    ("left_deg", 40, 90), ("right_deg", 40, 90),
]


def stats(v):
    n = len(v)
    m = sum(v) / n
    s = (sum((x - m) * (x - m) for x in v) / n) ** 0.5
    return m, min(v), max(v), s


def sensors(n):
    head = []
    front = []
    ir = []
    ref = []
    obstacle = 0
    for i in range(n):
        head.append(codey.light_sensor.get_value())
        front.append(rocky.color_ir_sensor.get_light_strength())
        ir.append(rocky.color_ir_sensor.get_reflected_infrared())
        ref.append(rocky.color_ir_sensor.get_reflected_light())
        if rocky.color_ir_sensor.is_obstacle_ahead():
            obstacle += 1
        time.sleep(DT)
    out = ["SENS", idx, n]
    for series in (head, front, ir, ref):
        m, lo, hi, s = stats(series)
        out += [round(m, 2), lo, hi, round(s, 2)]
    out.append(round(obstacle / n, 2))
    print(",".join([str(x) for x in out]))
    print("RAW_head," + str(idx) + "," + ",".join([str(x) for x in head]))
    print("RAW_front," + str(idx) + "," + ",".join([str(x) for x in front]))
    codey.display.show(int(out[3]))


def move():
    kind, speed, val = TESTS[test]
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
    print("TEST," + str(test) + "," + kind + "," + str(speed) + "," + str(val) + ","
          + str(codey.battery.get_percentage()))


def ir_send():
    for i in range(20):
        codey.ir.send("ping")
        time.sleep(0.3)
    codey.ir.send("end")
    print("SENT,20")
    codey.display.show(20)


def ir_recv():
    got = 0
    for i in range(300):
        m = codey.ir.receive()
        if m is None:
            time.sleep(0.05)
            continue
        m = str(m).strip()
        if m == "end":
            break
        if m == "ping":
            got += 1
    print("RECV," + str(got))
    codey.display.show(got)


def free():
    try:
        return gc.mem_free()
    except AttributeError:
        return -1


def now():
    try:
        return time.ticks_ms() / 1000
    except AttributeError:
        return time.time()


def layer(x, w, b, n_in, n_out, act):
    y = [0.0] * n_out
    for j in range(n_out):
        s = b[j]
        base = j * n_in
        for i in range(n_in):
            s += w[base + i] * x[i]
        if act:
            s = 1.0 if s > 1.0 else (-1.0 if s < -1.0 else s)
        y[j] = s
    return y


def loop_timing():
    R = 20
    gc.collect()
    free0 = free()
    w1 = array('f', [0.01] * (32 * 64))
    b1 = array('f', [0.0] * 64)
    w2 = array('f', [0.01] * (64 * 64))
    b2 = array('f', [0.0] * 64)
    w3 = array('f', [0.01] * (64 * 4))
    b3 = array('f', [0.0] * 4)
    free1 = free()
    x = [0.1] * 32
    t0 = now()
    for r in range(R):
        codey.light_sensor.get_value()
        rocky.color_ir_sensor.get_light_strength()
        rocky.color_ir_sensor.get_reflected_infrared()
        rocky.color_ir_sensor.is_obstacle_ahead()
    t_sens = (now() - t0) / R
    t0 = now()
    for r in range(R):
        h = layer(x, w1, b1, 32, 64, True)
        h = layer(h, w2, b2, 64, 64, True)
        layer(h, w3, b3, 64, 4, False)
    t_net = (now() - t0) / R
    print("LOOP,sensors_s," + str(t_sens) + ",net_s," + str(t_net)
          + ",mem_free_before," + str(free0) + ",mem_free_after_weights," + str(free1))
    codey.display.show(int(t_net * 1000))


@event.start
def start_cb():
    print("SENS,idx,n,head_mean,head_min,head_max,head_std,front_mean,front_min,front_max,front_std,"
          "ir_mean,ir_min,ir_max,ir_std,ref_mean,ref_min,ref_max,ref_std,obstacle_frac")
    codey.display.show(mode)


@event.button_c_pressed
def c_cb():
    global mode
    mode = (mode + 1) % MODES
    codey.display.show(mode)


@event.button_b_pressed
def b_cb():
    global idx, test
    if mode == 2:
        test = (test + 1) % len(TESTS)
        codey.display.show(test)
    else:
        idx += 1
        codey.display.show(idx)


@event.button_a_pressed
def a_cb():
    if mode == 0:
        sensors(N)
    elif mode == 1:
        sensors(100)
    elif mode == 2:
        move()
    elif mode == 3:
        ir_send()
    elif mode == 4:
        ir_recv()
    elif mode == 5:
        loop_timing()
