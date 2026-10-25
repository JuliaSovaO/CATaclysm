import codey
import rocky
import time
import event

N = 20
DT = 0.1
idx = 0


def stats(v):
    n = len(v)
    m = sum(v) / n
    s = (sum((x - m) * (x - m) for x in v) / n) ** 0.5
    return m, min(v), max(v), s


def measure(n):
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
    out = [idx, n]
    for series in (head, front, ir, ref):
        m, lo, hi, s = stats(series)
        out += [round(m, 2), lo, hi, round(s, 2)]
    out.append(round(obstacle / n, 2))
    print(",".join([str(x) for x in out]))
    print("RAW_head," + str(idx) + "," + ",".join([str(x) for x in head]))
    print("RAW_front," + str(idx) + "," + ",".join([str(x) for x in front]))
    return out


@event.start
def start_cb():
    print("idx,n,head_mean,head_min,head_max,head_std,front_mean,front_min,front_max,front_std,"
          "ir_mean,ir_min,ir_max,ir_std,ref_mean,ref_min,ref_max,ref_std,obstacle_frac")
    codey.display.show(idx)


@event.button_a_pressed
def a_cb():
    out = measure(N)
    codey.display.show(int(out[2]))


@event.button_b_pressed
def b_cb():
    global idx
    idx += 1
    codey.display.show(idx)


@event.button_c_pressed
def c_cb():
    out = measure(100)
    codey.display.show(int(out[2]))
