import codey
import rocky
import time
import event
import gc
from array import array

R = 20
IN, H1, H2, OUT = 32, 64, 64, 4


def now():
    try:
        return time.ticks_ms() / 1000
    except AttributeError:
        return time.time()


def free():
    try:
        return gc.mem_free()
    except AttributeError:
        return -1


def read_sensors():
    return (codey.light_sensor.get_value(),
            rocky.color_ir_sensor.get_light_strength(),
            rocky.color_ir_sensor.get_reflected_infrared(),
            rocky.color_ir_sensor.is_obstacle_ahead())


def layer(x, w, b, n_in, n_out, act):
    y = [0.0] * n_out
    for j in range(n_out):
        s = b[j]
        base = j * n_in
        for i in range(n_in):
            s += w[base + i] * x[i]
        y[j] = (s if s < 1.0 else 1.0) if act else s
        if act and y[j] < -1.0:
            y[j] = -1.0
    return y


@event.button_a_pressed
def a_cb():
    gc.collect()
    free0 = free()
    w1 = array('f', [0.01] * (IN * H1))
    b1 = array('f', [0.0] * H1)
    w2 = array('f', [0.01] * (H1 * H2))
    b2 = array('f', [0.0] * H2)
    w3 = array('f', [0.01] * (H2 * OUT))
    b3 = array('f', [0.0] * OUT)
    free1 = free()
    x = [0.1] * IN

    t0 = now()
    for r in range(R):
        read_sensors()
    t_sens = (now() - t0) / R

    t0 = now()
    for r in range(R):
        h = layer(x, w1, b1, IN, H1, True)
        h = layer(h, w2, b2, H1, H2, True)
        layer(h, w3, b3, H2, OUT, False)
    t_net = (now() - t0) / R

    print("LOOP,sensors_s," + str(t_sens) + ",net_s," + str(t_net)
          + ",mem_free_before," + str(free0) + ",mem_free_after_weights," + str(free1))
    codey.display.show(int(t_net * 1000))
