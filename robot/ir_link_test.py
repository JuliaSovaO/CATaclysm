import codey
import time
import event

COUNT = 20


@event.button_a_pressed
def send_cb():
    for i in range(COUNT):
        codey.ir.send("ping")
        time.sleep(0.3)
    codey.ir.send("end")
    print("SENT," + str(COUNT))
    codey.display.show(COUNT)


@event.button_b_pressed
def recv_cb():
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
