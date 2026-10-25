import re
import sys

LABELS = r"(?:SENS|RAW_head|RAW_front|TEST|SENT|RECV|LOOP)"


def decode(text):
    bs = bytes(int(t, 16) for t in re.findall(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{2}(?![0-9A-Fa-f])", text))
    return bs.decode("latin-1").replace("\r", "")


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    show_all = "--all" in argv
    out = None
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
        args = [a for a in args if a != out]
    text = open(args[0], encoding="utf-8").read() if args else sys.stdin.read()
    lines = []
    for line in decode(text).split("\n"):
        m = re.search(LABELS + r",.*", line)
        if m and '"' not in m.group(0) and "str(" not in m.group(0):
            lines.append(m.group(0).strip())
        elif show_all and "str(" not in line:
            clean = "".join(ch for ch in line if ch.isprintable()).strip()
            if clean:
                lines.append(clean)
    result = "\n".join(lines)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(result + "\n")
        print("Збережено " + str(len(lines)) + " рядків у " + out)
    else:
        print(result)


if __name__ == "__main__":
    main(sys.argv[1:])
