"""
Run a resumable DeepSeek job only in DeepSeek's off-peak hours, pausing it before peak hours begin.

Peak (https://api-docs.deepseek.com/zh-cn/quick_start/pricing/, checked 2026-11-02): Beijing time,
Monday-Friday 09:00-12:00 and 14:00-18:00, excluding Chinese public holidays; every other hour is off-peak
at half price. Holidays are treated as peak here (conservative).

Usage: python offpeak_runner.py --out <jsonl the job appends to> -- <command ...>
The job must resume from its output file (bench/run.py and agentenv/run.py do). When it is stopped, a
partially written last line is cut off so the next resumed record starts on a fresh line.
"""
import argparse, datetime, os, subprocess, sys, time

BJ = datetime.timezone(datetime.timedelta(hours=8))
PEAK = [(9, 12), (14, 18)]
MARGIN = datetime.timedelta(minutes=5)       # stop this long before a peak window opens

def peak_windows(day):
    if day.weekday() >= 5: return []
    return [(datetime.datetime.combine(day, datetime.time(a), BJ), datetime.datetime.combine(day, datetime.time(b), BJ))
            for a, b in PEAK]

def windows_around(now):
    return [w for d in (-1, 0, 1, 2, 3) for w in peak_windows((now + datetime.timedelta(days=d)).date())]

def in_peak(now):
    return any(a - MARGIN <= now < b for a, b in windows_around(now))

def next_peak_start(now):
    return min(a - MARGIN for a, b in windows_around(now) if a - MARGIN > now)

def peak_end(now):
    return max(b for a, b in windows_around(now) if a - MARGIN <= now < b)

def trim_partial(path):
    if not os.path.exists(path): return
    with open(path, "rb+") as f:
        data = f.read()
        if data and not data.endswith(b"\n"):
            f.truncate(data.rfind(b"\n") + 1)
            print(f"trimmed partial line from {path}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    while True:
        now = datetime.datetime.now(BJ)
        if in_peak(now):
            end = peak_end(now)
            print(f"{now:%m-%d %H:%M} BJ peak: waiting until {end:%m-%d %H:%M}", flush=True)
            time.sleep((end - now).total_seconds() + 30)
            continue
        stop_at = next_peak_start(now)
        print(f"{now:%m-%d %H:%M} BJ off-peak: running until done or {stop_at:%m-%d %H:%M}", flush=True)
        p = subprocess.Popen(cmd)
        while p.poll() is None and datetime.datetime.now(BJ) < stop_at:
            time.sleep(20)
        if p.poll() is None:
            p.kill(); p.wait()
            trim_partial(a.out)
            print(f"{datetime.datetime.now(BJ):%m-%d %H:%M} BJ paused for peak", flush=True)
            continue
        trim_partial(a.out)
        print(f"job exited {p.returncode}", flush=True)
        sys.exit(p.returncode)

if __name__ == "__main__":
    main()
