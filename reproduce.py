"""Rebuild the paper's tables, number macros and figures from the stored result files and logs. No model call, no network.

    python reproduce.py            # all generators
    python reproduce.py --quick    # skip the generators that re-solve the exact oracle (minutes)

Run from the root of the unpacked archive (the folder holding raea4/ and paper/). Each paper/scripts/make_*.py is
run as a separate process. Every file under paper/tables is hashed before and after: a table that the generators
rewrite must come out identical to the packaged one. The only lines left out of the comparison are run-record macros
(names ending in Sha, ShaShort, Date, Rows or ApiErrors): they record hashes, modification dates and row counts of the
authors' original log files, which the anonymised copies in this archive cannot reproduce (see README, "What was removed").
Some of them are printed in the manuscript (e.g. \beRows, \agFlashApiErrors, \agTerraUsageLimitRows) and in the resource
tables, so after the generators have run each such line is put back to its packaged value: the tables are left exactly as
packaged, and these few counts and dates are records of the original runs, not recomputed here. Figures are PDF files with embedded creation metadata, so
they are checked for being written, not for identical bytes. Exit status 0 means every generator ran and no table changed.
"""
import hashlib, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(ROOT, "paper", "scripts")
TABLES = os.path.join(ROOT, "paper", "tables")
FIGS = os.path.join(ROOT, "paper", "figs")
SLOW = {"make_agent_llm_tables.py"}               # re-solves the exact oracle on the stored trajectories (minutes)
MAIN = ["make_tables.py", "make_agent_llm_tables.py", "make_defaults_table.py", "make_anatomy_table.py",
        "make_anatomy_confirm_table.py", "make_k2r_strict_outputs.py", "make_example_outputs.py", "make_x_outputs.py",
        "make_w_outputs.py", "make_d2_outputs.py", "make_d3_outputs.py", "make_v_outputs.py", "make_s_outputs.py",
        "make_m_outputs.py", "make_review_outputs.py", "make_overview_fig.py", "make_s_fig.py", "make_u_outputs.py", "make_hd_outputs.py", "make_be_outputs.py"]   # main-text numbers, tables and figures first


RECORD = re.compile(rb"^\\providecommand\{\\[A-Za-z]*(?:Sha|ShaShort|Date|Rows|ApiErrors)\}")
IGNORED = [0]


def digest(p, records):
    raw = open(p, "rb").read().replace(b"\r\n", b"\n")   # CRLF/LF-independent
    if records and p.endswith(".tex"):
        lines = raw.split(b"\n"); kept = [l for l in lines if not RECORD.match(l)]
        IGNORED[0] += len(lines) - len(kept); raw = b"\n".join(kept)
    return hashlib.sha256(raw).hexdigest()


def record_lines(folder):
    """{(file, macro name): packaged line} for every run-record macro under folder."""
    out = {}
    for d, _, files in os.walk(folder):
        for f in files:
            if not f.endswith(".tex"): continue
            p = os.path.join(d, f)
            for l in open(p, "rb").read().replace(b"\r\n", b"\n").split(b"\n"):
                if RECORD.match(l): out[(p, l.split(b"}", 1)[0])] = l
    return out


def restore_records(saved):
    """Put every run-record macro line back to its packaged value (the generators recompute them from the anonymised logs)."""
    files = {}
    for (p, key), line in saved.items(): files.setdefault(p, {})[key] = line
    n = 0
    for p, keep in files.items():
        if not os.path.exists(p): continue
        raw = open(p, "rb").read(); crlf = b"\r\n" in raw
        lines = raw.replace(b"\r\n", b"\n").split(b"\n")
        new = [keep.get(l.split(b"}", 1)[0], l) if RECORD.match(l) else l for l in lines]
        if new != lines:
            n += sum(a != b for a, b in zip(new, lines))
            out = b"\n".join(new)
            open(p, "wb").write(out.replace(b"\n", b"\r\n") if crlf else out)
    return n


def snap(folder, records=False):
    out = {}
    for d, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(d, f)
            out[os.path.relpath(p, folder)] = (digest(p, records), os.path.getmtime(p))
    return out


def main():
    quick = "--quick" in sys.argv
    for need in ("numpy", "matplotlib"):
        try: __import__(need)
        except ImportError: sys.exit(f"missing package: {need} (see requirements.txt)")
    os.makedirs(FIGS, exist_ok=True)
    names = sorted(f for f in os.listdir(SCRIPTS) if f.startswith("make_") and f.endswith(".py"))
    names = [f for f in MAIN if f in names] + [f for f in names if f not in MAIN]
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    t0, f0 = snap(TABLES, True), snap(FIGS)
    saved = record_lines(TABLES)
    failed = []
    for f in names:
        if quick and f in SLOW:
            print(f"skip   {f} (--quick)"); continue
        t = time.time()
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, f)], cwd=ROOT, env=env, capture_output=True, text=True)
        print(f"{'ok    ' if r.returncode == 0 else 'FAILED'} {f}  ({time.time() - t:.0f} s)", flush=True)
        if r.returncode != 0:
            failed.append(f); print("    " + (r.stderr or r.stdout).strip().replace("\n", "\n    ")[-1500:])
    restored = restore_records(saved)
    IGNORED[0] = 0
    t1, f1 = snap(TABLES, True), snap(FIGS)
    rewritten = [k for k in t1 if k in t0 and t1[k][1] != t0[k][1]]
    changed = sorted(k for k in t1 if k in t0 and t1[k][0] != t0[k][0])
    new = sorted(k for k in t1 if k not in t0)
    figs = sorted(k for k in f1 if k not in f0 or f1[k][1] != f0[k][1])
    print(f"\ntables: {len(t1)} files, {len(rewritten)} rewritten, {len(changed)} changed, {len(new)} new; {IGNORED[0]} run-record lines not compared ({restored} put back to their packaged values)")
    for k in changed: print("  CHANGED", k)
    for k in new: print("  NEW", k)
    print(f"figures written: {len(figs)}: {', '.join(figs)}")
    if failed: print("failed generators:", ", ".join(failed))
    sys.exit(1 if failed or changed else 0)


if __name__ == "__main__":
    main()
