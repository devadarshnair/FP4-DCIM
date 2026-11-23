"""e20 CACTI cross-check [EST]: CACTI 7 (HewlettPackard/cacti, commit given by --src) model of a 1 KB,
32-bit word, 1RW + 1R RAM at 130 nm (CACTI interpolates its 180 nm and 90 nm technology files), for the
ITRS high-performance device flavour (the only one CACTI's 180 nm file defines). CACTI uses its own interpolated Vdd; energies are also reported rescaled to
1.8 V by (1.8 / Vdd_cacti)^2. This is an independent analytical model, not a characterization of the
OpenRAM circuit: it bounds what a conventional compiled SRAM of the same size and node would cost.

Run: python scripts/safe_run.py --max-rss-mb 1500 -- python experiments/e20_sram_energy/cacti.py --src ~/tmp/cacti_src
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

CFG = REPO_ROOT / "experiments/e20_sram_energy/cacti_1kb_1rw1r_130nm.cfg"


def num(pat: str, txt: str) -> float | None:
    m = re.search(pat, txt)
    return float(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    args = ap.parse_args()
    src = args.src.expanduser()
    sha = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cfg = {"cacti_src": str(src), "cacti_commit": sha, "cfg": str(CFG.relative_to(REPO_ROOT)),
           "flavours": ["itrs-hp"],
           "why_hp_only": "tech_params/180nm.dat defines only the hp flavour (lstp/lop rows are 0), so 130 nm "
                          "interpolation is meaningful for itrs-hp only",
           "vdd_rule": "parameter.cc: alpha = (130-90)/(180-90); Vdd = alpha*1.5 + (1-alpha)*1.2 = 1.333 V"}
    if not (src / "cacti").exists():
        subprocess.run(["make", "-j1"], cwd=src, check=True, capture_output=True)
    base = CFG.read_text()
    with start_run("e20_cacti", cfg) as run:
        rows, m = [], {}
        for fl in cfg["flavours"]:
            t = re.sub(r'^-Data array cell type - "[^"]+"', f'-Data array cell type - "{fl}"', base, flags=re.M)
            t = re.sub(r'^-Data array peripheral type - "[^"]+"', f'-Data array peripheral type - "{fl}"', t, flags=re.M)
            p = run.dir / f"cacti_{fl}.cfg"
            p.write_text(t)
            r = subprocess.run(["./cacti", "-infile", str(p)], cwd=src, capture_output=True, text=True)
            out = r.stdout + r.stderr
            (run.dir / f"cacti_{fl}.out").write_text(out)
            rd = num(r"Total dynamic read energy per access \(nJ\):\s*([\d.eE+-]+)", out)
            wr = num(r"Total dynamic write energy per access \(nJ\):\s*([\d.eE+-]+)", out)
            lk = num(r"Total leakage power of a bank \(mW\):\s*([\d.eE+-]+)", out)
            alpha = (130 - 90) / (180 - 90)
            vdd = num(r"Vdd\s*[:=]\s*([\d.]+)", out) or alpha * 1.5 + (1 - alpha) * 1.2
            row = {"flavour": fl, "read_pj": rd * 1e3 if rd else None, "write_pj": wr * 1e3 if wr else None,
                   "leak_uw": lk * 1e3 if lk else None, "vdd_cacti": vdd}
            if rd and vdd:
                row["read_pj_at_1v8"] = row["read_pj"] * (1.8 / vdd) ** 2
                row["write_pj_at_1v8"] = row["write_pj"] * (1.8 / vdd) ** 2
            rows.append(row)
            m[fl] = row
        run.save_csv("cacti.csv", rows)
        run.log_metrics(m)
        print(rows)


if __name__ == "__main__":
    main()
