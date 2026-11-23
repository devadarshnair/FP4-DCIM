"""e01 - integerization cost per element format (paper Fig. F2, analytic half; contribution C-1).

Output: results/e01_integerization/<run>/integerize.csv + metrics.json
Run:    python experiments/e01_integerization/run.py [--paper]
"""

from __future__ import annotations

import argparse
import math

from cimfp4.formats.integerize import FORMATS, integerize
from cimfp4.utils.runlog import start_run

BLOCKS = {"mx": 32, "nv": 16}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    cfg = {"formats": list(FORMATS), "blocks": BLOCKS}
    with start_run("e01_integerization", cfg, paper=args.paper) as run:
        rows = []
        for name in FORMATS:
            for tag, bs in BLOCKS.items():
                r = integerize(name, bs)
                rows.append({
                    "format": name, "block": bs, "block_kind": tag, "code_bits": r.width,
                    "lsb_log2": int(math.log2(r.lsb)), "max_int": r.max_int,
                    "int_bits_signed": r.int_bits_signed, "extra_bits": r.extra_bits,
                    "product_bits": r.product_bits, "block_sum_unsigned": r.block_sum_unsigned,
                    "block_sum_signed": r.block_sum_signed,
                })
        run.save_csv("integerize.csv", rows)
        get = {(r["format"], r["block"]): r for r in rows}
        run.log_metrics({
            "e2m1_extra_bits": get[("E2M1", 32)]["extra_bits"],
            "e2m3_extra_bits": get[("E2M3", 32)]["extra_bits"],
            "e3m2_extra_bits": get[("E3M2", 32)]["extra_bits"],
            "e4m3_extra_bits": get[("E4M3", 32)]["extra_bits"],
            "e5m2_extra_bits": get[("E5M2", 32)]["extra_bits"],
            "mx_block_sum_signed_bits": get[("E2M1", 32)]["block_sum_signed"],
            "nv_block_sum_signed_bits": get[("E2M1", 16)]["block_sum_signed"],
        })
        for r in rows:
            if r["block"] == 32:
                print(f"{r['format']:5s} code {r['code_bits']:2d}b -> int {r['int_bits_signed']:3d}b "
                      f"(+{r['extra_bits']}), exact 32-sum {r['block_sum_signed']}b")
    print("results in", run.dir)


if __name__ == "__main__":
    main()
