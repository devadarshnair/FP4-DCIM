"""e37 - T6.3: MIDAS Fig. 13 / Fig. 22 steps and MXFP6 early stop against tolerances frozen in
a pre-registration note (committed before this script was run). CPU numpy only.

Run: python experiments/e37_midas_ablation/run.py
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path

import numpy as np
import yaml

from cimfp4.formats.mx_quant import E3M2_EMAX, floor_log2, rne_e3m2
from cimfp4.golden.l2_midas_like import FORMATS, MidasLikeMacro, block_integer_reference
from cimfp4.llm.capture import load_capture
from cimfp4.power.liberty import load_cells
from cimfp4.power.midas_model import MidasAssumptions, macro_record
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

CAPTURE = REPO_ROOT / "data/captures/qwen2.5-0.5b@060db6499f32/prefill"
PREREG = REPO_ROOT / "docs/spec/T6_3_prereg.md"
TOL = 0.25
P = "sky130_fd_sc_hd__"
ALPHA, ALPHA_MEPU, VDD = 0.3, 0.05, 1.8
N_LCA, ROWS = 512, 32


def band(target: float) -> tuple[float, float]:
    return target * (1 - TOL), target * (1 + TOL)


def verdict(name: str, model: float, target: float) -> dict:
    lo, hi = band(target)
    return {"check": name, "model": model, "target": target, "lo": lo, "hi": hi,
            "rel_err": model / target - 1, "verdict": "PASS" if lo <= model <= hi else "FAIL"}


def expo_logic(fmt: str, cycles_early: float, cells) -> dict:
    """Energy (fJ/op) and area (um2) of exponent-related logic for the three variants (prereg model)."""
    f = FORMATS[fmt]
    eb, mb = f["ebits"], f["mbits"]
    w = eb + 1
    top, bottom = 2 * ((1 << eb) - 1) + mb, 2
    cyc_full = top - bottom + 1
    fa, mux, and2, dff, ha = (cells[k] for k in ("fa_1", "mux2_1", "and2_1", "dfxtp_1", "ha_1"))
    e = lambda c: c.toggle_energy_fj(vdd=VDD)  # noqa: E731
    clk = dff.clock_energy_fj()
    # baseline, once per op
    base_lca_e = ALPHA * (eb * e(fa) + w * e(fa) + (mb + 1) * w * e(mux))
    tree_e = ALPHA * (N_LCA - 1) * (w * e(fa) + w * e(mux))
    base_lca_a = eb * fa.area + w * fa.area + (mb + 1) * w * mux.area
    tree_a = (N_LCA - 1) * (w * fa.area + w * mux.area)
    # TAOR per-cycle logic
    def taor_cycle_e() -> float:
        vrdu = N_LCA * (3 * e(fa) * ALPHA * 0.5 + 2 * e(and2) * ALPHA)
        lcg = ROWS * (4 * e(fa) * ALPHA + 4 * (clk + ALPHA * e(dff)))
        cnt = 4 * (clk + ALPHA * e(dff)) + 4 * ALPHA * e(ha)
        return vrdu + lcg + cnt
    taor_a = N_LCA * (3 * fa.area + 2 * and2.area) + ROWS * (4 * fa.area + 4 * dff.area) + 4 * dff.area + 4 * ha.area
    mepu_cycle_e = (N_LCA - 1) * e(and2) * ALPHA_MEPU
    mepu_a = (N_LCA - 1) * and2.area
    return {
        "cycles_full": cyc_full,
        "energy": {
            "baseline": N_LCA * base_lca_e + tree_e,
            "taor": cyc_full * taor_cycle_e() + tree_e,
            "taor_mepd": cycles_early * (taor_cycle_e() + mepu_cycle_e),
        },
        "area": {"baseline": N_LCA * base_lca_a + tree_a, "taor": taor_a + tree_a, "taor_mepd": taor_a + mepu_a},
    }


def reductions(x: dict) -> dict:
    return {
        "taor_power": 1 - x["energy"]["taor"] / x["energy"]["baseline"],
        "taor_area": 1 - x["area"]["taor"] / x["area"]["baseline"],
        "taor_mepd_power": 1 - x["energy"]["taor_mepd"] / x["energy"]["baseline"],
        "taor_mepd_area": 1 - x["area"]["taor_mepd"] / x["area"]["baseline"],
    }


def encode_e3m2(q: np.ndarray) -> np.ndarray:
    """E3M2 grid values (|q| <= 28) -> 6-bit codes (s,E3,m2)."""
    a = np.abs(q)
    e = floor_log2(a)
    sub = a < 0.25
    ebits = np.where(sub, 0, e + 3)
    m = np.where(sub, np.rint(a / 2.0**-4), np.rint((a / np.ldexp(1.0, np.where(sub, 0, e).astype(np.int32)) - 1) * 4))
    m = np.where(a == 0, 0, m).astype(np.int64)
    ebits = np.where(a == 0, 0, ebits).astype(np.int64)
    return (((q < 0).astype(np.int64) << 5) | (ebits << 2) | m).astype(np.int64)


def mxfp6_codes(x: np.ndarray) -> np.ndarray:
    xb = x.reshape(-1, 32)
    amax = np.abs(xb).max(1)
    xe = np.where(amax > 0, floor_log2(amax) - E3M2_EMAX, -127)
    return encode_e3m2(rne_e3m2(xb / np.ldexp(1.0, xe.astype(np.int32))[:, None]))


def proxy_latency(rng, kind: str, ops: int = 256) -> float:
    def draw(shape):
        return rng.standard_normal(shape) if kind == "gauss" else rng.standard_t(3, shape)
    cyc = []
    mac_e, mac_f = MidasLikeMacro("mxfp6"), MidasLikeMacro("mxfp6", early_stop=False)
    for _ in range(8):
        wc = mxfp6_codes(draw((16, 32)).ravel()).reshape(16, 32)
        xc = mxfp6_codes(rng.standard_normal((ops, 32)).ravel() if kind == "gauss" else rng.standard_t(3, (ops, 32)).ravel()).reshape(ops, 32)
        _, a = mac_e.run(wc, xc)
        _, b = mac_f.run(wc, xc)
        cyc.append((a["cycles"].mean(), b["cycles"].mean()))
    c = np.array(cyc)
    return float(1 - c[:, 0].mean() / c[:, 1].mean())


def main() -> None:
    seed_all(0)
    targets = yaml.safe_load((REPO_ROOT / "docs/spec/midas_targets.yaml").read_text())
    f13, f22 = targets["fig13_exponent_logic"], targets["fig22_system_ee_steps"]
    cells_all = load_cells([P + c for c in ("fa_1", "mux2_1", "and2_1", "dfxtp_1", "ha_1")])
    cells = {k[len(P):]: v for k, v in cells_all.items()}
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    prereg_commit = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(PREREG)], capture_output=True,
                                   text=True, cwd=REPO_ROOT).stdout.strip()
    files = sorted(Path(CAPTURE).glob("*.npz"))
    cfg = {"tolerance_rel": TOL, "prereg": str(PREREG.relative_to(REPO_ROOT)), "prereg_sha256": prereg_sha,
           "prereg_commit": prereg_commit, "capture": str(CAPTURE.relative_to(REPO_ROOT)),
           "alpha": ALPHA, "alpha_mepu": ALPHA_MEPU, "vdd": VDD, "tiles": 4, "tokens": 256, "seed": 0,
           "targets_file": "docs/spec/midas_targets.yaml", "source": "refs/MIDAS.pdf (choi2026midas)"}
    with start_run("e37_midas_ablation", cfg) as run:
        rows: list[dict] = []
        # ---- C0 exactness with early stop on
        rng = np.random.default_rng(1)
        mism = {}
        for fmt in ("mxfp4", "mxfp6"):
            nb = 1 << FORMATS[fmt]["code_bits"]
            m = 0
            for _ in range(20):
                wc = rng.integers(0, nb, (16, 32))
                xc = rng.integers(0, nb, (64, 32))
                # sparsify some tiles so early stop actually triggers
                xc = np.where(rng.random(xc.shape) < 0.5, rng.integers(0, 2, xc.shape), xc)
                p, _ = MidasLikeMacro(fmt).run(wc, xc)
                m += int((p != block_integer_reference(wc, xc, fmt)).sum())
            mism[fmt] = m
        c0 = all(v == 0 for v in mism.values())
        rows.append({"check": "C0_lossless_exact", "model": sum(mism.values()), "target": 0, "lo": 0, "hi": 0,
                     "rel_err": "", "verdict": "PASS" if c0 else "FAIL"})
        # ---- activity on the e19 sample (MXFP4)
        rng = np.random.default_rng(0)
        acts = []
        for f in files:
            qw, qa = load_capture(f, "mxfp4")
            wb, xb = qw.blocked_codes(), qa.blocked_codes()
            for _ in range(4):
                b = int(rng.integers(0, wb.shape[1]))
                n0 = int(rng.integers(0, wb.shape[0] - 16))
                tok = rng.choice(xb.shape[0], 256, replace=False)
                _, act = MidasLikeMacro("mxfp4").run(wb[n0 : n0 + 16, b], xb[tok, b])
                acts.append(act)
        act = {k: np.concatenate([x[k] for x in acts]) for k in acts[0] if k != "c_end"}
        rec = macro_record(act, MidasAssumptions())
        cyc4 = float(act["cycles"].mean())
        x4 = expo_logic("mxfp4", cyc4, cells)
        r4 = reductions(x4)
        # NOTE: N_LCA assumes 16 columns; activity ran 16-col tiles.
        rows += [
            verdict("C2_taor_power_pct", 100 * r4["taor_power"], f13["taor"]["power_reduction_pct"]),
            verdict("C3_taor_area_pct", 100 * r4["taor_area"], f13["taor"]["area_reduction_pct"]),
            verdict("C4_taor_mepd_power_pct", 100 * r4["taor_mepd_power"], f13["taor_mepd"]["power_reduction_pct"]),
            verdict("C5_taor_mepd_area_pct", 100 * r4["taor_mepd_area"], f13["taor_mepd"]["area_reduction_pct"]),
        ]
        e2 = sum(rec["energy_fj_per_op"].values())
        e1 = e2 - x4["energy"]["taor_mepd"] + x4["energy"]["baseline"]
        step2 = 100 * (e1 / e2 - 1)
        rows.append(verdict("C6_fig22_step2_gain_pct", step2, 100 * (f22["taor_mepd"] / f22["stca_saca"] - 1)))
        rows.append({"check": "C1_mxfp6_mepd_latency_pct", "model": "", "target": 22.5, "lo": band(22.5)[0], "hi": band(22.5)[1],
                     "rel_err": "", "verdict": "NE (needs GPT2 MXFP6 activations; model run)"})
        rows.append({"check": "C7_fig22_step1_stca_saca", "model": "", "target": 72.0, "lo": "", "hi": "", "rel_err": "",
                     "verdict": "NE (no pre-aligned 2C FP-CIM baseline mapping/dataflow model)"})
        rows.append({"check": "C8_fig22_step3_sadb_sscr", "model": "", "target": 17.3, "lo": "", "hi": "", "rel_err": "",
                     "verdict": "NE (no approximation model; L2 has approximation off)"})
        run.save_csv("checks.csv", rows)
        # ---- informational / sensitivity (non-gating, [EST])
        rng = np.random.default_rng(2)
        x6 = expo_logic("mxfp6", 15.0, cells)  # sensitivity with full-range cycles only for area; energy uses proxy below
        info = {
            "mxfp4_cycles_per_op_early": cyc4,
            "mxfp4_cycles_full": x4["cycles_full"],
            "mxfp4_mepd_latency_reduction_pct_EST": 100 * (1 - cyc4 / x4["cycles_full"]),
            "mxfp4_expo_energy_fj_per_op": x4["energy"],
            "mxfp4_expo_area_um2": x4["area"],
            "mxfp4_reductions_pct": {k: 100 * v for k, v in r4.items()},
            "macro_energy_fj_per_op_e19_model": e2,
            "fig22_step2_target_pct": 100 * (f22["taor_mepd"] / f22["stca_saca"] - 1),
            "proxy_mxfp6_latency_reduction_pct_EST": {k: 100 * proxy_latency(rng, k) for k in ("gauss", "student_t3")},
            "mxfp6_sensitivity_area_reductions_pct_EST": {k: 100 * v for k, v in reductions(x6).items() if "area" in k},
            "c0_mismatches": mism,
        }
        gating = [r for r in rows]
        verd = {r["check"]: r["verdict"] for r in gating}
        all_pass = all(v == "PASS" for v in verd.values())
        run.log_metrics({"label": "MIDAS-like" if all_pass else "MIDAS-inspired", "verdicts": verd,
                         "n_pass": sum(v == "PASS" for v in verd.values()),
                         "n_fail": sum(v == "FAIL" for v in verd.values()),
                         "n_ne": sum(v.startswith("NE") for v in verd.values()), "info": info})
        for r in rows:
            print(r["check"], r["model"] if r["model"] == "" else round(float(r["model"]), 2), r["target"], r["verdict"])
        print(info)


if __name__ == "__main__":
    main()
