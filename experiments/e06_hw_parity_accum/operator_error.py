"""e06b - operator-level accumulator adequacy and fake-quant order sensitivity (D1 numerics; CPU only).

Why: perplexity deltas of ±0.01-0.02 between accumulation modes are at the noise floor of the NVFP4 fp32
fake-quant reference itself (its GEMM rounds, so its result depends on the GPU's summation order, and the next
layer's activation re-quantization occasionally flips a code). This experiment measures the accumulator at the
operator level on real captured layers instead:

(a) for every (W, Kt) of the e06 sweep and both formats: fraction of outputs bit-identical to L1 Exact,
    max |y - exact| / rms(exact), and the fraction of rows that provably needed no truncation (fast path);
(b) fp32 fake-quant GEMM vs L1 Exact (same statistics);
(c) order sensitivity: the fp32 fake-quant GEMM in two summation orders (single GEMM vs 256-wide K chunks),
    fraction of outputs bit-identical, and how many next-layer element codes / block scales / tensor scales
    the reordering changes after re-quantization.

Inputs: Qwen2.5-0.5B prefill captures (21 modules, MXFP4 and NVFP4 codes), first `--tokens` tokens.
Run: python experiments/e06_hw_parity_accum/operator_error.py [--tokens 256]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cimfp4.formats import mx_quant
from cimfp4.golden.accum import Exact, Hier, Window
from cimfp4.kernels.fakequant import no_tf32
from cimfp4.kernels.torch_accum import linear_l1, mx_int, nv_int
from cimfp4.llm.capture import load_capture
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

CAPTURE = REPO_ROOT / "data/captures/qwen2.5-0.5b@060db6499f32/prefill"
WS = (24, 28, 32, 36, 40, 48)
KTS = (128, 512, 2048, None)


def _operands(path: Path, fmt: str, tokens: int):
    qw, qa = load_capture(path, fmt)
    w = torch.from_numpy(mx_quant.dequantize(qw)).float()  # exact dequantized values (fp32-representable)
    x = torch.from_numpy(mx_quant.dequantize(qa)[:tokens]).float()
    if fmt == "mxfp4":
        wi, wm = mx_int(w)
        return x, w, (wi, wm, fmt), 1.0
    wi, uw, kw, ws = nv_int(w)
    return x, w, (wi, (uw, kw), fmt), ws


def _stats(y: torch.Tensor, ref: torch.Tensor) -> tuple[int, float]:
    rms = ref.double().pow(2).mean().sqrt().clamp_min(1e-300)
    return int((y == ref).sum()), float(((y.double() - ref.double()).abs() / rms).max())


def _fq_gemm(x, w, fmt, chunk=None):
    from cimfp4.kernels.torch_quant import fq_mxfp4, fq_nvfp4

    fq = (lambda t: fq_mxfp4(t, "ocp_v11")) if fmt == "mxfp4" else (lambda t: fq_nvfp4(t, "hw"))
    xq, wq = fq(x), fq(w)
    with no_tf32():
        if chunk is None:
            return xq @ wq.T
        return sum(xq[:, s : s + chunk] @ wq[:, s : s + chunk].T for s in range(0, xq.shape[1], chunk))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    seed_all(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    files = sorted(CAPTURE.glob("*.npz"))
    cfg = {**vars(a), "capture": str(CAPTURE.relative_to(REPO_ROOT)), "n_modules": len(files), "Ws": WS, "Kts": KTS}
    with start_run("e06b_operator_error", cfg, paper=a.paper) as run:
        grid, ref_rows, order_rows = [], [], []
        for fmt in ("mxfp4", "nvfp4"):
            bs = 32 if fmt == "mxfp4" else 16
            acc = {}
            fq_same = fq_n = 0
            fq_max = 0.0
            flips = {"el": 0, "el_n": 0, "bl": 0, "bl_n": 0, "ts": 0, "ts_n": 0, "same": 0, "n": 0}
            for f in files:
                x, w, args, ws = _operands(f, fmt, a.tokens)
                ex = linear_l1(x, *args, Exact(), w_tensor_scale=ws)
                for wd in WS:
                    for kt in KTS:
                        mode = Window(wd) if kt is None else Hier(kt // bs, wd, reducer="fp32")
                        st: dict = {}
                        try:
                            y = linear_l1(x, *args, mode, w_tensor_scale=ws, stats=st)
                            same, mx = _stats(y, ex)
                            ovf = 0
                        except AssertionError:
                            same, mx, ovf = 0, float("inf"), 1
                        r = acc.setdefault((wd, kt), {"same": 0, "n": 0, "max": 0.0, "fast": 0, "rows": 0, "ovf": 0})
                        r["same"] += same
                        r["n"] += ex.numel()
                        r["max"] = max(r["max"], mx)
                        r["fast"] += st.get("fast_rows", 0)
                        r["rows"] += st.get("rows", 0)
                        r["ovf"] += ovf
                y1 = _fq_gemm(x, w, fmt)
                same, mx = _stats(y1, ex)
                fq_same, fq_n, fq_max = fq_same + same, fq_n + ex.numel(), max(fq_max, mx)
                y2 = _fq_gemm(x, w, fmt, chunk=256)
                flips["same"] += int((y1 == y2).sum())
                flips["n"] += y1.numel()
                n = y1.shape[1] - y1.shape[1] % bs
                if fmt == "mxfp4":
                    (i1, e1), (i2, e2) = mx_int(y1[:, :n]), mx_int(y2[:, :n])
                    flips["el"] += int((i1 != i2).sum())
                    flips["bl"] += int((e1 != e2).sum())
                else:
                    i1, u1, k1, s1 = nv_int(y1[:, :n])
                    i2, u2, k2, s2 = nv_int(y2[:, :n])
                    flips["el"] += int((i1 != i2).sum())
                    flips["bl"] += int(((u1 != u2) | (k1 != k2)).sum())
                    flips["ts"] += int(s1 != s2)
                flips["el_n"] += i1.numel()
                flips["bl_n"] += i1.shape[0] * i1.shape[1]
                flips["ts_n"] += 1
                print(f"[{fmt}] {f.stem} done", flush=True)
            for (wd, kt), r in acc.items():
                grid.append({"fmt": fmt, "W": wd, "Kt": kt or "K", "bit_identical": r["same"] / r["n"],
                             "max_err_over_rms": r["max"], "no_truncation_rows": r["fast"] / max(r["rows"], 1),
                             "overflow_modules": r["ovf"]})
            ref_rows.append({"fmt": fmt, "fakequant_bit_identical": fq_same / fq_n, "fakequant_max_err_over_rms": fq_max})
            order_rows.append({"fmt": fmt, "reorder_bit_identical": flips["same"] / flips["n"],
                               "element_code_changes": flips["el"], "element_codes": flips["el_n"],
                               "block_scale_changes": flips["bl"], "block_scales": flips["bl_n"],
                               "tensor_scale_changes": flips["ts"], "tensors": flips["ts_n"]})
        run.save_csv("operator_grid.csv", grid)
        run.save_csv("fakequant_vs_exact.csv", ref_rows)
        run.save_csv("order_sensitivity.csv", order_rows)
        hw = {r["fmt"]: r for r in grid if r["W"] == 40 and r["Kt"] == 512}
        wstar = {}
        for fmt in ("mxfp4", "nvfp4"):
            ok = [wd for wd in WS if all(r["bit_identical"] == 1.0 for r in grid if r["fmt"] == fmt and r["W"] == wd
                                         and r["Kt"] != "K")]
            wstar[fmt] = min(ok) if ok else None
        run.log_metrics(
            {f"{f}_hw40_kt512_bit_identical": hw[f]["bit_identical"] for f in hw}
            | {f"{f}_hw40_kt512_max_err_over_rms": hw[f]["max_err_over_rms"] for f in hw}
            | {f"{r['fmt']}_fakequant_bit_identical": r["fakequant_bit_identical"] for r in ref_rows}
            | {f"{r['fmt']}_fakequant_max_err_over_rms": r["fakequant_max_err_over_rms"] for r in ref_rows}
            | {f"{r['fmt']}_reorder_bit_identical": r["reorder_bit_identical"] for r in order_rows}
            | {f"{r['fmt']}_reorder_element_code_changes": r["element_code_changes"] for r in order_rows}
            | {f"{f}_min_W_bit_identical_all_finite_Kt": v for f, v in wstar.items()}
        )
        for r in grid:
            if r["Kt"] in (512, "K"):
                print(f"{r['fmt']} W{r['W']:2d} Kt{r['Kt']}: identical {r['bit_identical']:.6f} "
                      f"max {r['max_err_over_rms']:.2e} no-trunc {r['no_truncation_rows']:.3f}")
        print(ref_rows, order_rows, wstar)


if __name__ == "__main__":
    main()
