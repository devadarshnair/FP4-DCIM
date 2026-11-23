"""GPU implementation of the L1 accumulation modes for whole-model evaluation (WP-16).

A Linear layer y = x·Wᵀ is decomposed into exact block terms P_b·2^E_b (integer block sums from fp32
batched matmuls of the ×2 integer element values - exact because |S| ≤ 4608 < 2^24) and accumulated with
the same semantics as golden.accum: Exact (float64; exact at realistic spreads), FP32Seq, Window(W),
Hier(tile_blocks, W, reducer ∈ {"exact", "fp32"}), AlignTrunc(F). Bit-identical to golden.l1_dot on
small shapes (tests/kernels/test_torch_accum.py).
"""

from __future__ import annotations

import math

import torch

from ..golden.accum import AlignTrunc, Exact, FP32Seq, Hier, Window
from ..kernels.fakequant import no_tf32
from .torch_quant import _e4m3_rne_satfinite, _floor_log2, _rne_e2m1

GRID2 = (0, 1, 2, 3, 4, 6, 8, 12)


# ------------------------------------------------------------------------------- integer operands
def mx_int(x: torch.Tensor, rule: str = "ocp_v11"):
    """MXFP4 quantization -> (ints ×2 as float32 (..., nb, 32), block exponent int64 (..., nb))."""
    v = x.to(torch.float64).reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    amax = v.abs().amax(-1)
    e = _floor_log2(amax)
    frac = amax / torch.ldexp(torch.ones_like(amax), e)
    adj = {
        "ocp_floor": torch.zeros_like(e),
        "ocp_v11": (frac >= 1.75).to(torch.int64),
        "rceil": (frac > 1.5).to(torch.int64),
        "ceil": (frac > 1.0).to(torch.int64),
    }[rule]
    xexp = torch.clamp(torch.where(amax > 0, e - 2 + adj, torch.full_like(e, -127)), -127, 127)
    s = torch.ldexp(torch.ones_like(amax), xexp)[..., None]
    q = _rne_e2m1(v.abs() / s) * torch.sign(v)
    return (q * 2).to(torch.float32), xexp


def nv_int(x: torch.Tensor, convention: str = "hw", tensor_amax: float | None = None):
    """NVFP4 -> (ints ×2 float32 (..., nb, 16), scale significand u int64, exponent k int64, s_dec float).

    tensor_amax: static (calibrated) tensor amax, as in fq_nvfp4/quantize_nvfp4; None -> dynamic amax of x.
    """
    v32 = x.to(torch.float32)
    t_amax = torch.tensor(float(v32.abs().max()) if tensor_amax is None else tensor_amax, dtype=torch.float32)
    s_dec = (
        torch.tensor(1.0, dtype=torch.float32)
        if t_amax == 0
        else (t_amax / torch.tensor(2688.0)).to(torch.float32)
    )
    s_enc = (torch.tensor(1.0, dtype=torch.float32) / s_dec).to(torch.float32)
    vb = v32.reshape(*x.shape[:-1], x.shape[-1] // 16, 16)
    amax_b = vb.abs().amax(-1)
    if convention == "hw":
        c = (s_enc / torch.tensor(6.0, dtype=torch.float32)).to(torch.float32)
        scale = _e4m3_rne_satfinite(amax_b.to(torch.float64) * float(c))
    else:
        scale = _e4m3_rne_satfinite(
            ((amax_b / torch.tensor(6.0, dtype=torch.float32)) * s_enc).to(torch.float32).to(torch.float64)
        )
    d64 = scale * float(s_dec)
    a = torch.where(
        d64[..., None] > 0,
        vb.abs().to(torch.float64) / d64[..., None],
        torch.zeros_like(vb, dtype=torch.float64),
    )
    q = _rne_e2m1(a) * torch.sign(vb.to(torch.float64))
    # scale = u * 2^k with u in [0, 15]
    m, e = torch.frexp(scale)  # scale = m * 2^e, m in [0.5, 1)
    k = torch.where(scale > 0, e - 4, torch.full_like(e, -9)).to(torch.int64)
    u = torch.where(scale > 0, (m * 16).round(), torch.zeros_like(m)).to(torch.int64)
    sub = (u < 8) & (scale > 0)  # E4M3 subnormal: u = scale / 2^-9
    u = torch.where(sub, (scale / 2.0**-9).round().to(torch.int64), u)
    k = torch.where(sub, torch.full_like(k, -9), k)
    return (q * 2).to(torch.float32), u, k, float(s_dec)


# ------------------------------------------------------------------------------- block terms
def block_terms(xi, wi, x_meta, w_meta, fmt: str):
    """-> P (T, N, nb) int64, E (T, N, nb) int64, ex (T, nb), ew (N, nb) exponents."""
    with no_tf32():
        s = torch.einsum("tbk,nbk->tnb", xi, wi)  # exact integer sums in fp32
    s = s.round().to(torch.int64)
    if fmt == "mxfp4":
        ex, ew = x_meta, w_meta
        p = s
    else:
        (ux, ex), (uw, ew) = x_meta, w_meta
        p = s * ux[:, None, :] * uw[None, :, :]
    e = ex[:, None, :] + ew[None, :, :] - 2
    return p, e, ex, ew


def _ldexp_i(p: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    return torch.ldexp(p.to(torch.float64), e)


def accumulate(p, e, ex, ew, mode, int_bits: int) -> torch.Tensor:
    t, n, nb = p.shape
    if isinstance(mode, Exact):
        return _ldexp_i(p, e).sum(-1).to(torch.float32)  # float64: exact at realistic spreads
    if isinstance(mode, FP32Seq):
        acc = torch.zeros((t, n), dtype=torch.float32, device=p.device)
        for b in range(nb):
            acc = acc + _ldexp_i(p[..., b], e[..., b]).to(torch.float32)
        return acc
    if isinstance(mode, Window):
        mode = Hier(nb, mode.width, mode.guard, "exact")
    if isinstance(mode, Hier):
        tb = mode.tile_blocks
        guard = mode.guard if mode.guard is not None else max(1, math.ceil(math.log2(max(tb, 2))))
        tiles = []
        for s in range(0, nb, tb):
            pt, et = p[..., s : s + tb], e[..., s : s + tb]
            anchor = ex[:, s : s + tb].amax(-1)[:, None] + ew[:, s : s + tb].amax(-1)[None, :] - 2
            lsb = anchor + int_bits + guard - mode.window + 1
            sh = et - lsb[..., None]
            term = torch.where(sh >= 0, pt << sh.clamp(0, 62), pt >> (-sh).clamp(0, 63))
            acc = term.sum(-1)
            lim = 1 << (mode.window - 1)
            if ((acc >= lim) | (acc < -lim)).any():
                raise AssertionError("tile window overflow")
            tiles.append((acc, lsb))
        if mode.reducer == "fp32":
            out = torch.zeros((t, n), dtype=torch.float32, device=p.device)
            for acc, lsb in tiles:
                out = out + _ldexp_i(acc, lsb).to(torch.float32)
            return out
        return sum(_ldexp_i(acc, lsb) for acc, lsb in tiles).to(torch.float32)  # exact reducer (float64)
    if isinstance(mode, AlignTrunc):
        f = mode.frac
        acc_n = torch.zeros((t, n), dtype=torch.int64, device=p.device)
        acc_l = torch.zeros((t, n), dtype=torch.int64, device=p.device)
        for b in range(nb):
            pb, eb = p[..., b], e[..., b]
            msb_a = torch.where(
                acc_n != 0,
                _floor_log2(acc_n.abs().to(torch.float64)) + acc_l,
                torch.full_like(acc_l, -(1 << 40)),
            )
            msb_t = torch.where(
                pb != 0, _floor_log2(pb.abs().to(torch.float64)) + eb, torch.full_like(eb, -(1 << 40))
            )
            lsb = torch.maximum(msb_a, msb_t) - f

            def trunc(n_, l_, lsb=lsb):
                d = l_ - lsb
                left = n_ << d.clamp(0, 62)
                right = n_.abs() >> (-d).clamp(0, 63)
                return torch.where(d >= 0, left, torch.sign(n_) * right)

            new = trunc(acc_n, acc_l) + trunc(pb, eb)
            empty = (acc_n == 0) & (pb == 0)
            acc_n = torch.where(empty, torch.zeros_like(new), new)
            acc_l = torch.where(empty, torch.zeros_like(lsb), lsb)
        return _ldexp_i(acc_n, acc_l).to(torch.float32)
    raise TypeError(mode)


# ------------------------------------------------------------------------------- fast exact path
# Folding each block's scale into its integer elements turns a tile's Σ_b P_b·2^E_b into one float64 GEMM.
# Every product is exact in float64 (≤ 13/26 significant bits); every partial sum is an integer multiple of
# 2^Emin, so the GEMM is exact (in any summation order) whenever K·Pmax·2^(Emax-Emin) < 2^53 (condition A).
# For Window/Hier the slow path truncates terms below the window LSB; if no non-zero term lies below it
# (condition B), the window accumulator equals the exact tile sum. Rows violating A or B use the slow path,
# so results are bit-identical to the slow path by construction (tests/kernels/test_torch_accum_fast.py).
_PMAX = {"mxfp4": 144.0, "nvfp4": 144.0 * 225.0}  # max |element-int product| (ints ×2; NV × u_x·u_w)


def _scaled(ints, exp, sig):
    v = ints.to(torch.float64) * torch.ldexp(torch.ones_like(exp, dtype=torch.float64), exp)[..., None]
    return v if sig is None else v * sig.to(torch.float64)[..., None]


def _spreads(exp, nz):
    """Per row: (max_b exp over all blocks, max - min over non-zero blocks) for a tile slice."""
    big = 1 << 20
    emax_all = exp.amax(-1)
    emax_nz = torch.where(nz, exp, torch.full_like(exp, -big)).amax(-1)
    emin_nz = torch.where(nz, exp, torch.full_like(exp, big)).amin(-1)
    has = nz.any(-1)
    spread = torch.where(has, emax_nz - emin_nz, torch.zeros_like(emax_nz))
    below = torch.where(has, emax_all - emin_nz, torch.zeros_like(emax_nz))  # distance of lowest nz term
    return spread, below


def _fast_accumulate(xi, ex, xs, wi, ew, ws, mode, int_bits, fmt):
    """-> (y float32 (T, N), ok (T,) rows computed exactly by the fast path)."""
    t, nb, bs = xi.shape
    x_nz = (xi != 0).any(-1) if xs is None else ((xi != 0).any(-1) & (xs > 0))
    w_nz = (wi != 0).any(-1) if ws is None else ((wi != 0).any(-1) & (ws > 0))
    X, W = _scaled(xi, ex, xs), _scaled(wi, ew, ws)
    if isinstance(mode, Exact):
        tiles, window = [(0, nb)], None
    else:
        if isinstance(mode, Window):
            mode = Hier(nb, mode.width, mode.guard, "exact")
        tb = mode.tile_blocks
        guard = mode.guard if mode.guard is not None else max(1, math.ceil(math.log2(max(tb, 2))))
        tiles, window = [(s, min(s + tb, nb)) for s in range(0, nb, tb)], (mode.window, guard)
    ok = torch.ones(t, dtype=torch.bool, device=xi.device)
    vals = []
    for s0, s1 in tiles:
        sx, bx = _spreads(ex[:, s0:s1], x_nz[:, s0:s1])
        sw, bw = _spreads(ew[:, s0:s1], w_nz[:, s0:s1])
        log_k = math.log2((s1 - s0) * bs * _PMAX[fmt])
        ok &= (log_k + sx.double() + float(sw.max()) < 52.0)  # condition A (1 bit margin)
        with no_tf32():
            v = (X[:, s0:s1].reshape(t, -1) @ W[:, s0:s1].reshape(W.shape[0], -1).T) * 0.25
        if window is not None:
            width, guard = window
            ok &= (bx + int(bw.max()) <= width - int_bits - guard - 1)  # condition B: no truncated term
            anchor = ex[:, s0:s1].amax(-1)[:, None] + ew[:, s0:s1].amax(-1)[None, :] - 2
            lsb = anchor + int_bits + guard - width + 1
            acc = torch.ldexp(v, -lsb)
            lim = float(1 << (width - 1))
            if (((acc >= lim) | (acc < -lim)) & ok[:, None]).any():
                raise AssertionError("tile window overflow")
        vals.append(v)
    if isinstance(mode, Exact):
        return vals[0].to(torch.float32), ok
    if mode.reducer == "fp32":
        out = torch.zeros(vals[0].shape, dtype=torch.float32, device=xi.device)
        for v in vals:
            out = out + v.to(torch.float32)
        return out, ok
    return sum(vals).to(torch.float32), ok


def _block_sum(xi, wi, b, xs, ws):
    """Exact integer block sums P_b (T, N) int64 for block b (fp32 GEMM of ints ×2: |S| ≤ 4608 < 2^24)."""
    with no_tf32():
        s = (xi[:, b, :] @ wi[:, b, :].T).round().to(torch.int64)
    return s if xs is None else s * xs[:, b, None] * ws[None, :, b]


def _blockloop_accumulate(xi, ex, xs, wi, ew, ws, mode, int_bits):
    """Same semantics and operation order as `accumulate`, one block at a time (no (T, N, nb) tensor)."""
    t, nb, _ = xi.shape
    n = wi.shape[0]
    dev = xi.device

    def e_of(b):
        return ex[:, b, None] + ew[None, :, b] - 2

    if isinstance(mode, Exact):  # float64 sum in block order (matches accumulate's .sum(-1) only if exact)
        raise TypeError("Exact uses the GEMM path or the reference path")
    if isinstance(mode, FP32Seq):
        acc = torch.zeros((t, n), dtype=torch.float32, device=dev)
        for b in range(nb):
            acc = acc + _ldexp_i(_block_sum(xi, wi, b, xs, ws), e_of(b)).to(torch.float32)
        return acc
    if isinstance(mode, Window):
        mode = Hier(nb, mode.width, mode.guard, "exact")
    if isinstance(mode, Hier):
        tb = mode.tile_blocks
        guard = mode.guard if mode.guard is not None else max(1, math.ceil(math.log2(max(tb, 2))))
        tiles = []
        for s in range(0, nb, tb):
            anchor = ex[:, s : s + tb].amax(-1)[:, None] + ew[:, s : s + tb].amax(-1)[None, :] - 2
            lsb = anchor + int_bits + guard - mode.window + 1
            acc = torch.zeros((t, n), dtype=torch.int64, device=dev)
            for b in range(s, min(s + tb, nb)):
                pt, sh = _block_sum(xi, wi, b, xs, ws), e_of(b) - lsb
                acc = acc + torch.where(sh >= 0, pt << sh.clamp(0, 62), pt >> (-sh).clamp(0, 63))
            lim = 1 << (mode.window - 1)
            if ((acc >= lim) | (acc < -lim)).any():
                raise AssertionError("tile window overflow")
            tiles.append((acc, lsb))
        if mode.reducer == "fp32":
            out = torch.zeros((t, n), dtype=torch.float32, device=dev)
            for acc, lsb in tiles:
                out = out + _ldexp_i(acc, lsb).to(torch.float32)
            return out
        return sum(_ldexp_i(acc, lsb) for acc, lsb in tiles).to(torch.float32)
    if isinstance(mode, AlignTrunc):
        f = mode.frac
        acc_n = torch.zeros((t, n), dtype=torch.int64, device=dev)
        acc_l = torch.zeros((t, n), dtype=torch.int64, device=dev)
        for b in range(nb):
            pb, eb = _block_sum(xi, wi, b, xs, ws), e_of(b)
            neg = -(1 << 40)
            msb_a = torch.where(
                acc_n != 0, _floor_log2(acc_n.abs().to(torch.float64)) + acc_l, torch.full_like(acc_l, neg)
            )
            msb_t = torch.where(
                pb != 0, _floor_log2(pb.abs().to(torch.float64)) + eb, torch.full_like(eb, neg)
            )
            lsb = torch.maximum(msb_a, msb_t) - f

            def trunc(n_, l_, lsb=lsb):
                d = l_ - lsb
                right = torch.sign(n_) * (n_.abs() >> (-d).clamp(0, 63))
                return torch.where(d >= 0, n_ << d.clamp(0, 62), right)

            new = trunc(acc_n, acc_l) + trunc(pb, eb)
            empty = (acc_n == 0) & (pb == 0)
            acc_n = torch.where(empty, torch.zeros_like(new), new)
            acc_l = torch.where(empty, torch.zeros_like(lsb), lsb)
        return _ldexp_i(acc_n, acc_l).to(torch.float32)
    raise TypeError(mode)


def linear_l1(
    x: torch.Tensor,
    w_ints,
    w_meta,
    fmt: str,
    mode,
    rule: str = "ocp_v11",
    nv_conv: str = "hw",
    w_tensor_scale: float = 1.0,
    chunk: int = 512,
    x_tensor_amax: float | None = None,
    fast: bool = True,
    stats: dict | None = None,
) -> torch.Tensor:
    """y = x·Wᵀ with the given accumulation mode. w_ints (N, nb, bs) float32; w_meta as from mx_int/nv_int.

    fast=True uses the exact float64-GEMM path for Exact/Window/Hier rows that provably need no rounding or
    truncation and the slow block-term path for all other rows/modes (bit-identical either way).
    x_tensor_amax: NVFP4 static (calibrated) activation tensor amax (T6.6); None -> dynamic amax of x.
    """
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if fmt == "mxfp4":
        xi, xm = mx_int(x2, rule)
        post, int_bits = 1.0, 14
    else:
        xi, ux, kx, xs = nv_int(x2, nv_conv, x_tensor_amax)
        xm = (ux, kx)
        post, int_bits = (
            float(torch.tensor(xs, dtype=torch.float32) * torch.tensor(w_tensor_scale, dtype=torch.float32)),
            21,
        )
    n = w_ints.shape[0]
    if fast and isinstance(mode, (Exact, Window, Hier)):
        if fmt == "mxfp4":
            y, ok = _fast_accumulate(xi, xm, None, w_ints, w_meta, None, mode, int_bits, fmt)
        else:
            y, ok = _fast_accumulate(xi, kx, ux, w_ints, w_meta[1], w_meta[0], mode, int_bits, fmt)
        if stats is not None:
            stats["rows"] = stats.get("rows", 0) + int(ok.numel())
            stats["fast_rows"] = stats.get("fast_rows", 0) + int(ok.sum())
        bad = (~ok).nonzero().flatten()
        if bad.numel():  # same integer operands (same NV tensor scale)
            xm_b = xm[bad] if fmt == "mxfp4" else (xm[0][bad], xm[1][bad])
            if isinstance(mode, Exact):  # float64 block-order sum: keep the reference semantics
                y[bad] = _slow_accumulate(xi[bad], xm_b, w_ints, w_meta, fmt, mode, int_bits, chunk)
            else:
                y[bad] = _blockloop(xi[bad], xm_b, w_ints, w_meta, fmt, mode, int_bits)
    elif fast:
        y = _blockloop(xi, xm, w_ints, w_meta, fmt, mode, int_bits)
    else:
        y = _slow_accumulate(xi, xm, w_ints, w_meta, fmt, mode, int_bits, chunk)
    if fmt == "nvfp4":
        y = (y * torch.tensor(post, dtype=torch.float32)).to(torch.float32)
    return y.reshape(*lead, n)


def _blockloop(xi, xm, w_ints, w_meta, fmt, mode, int_bits):
    if fmt == "mxfp4":
        return _blockloop_accumulate(xi, xm, None, w_ints, w_meta, None, mode, int_bits)
    return _blockloop_accumulate(xi, xm[1], xm[0], w_ints, w_meta[1], w_meta[0], mode, int_bits)


def _slow_accumulate(xi, xm, w_ints, w_meta, fmt, mode, int_bits, chunk):
    """Reference block-term path (before the NVFP4 tensor post-scale)."""
    outs = []
    n = w_ints.shape[0]
    for c0 in range(0, n, chunk):
        wm = (
            w_meta[c0 : c0 + chunk]
            if fmt == "mxfp4"
            else (w_meta[0][c0 : c0 + chunk], w_meta[1][c0 : c0 + chunk])
        )
        p, e, ex, ew = block_terms(xi, w_ints[c0 : c0 + chunk], xm, wm, fmt)
        outs.append(
            accumulate(p, e, ex if fmt == "mxfp4" else xm[1], ew if fmt == "mxfp4" else wm[1], mode, int_bits)
        )
    return torch.cat(outs, dim=1)
