"""Power-stimulus programs for the column-group designs (WP-28, G1 setup).

A *program* = weight sets (codes + exponents for 4 columns × M sets) + an op list, saved as .npz and
consumed identically by every design's cocotb harness (tb/cocotb/test_column_top.py, CT_PROGRAM).

Operating point (G1, reuse R): a weight set stays latched while R different activation vectors
(different tokens, same K-block) stream through it, then the next set is selected. Each op starts a
new partial sum (acc_clear = 1): cross-block accumulation happens in the accumulator memory, which is
identical for all designs and outside the compared boundary (disclosed in the paper).

Sources: real RTN-quantized captures (WP-14) or a 50 %-random control (uniform E2M1 codes, 50 % bit
toggle probability), as required by the G1 setup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..formats.types import FP4Block

NCOL = 4


def program_from_capture(
    qw: FP4Block, qa: FP4Block, rng: np.random.Generator, m: int = 16, sets_used: int = 8, reuse: int = 256
) -> dict:
    """Weights: 4 consecutive output rows; set s holds K-block b_s. Activations: `reuse` token rows."""
    r0 = int(rng.integers(0, qw.codes.shape[0] - NCOL))
    blocks = rng.choice(qw.num_blocks, size=min(m, qw.num_blocks), replace=False)
    sets = np.zeros((m, NCOL, 32), np.uint8)
    ew = np.zeros((m, NCOL), np.int64)
    for s, b in enumerate(blocks):
        sets[s] = qw.blocked_codes()[r0 : r0 + NCOL, b]
        ew[s] = qw.scale_exp[r0 : r0 + NCOL, b]
    tokens = rng.choice(qa.codes.shape[0], size=reuse, replace=qa.codes.shape[0] < reuse)
    ops = []
    for s in range(min(sets_used, len(blocks))):
        b = blocks[s]
        for t in tokens:
            ex = int(qa.scale_exp[t, b])
            ops.append((qa.blocked_codes()[t, b], ex, ex, s))
    return _pack(sets, ew, ops)


def program_random(rng: np.random.Generator, m: int = 16, sets_used: int = 8, reuse: int = 256) -> dict:
    """50 %-random control: uniform codes (every bit toggles with p = 0.5), exponents in a narrow range."""
    sets = rng.integers(0, 16, (m, NCOL, 32)).astype(np.uint8)
    ew = rng.integers(-3, 1, (m, NCOL))
    ops = []
    for s in range(sets_used):
        for _ in range(reuse):
            ex = int(rng.integers(-3, 1))
            ops.append((rng.integers(0, 16, 32).astype(np.uint8), ex, ex, s))
    return _pack(sets, ew, ops)


def _pack(sets, ew, ops) -> dict:
    maxw = ew.max(axis=0)
    return {
        "sets": sets,
        "ew": ew,
        "maxw": maxw,
        "x": np.array([o[0] for o in ops], np.uint8),
        "e_x": np.array([o[1] for o in ops], np.int64),
        "max_e_x": np.array([o[2] for o in ops], np.int64),
        "set": np.array([o[3] for o in ops], np.int64),
        "clear": np.ones(len(ops), bool),
    }


def save_program(path: Path, prog: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **prog)
    return path


def load_program(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------------------------------
# Macro-level programs (32 columns; macro_top / bs_macro_top; tb/cocotb/test_macro_power.py). Same
# operating point as the column-group programs above: set s holds one 32-row K-slice of 32 output
# rows, `reuse` token rows stream through it, every op clears (cross-op accumulation is outside the
# compared boundary). MXFP4: one 32-block per op. NVFP4: two 16-blocks per op (E4M3 scales as
# (u, k) significand/exponent pairs), the same field layout as the macro scale RF.
MCOL = 32


def macro_program_from_capture(
    qw: FP4Block, qa: FP4Block, rng: np.random.Generator, m: int = 16, sets_used: int = 2, reuse: int = 256
) -> dict:
    from ..formats.codecs import e4m3_significand

    fmt = qw.fmt
    k = qw.codes.shape[1]
    nslice = k // 32
    r0 = int(rng.integers(0, qw.codes.shape[0] - MCOL + 1))
    slices = rng.choice(nslice, size=min(m, nslice), replace=False)
    tokens = rng.choice(qa.codes.shape[0], size=reuse, replace=qa.codes.shape[0] < reuse)
    sets = np.zeros((m, MCOL, 32), np.uint8)
    e0, e1 = np.zeros((m, MCOL), np.int64), np.zeros((m, MCOL), np.int64)
    u0, u1 = np.ones((m, MCOL), np.int64), np.ones((m, MCOL), np.int64)
    if fmt == "nvfp4":
        uw, kw = e4m3_significand(qw.scale_e4m3)
        ux, kx = e4m3_significand(qa.scale_e4m3)
    for s, b in enumerate(slices):
        sets[s] = qw.codes[r0 : r0 + MCOL, 32 * b : 32 * (b + 1)]
        if fmt == "mxfp4":
            e0[s] = qw.scale_exp[r0 : r0 + MCOL, b]
        else:
            e0[s], e1[s] = kw[r0 : r0 + MCOL, 2 * b], kw[r0 : r0 + MCOL, 2 * b + 1]
            u0[s], u1[s] = uw[r0 : r0 + MCOL, 2 * b], uw[r0 : r0 + MCOL, 2 * b + 1]
    ops = []
    for s in range(min(sets_used, len(slices))):
        b = slices[s]
        for t in tokens:
            x = qa.codes[t, 32 * b : 32 * (b + 1)]
            if fmt == "mxfp4":
                ex = int(qa.scale_exp[t, b])
                ops.append((x, ex, 0, 1, 1, ex, s))
            else:
                ka, kb = int(kx[t, 2 * b]), int(kx[t, 2 * b + 1])
                ops.append((x, ka, kb, int(ux[t, 2 * b]), int(ux[t, 2 * b + 1]), max(ka, kb), s))
    return _pack_macro(sets, e0, e1, u0, u1, ops, fmt)


def macro_program_random(
    rng: np.random.Generator, fmt: str = "mxfp4", m: int = 16, sets_used: int = 2, reuse: int = 256
) -> dict:
    """50 %-random control at macro level: uniform codes, narrow exponent range, random E4M3 significands."""
    sets = rng.integers(0, 16, (m, MCOL, 32)).astype(np.uint8)
    e0 = rng.integers(-3, 1, (m, MCOL))
    e1 = rng.integers(-3, 1, (m, MCOL)) if fmt == "nvfp4" else np.zeros((m, MCOL), np.int64)
    u0 = rng.integers(8, 16, (m, MCOL)) if fmt == "nvfp4" else np.ones((m, MCOL), np.int64)
    u1 = rng.integers(8, 16, (m, MCOL)) if fmt == "nvfp4" else np.ones((m, MCOL), np.int64)
    ops = []
    for s in range(sets_used):
        for _ in range(reuse):
            x = rng.integers(0, 16, 32).astype(np.uint8)
            ka = int(rng.integers(-3, 1))
            if fmt == "mxfp4":
                ops.append((x, ka, 0, 1, 1, ka, s))
            else:
                kb = int(rng.integers(-3, 1))
                ops.append((x, ka, kb, int(rng.integers(8, 16)), int(rng.integers(8, 16)), max(ka, kb), s))
    return _pack_macro(sets, e0, e1, u0, u1, ops, fmt)


def _pack_macro(sets, e0, e1, u0, u1, ops, fmt) -> dict:
    maxw = np.maximum(e0, e1).max(axis=0) if fmt == "nvfp4" else e0.max(axis=0)
    return {
        "fmt": np.array(fmt),
        "sets": sets,
        "e0": e0.astype(np.int64),
        "e1": e1.astype(np.int64),
        "u0": u0.astype(np.int64),
        "u1": u1.astype(np.int64),
        "maxw": maxw.astype(np.int64),
        "x": np.array([o[0] for o in ops], np.uint8),
        "e_x0": np.array([o[1] for o in ops], np.int64),
        "e_x1": np.array([o[2] for o in ops], np.int64),
        "u_x0": np.array([o[3] for o in ops], np.int64),
        "u_x1": np.array([o[4] for o in ops], np.int64),
        "max_e_x": np.array([o[5] for o in ops], np.int64),
        "set": np.array([o[6] for o in ops], np.int64),
        "clear": np.ones(len(ops), bool),
    }


def macro_program_mixed(mx: dict, nv: dict, chunk: int = 64) -> dict:
    """T4.2 mixed-format stream: the MXFP4 and NVFP4 programs of the same window merged into one program for one
    macro. The MX sets keep indices 0..S-1 and the NV sets move to S..2S-1 (S = sets used per format; 2S <= m), so
    both formats' weights are resident at once. The two op streams are interleaved in chunks of `chunk` ops (MX
    chunk, NV chunk, ...; each stream keeps its own order), `op_nv` flags the NV ops, and maxw is the per-column max
    of both programs (the static anchor must cover both formats; the references use the same value). Every op
    clears, so each op's expected value depends only on its own format's reference."""
    if not (bool(mx["clear"].all()) and bool(nv["clear"].all())):
        raise ValueError("mixed stream needs clear = 1 on every op")
    s_mx, s_nv = np.asarray(mx["set"]), np.asarray(nv["set"])
    used = int(max(s_mx.max(), s_nv.max())) + 1
    m = mx["sets"].shape[0]
    if 2 * used > m:
        raise ValueError(f"{2 * used} sets needed, macro has {m}")
    sets = np.zeros_like(mx["sets"])
    sets[:used], sets[used : 2 * used] = mx["sets"][:used], nv["sets"][:used]
    f = {}
    for k in ("e0", "e1", "u0", "u1"):
        a = np.array(mx[k][:m], copy=True)
        a[used : 2 * used] = nv[k][:used]
        f[k] = a.astype(np.int64)
    order = []  # (is_nv, op index)
    i = j = 0
    while i < len(s_mx) or j < len(s_nv):
        order += [(0, q) for q in range(i, min(i + chunk, len(s_mx)))]; i += chunk
        order += [(1, q) for q in range(j, min(j + chunk, len(s_nv)))]; j += chunk
    src = (mx, nv)
    pick = lambda k: np.array([src[n][k][q] for n, q in order])
    prog = {"fmt": np.array("mixed"), "sets": sets, **f,
            "maxw": np.maximum(mx["maxw"], nv["maxw"]).astype(np.int64),
            "x": pick("x").astype(np.uint8)}
    for k in ("e_x0", "e_x1", "u_x0", "u_x1", "max_e_x"):
        prog[k] = pick(k).astype(np.int64)
    prog["set"] = np.array([src[n]["set"][q] + used * n for n, q in order], np.int64)
    prog["clear"] = np.ones(len(order), bool)
    prog["op_nv"] = np.array([n for n, _ in order], bool)
    prog["mixed_chunk"] = np.array(chunk)
    return prog
