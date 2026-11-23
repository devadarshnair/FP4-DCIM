"""Inter-block accumulation precision study for MXFP4 / NVFP4 dot products (DERIVED, synthetic).
Exact intra-block integer sums; inter-block accumulation in reduced precision."""
import numpy as np
import os
rng = np.random.default_rng(1)
GRID = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])

def rtne_fp4(x):
    a = np.minimum(np.abs(x), 6.0)
    idx = np.clip(np.searchsorted(GRID, a), 1, 7)
    lo, hi = GRID[idx - 1], GRID[idx]
    mid = (lo + hi) / 2
    pick_hi = (a > mid) | ((a == mid) & ((idx % 2) == 0))
    code = np.where(pick_hi, idx, idx - 1)
    code = np.where(a == 0, 0, code)
    return np.sign(x) * GRID[code]

def mx(blocks):
    amax = np.maximum(np.abs(blocks).max(axis=-1, keepdims=True), 1e-38)
    e = np.ceil(np.log2(amax / 6.0))
    return rtne_fp4(blocks / 2.0 ** e), e      # elements (real), scale exponent

def round_sig(x, p, mode):
    """round x to p-bit significand float (unbounded exponent)"""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    nz = x != 0
    e = np.floor(np.log2(np.abs(x[nz])))
    ulp = 2.0 ** (e - (p - 1))
    if mode == "rne":
        out[nz] = np.round(x[nz] / ulp) * ulp   # numpy round = half-even
    else:  # rz
        out[nz] = np.trunc(x[nz] / ulp) * ulp
    return out

K, B = 4096, 32
nb = K // B
T = 4000   # number of dot products
# activations: per-channel lognormal std (outlier channels), weights gaussian
sig_ch = 2.0 ** rng.normal(0, 1.5, size=K)
A = rng.standard_normal((T, K)) * sig_ch
W = rng.standard_normal((T, K)) * 0.02
qa, ea = mx(A.reshape(T, nb, B))
qw, ew = mx(W.reshape(T, nb, B))
part = (qa * qw).sum(-1) * 4           # exact integers in units of 2^-2
assert np.allclose(part, np.round(part))
pexp = ea[..., 0] + ew[..., 0] - 2
exact = (part * 2.0 ** pexp).sum(-1)
ref = (A * W).sum(-1)
qerr = np.sqrt(np.mean((exact - ref) ** 2) / np.mean(ref ** 2))
span = (pexp.max(-1) - pexp.min(-1))
print(f"MXFP4 K={K}: quantization rel-RMS error vs FP64 = {qerr:.4f} ({20*np.log10(1/qerr):.1f} dB)")
print(f"block-partial |int| max={np.abs(part).max():.0f}; exponent span per dot product: median={np.median(span):.0f}, p99={np.percentile(span,99):.0f}, max={span.max():.0f}")
print(f"total bits for exact fixed-point accumulator (14b partial + span + log2(nb)): p99={14+np.percentile(span,99)+np.log2(nb):.0f}, max={14+span.max()+np.log2(nb):.0f}")

for p in (8, 11, 12, 14, 16, 20, 24):
    for mode in ("rne", "rz"):
        acc = np.zeros(T)
        for j in range(nb):
            acc = round_sig(acc + part[:, j] * 2.0 ** pexp[:, j], p, mode)
        aerr = np.sqrt(np.mean((acc - exact) ** 2) / np.mean(exact ** 2))
        print(f"  sequential acc p={p:2d} {mode}: acc rel-RMS err={aerr:.2e}  ratio to quant err={aerr/qerr:.3f}")

# fixed-window fused (Blackwell-like): align all partials to max exponent, truncate to F fractional bits
for F in (8, 12, 16, 20, 25, 35):
    emax = (np.floor(np.log2(np.abs(part) + 1e-30)) + pexp).max(-1, keepdims=True)
    terms = part * 2.0 ** pexp
    trunc = np.trunc(terms / 2.0 ** (emax - F)) * 2.0 ** (emax - F)
    s = trunc.sum(-1)
    aerr = np.sqrt(np.mean((s - exact) ** 2) / np.mean(exact ** 2))
    print(f"  fused aligned-to-max, F={F:2d} frac bits (RZ): rel err={aerr:.2e} ratio={aerr/qerr:.4f}")

# 24-bit fixed-point accumulator with static per-layer binary point (blueprint): choose LSB so that
# max |exact| over calibration fits in 23 bits magnitude
for bits in (24, 32):
    lsb = 2.0 ** (np.ceil(np.log2(np.abs(exact).max())) - (bits - 1))
    acc = np.trunc((part * 2.0 ** pexp) / lsb).sum(-1) * lsb
    aerr = np.sqrt(np.mean((acc - exact) ** 2) / np.mean(exact ** 2))
    print(f"  {bits}-bit fixed acc, static LSB=2^{np.log2(lsb):.0f} (truncate each partial): rel err={aerr:.2e} ratio={aerr/qerr:.4f}")
