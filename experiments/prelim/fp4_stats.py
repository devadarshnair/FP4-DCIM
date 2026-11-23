"""Synthetic E2M1 code statistics + accumulator range math for MXFP4 / NVFP4.
DERIVED numbers for report 02-formats-quantization.md. Not real-model data."""
import numpy as np
rng = np.random.default_rng(0)

GRID = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])

def rtne_fp4(x):
    """round |x| to E2M1 grid with ties-to-even (on code LSB), saturate at 6"""
    a = np.minimum(np.abs(x), 6.0)
    idx = np.searchsorted(GRID, a)            # GRID[idx-1] < a <= GRID[idx]
    idx = np.clip(idx, 1, 7)
    lo, hi = GRID[idx - 1], GRID[idx]
    mid = (lo + hi) / 2
    pick_hi = (a > mid) | ((a == mid) & ((idx % 2) == 0))  # even code index
    code = np.where(pick_hi, idx, idx - 1)
    code = np.where(a == 0, 0, code)
    return np.sign(x) * GRID[code], code

def mx_quant(blocks, mode):
    amax = np.abs(blocks).max(axis=1, keepdims=True)
    amax = np.where(amax == 0, 1e-30, amax)
    if mode == "floor":      # OCP v1.0
        e = np.floor(np.log2(amax)) - 2
    elif mode == "rceil":    # NVIDIA/ModelOpt: ceil(log2(amax/6))
        e = np.ceil(np.log2(amax / 6.0))
    s = 2.0 ** e
    q, code = rtne_fp4(blocks / s)
    sat = (np.abs(blocks / s) > 6.0).mean()
    return q * s, code, sat

def e4m3_round(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.floor(np.log2(np.maximum(x, 2.0 ** -9)))
    e = np.maximum(e, -6)                     # subnormal spacing 2^-9
    step = 2.0 ** (e - 3)
    return np.minimum(np.round(x / step) * step, 448.0)

def nv_quant(blocks, gamax):
    s_enc = 6 * 448 / gamax
    amax = np.abs(blocks).max(axis=1, keepdims=True)
    sdec_b = e4m3_round(amax / 6 * s_enc)
    sdec_b = np.where(sdec_b == 0, 2.0 ** -9, sdec_b)
    scale = sdec_b / s_enc
    q, code = rtne_fp4(blocks / scale)
    return q * scale, code

def stats(name, x, blk):
    blocks = x.reshape(-1, blk)
    res = {}
    if blk == 32:
        for m in ["floor", "rceil"]:
            dq, code, sat = mx_quant(blocks, m)
            res[f"MX-{m}"] = (dq, code, sat)
    else:
        dq, code = nv_quant(blocks, np.abs(x).max())
        res["NVFP4"] = (dq, code, 0.0)
    for k, (dq, code, sat) in res.items():
        hist = np.bincount(code.ravel(), minlength=8) / code.size
        qsnr = 10 * np.log10((blocks ** 2).sum() / ((blocks - dq) ** 2).sum())
        nz_blk = (code == 0).mean(axis=1)
        print(f"{name:10s} blk={blk:2d} {k:9s} QSNR={qsnr:5.2f}dB zero={hist[0]*100:5.1f}% sat(pre-clip)={sat*100:4.1f}% "
              f"hist(|code| 0,.5,1,1.5,2,3,4,6)%=" + " ".join(f"{h*100:4.1f}" for h in hist))
    return res

N = 32 * 200000
dists = {
    "gauss": rng.standard_normal(N),
    "laplace": rng.laplace(size=N),
    "student3": rng.standard_t(3, size=N),
}
for n, x in dists.items():
    for blk in (32, 16):
        stats(n, x, blk)

# product-level stats: P(product == 0) for independent W,A codes
print()
for n, x in dists.items():
    _, cw, _ = mx_quant(x.reshape(-1, 32), "rceil")
    y = rng.permutation(x)
    _, ca, _ = mx_quant(y.reshape(-1, 32), "rceil")
    pz = ((cw == 0) | (ca == 0)).mean()
    print(f"{n}: P(W=0)={np.mean(cw==0):.3f} P(A=0)={np.mean(ca==0):.3f} P(prod=0)={pz:.3f}")

# Accumulator range math (exact): integer product of (2*|w|)*(2*|a|) in {0..144}
m2 = (2 * GRID).astype(int)          # 0 1 2 3 4 6 8 12
prods = sorted(set(int(a * b) for a in m2 for b in m2))
print("\nDistinct |4*w*a| products:", prods, "count", len(prods))
for K in (16, 32):
    mx = 144 * K
    print(f"K={K}: max |sum| (units of 1/4) = {mx}, unsigned bits={int(np.ceil(np.log2(mx+1)))}, "
          f"signed 2's comp bits={int(np.ceil(np.log2(mx+1)))+1}")
# fractional resolution: smallest nonzero product = 0.5*0.5=0.25 -> 2 fractional bits in real units
