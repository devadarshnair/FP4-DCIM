import numpy as np, runpy, os
# Re-use functions from acc_sim by exec of its defs only
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "acc_sim.py")).read().split("K, B = 4096, 32")[0]
g = {}; exec(src, g)
mx, round_sig = g["mx"], g["round_sig"]
rng = np.random.default_rng(2)
K, B, T = 4096, 32, 3000; nb = K // B
sig_ch = 2.0 ** rng.normal(0, 1.0, size=K)
out_ch = rng.choice(K, 6, replace=False); sig_ch[out_ch] *= 300.0     # massive-activation channels
A = rng.standard_normal((T, K)) * sig_ch
# make 2% of tokens have an extra 1000x 'attention-sink' style spike in one channel
spike = rng.random(T) < 0.02; A[spike, out_ch[0]] *= 1000
W = rng.standard_normal((T, K)) * 0.02 * 2.0 ** rng.normal(0, 0.5, size=K)
qa, ea = mx(A.reshape(T, nb, B)); qw, ew = mx(W.reshape(T, nb, B))
part = (qa * qw).sum(-1) * 4; pexp = ea[..., 0] + ew[..., 0] - 2
exact = (part * 2.0 ** pexp).sum(-1); ref = (A * W).sum(-1)
qerr = np.sqrt(np.mean((exact - ref) ** 2) / np.mean(ref ** 2))
nzp = np.where(part != 0, pexp, np.nan)
span = np.nanmax(nzp, -1) - np.nanmin(nzp, -1)
print(f"[outlier model] quant rel err={qerr:.3f}; exp span median={np.median(span):.0f} p99={np.percentile(span,99):.0f} max={np.max(span):.0f}")
print(f"  exact fixed-point width per dot product (14+span+7): p99={14+np.percentile(span,99)+7:.0f}, max={14+np.max(span)+7:.0f}")
# per-layer static window: global min/max exponent over whole layer
print(f"  layer-wide exponent range of nonzero partials: {np.nanmin(nzp):.0f}..{np.nanmax(nzp):.0f} -> static exact width {14+np.nanmax(nzp)-np.nanmin(nzp)+7:.0f} bits")
for p in (8, 10, 11, 12, 14, 16):
    for mode in ("rne", "rz"):
        acc = np.zeros(T)
        for j in range(nb):
            acc = round_sig(acc + part[:, j] * 2.0 ** pexp[:, j], p, mode)
        e = np.sqrt(np.mean((acc - exact) ** 2) / np.mean(exact ** 2))
        print(f"  seq float acc p={p:2d} {mode}: rel err={e:.2e} ({e/qerr*100:.2f}% of quant err)")
for bits in (16, 20, 24, 32):
    lsb = 2.0 ** (np.ceil(np.log2(np.abs(exact).max())) - (bits - 1))
    acc = np.trunc((part * 2.0 ** pexp) / lsb).sum(-1) * lsb
    e = np.sqrt(np.mean((acc - exact) ** 2) / np.mean(exact ** 2))
    med = np.median(np.abs(acc - exact) / (np.abs(exact) + 1e-30))
    print(f"  {bits}b fixed acc, static LSB (no overflow): rms rel err={e:.2e} median per-output rel err={med:.2e}")
