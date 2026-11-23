import numpy as np
E2M1_GRID = np.array([0,0.5,1,1.5,2,3,4,6])
def round_e2m1(y):
    """round |y| to nearest E2M1 magnitude, ties-to-even-mantissa, saturate at 6. returns signed values"""
    a = np.minimum(np.abs(y), 6.0)
    # candidate indices
    idx = np.searchsorted(E2M1_GRID, a)            # first grid >= a
    idx = np.clip(idx,1,7)
    lo = E2M1_GRID[idx-1]; hi = E2M1_GRID[idx]
    dlo = a-lo; dhi = hi-a
    # tie -> even code (code index even => mantissa bit 0)
    pick_hi = (dhi < dlo) | ((dhi==dlo) & (idx%2==0))
    q = np.where(pick_hi, hi, lo)
    return np.sign(y)*q
def mx_quant(x, k=32):
    """OCP MX: x shape (...,K); blocks along last axis. returns (elements, scale_exp)"""
    s = x.reshape(*x.shape[:-1], -1, k)
    amax = np.abs(s).max(-1, keepdims=True)
    amax = np.where(amax==0, 1e-30, amax)
    e = np.floor(np.log2(amax)) - 2      # emax_elem for E2M1 = 2
    q = round_e2m1(s / 2.0**e)
    return q.reshape(x.shape), e.squeeze(-1)
def e4m3_round(v):
    v = np.asarray(v, dtype=np.float64)
    e = np.floor(np.log2(np.maximum(v,2**-9)))
    e = np.clip(e, -6, 8)
    m = np.round(v/2.0**e*8)/8
    return np.minimum(m*2.0**e, 448.0)
def nv_quant(x, k=16):
    s = x.reshape(*x.shape[:-1], -1, k)
    amax = np.abs(s).max(-1, keepdims=True)
    sc = e4m3_round(np.where(amax==0,1e-30,amax)/6.0)
    sc = np.where(sc==0, 2**-9, sc)
    q = round_e2m1(s / sc)
    return q.reshape(x.shape), sc.squeeze(-1)
def gen(dist, shape, rng, nchan=None):
    if dist=="gauss": return rng.standard_normal(shape)
    if dist=="laplace": return rng.laplace(size=shape)
    if dist=="t3": return rng.standard_t(3, size=shape)
    if dist=="llm":   # heavy-tailed + persistent outlier channels (last axis = channel)
        x = rng.standard_t(4, size=shape)
        C = shape[-1]; ch = np.ones(C); o = rng.choice(C, max(1,C//100), replace=False); ch[o]=30.0
        ch *= np.exp(0.5*rng.standard_normal(C))   # per-channel magnitude spread
        return x*ch
    if dist=="relu": return np.maximum(rng.standard_normal(shape),0)  # e.g. post-ReLU/GELU-ish one-sided
    raise ValueError(dist)
