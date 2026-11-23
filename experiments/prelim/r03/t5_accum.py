# Inter-block accumulation: exact vs FP32 vs anchored fixed-point window (W bits)
import numpy as np
from quant import *
rng=np.random.default_rng(11)
K=4096; NB=K//32; NDOT=300
def blocks_mx(x,w):
    xq,xe=mx_quant(x); wq,we=mx_quant(w)
    P=(xq*2).astype(np.int64)*(wq*2).astype(np.int64)        # products *4
    S=P.reshape(NB,32).sum(1)                                  # exact 14b block sums
    E=(xe+we-2).astype(np.int64)                               # value = S*2^E
    return S,E,xe.astype(int),we.astype(int)
def exact(S,E):
    m=E.min(); return sum(int(s)<<int(e-m) for s,e in zip(S,E)), int(m)
def to_float(N,m): return float(N)*2.0**m if N!=0 else 0.0
res={}
Ws=[16,20,24,28,32,40]
errs={k:[] for k in ["fp32"]+[f"W{w}" for w in Ws]+["bf16out"]}
spread=[]; ranges=[]
for t in range(NDOT):
    x=gen("llm",(1,K),rng)[0]*np.exp(rng.normal(0,2)); w=rng.standard_normal(K)*0.02*np.exp(rng.normal(0,0.5,size=K//256).repeat(256))
    S,E,xe,we=blocks_mx(x[None],w[None])
    S=S.ravel(); E=E.ravel()
    N,m=exact(S,E); ref=to_float(N,m)
    if ref==0: continue
    scale=np.sum(np.abs(S)*2.0**E)   # sum |block| for condition-normalised error
    spread.append(E.max()-E.min()); ranges.append((E.min(),E.max()))
    # FP32 sequential
    acc=np.float32(0)
    for s,e in zip(S,E): acc=np.float32(acc+np.float32(np.ldexp(float(s),int(e))))
    errs["fp32"].append(abs(float(acc)-ref)/scale)
    # anchored window: top anchor from max_ex+max_we known before compute (+14b block + 7b guard for 128 blocks)
    Etop=xe.max()+we.max()-2+14+7
    for W in Ws:
        A=Etop-(W-1)
        acc=0
        for s,e in zip(S,E):
            sh=int(e-A)
            acc+= (int(s)<<sh) if sh>=0 else (int(s)>>(-sh))   # truncation (floor) on right shift
        assert abs(acc) < 2**(W-1)
        errs[f"W{W}"].append(abs(acc*2.0**A-ref)/scale)
    bf=np.float32(ref).view(np.uint32); bf=((int(bf)+0x7FFF+((int(bf)>>16)&1))>>16)<<16
    errs["bf16out"].append(abs(float(np.uint32(bf).view(np.float32))-ref)/scale)
print(f"dot products: {len(spread)}, K={K}, blocks={NB}; block-exponent spread within a dot product: median {np.median(spread):.0f}, max {np.max(spread)}")
print("block exponent (E_w+E_x-2, unbiased) range over all dots:",min(r[0] for r in ranges),"..",max(r[1] for r in ranges))
print("error normalised by sum|block contributions| (median / 99th pct / max):")
for k,v in errs.items():
    v=np.array(v); print(f"  {k:8s} {np.median(v):.2e} {np.percentile(v,99):.2e} {v.max():.2e}")
