# Per-MAC switching estimate: multiplier netlist (gate-level, from ABC) + row bus + tree (from t4_toggle)
import numpy as np
from quant import *
from blifsim import load, sim
import t4_toggle as TT
rng=np.random.default_rng(3)
def m2(c): return np.array([0,1,2,3,4,6,8,12])[c]
def codes(q):  # signed e2m1 value -> (s, magcode)
    mag=np.abs(q)*2; lut={0:0,1:1,2:2,3:3,4:4,6:5,8:6,12:7}
    mc=np.vectorize(lut.get)(mag.astype(int)); return (q<0), mc
V2=load("v2_static.blif"); V3=load("v3_split.blif"); V9=load("v9_tcbcast.blif"); V4=None
import subprocess
def bitsd(name,val,n): return {f"{name}[{i}]":((val>>i)&1).astype(bool) for i in range(n)}
T=4000; N=32; ncols=4
print(f"{'dist':8s} | mult gate-node toggles per MAC: {'v3 split':>9s} {'v9 TCbc':>8s} | bcast-wire toggles/row-cycle: v3(11w) v9(24w)")
for d in ["gauss","llm","relu"]:
    x=gen(d,(T,256),rng); xq,_=mx_quant(x); w=rng.standard_normal((ncols,256)); wq,_=mx_quant(w)
    xs,xc=codes(xq[:,:N]); ws,wc=codes(wq[:,:N])
    tot3=tot9=0; bw3=bw9=0
    for r in range(N):
        Xp=m2(xc[:,r]); X3=3*Xp; sx=xs[:,r]
        Xs=np.where(sx,-Xp,Xp); P1=Xs&31; N1=(-Xs)&31; P3=(3*Xs)&127; N3=(-3*Xs)&127
        b3={**bitsd("X",Xp,4),**bitsd("X3",X3,6),"sx":sx}
        b9={**bitsd("P1",P1,5),**bitsd("N1",N1,5),**bitsd("P3",P3,7),**bitsd("N3",N3,7)}
        bw3+=sum(np.count_nonzero(a[1:]!=a[:-1]) for a in b3.values())
        bw9+=sum(np.count_nonzero(a[1:]!=a[:-1]) for a in b9.values())
        for c in range(ncols):
            mc=wc[c,r]; s=ws[c,r]; e=(mc>>1)&3; m=mc&1; h=int(e!=0)
            awo=[int(e<2), int(e==2), int(e==3)]
            st=lambda v: np.full(T,bool(v))
            A1,A2,A3=(not h) and m, h and not m, h and m
            i3={**b3,"s1":st(A1),"s2":st(A2),"s3":st(A3),"sw":st(s),**{f"awo[{k}]":st(awo[k]) for k in range(3)}}
            _,t3=sim(V3,i3); tot3+=t3
            i9={**b9,"sp1":st(A1 and not s),"sn1":st(A1 and s),"sp2":st(A2 and not s),"sn2":st(A2 and s),"sp3":st(A3 and not s),"sn3":st(A3 and s),**{f"awo[{k}]":st(awo[k]) for k in range(3)}}
            _,t9=sim(V9,i9); tot9+=t9
    n=(T-1)*N*ncols
    print(f"{d:8s} | {'':31s} {tot3/n:9.2f} {tot9/n:8.2f} | {bw3/(T-1)/N:.2f}  {bw9/(T-1)/N:.2f}  (per row per cycle, per wire-row)")
