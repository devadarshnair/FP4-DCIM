# Gate-level (zero-delay) switching-activity model of column adder trees.
# Compares: (TCN) 2's complement, naive sign extension to full width
#           (TC)  2's complement 9b with sign-extension elimination (invert MSB + constant)
#           (OC)  1's-complement row output (m XOR s) + s as LSB carry-in bits, with SEE
#           (SPL) dual-polarity: two unsigned 8b Dadda trees + final subtractor
#           (SPLs) shared-tree sign-split variant is not possible; omitted
# Weight-stationary: each column has fixed weights; activations change every cycle.
import numpy as np, sys
from quant import *
DADDA=[2,3,4,6,9,13,19,28,42,63]
class Net:
    def __init__(s): s.tog=0; s.nfa=0; s.nha=0; s.fatog=0; s.cells=[]
    def t(s,x): return int(np.count_nonzero(x[1:]!=x[:-1])) if x.ndim==1 else 0
def dadda(cols, net):
    """cols: list (LSB first) of lists of bool arrays (or python bool constants). reduce to height 2."""
    cols=[list(c) for c in cols]
    def h(): return max(len(c) for c in cols)
    ds=[d for d in DADDA if d < h()]
    for d in reversed(ds):
        i=0
        while i < len(cols):
            while len(cols[i]) > d:
                if i+1==len(cols): cols.append([])
                if len(cols[i])==d+1:
                    a,b=cols[i].pop(0),cols[i].pop(0)
                    sm=a^b; co=a&b; net.nha+=1
                    net.fatog+= 0.5*(net.t(sm)+net.t(co))
                else:
                    a,b,c=cols[i].pop(0),cols[i].pop(0),cols[i].pop(0)
                    sm=a^b^c; co=(a&b)|(a&c)|(b&c); net.nfa+=1
                    net.fatog+= net.t(sm)+net.t(co)
                cols[i].append(sm); cols[i+1].append(co)
            i+=1
    return cols
def cpa(cols, net, width):
    """ripple-carry add of the two remaining rows, mod 2^width; count toggles of sum & carry nodes"""
    T=None
    for c in cols:
        for x in c:
            if isinstance(x,np.ndarray): T=len(x); break
        if T: break
    z=np.zeros(T,bool); carry=z; out=[]
    for i in range(width):
        c=cols[i] if i < len(cols) else []
        c=[x if isinstance(x,np.ndarray) else np.full(T,bool(x)) for x in c]+[z,z]
        a,b=c[0],c[1]
        s=a^b^carry; co=(a&b)|(a&carry)|(b&carry)
        net.fatog+= net.t(s)+net.t(co); net.nfa+=1
        out.append(s); carry=co
    return out
def bits(v, n):  # v int array (T,) -> list of n bool arrays (2's complement)
    return [((v>>i)&1).astype(bool) for i in range(n)]
def bus_tog(blist): return sum(int(np.count_nonzero(b[1:]!=b[:-1])) for b in blist)

def run(P, scheme, N=32):
    """P: (T,N) int signed products*4 for one column over time"""
    T=P.shape[0]; net=Net(); bus=0
    m=np.abs(P); s=(P<0)
    if scheme=="TCN":  # sign-extend to 14 bits
        W=14; cols=[[] for _ in range(W)]
        for r in range(N):
            b=bits(P[:,r] & ((1<<W)-1), W); bus+=bus_tog(b[:9])  # physical bus 9b; extension = fanout wires
            for i in range(W): cols[i].append(b[i])
        cols=dadda(cols,net); out=cpa(cols,net,W)
    elif scheme in ("TC","OC"):
        W=14; cols=[[] for _ in range(W)]
        for r in range(N):
            if scheme=="TC": b=bits(P[:,r] & 0x1FF, 9)
            else:            b=bits(np.where(s[:,r], (~m[:,r])&0x1FF, m[:,r]), 9)
            bus+=bus_tog(b)
            b[8]=~b[8]   # sign-extension elimination: invert MSB, add constant
            for i in range(9): cols[i].append(b[i])
            if scheme=="OC": cols[0].append(s[:,r]); bus+=bus_tog([s[:,r]])
        # constant = -N*2^8 mod 2^W ; add its bits as constant inputs
        const=(-N*256) & ((1<<W)-1)
        for i in range(W):
            if (const>>i)&1: cols[i].append(np.ones(T,bool))
        cols=dadda(cols,net); out=cpa(cols,net,W)
    elif scheme=="SPL":
        W=13; tot=[]
        for pol in (False,True):
            cols=[[] for _ in range(W)]
            for r in range(N):
                v=np.where(s[:,r]==pol, m[:,r], 0)
                b=bits(v,8); bus+=bus_tog(b)
                for i in range(8): cols[i].append(b[i])
            cols=dadda(cols,net); tot.append(cpa(cols,net,W))
        # final subtract pos - neg  (14b): pos + ~neg + 1
        posb=tot[0]+[np.zeros(T,bool)]; negb=[~x for x in tot[1]]+[np.ones(T,bool)]
        cols=[[posb[i],negb[i]] for i in range(14)]; cols[0].append(np.ones(T,bool))
        cols=dadda(cols,net); out=cpa(cols,net,14)
    # verify
    W=len(out); val=sum((o.astype(np.int64)<<i) for i,o in enumerate(out))
    val=np.where(val>=(1<<(W-1)), val-(1<<W), val)
    assert np.array_equal(val, P.sum(1)), scheme
    return bus, net.fatog, net.nfa, net.nha

def products(dist, fmt, T, ncols, rng, N):
    qf = mx_quant if fmt=="MXFP4" else nv_quant
    kb = 32 if fmt=="MXFP4" else 16
    # activations: T tokens x K channels; weights: K x ncols (gaussian), quantize along K
    K=N
    Kfull=max(N,kb)*8
    x=gen(dist,(T,Kfull),rng); xq,_=qf(x)
    w=rng.standard_normal((ncols,Kfull)); wq,_=qf(w)
    xq=xq[:,:N]; wq=wq[:,:N]
    Xi=(xq*2).astype(np.int64); Wi=(wq*2).astype(np.int64)
    return [(Xi*Wi[c][None,:]) for c in range(ncols)]   # list of (T,N)

if __name__=="__main__":
    rng=np.random.default_rng(7)
    T=6000; ncols=6
    print("per-column, per-cycle averages (32-row column).  FA-eq toggles = FA sum+carry output toggles (HA weighted 0.5)")
    print(f"{'dist':8s} {'fmt':6s} {'N':>3s} | {'scheme':5s} {'rowbus tog':>10s} {'tree tog':>9s} {'total':>8s} {'vs TC':>7s} | FA  HA")
    for fmt,N in (("MXFP4",32),("NVFP4",16)):
      for d in ["gauss","laplace","t3","llm","relu"]:
        Ps=products(d,fmt,T,ncols,rng,N)
        resd={}
        for sch in ("TCN","TC","OC","SPL"):
            acc=np.zeros(4)
            for P in Ps: acc+=np.array(run(P,sch,N))
            bus,ft,nfa,nha=acc/ncols; resd[sch]=(bus/(T-1),ft/(T-1),nfa,nha)
        base=resd["TC"][0]+resd["TC"][1]
        for sch,(b,f,nfa,nha) in resd.items():
            print(f"{d:8s} {fmt:6s} {N:3d} | {sch:5s} {b:10.1f} {f:9.1f} {b+f:8.1f} {100*((b+f)/base-1):+6.1f}% | {nfa:.0f} {nha:.0f}")
        sys.stdout.flush()
