# Bit-serial (TSMC-style) INT DCIM baseline on the same E2M1 data:
# weights stored as 5b 2's comp W''=2*w in [-12,12]; activations X''=2*x fed LSB-first, 5 cycles (MSB weighted -16).
# per cycle, per weight-bit column: 32 one-bit products -> popcount Dadda tree -> 6b; then per bit column shift-accumulate.
import numpy as np
import t4_toggle as TT
from quant import *
rng=np.random.default_rng(5)
T=1500; N=32; ncols=3
def run_dist(d):
    x=gen(d,(T,256),rng); xq,_=mx_quant(x); w=rng.standard_normal((ncols,256)); wq,_=mx_quant(w)
    X=(xq[:,:N]*2).astype(np.int64); Wt=(wq[:,:N]*2).astype(np.int64)
    tot_tree=0; tot_acc=0; tot_and=0; cyc=0
    for c in range(ncols):
        Wb=[((Wt[c]>>b)&1).astype(bool) for b in range(5)]      # static weight bits per row
        # build cycle stream: for each token t, cycles j=0..4 with activation bit j
        xb=np.stack([((X>>j)&1).astype(bool) for j in range(5)],1).reshape(T*5,N)  # (T*5, N)
        for b in range(5):
            pp=xb & Wb[b][None,:]                                   # AND/NOR in bitcell
            tot_and+=int(np.count_nonzero(pp[1:]!=pp[:-1]))
            net=TT.Net(); cols=[[pp[:,r] for r in range(N)]]+[[] for _ in range(6)]
            cols=TT.dadda(cols,net); out=TT.cpa(cols,net,6); tot_tree+=net.fatog
            # accumulator per bit column: acc_j = acc_{j-1} + (+/-)pc<<j ; model 12b ripple adder toggles
            pc=sum(o.astype(np.int64)<<i for i,o in enumerate(out))
            j=np.tile(np.arange(5),T); term=np.where(j==4,-(pc<<4),pc<<j)
            acc=np.zeros(T*5,np.int64); run=0
            accv=[]
            for t in range(T*5):
                run = term[t] if j[t]==0 else run+term[t]; accv.append(run)
            accv=np.array(accv)&0xFFF
            tot_acc+=sum(int(np.count_nonzero(((accv>>i)&1)[1:]!=((accv>>i)&1)[:-1])) for i in range(12))*2  # ~sum+carry
        # check
    macs=T*N*ncols
    return tot_tree/macs, tot_acc/macs, tot_and/macs
for d in ["gauss","llm","relu"]:
    tr,ac,an=run_dist(d)
    print(f"{d:6s}: bit-serial per MAC: popcount-tree FA-out toggles {tr:.2f}, accumulator ~{ac:.2f}, AND-node toggles {an:.2f}  (cycles/MAC-row: 5)")
