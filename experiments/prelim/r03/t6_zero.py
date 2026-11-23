import numpy as np
from quant import *
rng=np.random.default_rng(1)
K=4096; T=256
print(f"{'dist':8s} {'fmt':6s} {'P(elem=0)':>9s} {'P(|e|<=0.5)':>11s}  code histogram |e| (0,.5,1,1.5,2,3,4,6)")
res={}
for d in ["gauss","laplace","t3","llm","relu"]:
    x=gen(d,(T,K),rng)
    for fmt,qf in (("MXFP4",mx_quant),("NVFP4",nv_quant)):
        q,_=qf(x)
        a=np.abs(q).ravel()
        h=[np.mean(a==g) for g in E2M1_GRID]
        res[(d,fmt)]=h[0]
        print(f"{d:8s} {fmt:6s} {h[0]:9.3f} {h[0]+h[1]:11.3f}  "+" ".join(f"{v:.3f}" for v in h))
print()
print("product-zero probability (weights gaussian, activations dist), P = 1-(1-pw)(1-px):")
for fmt in ("MXFP4","NVFP4"):
    pw=res[("gauss",fmt)]
    for d in ["gauss","laplace","t3","llm","relu"]:
        px=res[(d,fmt)]
        print(f"  {fmt} act={d:8s}: pw={pw:.3f} px={px:.3f} -> P(prod=0)={1-(1-pw)*(1-px):.3f}")
