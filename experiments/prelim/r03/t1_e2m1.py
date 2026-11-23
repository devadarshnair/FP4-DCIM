# Task 1: exhaustive E2M1 product enumeration and width derivation
from fractions import Fraction as F
from collections import Counter
import math

def e2m1(code):
    s=(code>>3)&1; e=(code>>1)&3; m=code&1
    mag = F(m,2) if e==0 else F(2+m,2)*2**(e-1)
    return (-1)**s*mag

vals=[e2m1(c) for c in range(16)]
print("E2M1 values:",[str(v) for v in vals])
print("2*|v| :", sorted(set(int(abs(v)*2) for v in vals)))
prods=Counter(); allint=True
for w in range(16):
  for x in range(16):
    p=vals[w]*vals[x]*4
    allint &= (p.denominator==1)
    prods[int(p)]+=1
mags=sorted(set(abs(k) for k in prods))
print("all 256 products*4 integer:",allint)
print("distinct |p*4|:",mags, "count",len(mags))
print("max |p*4| =",max(mags),"-> magnitude bits =",max(mags).bit_length())
# {1,3,9}*2^k structure
def odd(n):
    while n and n%2==0: n//=2
    return n
print("odd parts:",sorted(set(odd(m) for m in mags if m)))
print("k range:",sorted(set((m//odd(m)).bit_length()-1 for m in mags if m)))
print("histogram of signed p*4 over 256 code pairs (incl -0):",dict(sorted(prods.items())))
zero_pairs=sum(1 for w in range(16) for x in range(16) if vals[w]*vals[x]==0)
print("code pairs with zero product:",zero_pairs,"/256 =",zero_pairs/256)
# significand/shift decomposition: |v|*2 = A<<a, A in{0..3}, a in{0..2}
for c in range(8):
    e=(c>>1)&3; m=c&1; h=int(e!=0); A=2*h+m; a=max(e,1)-1
    assert A<<a == int(abs(vals[c])*2)
    print(f"code {c:03b}: |v|*2={int(abs(vals[c])*2):2d} = A({A:02b}) << a({a})")
for N in (16,32,64):
    mx=N*144
    pol=mx.bit_length()
    sgn=next(b for b in range(1,40) if -(1<<(b-1))<=-mx and mx<=(1<<(b-1))-1)
    print(f"N={N}: per-polarity max {mx} -> {pol} bits unsigned; signed sum in [-{mx},{mx}] -> {sgn} bits 2's comp")
# NVFP4 scale: E4M3 (bias 7), significand 1.mmm -> 4-bit integer 8..15 (normal) or 0..7 (subnormal)
sig=set()
for e in range(16):
  for m in range(8):
    if e==15 and m==7: continue # NaN
    sig.add((8+m) if e>0 else m)
print("E4M3 integer significands:",sorted(sig),"-> bits",max(sig).bit_length())
print("scale_w*scale_x significand product max:",15*15,"bits",(15*15).bit_length())
print("16-row signed sum (13b) x 8b scale product -> |max|",2304*225,"bits signed",(2304*225).bit_length()+1)
print("16-row signed sum (13b) x 4b single-scale -> |max|",2304*15,"bits signed",(2304*15).bit_length()+1)
# E8M0 exponent sum range
print("E8M0 unbiased in [-127,127]; sum in [-254,254] -> 9 bits signed; biased sum 0..508 (255=NaN excluded) -> 9 bits unsigned")
