import math
# ---------------- model configs ----------------
# (name, layers, d, n_heads, n_kv, head_dim, ffn, ffn_type, vocab, tied)
M = {
 'GPT2-124M':  (12, 768, 12,12, 64, 3072,'gelu',50257,True),
 'GPT2-L-774M':(36,1280, 20,20, 64, 5120,'gelu',50257,True),
 'Qwen2.5-0.5B':(24,896, 14,2, 64, 4864,'swiglu',151936,True),
 'Llama3.2-1B':(16,2048, 32,8, 64, 8192,'swiglu',128256,True),
 'Qwen2.5-1.5B':(28,1536,12,2,128, 8960,'swiglu',151936,True),
 'Llama3.2-3B':(28,3072, 24,8,128, 8192,'swiglu',128256,True),
 'SmolLM2-135M':(30,576, 9,3, 64, 1536,'swiglu',49152,True),
 'DeiT-Tiny':  (12, 192, 3,3, 64, 768,'gelu',0,True),
 'DeiT-Small': (12, 384, 6,6, 64, 1536,'gelu',0,True),
 'ViT-B/16':   (12, 768, 12,12,64, 3072,'gelu',0,True),
}
def lin_params(c):
    L,d,h,kv,hd,f,ft,V,tied=c
    attn = d*h*hd + 2*d*kv*hd + h*hd*d
    mlp = (3 if ft=='swiglu' else 2)*d*f
    return L*(attn+mlp), V*d
print("== Params and weight footprint ==")
print(f"{'model':14s} {'linear(M)':>9s} {'embed/head(M)':>13s} {'MXFP4 MB(all lin+head)':>22s} {'NVFP4 MB':>9s}")
for n,c in M.items():
    lp,ep=lin_params(c)
    tot=lp+ep
    print(f"{n:14s} {lp/1e6:9.1f} {ep/1e6:13.1f} {tot*4.25/8/1e6:22.1f} {tot*4.5/8/1e6:9.1f}")

# ---------------- capacity: area to hold all weights on chip ----------------
print("\n== On-chip area to hold all linear+head weights (MXFP4, 4.25b) ==")
dens = {'MIDAS 28nm CIM (16Kb/0.062mm2)':16384/0.062, 'TSMC 3nm DCIM ISSCC24 (3.78Mb/mm2)':3.78e6,
        'Ours 28nm (assume 4Kb/0.03mm2)':4096/0.03}
for n in ['SmolLM2-135M','GPT2-124M','Qwen2.5-0.5B','Llama3.2-1B','DeiT-Tiny','ViT-B/16']:
    lp,ep=lin_params(M[n]); bits=(lp+ep)*4.25
    print(n, {k: round(bits/v,1) for k,v in dens.items()}, 'mm2')

# ---------------- Decode roofline ----------------
c=M['Llama3.2-1B']; lp,ep=lin_params(c)
W=(lp+ep)            # weights touched per decode token (LM head included)
Wb=W*4.25/8
flop_tok=2*W
print(f"\n== Llama-3.2-1B decode: weights touched/token={W/1e9:.3f} G, bytes={Wb/1e9:.3f} GB, FLOP/token={flop_tok/1e9:.2f} G, AI(B=1)={flop_tok/Wb:.2f} FLOP/B")
for name,bw in [('LPDDR5-6400 x32',25.6e9),('LPDDR5-6400 x64',51.2e9),('Jetson AGX Orin 256b',204.8e9),('Jetson AGX Thor',273e9)]:
    print(f"  {name:22s} BW={bw/1e9:6.1f} GB/s -> B=1 upper bound {bw/Wb:6.1f} tok/s, needs only {bw/Wb*flop_tok/1e12:.3f} TFLOPS")

# KV cache bytes per token-position (all layers)
L,d,h,kv,hd,f,ft,V,_=c
kv_pos = L*2*kv*hd  # elements
print(f"  KV elems per context position = {kv_pos}  ({kv_pos/1024:.0f} KB @8b, {kv_pos*2/1024:.0f} KB @16b)")
attn_flop_per_ctx = L*2*2*h*hd   # QK and PV per decode token per context position
print(f"  attention FLOPs per decode token per context pos = {attn_flop_per_ctx}; at ctx 2048: {attn_flop_per_ctx*2048/1e6:.0f} MFLOP ({attn_flop_per_ctx*2048/flop_tok*100:.1f}% of linear)")

# ---------------- Energy per token vs reuse R ----------------
print("\n== Energy per weight-use vs reuse R (per 4.25b weight; FLOP=2) ==")
bits=4.25
E_dram_pJb = {'LPDDR5 4pJ/b':4.0,'LPDDR5 6.6pJ/b':6.6}
E_buf=0.3   # pJ/bit GMEM write+read staging (28nm, UNVERIFIED param)
E_wr=0.08   # pJ/bit CIM array write (UNVERIFIED param)
E_flop_sys=1/28.2e12*1e12  # pJ/FLOP, MIDAS system (excl DRAM) = 0.0355
E_flop_mac=1/96.3e12*1e12  # pJ/FLOP MIDAS macro only
f_macro=0.218
print(f"  MIDAS: macro {E_flop_mac*1000:.1f} fJ/FLOP, system {E_flop_sys*1000:.1f} fJ/FLOP")
for dn,ed in E_dram_pJb.items():
    print(f" DRAM={dn}")
    Eload=bits*(ed+E_buf+E_wr)
    print(f"   E_load per weight = {Eload:.2f} pJ ; on-chip compute per use = {2*E_flop_sys:.3f} pJ ; ratio = {Eload/(2*E_flop_sys):.0f}x")
    for R in [1,4,8,16,32,64,128,256,512,1024,2048,8192]:
        Ecim = Eload/R + 2*E_flop_sys
        # baseline digital: macro part k x worse, periphery same
        out=[]
        for k in [2,4,9.4]:
            Eon_base = 2*E_flop_sys*((1-f_macro)+f_macro*k)
            Eb = Eload/R + Eon_base
            out.append(Eb/Ecim)
        print(f"   R={R:5d}: eff TFLOPS/W incl DRAM = {2/Ecim:6.2f}; DRAM+load share = {Eload/R/Ecim*100:5.1f}%;  CIM gain vs digital (k=2,4,9.4): " + ", ".join(f"{x:.2f}x" for x in out))

# ---------------- macro utilisation w/ reload ----------------
print("\n== Macro (32x32 FP4) utilisation vs reuse R, write port width ==")
wbits=1024*4+32*8   # weights + E8M0 scales
for wp in [128,256,512]:
    Tr=math.ceil(wbits/wp)
    row=[]
    for R in [1,4,16,32,64,128]:
        nopp=R/(R+Tr); pp=min(1,R/Tr)
        row.append(f"R={R}:{nopp*100:.0f}%/{pp*100:.0f}%")
    print(f"  port {wp}b: reload {Tr} cyc | no-PP/PP util: "+"  ".join(row))

# ---------------- array sizing ----------------
print("\n== N macros @f: peak, ridge R, and BW to stay compute-bound ==")
for N in [8,16,32,64]:
    for fclk in [250e6,500e6]:
        peak=N*2048*fclk
        for bw in [25.6e9,51.2e9]:
            ridge_AI=peak/bw; R=ridge_AI/(2/(bits/8))
            print(f"  N={N:2d} f={fclk/1e6:.0f}MHz peak={peak/1e12:5.2f} TFLOPS  BW={bw/1e9:4.1f}GB/s -> ridge R={R:6.0f} tokens")
# MIDAS
print(f"\n MIDAS per-macro throughput: {0.69e12*0.062/1e9:.1f} GFLOPS @250MHz = {0.69e12*0.062/250e6:.0f} FLOP/cycle; 24 macros = {24*0.69e12*0.062/1e12:.2f} TFLOPS; weights on chip {24*16384/8/1024:.0f} KB")
print(f" Our macro: 2048 FLOP/cycle with 1024 stored weights -> compute/storage {2048/1024:.1f} FLOP/cyc/weight vs MIDAS {0.69e12*0.062/250e6/4096:.3f}")

# ---------------- prefill ----------------
print("\n== Llama-3.2-1B prefill (linear only, LM head only last token) + attention ==")
for Lp in [128,512,2048]:
    fl = 2*lp*Lp + 2*ep + L*2*h*hd*Lp*Lp   # causal QK+PV ~ 2*L^2*d_attn per layer (2 FLOP * L^2/2 * 2 ops)
    for N,fclk in [(16,500e6),(64,500e6)]:
        peak=N*2048*fclk
        t_c=fl/peak; t_m=(lp+ep)*bits/8/51.2e9
        print(f"  L={Lp:5d} FLOP={fl/1e12:6.2f} T ; N={N} -> compute {t_c*1e3:7.1f} ms vs weight-stream {t_m*1e3:5.1f} ms (if R=L)")

# ---------------- GMEM sizing for reuse R=T tokens ----------------
print("\n== GMEM bytes per token resident (Llama-3.2-1B FFN step) ==")
x_in = d*4.25/8; h_mid = f*4.25/8; resid = d*2
per_tok = x_in + h_mid + resid + d*4   # + FP32 out tile partials (upper bound)
print(f"  ~{per_tok/1024:.1f} KB/token -> T(R) achievable: 512KB:{512*1024/per_tok:.0f}, 1MB:{1024*1024/per_tok:.0f}, 2MB:{2*1024*1024/per_tok:.0f}")

# ---------------- batched decode: KV share ----------------
print("\n== Batched decode traffic per step (Llama-3.2-1B, ctx 1024, KV 8b) ==")
for B in [1,4,16,64,256]:
    wbytes=Wb; kvb=B*kv_pos*1024
    print(f"  B={B:3d}: weights {wbytes/1e6:6.0f} MB, KV {kvb/1e6:7.1f} MB -> KV share {kvb/(wbytes+kvb)*100:4.1f}%  weight-reuse R={B}")

# ---------------- tokens/J decode ----------------
print("\n== Decode energy/token Llama-3.2-1B B=1 ==")
for ed in [4.0,6.6]:
    e = Wb*8*(ed+E_buf+E_wr) + flop_tok*E_flop_sys
    e = e*1e-12
    print(f"  DRAM {ed} pJ/b: {e*1e3:.1f} mJ/token -> {1/e:.0f} tok/J (compute part {flop_tok*E_flop_sys*1e-12*1e6:.0f} uJ)")
