# D1 rounding points - where every rounding happens (frozen 2026-11-06)

Fixes exactly where element rounding, block-scale rounding, NV tensor scaling, inter-block alignment, and
final output rounding occur. Complements `numerics.md` (datapath widths). Scope: standard 1D K-block scaling only.

| # | Stage | MXFP4 | NVFP4 | Exact? | Where implemented / verified |
|---|---|---|---|---|---|
| R1 | Tensor scale | - | s_dec = fp32(amax_tensor / 2688), s_enc = fp32(1 / s_dec) (one fp32 rounding each) | rounds | `formats/nv_quant.py`; weights offline, activations per tensor in the quantizer |
| R2 | Block scale | UE8M0 2^X, X = ⌊log₂ amax⌋ - 2 + adj (OCP v1.1 expAdj at 1.m ≥ 1.75) - integer, no mantissa rounding | E4M3_RNE_satfinite(amax_b · c), c = fp32(s_enc / 6) ("hw": one rounding of the exact product) | MX exact; NV one RNE | `mx_quant`, `nv_quant`, `act_quant_{mx,nv}.sv` (bit-exact, 10⁶ blocks) |
| R3 | Element | E2M1 = RNE(v / 2^X), saturate at 6 | E2M1 = RNE(v / (scale · s_dec)) evaluated exactly in float64, saturate at 6 | one RNE (the only lossy step of quantization) | same files; torch `torch_quant.py` bitwise equal |
| R4 | Products and intra-block sum | (2x)(2w) integers, Σ over 32 -> 14-bit exact | two 16-element half sums (12-bit exact) × scale significands u_x·u_w | **exact** | `leaf_*`, generated trees, `column.sv`; L1/L2 golden |
| R5 | Inter-block alignment | shift to a W = 40 window anchored at max e_w + max e_x of the Kt = 512 tile; bits below the LSB truncated (floor) | same window, NV exponents k_x + k_w, int_bits 21 | **exact on all real data** (e06b: MX exact for W ≥ 36 at every Kt; NV no truncated row at W ≥ 40, Kt ≤ 512); overflow is asserted, never wraps | `column.sv` window accumulator; `golden/accum.py` HIER; `torch_accum.py` |
| R6 | Tile reduction | each tile's integer -> fp32 (RNE), tiles summed sequentially in fp32 | same | rounds (≤ 1 ulp per tile; NV max 5.0e-6 × rms, 0.0085 % of outputs differ from exact) | HIER `reducer="fp32"` (outside the macro: cluster reducer) |
| R7 | NV tensor post-scale | - | y · fp32(s_dec,x · s_dec,w) in fp32 | one rounding | system post-processing (outside the macro) |
| R8 | Layer output | fp32 -> bf16 (RNE) at the Linear output; bias added in fp32 before | same | rounds | `llm/cim_linear.py` |
| - | Attention BMMs, lm_head | MXFP8 on the digital attention engine; not on the macro | same | - | `llm/attn_quant.py` |

## Reference paths and what they are allowed to claim

- **L1 Exact / HIER (GPU `torch_accum.py`, numpy `golden`)** implement R1-R8 exactly as above and are
  order-independent (exact integer sums; float64 GEMM tiles used only where a per-row bound proves no
  rounding). They are the canonical reference for the hardware contract and are bitwise portable.
- **fp32 fake-quant (`CIMLinear` backend `fakequant`)** multiplies dequantized operands in an fp32 GEMM. For
  MXFP4 this is exact (power-of-two scales; 100 % of real outputs equal exact, any summation order). For
  NVFP4 it adds an extra, order-dependent rounding inside the GEMM (only 13.8 % of real outputs equal exact;
  max error 1.5e-5 × rms, 3× the hardware path's). NVFP4 fake-quant numbers are therefore a fast
  approximation of the contract, reproducible bitwise only on the same GPU.

## Accuracy consequence (measured)

Full-pass WikiText-2 token-PPL, HW-HIER(40, 512) vs fp32 fake-quant (e06): MXFP4 Δ = 0.0000 on SmolLM2-360M
and Qwen2.5-0.5B; NVFP4 Δ = -0.0044 / +0.0085 (vs L1 Exact: +0.0054 / -0.0014). The NVFP4 deltas are of the
same size as the fake-quant reference's own order sensitivity (cross-GPU 0.14 %), so the operator-level
statistics above, not these PPL deltas, are the evidence that the accumulator is adequate.
