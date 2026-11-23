# D2 numerical contract - MIDAS-inspired INT4 surrogate (exact INT5 stays the control)

Status: **frozen v1.1, 2026-11-06**. Changes need a new version.

v1.1 change (same day, before any accuracy run): primary scale rule `ceil` -> `floor_sat`. On all 168 Qwen2.5-0.5B
weight matrices `floor_sat` gives 17.38 dB mean SQNR vs 16.27 dB for `ceil` (+1.1 dB); keeping `ceil` would make
the D2 baseline needlessly weak. `ceil` is retained as an ablation.

## 1. Why INT4 cannot be exact

An E2M1 value v ∈ {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. The exact integer the hardware multiplies is 2v ∈
{0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}: it spans [-12, 12] and needs **signed INT5** (two's complement [-16, 15], or
sign + 4-bit magnitude). Any signed 4-bit representation (two's complement [-8, 7] or sign-magnitude [-7, 7])
must therefore clip, rescale, or re-encode. The existing bit-serial **INT5** baseline (`rtl/baselines/bitserial_int5/`,
`golden/l2_bitserial.py`) is the *exact* control and is never renamed INT4.

## 2. The surrogate: MXINT4-SM (primary)

A block format with the same block size, scale type, and code count as MXFP4, but uniformly spaced elements:

| Field | Definition |
|---|---|
| Block | 32 consecutive elements along K (identical to MXFP4) |
| Shared scale | UE8M0 power of two 2^X (identical encoding to MXFP4) |
| Element | sign + 3-bit magnitude m ∈ {0,...,7} (sign-magnitude, as MIDAS's SACA datapath uses SM operands) |
| Value | (-1)^s · m · 2^X |
| Scale rule | `floor_sat` (primary): X = ⌊log₂ amax⌋ - 2, so amax / 2^X ∈ [4, 8) - the OCP MXINT8 pattern; amax = 0 -> X = -127. `ceil` (ablation): X = ⌈log₂(amax / 7)⌉, never clips |
| Rounding | m = min(RNE(|v| / 2^X), 7): values in (7.5, 8)·2^X saturate to 7 (error ≤ 1 step there, ≤ ½ step elsewhere) |
| Zero / NaN | ±0 -> m = 0; NaN inputs are rejected (same as the L1 golden model) |

This is a **project-specific format**, not an OCP format. OCP MX v1 defines MXINT8 (two's complement with an
implicit 2^-6) but no 4-bit integer element; MXINT4-SM follows the same shared-scale pattern. It is
quantized **from the original high-precision tensor**, so it is accuracy-preserving *under its own scale*;
it is not a mapping of MXFP4 codes.

Why this choice (vs. alternatives in §3): same 16-code budget as E2M1 (15 distinct values each), the same
block/scale hardware, the best measured scale rule, and a sign-magnitude element that a MIDAS-style bit-serial schedule
can process in **3 magnitude cycles** with no per-element alignment. It therefore isolates the two questions
D2 exists to answer: what the FP4 element grid buys in accuracy, and what bit-parallel compute buys in energy.

## 3. Ablations (lossy recodes of existing MXFP4 codes - reported, never used as the main surrogate)

| Name | Map 2v -> INT4 | Loss |
|---|---|---|
| `clip` | 2v clipped to [-7, 7] | ±8, ±12 -> ±7 (top two magnitudes destroyed) |
| `half` | RNE(2v / 2) ∈ {0, 0/1, 1, 2, 2, 3, 4, 6}, scale × 2 | 0.5 -> 0, 1.5 -> 2 (ties to even) |

Both are exact tables over the 15 E2M1 values and are tested exhaustively.

## 4. Arithmetic contract (golden L1 for D2)

- Block product: Σ_r s_r · m_x,r · m_w,r ∈ [-32·49, 32·49] = [-1568, 1568] -> 12-bit signed block sum.
- Block term exponent: X_x + X_w (no -2 offset, unlike the ×4 E2M1 integerization).
- Inter-block accumulation: the same `AccumMode` semantics as D1 (Exact / Window / Hier / FP32Seq), so the
  accumulator is held fixed across D1, D2 and the INT5 control. INT_BITS for MXINT4-SM = 12.
- Bit-serial schedule (for the L2/RTL model, later): activation magnitude bits fed LSB-first over 3 cycles,
  sign applied per LCA (sign-magnitude), weights stationary; no exponent alignment inside a block.

## 5. What is held fixed in the D1 / D2 / INT5 comparison

Same process library and corner, same 32-row block and column count per macro, same SRAM weight-set
storage and macro boundary, same clock constraint, same accumulator mode and width, same real-LLM stimulus
(the INT4 operands are re-quantized from the same source tensors), and throughput normalised to MAC/s.

## 6. Acceptance tests for this contract

1. `floor_sat`: amax / 2^X ∈ [4, 8), |m| ≤ 7, error ≤ ½ step except saturated values (≤ 1 step); `ceil`: never
   clips and its scale is the smallest that achieves that.
2. RNE ties, zeros, ±amax, subnormal-range inputs; torch implementation bitwise equal to numpy.
3. Recode tables `clip` and `half` exact over all 16 E2M1 codes.
4. L1 block sums equal Σ v_x·v_w exactly; Exact accumulation equals a float64 reference.
5. Quantization error (SQNR) on real weights/activations reported for MXFP4, NVFP4, MXINT4-SM and the two
   recodes (experiment `d2_int4_error`), before any accuracy run or RTL.
