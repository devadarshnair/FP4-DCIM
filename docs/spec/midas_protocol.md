# MIDAS accuracy protocol and how we reproduce it (WP-19)

Source: MIDAS, JSSC 61(10), Oct 2026 [choi2026midas], **Table II "Benchmark Evaluation"** (p. 5367; an image in the
PDF, read from a 300-dpi render of `refs/MIDAS.pdf` p. 11 on 2026-11-01) and §VI text.

## What MIDAS reports (verbatim from Table II)

| | GPT2-Large | Llama3.2-1B | Qwen2-1.5B |
|---|---|---|---|
| Dataset | WikiText-2 | WikiText-2 | WikiText-2 |
| Parameters | 774M | 1B | 1.5B |
| Precision (IA/W) | MXFP4 / MXFP4 ⁽²⁾ | MXFP6 / MXFP6 | MXFP6 / MXFP6 |
| PPL (GT) | 20.11 (19.44) | 11.62 (10.99) | 11.12 (10.70) |
| System EE (TFLOPS/W) ⁽³⁾ | 28.2 | 16.2 | 17.3 |

Footnotes: ⁽¹⁾ MXFP8 for attention block operations (QKᵀ, AV) - applies to all columns; ⁽²⁾ MXFP6 for the projection
layer in the attention block; ⁽³⁾ 0.85 V, 50 MHz.

**Fig. 1(b) "GenAI Benchmark Evaluation Comparison"** (p. 5358; image, read from a 500-dpi render 2026-11-10): WikiText-2
PPL, panels labelled "@GPT2-Large, Token=1024" and "@Llama3.2-1B, Token=1024", bars FP / INT / MX at 8, 6, 4 bits.
Only the MX bars carry numbers: GPT2-L **19.65 / 19.66 / 22.85**, Llama3.2-1B **11.58 / 11.62 / 20.42** (8 / 6 / 4 bit),
baselines "(FP32) PPL" 19.44 / 10.99. The FP and INT bars have no values and the axis no ticks (not citable as
numbers). Element formats, block size and attention precision of these bars are not stated; the 6-bit Llama bar
equals Table II's MXFP6 11.62.

**Eq. (1)** (§IV-A, p. 5362): e_shared = ⌊log₂ max|x|⌋ - e_max,elem, i.e. the OCP v1.0 floor rule (our `ocp_floor`), not our
default `ocp_v11`.

**SQNR plots (approximate, read off the figures):** Fig. 2(a) (p. 5358) GPT2-L "Truncated Bit # vs Accuracy Drop" (IA and W
truncation): MXFP4 ≈ 48 / 14 / 5 / 1 dB at 0-3 truncated bits, MXFP6 ≈ 48 / 47 / 45 / 35 / 25 / 16 / 7 dB at 0-6,
"threshold for acceptable accuracy" ≈ 37 dB. Fig. 18 (p. 5366; MXFP6, GPT2-L, 0.85 V 50 MHz): SADB+SSCR energy efficiency vs
average approximated bits with a 37 dB SQNR line; text: +12.4 % over SADB alone at 37 dB, up to +27.3 % at an
average of six approximated bits. The SQNR definition (which tensor, reference) is not stated.

**Element formats** (§VI, p. 5366): MXFP4 = (1,2,1) E2M1, MXFP6 = (1,3,2) E3M2 (= our `mxfp6`). **Table II text**
(p. 5367) calls the values "measured perplexity".

Not stated anywhere in the paper: token vs word PPL (inferred token), rounding of elements, RTN vs GPTQ,
whether lm_head/embeddings are quantized, whether the PPL rows are silicon-in-the-loop or software emulation.

## Consequences for our comparison (revised 2026-11-11 after e25)

1. **MIDAS FP4 points.** Fig. 1(b) adds all-MX 4-bit bars (GPT2-L 22.85, Llama3.2-1B 20.42; element format, block
   size and BMM precision unstated, read as MXFP4 with Table II's MXFP8 BMMs). Table II's GPT2-L 20.11 is a mix
   (MLP MXFP4, attention projections MXFP6, BMMs MXFP8). Llama/Qwen Table II rows are MXFP6 W/A.
2. **Protocol:** token-PPL, non-overlapping 1024-token windows ("Token=1024" in Fig. 1(b)). BF16 at 1024:
   GPT2-L 19.45 (GT 19.44), Qwen2-1.5B 10.64 (GT 10.70). Llama3.2-1B pending access to the gated HF checkpoint.
3. **Scale rule = OCP floor** (eq. (1)), confirmed by data: GPT2-L all-MXFP4 with `ocp_floor` = 22.99 (+0.6 % vs 22.85),
   with `ocp_v11` = 21.94 (-4.0 %). Every MIDAS-comparison row uses `ocp_floor` for linears **and** MXFP8 BMM operands
   (`experiments/e25_midas_accuracy`); our D1 default stays `ocp_v11`, which must be stated as a difference.
4. **Matches (|Δ| ≤ 1 %):** GPT2-L MX 8/6/4-bit (Fig. 1(b)) -0.18 / +0.47 / +0.63 %; Qwen2-1.5B MXFP6 11.12 with K
   unquantized and Q/P/V MXFP8 floor = 11.126 (+0.06 %), all BMMs BF16 = 11.057 (-0.56 %).
5. **Mismatch, cause unknown:** GPT2-L Table II mix 20.11 vs our 21.00 (+4.4 %; BF16 BMMs 20.90, lm_head MXFP4 21.22,
   all-`ocp_v11` 20.61). The all-MXFP6 and all-MXFP4 endpoints match, so the difference is in which linears MIDAS ran
   at MXFP6 or in something else the footnote does not state. Never claim parity with 20.11.
6. **Qwen2-1.5B attention K (attention-K failure).** The literal footnote (1) reading (K in MXFP8 E4M3) gives 86.9
   (floor) / 98.4 (v11). MIDAS's 11.12 is reproduced only when K is not E4M3/E5M2-quantized (BF16, MXINT8 or offset-
   removed K all recover). Report the K-BF16 row as "MIDAS-matching", the literal row next to it, and the K policy in
   the footnote.
7. MIDAS calls its Table II values "measured perplexity"; whether SADB/SSCR approximations were enabled is not stated.
   Ours are bit-exact fake-quant (no approximation). Where we match, any approximation cost is inside the 1 % band;
   where ours is higher (item 5) approximations cannot be the reason (they only add error).
