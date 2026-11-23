# T6.3 pre-registration: MIDAS Fig. 13 / Fig. 22 steps and the MXFP6 early stop (FROZEN once committed)

Written before any T6.3 number exists. Run: `experiments/e37_midas_ablation/run.py` -> `results/e37_midas_ablation/<run>`.
The run config records this file's sha256 and the commit that contains it. Source for every target: `refs/MIDAS.pdf`
(choi2026midas, opened; text §IV-D p. 5363 for Fig. 13, §VII p. 5367 for Fig. 22), targets in `docs/spec/midas_targets.yaml`.
Scope: behavioural + cost model only (CPU numpy, no model, no GPU, no RTL).

## Tolerance and rule
- Tolerance for every numeric check: **+-25 % relative** to the published value (the WP-37 criterion already frozen in
  `midas_targets.yaml`, header). Targets are read from figure text/captions: 11.3 %, 8.7 %, 68.8 %, 63.2 % (Fig. 13),
  +34.3 % (Fig. 22 step 2), 22.5 % (MEPD latency, MXFP6).
- Each check is PASS (inside the band), FAIL (outside), or NE (not evaluable with the allowed inputs; reason given). NE is not a pass.
- Verdict: label **MIDAS-like** only if C0 and every one of C1-C8 is PASS. Otherwise the label stays **MIDAS-inspired**. Because
  C1, C7 and C8 are already known NE, the label is predetermined to stay MIDAS-inspired; the run's value is the per-step verdicts.
- No retuning after seeing results: model constants below are fixed; a FAIL is reported, not repaired. Sensitivity rows are
  informational and never gate.

## Checks
- **C0 (precondition, exact):** with MEPD early stop on, `MidasLikeMacro.run` output equals `block_integer_reference` for
  MXFP4 and MXFP6 on random code tiles (0 mismatches). A failure voids every other check (MIDAS calls MEPD lossless).
- **C1 MXFP6 MEPD latency reduction = 22.5 % (avg over GPT2 ifmaps):** requires real MXFP6 activations of GPT2. The repo
  captures hold only Qwen2.5-0.5B / SmolLM2 MXFP4/NVFP4 codes (no raw tensors, no MXFP6 codes, no GPT2), so a faithful check needs a
  model run. **NE** (stop condition: model needed). Informational proxy only, labelled [EST]: synthetic Gaussian and Student-t(3)
  activations x Gaussian weights, MXFP6 (E3M2, OCP floor scale), reduction = 1 - cycles(early)/cycles(full range 15). Not gating.
- **C2-C5 exponent-logic reductions (Fig. 13), format MXFP4 (the Fig. 22 operating format; Fig. 13's format is not stated, so
  MXFP6 is a non-gating sensitivity):** reduction = 1 - X/Baseline, X in {TAOR, TAOR+MEPD}, energy per op as power proxy.
  C2 TAOR power 11.3; C3 TAOR area 8.7; C4 TAOR+MEPD power 68.8; C5 TAOR+MEPD area 63.2.
- **C6 Fig. 22 step 2 (TAOR+MEPD) = +34.3 % EE:** macro-level proxy (not system EE): E1/E2 - 1 with E2 = e19 macro model energy per op
  (MXFP4, same captures) and E1 = E2 - E_expo(TAOR+MEPD) + E_expo(baseline). [EST]
- **C7 Fig. 22 step 1 (STCA+SACA) = +72.0 %:** needs a model of the conventional IA-W pre-aligned 2C FP-CIM mapping and dataflow
  (cell utilisation, array-level cost, a Guo-style macro). Not in the repo. **NE.**
- **C8 Fig. 22 step 3 (SADB+SSCR) = +17.3 %:** needs the approximation model (sense-amp reference selection, SSCR regrouping)
  at the paper's SQNR/accuracy operating point; the L2 model has approximation off. **NE.**

## Exponent-logic model (fixed; sky130 hd Liberty cells via `power/liberty.py`, vdd 1.8, alpha = 0.3 as in e19, no fitting)
n = 512 LCAs (32 rows x 16 cols); eb, mb = format exponent/mantissa bits; W = eb + 1; cycles/op and activity from
`MidasLikeMacro` on the e19 sample (Qwen2.5-0.5B MXFP4 captures, tiles 4, tokens 256, seed 0; MXFP6 sensitivity: random E3M2 codes, same shape).
Cell energy = Liberty toggle energy x alpha; flop clock energy counted per cycle as in e19. Area = Liberty cell area x count.
- **Baseline** (conventional IA-W independent pre-alignment, max-exponent pre-detection), once per op:
  per LCA: eb FA (exponent add) + W FA (Emax - Esum) + (mb+1)*W mux2 (alignment shifter); max-detect tree (n-1) x (W FA + W mux2).
  Sign->2C conversion is excluded from "exponent-related logic".
- **TAOR** (max tree retained, no early stop; cycles = top - bottom + 1): per LCA per cycle: 3 FA x 0.5 (VRDU, e19 convention) + 2 and2 (valid compare);
  per row per cycle: 4 FA (LCG) + 4 dfxtp; global 4-bit counter (4 dfxtp + 4 ha) per cycle; plus the max-detect tree once per op.
- **TAOR+MEPD:** as TAOR without the max-detect tree and without its per-LCA align logic, plus MEPU = (n-1) and2 per cycle at alpha 0.05 (e19), cycles = measured with early stop.
- Power proxy = energy per op (fJ/op). Area counts the same elements (VRDU/LCG/counter/MEPU instances; flops for LCG latches).

Known weaknesses stated up front: std-cell sky130 proxies for a 28 nm custom macro; dynamic energy only; the baseline's
exact exponent organisation is not published in the text read, so C2-C6 test a plausible reconstruction, and a FAIL may reflect that
rather than MIDAS. The ratio form cancels the supply and technology scale but not cell-mix differences.
