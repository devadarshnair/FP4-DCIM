# Verification closure plan: functional coverage bins and formal obligations (T1.4)

Scope: the v2 MXFP4/NVFP4 column and macro (`rtl/src/column.sv`, `column_top.sv`, `macro_top.sv`) and the two
baselines that appear in paper claims: (a) bit-serial INT5 (`rtl/baselines/bitserial_int5/`) and (c) DPU
(`dpu_macro` = `macro_top` with `LEAF = GENERIC`, same column/alignment/accumulator RTL). Numerics contract:
[`numerics.md`](numerics.md). Driver: `scripts/t14_verif_closure.py {formal,coverage}`; results:
`results/verif_closure/<run>/{formal,coverage}/`.

**Acceptance (T1.4):** every legal bin hit, or waived in writing below; every illegal bin at 0 hits; each
formal obligation PASS (or an explicitly waived method with a replacement proof).

## 1. How coverage is collected

- Monitors are SystemVerilog `cover property` statements in `tb/cov/cov_*.sv`, attached with `bind`
  (`tb/cov/bind_*.sv`) so RTL, testbenches and scoreboards are unchanged. Verilator builds them with
  `--coverage-user` (functional bins) and `--coverage-line` (supplementary line coverage).
- `tb/cov/t14_cov_plugin.py` is a pytest plugin that wraps `cocotb_tools.runner.get_runner`: the existing cocotb
  regressions run unmodified, each writes one `coverage.dat`; the driver sums bins over all instances (32 columns per
  macro) and over the tests of a design family. A bin hit in a test counts only because every one of these tests
  checks the DUT bit-exactly against the golden reference (`column_reference` = L2 = L1 Window; `nv_column_reference`;
  the INT5/DPU references) - a passing test with a hit bin means the scenario was exercised **and** checked.
- Small seeds only (the 1e5/1e6 regressions are T1.1/T1.3 and are not rerun here). Job list: `COV_JOBS` in the
  driver (column corners/random/capture, NV pairs/random, column_top M=16 and set-pattern, macro MX/NV M=16 and
  set-pattern, the new directed `test_macro_mixed` on `macro_top` and `dpu_macro`, DPU MX/NV, INT5 column/macro
  M=16 and set-pattern).
- Design families: **v2** (column, column_top, macro_top), **dpu** (dpu_macro), **int5** (bs_column_top,
  bs_macro_top). The column bins exist in both v2 and dpu because the DPU reuses `column.sv`.

## 2. Bins

`cov_column` (every `column` instance; stage 1 = operands in, stage 2 = after the carry-save register).

| Bin | Meaning (stage) | Why it matters |
|---|---|---|
| F_MX / F_NV | op issued in MXFP4 / NVFP4 mode (1) | both formats |
| F_MX_AFTER_NV / F_NV_AFTER_MX | op whose mode differs from the previous op (1) | `r_nv` and the MX/NV pipeline-register enables |
| F_TOGGLE_B2B | mode differs from the op in the **previous cycle** (1) | hazard on the shared stage-2 adder |
| F_B2B / F_RESUME | op in consecutive cycles / after an idle gap (1) | back-to-back issue, restart |
| F_CLEAR / F_CLEAR_B2B / F_ACCUM / F_ACCUM_NV | new dot product; new dot product right after another op; accumulate (MX, NV) | accumulator clear vs accumulate |
| E_W_MIN / E_W_MAX / E_X_MIN / E_X_MAX | UE8M0 0x00 (-127) / 0xFE (+127) on weight / activation (1) | scale extremes |
| E_W_NEG / E_W_POS | negative / positive weight exponent | sign of exponent arithmetic |
| U_ZERO / U_SUB / U_MAX / U_HALVES_DIFF | NV scale significand 0; subnormal (0<u<8); 15×15 = 225; halves differ | E4M3 significand product |
| K_NEG / K_POS / K_MAX / K_MIN | NV k exponent < 0 / > 0 / = 5 / = -9 (ends of the legal E4M3 range) | NV scale exponent extremes |
| A_D0 / A_LEFT / A_BASE / A_RIGHT | MX alignment d = 0; 0<d<BASE; d = BASE (no shift); BASE<d<W-1 | left shift, exact, truncating right shift |
| A_SAT_EQ / A_SAT_GT | d = W-1; d > W-1 (saturation logic) | anchor/alignment extremes |
| A_NV_D0 / A_NV_SPLIT / A_NV_RIGHT / A_NV_SAT | NV halves both d = 0; d0 ≠ d1; right shift; saturation | per-half alignment |
| S_MX_MAX / S_MX_MIN / S_MX_ZERO / S_MX_NEG | MX block sum +4608 / -4608 / 0 / < 0 (2) | tree/CPA extremes |
| S_NV_MAX / S_NV_MIN / S_NV_NEG | NV half sum +2304 / -2304 / < 0 (2) | |
| T_FLOOR_M1 / T_FLOOR_0 | right-shifted term floors to -1 / to 0 (2) | floor rounding of truncated terms |
| N_IN / N_IN_NV / N_STICKY / N_CLEARED | NaN in (MX / NV); NaN flag carried to a later op; cleared by a new dot product | NaN propagation |
| W_GUARD_USED / W_TOP2 | \|acc\| above one maximal block term; \|acc\| ≥ 2^(W-3) (2) | guard bits / window top |
| W_SIGN_FLIP / W_CANCEL0 | accumulator changes sign / cancels to exactly 0 inside a dot product | two's-complement window |
| **ILL_OVF / ILL_DNEG** | window overflow flag; anchor below a block exponent | **illegal: 0 hits required** |

`cov_ctrl` (macro_top / column_top / INT5 tops: op and set-switch control).

| Bin | Meaning |
|---|---|
| C_FIRE / C_STALL / C_STALL_LOADING | op accepted; op held (not ready); op held while a set load is pending |
| C_SWITCH / C_SWITCH_AFTER_FIRE / C_FIRST_AFTER_SW | weight-set switch (load_req); switch after ops ran; first op after a switch |
| C_B2B_SAME_SET | back-to-back ops on the same set (steady-state cadence) |
| C_SET_ZERO / C_SET_MAX | op on set 0 / on set M-1 |
| C_CLEAR / C_X_NAN / C_W_NAN | op with clear; activation-scale NaN; weight-scale NaN record of the current set |
| C_MX / C_NV / C_NV_TOGGLE / C_NV_TOGGLE_B2B / C_NV_TOGGLE_SAMESET | per-op `op_nv`: MX, NV, toggle, toggle in consecutive cycles, toggle without a set switch |

`cov_bs_column` (INT5 bit-serial column).

| Bin | Meaning |
|---|---|
| BS_ACCEPT_IDLE / BS_B2B | block accepted from idle / on bit 4 of the previous block (5 cycles/op, after the back-to-back fix) |
| BS_CLEAR / BS_NAN / BS_N_STICKY | clear; NaN in; NaN carried |
| BS_D0 / BS_D_RIGHT / BS_D_SAT | alignment d = 0; right shift; saturation |
| BS_BLK_MAX / BS_BLK_MIN / BS_BLK_NEG | block sum ±4608, negative |
| **ILL_BS_OVF** | **illegal** |

Per-bin hit counts and the tests that hit each bin: §5 (generated from the run's `coverage/bins.csv`).

**Gaps handed over from T1.3** (`results/macro_regression/20261111T142218298731Z_8fbda3b`, op_nv fixed per
format, no dedicated scale corners) are closed by bins plus the directed `tb/cocotb/test_macro_mixed.py`
(macro_top, dpu_macro, and an MX-only stream on bs_macro_top): per-op `op_nv` toggling -> F_MX_AFTER_NV, F_NV_AFTER_MX, F_TOGGLE_B2B, C_NV_TOGGLE,
C_NV_TOGGLE_B2B, C_NV_TOGGLE_SAMESET; UE8M0 max/min -> E_X_MAX/E_X_MIN (macro, activation side) and E_W_MAX/E_W_MIN
(column corner suite); E4M3 zero/subnormal/max scale -> U_ZERO, U_SUB, U_MAX, K_MAX, K_MIN; NaN scale -> C_W_NAN,
C_X_NAN, N_IN, N_STICKY, N_CLEARED with `nan_out` checked on every column (the macro regressions check `acc`
only); NV NaN scale -> N_IN_NV, checked since T1.5 (`nan_out` vs `nv_column_nan`).

## 3. Formal obligations

Tools available locally: Yosys 0.69 (`sat`, MiniSAT) and `yosys-smtbmc`. **Not installed: SBY, EQY, and any SMT
solver (z3, yices, boolector, bitwuzla).** The obligations therefore use Yosys `sat` plus an algebraic checker; no
large toolchain was installed.

| ID | Method | Statement |
|---|---|---|
| LEAF_SPEC_REGEN | Python | the LUT reference `tb/formal/gen/leaf_spec.sv` equals a fresh `emit_leaf_spec_sv()` from `cimfp4.formats.tables` |
| LEAF_FP4 | Yosys sat miter | `row_driver + leaf_mult` (the v2 leaf as used in `column.sv`) ≡ 256-entry E2M1×E2M1 LUT, all 8 input bits |
| LEAF_GENERIC | Yosys sat miter | `generic_unit` (DPU leaf) ≡ the same LUT |
| REGEN_<tree> | Python | `rtl/gen/<tree>.sv` is byte-identical to `emit_sv(build_column_tree(...))` |
| LINK_<tree> | Yosys sat (structural miter) | the committed monolith ≡ composition of the three generated slices h0 ∘ h1 ∘ merge |
| S_<tree>_merge | Yosys sat | merge slice: m_s + m_c ≡ a + b + c + d (mod 2^w) for all inputs |
| S_<tree>_h0/h1 | Yosys sat | half slice: s + c ≡ Σ_r val(r) (mod 2^w), val = signed(in)+s (oc), signed(in) (tc), in (unsigned) |
| A_<tree>_h0/h1 | algebraic | same contract as S_*_h*, by backward rewriting of the Yosys gate netlist |
| A_<tree>_mono_m / _h0 / _h1 | algebraic | the committed monolith: m ≡ Σ_32 val (mod 2^w) and h_k ≡ Σ_16 val (mod 2^w_half) |
| MUT_* | all | one-gate mutations of the oc slices must be rejected by every method (sanity) |

Trees: `oc` (v2 column), `unsigned` (INT5 popcount), `tc` (ablation, not instantiated); `dual` (ablation) by the
algebraic monolith check only.

**Composition.** LINK makes the slices the monolith; S/A make each slice meet its contract; modular addition gives
m ≡ (h0_s+h0_c) + (h1_s+h1_c) ≡ Σ_32 val (mod 2^w); the half-tree outputs are the low w_half bits of the full-width
half pairs. With LEAF_FP4 (|p| ≤ 144 per row, encoded as oc+s) the sums fit (|Σ| ≤ 4608 < 2^13), so the modular
statements are exact. What remains outside formal scope (stage 2: CPA, NV scale multiply, alignment shifter, window
accumulator, control) is covered by the bit-exact regressions and the bins above.

**Algebraic checker** (`src/cimfp4/rtlgen/alg_extract.py`): reads the Yosys JSON of the RTL itself (`techmap`
gate level), starts from Σ 2^i (o_i + p_i) and substitutes each gate's exact multilinear polynomial over Z/2^w
(NOT 1-a, AND ab, OR a+b-ab, XOR a+b-2ab, ...), cell by cell in reverse topological order. Multilinear polynomials
over Z/2^w are canonical for functions {0,1}^n -> Z/2^w, so equal normal forms are a complete proof of
`outputs ≡ spec (mod 2^w)` (function extraction / backward rewriting, as in Ciesielski et al. and Kaufmann-Biere
AMulet; not yet in `references.bib`, cite only after opening). Soundness checks: the MUT_ALG mutation is rejected,
and `tests/rtlgen/test_tree_slices.py` rejects a mutated monolith.

### Waiver: monolithic SAT/EQY equivalence of the 32-input tree

Proving the 32-input carry-save tree against a word-level sum with a SAT/SMT equivalence engine (EQY/SBY or Yosys
`sat`) in one piece is waived. Rationale: (i) known intractable for SAT on multi-operand adders with a differently
structured reference (R04/R07); (ii) measured here: even one 16-row half slice of the v2 tree (10,722 variables,
28,812 clauses) did not finish in 1,200 s with Yosys/MiniSAT (feasibility probe 2026-11-11) and is recorded as
TIMEOUT in the formal run; (iii) it is replaced by stronger evidence - the compositional chain LINK + S_merge (SAT) +
A_h0/A_h1 (algebraic) - and, in addition, a direct algebraic proof of the monolith (A_*_mono_*). The INT5 popcount
tree is small enough for SAT on every slice.

## 4. Waivers (bins)

Machine-readable copy: `tb/cov/waivers.yaml` (the driver marks unhit waived bins WAIVED; a hit bin is reported HIT even if waived, so a waived-and-hit bin would need hand re-classification; the N_IN_NV waiver was removed on 2026-11-13 by T1.5).

**Withdrawn waiver (2026-11-13): A_NV_SAT.** The earlier waiver claimed NV saturation is unreachable because legal E4M3 k ∈ [-9, 5] gives d ≤ 28. That is wrong at macro level: the per-column anchor `max_e_w` is static and shared by MX and NV ops on the same column, so a column whose MX weight records reach UE8M0 +127 (column 31 in `test_macro_mixed`) has anchor 127 and every NV op on it saturates (d ≥ W-1). The bin is hit 364 times in both v2 and dpu and checked bit-exactly against `nv_column_reference`; no waiver is needed. (Whether such a column loses NV precision in practice is an anchor-policy question for the mapping, not a datapath bug: the RTL matches the numerics contract.)

| Family | Bin | Reason |
|---|---|---|
| ~~all~~ | ~~N_IN_NV~~ | **Waiver removed (T1.5, 2026-11-13).** `test_macro_mixed` now writes weight-scale NaN records from E4M3 0x7F/0xFF (`rf_record`), drives `op_x_nan` on NV ops, and checks `nan_out` of every column per op against `nv_column_nan` (stickiness, clear); the bin is hit and checked (1266 hits per family, run `results/verif_closure/20261113T182021367463Z_7541fa6`, partial `--only test_macro_mixed`). |
| int5 | C_B2B_SAME_SET | Unreachable by design: the bit-serial macro accepts one op per 5 cycles (`op_ready = !busy \|\| bit_i == 4`), so ops never fire in consecutive cycles; its back-to-back cadence is BS_B2B (hit). |
| int5 | C_NV, C_NV_TOGGLE, C_NV_TOGGLE_B2B, C_NV_TOGGLE_SAMESET | Baseline (a) is MXFP4-only (`op_nv` must be 0, numerics.md); NVFP4 is not part of the INT5 claims. |

## 5. Results (T1.4 closure, 2026-11-13)

**Coverage** - `results/verif_closure/20261113T131621669195Z_ca8507c` (clean tree at ca8507c; `coverage/bins.csv`,
`jobs.csv`, `line_coverage.csv`, `metrics.json`). 20/20 regressions passed bit-exactly (0 mismatches). 166 legal
bins over the three families (v2, dpu, int5): driver status 161 HIT, 5 WAIVED, 0 UNHIT; 5 illegal bins
(ILL_OVF, ILL_DNEG in v2 and dpu; ILL_BS_OVF in int5) at 0 hits. Re-classifying N_IN_NV (hit, output unchecked; §4)
gave **159 hit-and-checked + 7 waived (N_IN_NV ×2, INT5 C_B2B_SAME_SET and the four INT5 C_NV* bins) = 166/166
closed**; superseded 2026-11-13 by T1.5: N_IN_NV ×2 are now hit-and-checked (**161 hit-and-checked + 5 waived: INT5 C_B2B_SAME_SET and the four INT5 C_NV***, 166/166 closed; the affected jobs were re-run only: `results/verif_closure/20261113T182021367463Z_7541fa6`, 3 test_macro_mixed jobs, `t14_verif_closure.py coverage --only test_macro_mixed`). All T1.3 gap bins are hit in v2 and dpu: F_MX_AFTER_NV, F_NV_AFTER_MX, F_TOGGLE_B2B, C_NV_TOGGLE,
C_NV_TOGGLE_B2B, C_NV_TOGGLE_SAMESET (per-op `op_nv`), E_W/E_X_MIN/MAX (UE8M0 ±127), U_ZERO, U_SUB, U_MAX, K_MAX,
K_MIN (E4M3 zero/subnormal/max, k = 5 / -9), A_SAT_EQ/GT and A_NV_SAT (saturated anchors), C_W_NAN/C_X_NAN.
Supplementary line+branch coverage (not part of acceptance): v2 `column.sv` 24/24, `leaf_mult`, `row_driver`,
`generic_leaf`, `cim_icg` 100 %; `macro_top` 9/11 and `column_top` 10/12 miss only the false arm of `if (wr_en)`
inside an ICG-gated `always_ff` (the gated clock ticks only when the enable is 1, so the arm is structurally
unreachable); `wset_sram` 15/16, the INT5 tops/column (37/39, 16/19, 15/18) and the vendor SRAM model were not
analysed further.

Superseded/incomplete coverage runs: `20261111T161838316009Z_3244994` (aggregation bug, see its NOTE) and
`20261111T164943438539Z_10a4dad` (interrupted after 13/20 jobs, no metrics; NOTE, index status `incomplete`).

**Formal** - `results/verif_closure/20261113T140145733357Z_8609d56` (`formal/obligations.csv`, logs, generated
slices; tree dirty only in another task's `experiments/e29_sram_timing/`, which formal does not read). 38 rows:
34 PASS, 1 TIMEOUT (S_oc_h0, Yosys/MiniSAT 600 s), 3 NOT_RUN (S_oc_h1, S_tc_h0, S_tc_h1, same structure);
`all_required_pass = true`. Per obligation: LEAF_SPEC_REGEN, LEAF_FP4, LEAF_GENERIC PASS (leaf miters, exhaustive);
REGEN_oc/unsigned/tc/dual PASS (rtl/gen byte-identical to the generator); LINK_oc/unsigned/tc PASS (monolith ≡
h0 ∘ h1 ∘ merge); S_*_merge PASS (all three trees); S_unsigned_h0/h1 PASS; A_* (half slices and monolith, all four
trees) PASS; mutation checks MUT_ALG_h0, MUT_LINK, MUT_S_merge PASS (mutants rejected). The SAT half-slice
TIMEOUT/NOT_RUN rows are covered by the monolithic/half-slice SAT waiver in §3 and replaced by A_*_h0/h1 and
A_*_mono_* (algebraic, complete for the stated contract). `tests/rtlgen/test_tree_slices.py`: 5 passed.
Method note: EQY/SBY are not installed; the "compositional EQY" obligation of T1.4 is discharged with Yosys
`sat` structural miters (LINK, S_merge) plus algebraic extraction, as stated in §3. Earlier formal run
`20261111T143631444995Z_1213f4b` (same results, tree dirty in tb/docs) is superseded.
