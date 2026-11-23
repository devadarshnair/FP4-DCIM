# MXFP4 / NVFP4 hardware-cost table v0.2 (DRAFT, T2.7 + T2.11 numbers; not yet sent)

Purpose: the D1-side input to the partner's D3 offline oracle. Machine-readable twin:
[`hw_cost_table.yaml`](hw_cost_table.yaml) (version, date, sources). The YAML is **generated** by `scripts/gen_hw_cost_table.py` from the pinned runs (`--write`); `--check` verifies
that the YAML equals the regenerated one and that this file contains the same numbers. **Version 0.2 DRAFT, 2026-11-15** (energy numbers from 2026-11-14)
(T2.7): every energy/area number now comes from the T2.6 timing-driven (`td`) netlists (`docs/spec/synth_recipe.md`) measured in T2.7, on 9 real windows from both models (incl. one GPTQ-weight window) with MX and NV on the
**same** windows. v0.2 will be sent after review. No numbers from the partner
reports (`scale_dcim/`) are used. v0.1 values are listed in the changelog for comparison.

## Conditions (apply to every number)

- sky130_fd_sc_hd tt 1.8 V logic, OpenRAM TT 1.8 V SRAM; **40 MHz** power clock; **zero-delay** gate-level activity +
  OpenSTA power unless a cell says glitch-corrected [EST]. **Not post-route**, no clock tree, no parasitics.
  Netlist: flat Yosys `td` recipe (ABC `-D 20000 -constr`, OpenROAD repair at 25 ns, no placement); reg-to-reg
  timing at 25 ns is met in synthesis STA (v2 r2r fmax 47.4 MHz, ideal clock, no wires); macro port nets
  (`sc_wr_en`, `rst_n`, op broadcast) stay unbuffered (disclosed).
  The 40 MHz clock is simulation-feasible for the SRAM (e29) but above the silicon-verified 34 MHz at 1.7 V; at 34 MHz
  energy/op is unchanged (to first order), throughput x0.85 (33.20 GMAC/s) and the per-op time 30.8 ns.
- **Boundary:** the full 32x32 v2 `macro_top` (32 columns, 8 OpenRAM 1 KB weight stores, per-(set,column)
  scale RFs, write port, set-switch control; boundary wires [EST]). **Excludes** the activation quantizers,
  tile-anchor detector, and anything above the per-column window outputs (incl. the FP32 tensor scale).
- Workload (T2.12): 9 real prefill windows, 4 Qwen2.5-0.5B + 4 SmolLM2-360M RTN-capture windows (stratified) + 1
  GPTQ-weight window (Qwen layer-0 q_proj, weights from the format's GPTQ checkpoint), plus 1 random control; R=256 ops,
  M=16 sets, 2 set switches per window, 1024 MAC/op (`e21` run `20261114T023643822998Z_cada632` programs). **MX and NV
  use the same tensors, rows, slices and tokens per window** (v0.1 used different windows), so MX:NV is paired.
- SRAM corners: low / nominal / high from e20 (`sram_energy.yaml`); **high is [EST]** (2 x nominal), low is a
  device-only lower bound. As-built clocking (OpenRAM clock runs every cycle) is the primary policy.
  Wires are the nominal [EST] boundary-wire model, not varied across the SRAM columns.

## Decision unit

One common 32-element unit = **1 MXFP4 block (32 el, UE8M0)** or **2 NVFP4 blocks (2 x 16 el, E4M3, plus an
FP32 tensor scale per tensor)**. In the macro, one op processes 32 units (one per column, 1024 MACs), so
**unit value = fJ/MAC x 32**. MX and NV units occupy the same column slot and the same number of cycles.

## 1. Energy (macro boundary, MX and NV modes)

fJ/MAC (per-unit pJ in parentheses = x32/1000). Source: `e22` `grid.csv` (real windows, zero delay, as-built,
40 MHz, nominal wires), pinned run `results/e22_macro_compare/20261115T201935510694Z_94ba25d` (T2.11 rerun; the v2 rows are identical to the earlier `20261114T185702027770Z_be5d1aa`).

| Format | SRAM corner | Logic (comb+seq, measured) | SRAM (access+idle clock+leak) | Wires [EST] | **Total fJ/MAC** | **pJ / 32-el unit** |
|---|---|---|---|---|---|---|
| MXFP4 | low | 832.3 (26.63) | 62.7 (2.01) | 25.7 (0.82) | **920.6** | **29.46** |
| MXFP4 | nominal | 832.3 (26.63) | 74.0 (2.37) | 25.7 (0.82) | **932.0** | **29.82** |
| MXFP4 | high [EST] | 832.3 (26.63) | 152.2 (4.87) | 25.7 (0.82) | **1010.2** | **32.33** |
| NVFP4 | low | 1288.6 (41.23) | 62.7 (2.01) | 30.1 (0.96) | **1381.3** | **44.20** |
| NVFP4 | nominal | 1288.6 (41.23) | 74.0 (2.37) | 30.1 (0.96) | **1392.7** | **44.57** |
| NVFP4 | high [EST] | 1288.6 (41.23) | 152.2 (4.87) | 30.1 (0.96) | **1470.9** | **47.07** |

- NV - MX = **+456.3 fJ/MAC logic, +460.7 total; NV/MX = 1.49 [1.46, 1.53]** (paired bootstrap over the 9 windows) at
  nominal (+14.7 pJ per 32-el unit). v0.1: 1.47 on unpaired windows. SRAM terms are identical in both modes.
- The SRAM split at nominal: access 15.7, free-running idle clock 57.8, leakage 0.5 fJ/MAC (7.9 % of MX, 5.3 % of NV).
  Gated SRAM clocks would remove most of the idle-clock term (MX nominal 875.9 fJ/MAC) **[EST design change, not built]**.
- **Glitch-corrected [EST]** (tile SDF/zero factor of the `td` v2 tile = 1.973, `results/e13_sdf_td/20261114T073508435587Z_3a17c1d`,
  applied to the macro logic dynamic term): MXFP4 low/nominal/high = 1730 / 1742 / 1820 fJ/MAC. **NV glitch factor was not
  measured**, so no NV value is given.
- **Idle-NV overhead in MX mode:** +160 fJ/MAC at macro (e23, pre-T2.6 netlists, not re-measured in T2.7) and, at
  tile level with the `td` recipe, NV_GATE=0 +22.9 % -> NV_GATE=1 +8.2 % (T4.2 input). It is inside the 932.0
  above (NV_GATE=0 default). An MX-only (NV-stripped) macro estimate is ~772 fJ/MAC [DERIVED: 932.0 - 160].
- Scale-processing energy: the isolated E8M0-vs-E4M3 scale path was **not measured separately**. The best
  available bound is the whole NV-mode premium above: **+14.7 pJ / 32-el unit** (logic + wires).

## 2. Cycles / latency

| Item | MXFP4 | NVFP4 | Source |
|---|---|---|---|
| Cycles per op, amortized | 1.049 | 1.049 | e21 (537 cycles / 512 ops incl. 2 set switches) |
| Steady-state issue | 1 op/cycle, 32 units per op | same | e21, numerics.md |
| Throughput at 40 MHz | 39.05 GMAC/s (21.22 GFLOPS/mm2) | 39.05 GMAC/s | e22 `20261115T201935510694Z_94ba25d` `metrics.json` |
| Throughput at 34 MHz | 33.20 GMAC/s (18.04 GFLOPS/mm2) | 33.20 GMAC/s | e22 `20261115T201935510694Z_94ba25d` `metrics.json` |
| Time per op (32 units in parallel, amortized) | 26.2 ns at 40 MHz, 30.8 ns at 34 MHz | same | derived |
| Set switch (weight-set change) | 10 cycles, stalls (single buffer) | same | numerics.md (WP-67), e21 |
| fmax | synthesis-STA r2r 47.4 MHz (td flat v2, ideal clock, no wires; `results/e30_td_macro_v2_macro/20261114T014522099345Z_e3a93a6/metrics.json` `r2r_fmax_mhz`); the SRAM limit (34 MHz verified) binds. The old 69.7 MHz entry is removed | -- | e30 td synthesis, e29 SRAM timing (simulation) and the 34 MHz OpenRAM silicon reference |

NV costs **no extra cycles** in this macro; it pays in energy only (and in the already-built idle datapath).
Pipeline depth / end-to-end first-result latency: **not measured** (gap).

## 3. Area share

Same silicon for both formats (flat Yosys `td`, `results/e30_td_macro_v2_macro/20261114T014522099345Z_e3a93a6`;
v0.1 mixed flat and hier sources, fixed here):

| Item | mm2 | share |
|---|---|---|
| Macro total | 3.6807 | 100 % |
| 8 OpenRAM 1 KB stores | 1.5257 | 41.5 % |
| Logic incl. sequential | 2.1550 | 58.5 % (sequential 0.617 = 16.8 %) |
| **Per 32-el unit slot (1 column incl. SRAM share)** | **0.1150** | 1/32 |

- SRAM = `total_area_um2 - logic_area_um2` of the same `metrics.json` (8 stores); all area numbers in this file are **flat td**; the
  hierarchical td areas (v2 3.7786, INT5 2.6240, DPU 4.7553 mm2, `results/e30_td_macro_*_macro_hier/` pinned runs) are not used anywhere.
- The scale-RF share is no longer quoted from a hierarchical calibration run (v0.1 used 10.5 % of the hier macro);
  a flat per-block area split of the td netlist is not available (gap). The RF is provisioned for NV (25 b).
- Area is **identical for MX and NV**. A fixed-format MX-only macro area is not available (gap). No post-route area.

### 3b. Same-boundary comparison designs (all flat td, same recipe; zero-delay, as-built, nominal SRAM + wires, 40 MHz, real windows)

Source: e22 `20261115T201935510694Z_94ba25d` (`metrics.json`, `grid.csv`, `ratios_ci.csv`, `nv_row.csv`) and flat area/STA
`results/e30_td_macro_{v2,bs,dpu}_macro/<pinned>/metrics.json` (DPU: `results/e30_td_macro_dpu_macro/20261115T173244110818Z_cfa9ec7`,
measured, T2.11). INT5 is MX-only. DPU = generic-leaf digital MAC with the same local stores.

| Design | Area mm2 (flat) | r2r fmax MHz (synth STA) | Cycles/op | MX fJ/MAC | NV fJ/MAC | GFLOPS/mm2 @40 MHz |
|---|---|---|---|---|---|---|
| v2 (this work) | 3.6807 | 47.4 | 1.0488 | 931.99 | 1392.67 | 21.22 |
| INT5 bit-serial (MX only) | 2.5845 | 74.2 | 5.0547 | 2434.61 | n/a | 6.27 |
| DPU, measured flat | 4.9867 | 47.6 | 1.0488 | 1376.40 | 1840.29 | 15.66 |

v2 / INT5 (MX): **2.61 [2.59, 2.64]** zero-delay, 1.68 glitch-corrected [EST]; density 3.38x, throughput 4.82x.
v2 / DPU flat: MX **1.48 [1.47, 1.48]** (1.70 glitch-corrected [EST]), NV **1.32 [1.32, 1.32]** (1.51 [EST]); density 1.35x.
Boundary and clock mode of every ratio above: full `macro_top` (no activation quantizers, anchor or tile buffer), SRAM clock **free-running (as-built)**.
The conservative primary against INT5 is the **gated-SRAM-clock value, v2 / INT5 (MX) = 2.46 [EST]** (e22 `grid.csv` gated rows: 2158.2 / 876.0 fJ/MAC;
T2.10 `results/t210_sram_gating/`, CI [2.44, 2.48]); gated v2 / DPU flat = 1.51 MX, 1.33 NV [EST]. Gated values are an optimistic bound (ICG cost and write/reload cycles not modelled).
Adding the MX activation-quantizer bundle (+114 fJ/MAC, e28) gives 2.44 (free-running); prefill-2048 system level (e27/e28, gated) gives 2.40. These are different boundaries, not conflicting measurements.

DPU hier-to-flat logic energy penalty (measured): 1.338 MX / 1.258 NV. The earlier hierarchical DPU and the derived 1.191 flat-equivalent
rows are superseded (T2.11) and not quoted. The `[EST]` ratios use the tile SDF factors (`results/e13_sdf_td/20261114T073508435587Z_3a17c1d`),
not a macro SDF run.

## 4. Scale processing and metadata

| | MXFP4 | NVFP4 | Source |
|---|---|---|---|
| Scale format | UE8M0 per 32 el | E4M3 per 16 el, plus FP32 per tensor | spec |
| Scale bits per 32-el unit | 8 | 16 | spec |
| Bits per element incl. 4-b E2M1 code | 4.25 | 4.50 (+ 32 b/tensor, negligible) | derived from spec scale formats (numerics.md: UE8M0 per 32, E4M3 per 16) |
| Weight bits per unit | 128 | 128 | |
| RF bits provisioned per (set,column) | 25 (e0,e1 8 b; u0,u1 4 b; nan) | 25 | `macro_top.sv` SCW, `numerics.md` |
| RF bits per macro (M=16) | 12,800 | 12,800 | derived |
| Scale arithmetic in macro | integer add of exponents (alignment shift) | 8-b significand product + 21 b scaled half-sums + 2 alignments | numerics.md |
| Activation scale generation | `act_quant_mx` (UE8M0) | `act_quant_nv` (E4M3 RNE) | RTL verified; **no area/energy number** |

## 5. Format-switch cost

- A format is selected per op by `op_nv`; weights/scales for both formats live as sets in the same stores.
  **Per-op `op_nv` toggling cost was not measured** (all e21 windows are single-format; T4.2 mixed stream pending). By construction it adds
  no stall cycles, but mixed-stream energy (extra register toggling, partial sums across formats) is unmeasured.
- Switching the weight set (e.g., to hold different-format weights): 10 cycles stall and
  SRAM-only **1.61 / 4.11 / 8.22 nJ per macro** (low / nominal / high [EST]; 8 stores x e20 202/514/1028 pJ);
  full set-switch energy at nominal 4.20 nJ per macro (MX) / 4.20 nJ (NV) (`reload_fj_per_set_switch_nominal`, e22 T2.7).
  Per 32-el unit slot: 131 pJ nominal [DERIVED, 4.20 nJ / 32], amortized over R=256 ops already inside the 932.0 / 1392.7 above.
- Full reload from DRAM, mode-control logic, and conversion costs: not measured.

## Gaps for the oracle (do not fill with guesses)

1. DPU/INT5 comparison values are in section 3b (flat td, measured). INT5 bit-serial is MX-only (no NV). No post-route values.
2. Activation quantizer and tile-anchor energy/area are outside the boundary and unmeasured.
3. Isolated E8M0 vs E4M3 scale-path energy; FP32 tensor-scale cost; NV glitch factor.
4. Per-op format toggling cost and mixed-format streams (T4.2, user-owned later).
5. End-to-end pipeline latency; post-route fmax of the macro (only synthesis-STA r2r is available).
6. Data dependence: logic spread across the 9 real MX windows 777-893 fJ/MAC (NV 1255-1332); the GPTQ window is the
   lowest in both formats (777 / 1255).
7. Post-SDF/post-route values, whole-macro SRAM extraction (periphery schematic), CACTI cross-check, macro-boundary
   port buffering (unbuffered in `td`).
8. Idle-NV overhead at macro level with the td netlist and with NV_GATE=1 (T4.2).

## Changelog

- **v0.2 DRAFT edit 2026-11-15 (T2.4):** YAML now generated by `scripts/gen_hw_cost_table.py` (every number read from pinned results,
  `--check` compares YAML and MD); e22 pointer moved to the T2.11 rerun `20261115T201935510694Z_94ba25d` (v2 numbers unchanged); added
  section 3b with INT5 and the measured flat DPU (all flat td, replaces the hier/derived DPU rows); area sources stated as flat-only with
  the hier areas listed as unused; the 69.7 MHz fmax entry confirmed absent (only the td synthesis-STA 47.4 MHz and the SRAM limit are
  quoted); 34 MHz throughput 33.19 -> 33.20 (rounding) and 18.0 -> 18.04 GFLOPS/mm2; YAML tile NV_GATE=0 share 0.231 -> 0.229 (as in the MD).
  Values changed in the YAML only by this regeneration; the energy numbers are those of the 2026-11-14 draft.
- **v0.2 DRAFT (2026-11-14, T2.7):** all numbers re-measured on the T2.6 `td` netlists (flat Yosys, timing-driven), 9 real
  windows (2 models + 1 GPTQ-weight window), MX/NV paired on the same windows; area from the flat td run only (v0.1 mixed
  flat/hier sources, T2.4); fmax entry replaced by the td synthesis-STA r2r value and the SRAM clock limit; glitch factor
  from the td tile SDF run. Old -> new (nominal fJ/MAC): MX 946.7 -> 932.0, NV 1391.1 -> 1392.7, NV/MX 1.47 -> 1.49,
  glitch-corrected MX 1780 -> 1742 [EST], macro area 3.6255 -> 3.6807 mm2, GFLOPS/mm2 at 40 MHz 21.5 -> 21.2.
  Not yet sent.

- **v0.1 (2026-11-10, git 80bd20f):** first table. Energy/area/timing from e21/e22/e12 (flat), SRAM corners
  from e20, idle-NV from e23. Provisional.
