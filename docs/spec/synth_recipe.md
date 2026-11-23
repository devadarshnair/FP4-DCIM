# Synthesis recipe for energy netlists (T2.6)

Status: defined 2026-11-13 (T2.6). Implemented in `flows/synth/sweep.py`
(`SYNTH_RECIPE`), `flows/synth/td_repair.tcl` and the driver `experiments/e30_td_synth/run.py`.
Results: see §5 (filled from the e30 runs).

## 1. Why

The e12/e13 netlists behind every pre-T2.6 energy number (`base` recipe below) were mapped by ABC with no
delay target and no fanout buffering. At macro level their reg-to-reg paths are 132 ns (v2) and
73 ns (INT5), so they cannot run at the 40 MHz power clock. We also observed that the OpenLane2
synthesis netlist (e24) costs v2 +56 % logic energy on window `real0` (v2/INT5 4.79 -> 3.58). T2.6 needs one
recipe, applied the same way to every design, whose netlists meet 40 MHz (or say why not), plus a
recipe-sensitivity axis.

## 2. The recipe (`td`), identical for v2, bit-serial INT5 (fixed back-to-back RTL), DPU, and NV_GATE=0/1

Corner and libraries: sky130_fd_sc_hd `tt_025C_1v80`, OpenRAM `sky130_sram_1kbyte_1rw1r_32x256_8` TT_1p8V_25C
as a black box (Liberty timing/area), ideal clock, `flows/synth/constraints/column.sdc` (inputs driven by
`buf_2`, 0 ns I/O delay, 5 fF output load), period **25 ns (40 MHz)**.

Stage 1, Yosys (0.69, conda `cim`):
1. `synth -flatten -top <top>` (hierarchical `synth` only where flat Yosys does not fit the machine: DPU
   macro), same RTL sources and parameters as e12.
2. `dfflibmap -liberty <lib>` with the dont-use set.
3. `abc -D 20000 -constr <c> -dont_use ... -liberty <lib>`. `-D 20000` is the delay target in ps = 0.8 x
   period: 20 % margin for wires, clock skew/uncertainty and setup. `-constr` (`set_driving_cell
   sky130_fd_sc_hd__inv_2`, `set_load 5` fF) switches ABC to its constrained script, which adds
   `buffer; upsize {D}; dnsize {D}` (fanout buffering and gate sizing against the target).
4. Dont-use set: `sky130_fd_sc_hd__lpflow_*` (power-intent isolation/KAPWR cells; the `base` recipe used
   ~800 `lpflow_inputiso1p`/`lpflow_isobufsrc` per tile as plain logic), `clk*` (clock-tree buffers/inverters
   and delay buffers), `dly*`, `probe*`.
5. `opt_clean -purge; hilomap` (as e12).

Stage 2, OpenROAD (openlane2:2.3.10 image), `flows/synth/td_repair.tcl`, no placement (zero wire RC):
1. `set_max_fanout 10`, `set_max_transition 0.75` ns (the e24 OpenLane2 values), Liberty max capacitance.
2. `repair_design` (slew/cap/fanout), then `repair_timing -setup` at 25 ns.
   Measured limit: without placement `repair_design` inserts no fanout buffers (all instances sit at one
   location) and `buffer_ports` only puts a port buffer in front of the same fanout (worse slew, tile any-path
   slack 11.2 -> 6.9 ns), so it is not used. Fanout buffering of the combinational network, including
   flop-Q fanout, is ABC's `buffer` (stage 1). Residual DRVs are reported, not fixed: the reset and static
   write-enable port nets (`rst_n`, `ew_wr_en`, `maxw_wr_en`; constant during the compute window, so
   energy-neutral) and the SRAM `dout` fanout (6 vs the OpenRAM Liberty max_fanout 4).
3. Clock nets (including ICG outputs) are not touched: they are CTS's job, as in every synthesis-stage flow,
   and the power flow keeps the ideal clock (OpenSTA Clock group from the SDC), as in e13.

Checks reported per netlist: area (Yosys `stat -liberty` of the final netlist), cell/buffer/x1/lpflow counts,
OpenSTA (openlane2 image) any-path WNS and register-to-register WNS at 25 ns (`flows/synth/sta_macro.tcl`).
Power is measured with the unchanged e13 flow (zero-delay Icarus GLS bit-exact against the scoreboard, VCD
sliced to the compute window and rebased to t=0, OpenSTA `report_power`, OpenRAM Liberty power excluded, SRAM
per-access charge) on the `e13_power_rb` programs (8 real + 2 random windows).

## 3. Recipe-sensitivity axis

| recipe | what | role |
|---|---|---|
| `base` | e12: `abc -liberty`, no delay target, no buffering, no dont-use | the pre-T2.6 netlists, re-run on the current RTL (fixed INT5, NV_GATE param) |
| `td` | §2 | **the recipe for all energy numbers from T2.7 on** |
| `td_ol` | `td` + OpenLane2's `no_synth.cells` (copied to `flows/synth/constraints/ol_no_synth.cells`, SHA-256 `064a09ab...65ed9`) in the dont-use set: no x1 drive strengths, as the e24 OpenLane2 synthesis | explains the e24 +56 %: a cell-set (routability) choice, not timing |

## 4. Limits

- Synthesis-stage timing: no wires, ideal clock. The 20 % margin is a pre-layout allowance, not a guarantee;
  post-route timing is e24 (tile) and remains open for the macro.
- Zero-delay power (no glitches) as e13; the SDF glitch factors (e13_sdf_rb) are separate.
- The DPU macro is hierarchical Yosys: ABC runs per module with the same `-D`; OpenROAD links the
  hierarchy flat for repair and STA.

## 5. Results

### 5.1 Tile level (4-column tiles of e13/e24; runs `results/e30_td_synth/20261113T201935868931Z_0550fbd`, `results/e30_td_power/20261113T205342783613Z_65f5076`)

Area (std-cell logic, um2) and synthesis STA (ideal clock, no wires, tt) at 25 ns:

| tile | base area | td area | td_ol area | base r2r fmax | td r2r fmax | td_ol r2r fmax | td buffers |
|---|---|---|---|---|---|---|---|
| v2 (MX-folded) | 149,273 | 152,267 (+2.0 %) | 185,405 (+24.2 %) | 70.9 MHz | 70.6 MHz | 74.5 MHz | 731 |
| INT5 bit-serial (fixed) | 138,384 | 141,088 (+2.0 %) | 164,089 (+18.6 %) | 61.6 MHz | 75.4 MHz | 74.5 MHz | 651 |
| DPU | 221,281 | 225,464 (+1.9 %) | 292,873 (+32.4 %) | 72.9 MHz | 74.7 MHz | 74.4 MHz | 967 |
| v2 keepcol NV_GATE=0 | 216,119 | 221,908 | - | 47.2 MHz | 48.3 MHz | - | 991 |
| v2 keepcol NV_GATE=1 | 214,145 | 217,898 | - | 57.6 MHz | 60.1 MHz | - | 929 |

All tiles meet 25 ns with every recipe (WNS > 0 any-path and r2r); at tile level the unbuffered netlists were
already timing-feasible, so `td` changes little. Residual DRVs after `td` (reported, not fixed, §2): 67 fanout
nets per flat tile (3 port nets + 64 SRAM dout pins), 0 slew, 0 cap.

Zero-delay energy, e13 flow, mean of the 8 real `e13_power_rb` windows (fJ/MAC, logic + SRAM per-access; bootstrap
95 % CI of the paired ratio over windows):

| recipe | v2 | INT5 | DPU | EE v2/INT5 | EE v2/DPU | real0 v2/INT5 | real0 v2/DPU |
|---|---|---|---|---|---|---|---|
| e13_power_rb (old netlists, pre-fix INT5 6.05 cyc/op) | 697.4 | 3019.1 | 1126.2 | 4.34 | 1.61 | 4.79 | 1.53 |
| base, current RTL (INT5 5.05 cyc/op) | 699.3 | 2528.4 | 1127.4 | 3.63 [3.55, 3.74] | 1.61 [1.58, 1.63] | 3.99 | 1.52 |
| **td** | **708.7 (+1.3 %)** | **2540.9 (+0.5 %)** | **1149.6 (+2.0 %)** | **3.60 [3.52, 3.72]** | **1.62 [1.59, 1.64]** | 3.97 | 1.53 |
| td_ol | 923.3 (+32 %) | 2877.0 (+14 %) | 1519.6 (+35 %) | 3.13 [3.06, 3.21] | 1.64 [1.62, 1.66] | 3.43 | 1.56 |

Reading: (i) the timing-driven recipe costs +0.5-2.0 % energy and moves no ratio by more than 1 %; (ii) the
e24 "+56 %" for the OpenLane2 synthesis netlist is not timing closure: about half of it is the OL2 cell set (no x1
drive strengths; real0 v2 logic td 571 -> td_ol 735 vs OL2 889 fJ/MAC; the rest is OL2's `AREA 0` ABC script and
other OL2 settings, not separated here), and the cell set
penalises v2 and DPU more than INT5: the OL2 list also removes x1 flops, but x2 flops cost little more
(INT5 sequential 1554 -> 1583 fJ/MAC), while combinational energy rises 34-39 % (v2 545 -> 755, INT5 912 -> 1219,
DPU 987 -> 1351), and combinational logic is 77-87 % of v2/DPU logic energy vs 36 % of INT5;
(iii) the v2/INT5 drop from 4.34 to 3.63 is the INT5 back-to-back fix (6.05 -> 5.05 cycles/op), not the recipe.
Sequential/combinational split (td, fJ/MAC): v2 153/545, INT5 1554/912, DPU 152/987.

Idle-NV tax at tile level (td, MX-only stimulus, keepcol tile vs the NV-folded v2 tile): NV_GATE=0 +22.9 %
(871.0 fJ/MAC; sequential 258 vs 153), NV_GATE=1 +8.2 % (766.5; sequential 161): gating removes 64 % of the tax
and 12.0 % of the keepcol tile energy.

### 5.2 Macro level (32x32 macros, 8 OpenRAM stores; synthesis + STA only, no power; hierarchical Yosys for all three so the three are like-for-like and the DPU stays hierarchical)

Runs: `results/e30_td_macro_v2_macro_hier/20261114T012104246937Z_e3a93a6`,
`results/e30_td_macro_bs_macro_hier/20261114T012651433178Z_e3a93a6`,
`results/e30_td_macro_dpu_macro_hier/20261114T013055657046Z_e3a93a6`; old = `results/e12_macro_synth_*_hier/latest`
(base recipe, INT5 before the back-to-back fix).

| macro (hier) | logic area base -> td (mm2) | cells base -> td | td buffers | r2r min period base -> td | r2r 1/period (synthesis STA, ideal clock, no wires) base -> td |
|---|---|---|---|---|---|
| v2 | 2.207 -> 2.253 (+2.1 %) | 240,977 -> 251,526 | 13,446 | 164.9 -> **19.9 ns** | 6.1 -> **50.1 MHz** |
| INT5 bit-serial | 1.082 -> 1.098 (+1.5 %) | 113,620 -> 116,428 | 4,637 | 43.3 -> **13.4 ns** | 23.1 -> **74.7 MHz** |
| DPU | 3.196 -> 3.230 (+1.0 %) | 375,302 -> 384,489 | 12,641 | 127.9 -> **20.6 ns** | 7.8 -> **48.6 MHz** |

**Acceptance (T2.6): macro reg-to-reg WNS >= 0 at 25 ns tt holds for all three td netlists** (+5.06 / +11.62 /
+4.41 ns; synthesis stage: ideal clock, no wires, so the 20 % ABC margin is what is left for wires and skew; v2
uses 19.9 of the 20 ns target). Area ratios barely move (v2/INT5 logic 2.04 -> 2.05, DPU/v2 1.45 -> 1.43).

Not met, and why: the any-path WNS stays negative (-138.9 / -34.1 / -138.9 ns). The worst path is the
input port `sc_wr_en` (scale-RF write enable, fanout 12,801 in v2) into the ICG gating check; other unbuffered
port nets are `rst_n` (9,246), `maxw_wr_en`, and the per-op broadcast inputs `op_nv`, `op_u_x*` (fanout 160-192).
Port nets reach the column instances directly, so ABC never sees them, and placement-free `repair_design` does
not buffer (§2). These are macro-boundary nets: write-path controls are static during compute, and the per-op
broadcast inputs are the boundary wires already charged as [EST] in e21/e22. A post-placement `buffer_ports` +
`repair_design` (OpenLane2/e24 style) fixes them; their buffer-tree energy is not in any number here. Residual
fanout-DRV nets: 1,672 / 312 / 2,504.

Flat td netlists of v2 and INT5 (the flat flow of e21; flat v2 Yosys peaked 4.95 GB under `safe_run`, MemAvailable
min 1.2 GB; runs `results/e30_td_macro_v2_macro/20261114T014522099345Z_e3a93a6`,
`results/e30_td_macro_bs_macro/20261114T013951562875Z_e3a93a6`; old = `results/e12_macro_synth_{v2,bs}_macro/latest`):

| macro (flat) | logic area base -> td (mm2) | cells base -> td | td buffers | r2r min period base -> td | r2r 1/period (synthesis STA, ideal clock, no wires) td | r2r WNS @25 ns |
|---|---|---|---|---|---|---|
| v2 | 2.100 -> 2.155 (+2.6 %) | 262,972 -> 276,374 | 14,343 | 132.1 -> **21.1 ns** | **47.4 MHz** | +3.91 ns |
| INT5 bit-serial | 1.036 -> 1.059 (+2.2 %, incl. INT5 RTL fix) | 117,967 -> 125,379 | 5,116 | 73.1 -> **13.5 ns** | **74.2 MHz** | +11.51 ns |

Netlists for T2.7 (build artefacts, regenerate with the driver): `build/e30_td_synth/macro_td/{v2,bs}_macro_netlist.v`
(flat), `build/e30_td_synth/macro_td_hier/{v2,bs,dpu}_macro_netlist.v` (hierarchical Yosys, written flat by
OpenROAD), tiles `build/e30_td_synth/{base,td,td_ol}/*_netlist.v`.

## 6. What T2.7 must do with this

- Use `td` netlists for every energy number (e21 macro power for v2/INT5 flat, DPU hierarchical; e13/e15 tile).
- Expect the tile result to carry over: `td` adds 0.5-2 % energy at tile, so the macro ratios should move by
  about the same; the large moves are the INT5 back-to-back fix (6.05 -> 5.05 cycles/op) and any cell-set choice (`td_ol`).
- Keep `td_ol` (and `base`) as the recipe-sensitivity axis in the quoted range; quote `td` as primary.
- Macro boundary nets (port fanout) stay unbuffered; disclose, or add a placement-based `buffer_ports`/`repair_design`
  step if the macro goes through P&R.

Done 2026-11-14 (T2.7): macro e21 on `macro_td` (v2/INT5 flat) and `macro_td_hier` (DPU; v2 hier as calibration,
hier/flat 1.052), tile e13/e15 on `td/`, glitch factors from `e13_sdf_td`. `td_ol`/`base` were not re-run at macro level.
