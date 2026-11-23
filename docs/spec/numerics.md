# Numerics specification - v2 macro (WP-22, frozen 2026-10-31)

Widths are generated from the number formats by `src/cimfp4/rtlgen/pkg.py` (`python -m cimfp4.rtlgen`)
into `rtl/include/cim_pkg.sv` and the table [`numerics_table.md`](numerics_table.md). **Never hand-edit
widths in RTL**; change `pkg.py` and regenerate. Sources: OCP MX v1.0, NVFP4 recipe; R03 §8.

## Datapath (one weight column; MX mode)

| Stage | Signal | Width | Why |
|---|---|---|---|
| activation in | `x[3:0]` per row, `e_x` UE8M0 per 32-row block | 4 b, 8 b | E2M1 code; OCP MX v1.1 scale |
| row broadcast | `X' = 2|x|` (4 b), `3X'` (6 b), `s_x` | 11 wires/row | 2|x| ∈ [0,12]; 3·12 = 36 |
| weight store | M sets × 4 b per compute unit, SRAM | `SET_IDX_BITS` = 4 for M=16 | |
| leaf product | 9 b two's complement (or 8 b mag + sign) | `PROD_TC_BITS` = 9 | |p×4| ≤ 144 |
| tree | 32:2 carry-save, 2 × 14 b | `MX_SUM_S_BITS` = 14 | 32·144 = 4608 -> 13 b unsigned, 14 b signed |
| pipeline reg | carry-save output | 28 flops/column | |
| block result | S (14 b), exponent sum e_w + e_x - 2 (9 b), NaN flag | `MX_EXP_SUM_BITS` = 9 | [-256, 252] |
| alignment | d = (max_e_w,col - e_w)[static] + (max_e_x - e_x)[broadcast], saturating at W - 1; term = S << (BASE - d) or S >>> (d - BASE), BASE = W - 15 - guard | `SHIFT_BITS` = 6 (W=40) | anchor known before compute. **Correction:** R03's saturation at W - 14 would truncate terms that must floor to 0/-1 |
| accumulator | W-bit window, LSB = 2^(Etop - W + 1) | `WINDOW_BITS` = 40 | exact on LLM-like data (WP-05 test) |
| output | normalize once per output -> FP32 | | off critical path |

## NVFP4 mode differences

| Signal | Width | Note |
|---|---|---|
| two 16-row half-trees, each 2 × 13 b -> 13 b CPA | `NV_SUM_S_BITS` = 13 | halves stay separate until scaled |
| scale significand product u_w·u_x | 8 b | u ∈ [0,15] (E4M3 1.mmm·8 / 0.mmm·8), ≤ 225 |
| scaled half-sum S·u_w·u_x | 20 b signed | 16·144·225 = 518 400 |
| half exponent k_w + k_x - 2 | **6 b signed** | k ∈ [-9, 5] => [-20, 8]. **Correction:** R03 §8 listed 5 b, which cannot hold -20 |
| per-tensor FP32 scales | applied once per output | |

## Baseline (a): bit-serial INT5 column (WP-36)

Same storage (`wset_sram`, 4 columns per macro), same exponent path and W-bit window as v2. E2M1 -> INT5
(2v ∈ [-12, 12]) decoded statically at the latch; activations fed LSB-first over **5 cycles per 32-row
block** (bit 4 weighted -16); five 32-input popcount trees (`col_tree_unsigned_w1_r32`, 23 FA/12 HA each)
per column; combiner S_i = Σ_j ±(pc_j ≪ j); shift-add block accumulator; then align + window.
Throughput: **32 MAC / 5 cycles = 6.4 MAC/cycle/column** (v2: 32 MAC/cycle/column) => iso-throughput
comparison needs 5× the bit-serial columns (or 5× its clock). A set switch waits until all columns
finish the current block.

## Baseline (c) and generic-leaf ablation (WP-39)

- `generic_leaf`: E2M1 -> INT5 for both operands + plain 5×5 signed multiplier + SM/OC output; formally
  equal to the LUT spec. Synthesized (Yosys/ABC NAND mapping): **292 NAND + 129 NOT vs 91 NAND + 35 NOT
  for the FP4 AND-OR leaf (~3.3×)** - part of the gap is the generic leaf's sign-magnitude conversion.
- `column_top #(LEAF="GENERIC")` = v2 storage + generic leaf (waterfall ablation step).
- `dpu_column` (baseline c) = generic lanes + central weight SRAM; the weight bus (32 b/beat, 16 beats
  per set switch per 4-column group) is modeled in `golden/l2_digital.py`; the separate
  `central_wbuf.sv` of the plan is folded into the shared `wset_sram` (same macro), with distance
  represented in the power flow by wire capacitance.
- Wire model (`flows/pdk/wire_model.yaml`, from the sky130 tech LEF): ≈0.076-0.088 fF/µm per layer at
  minimum width (area + edge terms only; **no coupling => lower bound**).

## NVFP4 column mode (WP-31)

`nv_mode=1`: the half-tree carry-save outputs (2 × 2 × 13 b) are registered instead of the MX pair;
stage 2 forms S_h (13 b), P_h = S_h · (u_w,h · u_x,h) (8 b scale-significand product, 21 b signed),
aligns each half with its own d_h = (max_k_w - k_w,h) + (max_k_x - k_x,h) using
BASE_NV = W - 22 - GUARD, and adds both into the window. Plain signed multiplier (the Booth/fused
variant of R03 is an area/energy knob, not a numerics change). Verified: all 16×16 scale-significand
pairs (subnormal and normal E4M3), ≥10^5 random NV ops bit-exact vs the NV column model, which equals
L1 Window bitwise (480 NVFP4 dot products); MX regressions unchanged.

## Macro top (WP-32)

`macro_top`: 32 rows × 32 columns; one shared set of 32 row drivers (X′, 3X′, sign) feeds all columns;
8 weight-set stores (`sky130_sram_1kbyte_1rw1r_32x256_8`, 4 columns each = 8 × 8 Kb = 64 Kb at M=16,
matching `flows/pdk/sram_macros.yaml`) switch sets in parallel (10 cycles, single buffer); per-(set,
column) scale RF {e0, e1, u0, u1, nan} (25 b) and per-column max_e_w anchors, clock-gated. `op_nv`
selects MXFP4/NVFP4 per op. Verified bit-exact (all 32 columns) vs the per-column reference models for
MX and NV at M=1 and M=16 (SRAM), M=16 latch build, and real Qwen/SmolLM capture tiles.

## Ping-pong weight-set store (WP-67)

`wset_pingpong`: OpenRAM macro + two 512-b active buffers per 4-column group. The macro computes from
buf[sel] while a prefetch fills buf[~sel] through both SRAM ports (8 cycles). Set-switch latency:
**1 cycle** if the needed set was prefetched; **≤ 11 cycles** demand load from idle; **≤ 20 cycles**
worst case when a prefetch of another set is in flight (prefetch is not aborted). SRAM writes are
accepted whenever no read is in progress (write-while-compute). Verified on 10^4 random interleavings
(writes, prefetches, op set requests; snapshot semantics). Macro-level write port: 8 groups × 32 b =
256 b/cycle when all groups are written together (requirement ≥256 b for ping-pong at R≥16).

## Online activation quantizers and tile anchor (WP-33, WP-68)

- `act_quant_mx`: one 32-element bf16 block per cycle -> E2M1 codes + UE8M0 (OCP v1.1 expAdj or RCEIL);
  bit-identical to `mx_quant` (corner suite: all 7 ties, expAdj/RCEIL boundaries, all-zero, bf16
  subnormals, saturation, NaN/Inf -> 0xFF; random blocks).
- `act_quant_nv`: one 16-element bf16 block per cycle, **"hw" convention**: scale = E4M3_RNE_satfinite(exact
  amax·c), c = fp32(s_enc/6) config register (one rounding instead of the reference's two fp32 roundings);
  elements by exact comparison against the 7 midpoints (n/4)·u·S·2^(k+q). Bit-identical to
  `nv_quant(..., "hw")`; on LLM-like data the hw and reference conventions gave identical scales (0 of
  524,288 blocks differ).
- `anchor_detect`: tile-local max_e_x for AccumMode.Hier. **Latency = tile_blocks cycles; buffer =
  tile_blocks × (32·4 + 8) bits** of activation codes + exponents per tile (e.g. 16 blocks -> 2,176 b).
  Verified on 10^5 random tiles.

## HIER interface (window -> cluster reducer; WP-66)

K is split into tiles of `tile_blocks` consecutive 32-row blocks (one macro pass each). Each tile
accumulates in a W-bit window whose LSB is anchored on **tile-local** maxima only:
lsb_tile = (max_tile e_w + max_tile e_x - 2) + int_bits + guard - W + 1, guard = ⌈log2 tile_blocks⌉ -
so the max-e_x detector only needs to see one tile of activations (buffer = one tile; WP-68 records the
latency). Each macro emits per column per pass: window value (W b), lsb_tile (10 b signed), NaN flag.
Reducer options (L1 `Hier`): `exact` (lossless), `fp32` (sequential fp32 adds), `("window", Wr)`
(aligned to the max tile anchor, truncating). Verified: Hier(tile = K) ≡ Window; Hier(W=60, exact) ≡
Exact on LLM-like data; tile anchors unaffected by exponents outside the tile; L1 Hier bit-exact vs an
independent reference on 10^5 MX and 10^5 NVFP4 dot products.

## Storage (WP-63)

- Macro: `sky130_sram_1kbyte_1rw1r_32x256_8` (OpenRAM, 256 × 32 b, 1RW + 1R), one per **4 columns**:
  4 columns × M=16 sets × 32 rows × 4 b = 8192 b. Word address = set·16 + col·4 + k; word k of a column
  holds rows 8k..8k+7.
- Set switch: 16 words via both ports in 8 read cycles -> **10 cycles** from `load_req` to `busy` low
  (1 request + 8 reads + 1 capture); 512 b/set-switch/group => 64 b/cycle/group. Latch-store ablation
  (`wset_latch`): 2 cycles.
- Inventory (generated, `flows/pdk/sram_macros.yaml`): 0.191 mm², 42 Kb/mm², 0.599 pJ/bit/access at TT
  1.8 V 25 °C - the OpenRAM Liberty energy is a **placeholder** (constant for read/write/idle), so SRAM
  energy enters G1 as a band, not a point value. **Superseded for energy by e20:** characterized range in
  [`sram_energy.md`](sram_energy.md) / `sram_energy.yaml` (e20 SPICE; nominal 32 pJ per read word in a
  set switch, 514 pJ per set switch per macro, 7.1 pJ per clocked idle cycle).

## Weight write

256-bit write port (`WRITE_PORT_BITS`) = 64 weights/cycle; per column-block e_w (UE8M0) or E4M3; static
max_e_w,col registered at load time.

## Exceptions

- UE8M0 0xFF (NaN) on either operand -> NaN flag set, propagates to the output.
- **NVFP4 E4M3 block scales (T1.5).** Definition: E4M3 "fn" variant (bias 7, no Inf, one NaN pattern
  S.1111.111; max finite 448), as used for the NVFP4 block scale [abecassis2025nvfp4, §2] and the OCP
  E4M3 element encoding [rouhani2023ocpmx]; implemented in `cimfp4.formats.codecs`.
  | scale code | class | contract |
  |---|---|---|
  | `0x7F`, `0xFF` | NaN | block scale NaN: the NaN flag is set exactly as for UE8M0 0xFF (below); `u`, `k` are don't-care (golden stores 0) |
  | `0x00` | zero | `u = 0`: the 16-element block contributes exactly 0 (P_h = 0), **no NaN**, even if its element codes are non-zero; its `k` (= -9) still takes part in the max-`k` anchor like any finite code |
  | `0x01..0x07` | subnormal | exact: `u = m` (1..7), `k = -9`, value `m·2^-9` - no special handling, no flush-to-zero |
  | `0x08..0x7E` | normal | `u = 8+m`, `k = e-10` (value `(8+m)·2^(e-10)`), max 448 |
  | `0x80..0xFE` | reserved (sign set, not NaN) | block scales are amax-derived and non-negative; the quantizer never emits them. Not part of the contract: golden/stimulus code raises `ValueError` (`e4m3_scale_fields`); RTL behaviour undefined (the macro scale RF stores only `u,k` magnitudes, so the sign would be dropped silently - the writer must not drive them) |

  Propagation (same as MX): the macro scale RF holds one `nan` bit per (set, column) = **OR of the NaN
  state of both 16-element weight-block scales of that 32-row block**; the activation side is `op_x_nan` =
  OR of both activation-block scales. `nan_in = op_x_nan | rf.nan` is registered per op and **sticky**
  across ops until `acc_clear`; the dot product (and `nan_out`) is NaN iff any weight or activation scale
  code in it is NaN. NaN overrides the accumulator value (fp32 NaN at the output). The NaN flag is
  independent of format mode (`op_nv`). Per-tensor fp32 scales are finite by construction and are not
  covered. L1: `nv_nan_mask` / `l1_dot`; L2: `nv_column_nan`; tests `tests/formats/test_e4m3_scale_rules.py`.
  *RTL status:* the RTL implements NaN as a pre-decoded flag input (`sc_wr_nan`, `op_x_nan`); decoding
  0x7F/0xFF to that flag is the stimulus writer's job. **Confirmed (T1.5, cocotb `test_macro_mixed`, macro_top
  and dpu_macro):** NV `nan_out` per column equals `nv_column_nan` per op (weight-scale NaN records and
  `op_x_nan` on NV ops, sticky until `acc_clear`), non-NaN columns bit-exact; the scale RF holds k = -9 (zero and
  subnormal scales). **Writer rules** (`rf_record`, `anchor_maxw` in `tb/cocotb/test_macro_mixed.py`): reserved
  codes 0x80..0xFE are rejected (ValueError, nothing reaches the RF); the static `max_e_w` anchor is the max over
  all blocks of the column including zero-scale (u = 0, k = -9) blocks and NaN blocks (stored k = 0).
- E2M1 has no NaN/Inf; ±0 handled by the AND-OR leaf (outputs 0).
- Products are in units of 2^-2 (`PROD_EXP_OFFSET` = -2).

## Generated parameter table

See [`numerics_table.md`](numerics_table.md) (auto-generated, includes derived ranges).
