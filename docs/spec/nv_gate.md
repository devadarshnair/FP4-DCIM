# NV_GATE: idle-NV gating of the shared MX/NV column (T4.1)

Status: RTL written (T4.1, 2026-11-13); verification (NV_GATE=1 regression, macro power) is T4.2. Default `NV_GATE=0`.
Motivation: the e23 tile reconciliation measured an idle-NV tax in MX ops of about +106 fJ/MAC sequential
(~75 always-clocked NV pipeline flops per column) and +55 fJ/MAC combinational.

## Parameter plumbing
`NV_GATE` (int, default 0) is added to `column` -> `column_top` / `macro_top`, and to the DPU wrappers
`dpu_column` / `dpu_macro` (which instantiate the same `column_top`/`macro_top` with `LEAF=GENERIC`), so v2 and
DPU use one implementation. `column_top` ties `nv_mode=0`, so NV_GATE=1 there only gates permanently (useless but legal).
`rtl/gen` and `rtl/include` are untouched: the trees are shared and the gating lives in the hand-written `column.sv`.

## What is gated (NV_GATE=1, `nv_mode`=`op_nv` = 0)
(a) Clock gating, one existing-style `cim_icg` (sky130 `dlclkp_1`, enable = `nv_mode`, no new clock domain) per column
(32 per macro) driving only the NV-only stage-1 registers, 74 flops:
`r_h0s, r_h0c, r_h1s, r_h1c` (4x13 = 52, split half-tree carry-save outputs), `r_u0, r_u1` (2x8 = 16, E4M3 scale-significand
products), `r_d1` (6, second alignment amount). The load condition is unchanged (`nv_mode`), so register contents are
identical to NV_GATE=0 at every cycle where they are consumed; as before they hold across MX ops, so stage-2 NV logic
(13 b CPAs, 13x8 multipliers, second shifter) sees no input toggles in MX ops either way. They also drop their hold muxes
(plain `dfrtp` instead of enable flops). Always-clocked and not gated: `r_s, r_c` (MX), `r_d`, `r_valid, r_clear, r_nan, r_dneg, r_nv`, accumulator.
(b) Operand isolation (AND with `nv_mode`) of the NV-only stage-1 combinational logic: the two 4x4 E4M3 significand
multipliers (`u_w0,u_x0,u_w1,u_x1`, 16 ANDs) and the second exponent-difference adder (`e_w1,e_x1`, and the column-local copies
of `max_e_w,max_e_x` entering only `d1_raw`; 32 ANDs). `d_raw`/`d_sat` (MX/NV first block) stay un-isolated.
Isolation inputs are static or change per op, so the AND adds one gate delay on non-critical stage-1 exponent paths
(to be checked in T4.2 timing).

## Not gated (disclosed limitation)
The half-tree outputs `h0_*,h1_*` are internal nodes of the shared compressor tree that also produces `m_s,m_c`; they cannot
be isolated without gating the leaves/tree (MX needs them). The ~55 fJ combinational tax therefore only partly
disappears: the isolated part is the scale multipliers and the d1 adder.

## Expected effect [EST, unmeasured]
Sequential tax: clock-pin and enable-mux energy of 74 flops removed -> ~60-90 % of the +106 fJ/MAC, i.e. residual ~10-40 fJ.
Combinational tax: ~30-50 % of +55 fJ. Net idle-NV cost ~17 % of the v2 macro -> roughly 2-5 %.
Added: 32 ICGs + 48 ANDs per macro-column set (~0.2 % area, [EST]); the removed hold muxes largely offset this. NV-op energy
unchanged to first order (the ICG is enabled). All numbers become measured in T4.2.

## Verification status
- NV_GATE=0 equals as-built: Yosys sequential equivalence (`proc; flatten; async2sync; equiv_make; equiv_simple -seq 2;
  equiv_induct`) of `column` (NV_GATE=0) versus the pre-change `column.sv` was proven (4058 $equiv cells, 0 unproven, 37 s).
  Wrapper levels only forward a parameter with default 0.
- `verilator --lint-only -Wall` clean for NV_GATE=0 and 1 on column, column_top, macro_top, dpu_macro, dpu_column.
- DONE (T4.2, 2026-11-13, `results/nv_gate_verify/20261113T184402553793Z_0c165af`): directed tests 1-8 and 11 and lint with the SRAM model; originally pending: cocotb regressions of MX and NV streams including `op_nv` toggling for NV_GATE=1 (0 mismatches, results
  bit-identical to NV_GATE=0), gate-level gated-clock sim (ICG enable timing), synthesis/STA with ICG, macro power (e21/e22/e26 style),
  and the mixed-format stream.

## T4.2 directed tests (from the T4.1 review, 2026-11-13)
NV_GATE=1 vs NV_GATE=0 vs golden, bit-identical: (1) MX/NV/MX/NV toggling every op; (2) long MX run then NV (first NV cycle loads); (3) op_nv toggled with op_valid=0, then valid op; (4) op_nv changing on the falling edge vs just after the rising edge (latch transparency); (5) op_nv change during load_pending / set switch; (6) reset mid-stream with op_nv=1, then MX, then NV; (7) acc_clear coincident with an MX/NV transition; (8) mixed-format stream with random op_nv per op; (9) the same stream in gate-level / SYNTHESIS sim with the real dlclkp model; (10) STA clock-gating check on op_nv; (11) power sanity: zero r_h* toggles during MX. Also lint column_top/dpu_* and macro_top with the SRAM model on the include path.
