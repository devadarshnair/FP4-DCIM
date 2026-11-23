"""MIDAS-like macro power/area model (WP-37 part 2): activity (l2_midas_like) × per-action energy.

Per-action energies and areas come from the sky130 hd Liberty (power/liberty.py) and stated geometry -
they are NOT fitted to MIDAS's Fig. 21, so the share comparison in
experiments/e19_baselines/midas_validate.py is an out-of-sample fidelity test.
Components follow Fig. 21's partition exactly:

  cell_array      8T read ports: bit-line discharges (C_RBL·V² per 1-product) + cell area (bitcell µm²)
  mfa_adder_tree  per valid LCA-cycle MFA (sign-magnitude 3-input weighted add + SM->2C) and per column-cycle
                  LPAT + HPAT (32-input trees) + shift accumulator
  lca_peripheral  per valid LCA-cycle 3 sense amplifiers; per LCA-cycle VRDU (3-b subtract + start/end flags);
                  sign/exponent latches (area)
  left_peripheral per row-cycle CIB shift flops + LCG (4-b subtract, latch); per active row-cycle WL drive
                  (3 WLs × 16 LCAs gate load); row decoder (area)
  left_hole       4-b global counter + controller flops per cycle
  etc             MEPU AND-tree over 512 end flags per cycle; global BL driver per cycle

Dynamic energy only: leakage is not modelled (28 nm leakage cannot be inferred from sky130 Liberty), so
shares compare dynamic power. Activity factors α (toggle probability per active node-cycle) are stated
parameters with defaults from our own measured toggle statistics (~0.3 for data nets); all
assumptions are returned in the record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .liberty import load_cells

P = "sky130_fd_sc_hd__"
CELLS = ["fa_1", "ha_1", "and2_1", "nand2_1", "xor2_1", "dfxtp_1", "dlxtp_1", "inv_1", "buf_4", "mux2_1"]


@dataclass
class MidasAssumptions:
    vdd: float = 1.8
    c_rbl_ff: float = 8.0                # 8T read bit line of an LCA: ~11 cells × 0.5 fF drain + 2.5 fF wire
    c_wl_ff_per_lca: float = 1.5         # read-WL gate load per LCA on a row (2 read-port gates + wire)
    bitcell_um2: float = 2.0             # sky130 8T bitcell estimate (6T foundry cell ≈ 1.0-1.5 µm²)
    alpha_data: float = 0.3              # data-net toggle probability per active cycle
    mfa_fa: int = 4                      # MFA ≈ 3-input weighted add (2 FA) + SM->2C (≈2 FA-equivalents)
    tree_bits_lbs: int = 3               # LPAT input width (MXFP4: LBS columns)
    tree_bits_hbs: int = 3               # HPAT input width
    acc_bits: int = 20                   # per-column shift accumulator width
    cib_flops_per_row: int = 6           # sign + exponent + mantissa bits per row (MXFP4/6 modes)
    sa_cell: str = "dlxtp_1"             # latch-type sense-amplifier proxy


def _cells():
    return {k: v for k, v in zip(CELLS, load_cells([P + c for c in CELLS]).values(), strict=True)}


def macro_record(act: dict, a: MidasAssumptions | None = None, rows: int = 32, cols: int = 16,
                 cells_per_lca: int = 32) -> dict:
    """Energy (fJ/op) and area (µm²) per Fig. 21 component, averaged over the ops in `act`."""
    a = a or MidasAssumptions()
    c = _cells()
    v2 = a.vdd**2
    fa = c["fa_1"].toggle_energy_fj(vdd=a.vdd)
    dff, lat, inv = c["dfxtp_1"], c[a.sa_cell], c["inv_1"]
    dff_clk = dff.clock_energy_fj()
    and_e = c["and2_1"].toggle_energy_fj(vdd=a.vdd)
    n_ops = len(act["cycles"])
    cyc = act["cycles"].mean()
    valid = act["valid_lca_cycles"].mean()
    rows_act = act["active_row_cycles"].mean()
    ones = act["bl_ones"].mean()
    lca_cycles = act["lca_cycles_total"].mean()
    ncol = int(round(act["lca_cycles_total"][0] / max(act["cycles"][0], 1) / rows)) or cols
    tree_fa = (rows - 1) * (a.tree_bits_lbs + a.tree_bits_hbs + 2)  # two 32-input trees, ~log growth absorbed
    e = {
        "cell_array": ones * a.c_rbl_ff * v2,
        "mfa_adder_tree": valid * a.mfa_fa * fa * a.alpha_data
        + cyc * ncol * (tree_fa * fa * a.alpha_data + a.acc_bits * (dff_clk + a.alpha_data * fa)),
        "lca_peripheral": valid * 3 * lat.toggle_energy_fj(vdd=a.vdd) * a.alpha_data
        + lca_cycles * (3 * fa * a.alpha_data * 0.5),  # VRDU 3-b subtract, ~half the cycles toggle
        "left_peripheral": cyc * rows * (
            a.cib_flops_per_row * (dff_clk + a.alpha_data * dff.toggle_energy_fj(vdd=a.vdd))
            + 4 * fa * a.alpha_data
        )
        + rows_act * 3 * ncol * a.c_wl_ff_per_lca * v2,
        "left_hole": cyc * (8 * dff_clk + 8 * a.alpha_data * fa),
        "etc": cyc * ((rows * ncol - 1) * and_e * 0.05 + 10 * inv.toggle_energy_fj(vdd=a.vdd)),
    }
    lcas = rows * ncol
    area = {
        "cell_array": lcas * cells_per_lca * a.bitcell_um2,
        "mfa_adder_tree": (lcas * a.mfa_fa + ncol * tree_fa) * c["fa_1"].area + ncol * a.acc_bits * dff.area,
        "lca_peripheral": lcas * (3 * lat.area + 3 * c["fa_1"].area + 2 * dff.area + 4 * lat.area),
        "left_peripheral": rows * (a.cib_flops_per_row * dff.area + 4 * c["fa_1"].area + 3 * c["buf_4"].area
                                   + 6 * c["nand2_1"].area),
        "left_hole": 8 * dff.area + 30 * c["nand2_1"].area,
        "etc": (lcas - 1) * c["and2_1"].area + 10 * c["buf_4"].area,
    }
    tot_e, tot_a = sum(e.values()), sum(area.values())
    return {
        "n_ops": n_ops,
        "cycles_per_op": float(cyc),
        "energy_fj_per_op": e,
        "area_um2": area,
        "power_share_pct": {k: 100 * v / tot_e for k, v in e.items()},
        "area_share_pct": {k: 100 * v / tot_a for k, v in area.items()},
        "fj_per_mac": tot_e / (rows * ncol),
        "assumptions": asdict(a),
    }


def compare_shares(rec: dict, targets: dict) -> dict:
    """|model - MIDAS Fig. 21| in percentage points per component, and pass at ±5 pp."""
    out = {}
    for comp, t in targets.items():
        dp = rec["power_share_pct"][comp] - t["power"]
        da = rec["area_share_pct"][comp] - t["area"]
        out[comp] = {"power_model": rec["power_share_pct"][comp], "power_midas": t["power"], "power_dpp": dp,
                     "area_model": rec["area_share_pct"][comp], "area_midas": t["area"], "area_dpp": da,
                     "pass": abs(dp) <= 5 and abs(da) <= 5}
    return out


def shares_pass(cmp: dict) -> bool:
    return all(v["pass"] for v in cmp.values())
