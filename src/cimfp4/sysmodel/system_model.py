"""Analytical system-level model of a weight-stationary array of v2 / INT5 bit-serial / DPU macros (e25).

Only measured per-op costs enter (all loaded from results/ and docs/spec/sram_energy.yaml by `load_inputs`);
everything else is an explicit assumption in `Assumptions`. Scope: the *weight GEMMs* (q,k,v,o,gate,up,down,
optional lm_head) executed on the 32x32 macro. Optional (e27/T2.2): e28 activation quantizer, anchor, tile buffer and
HIER reducer (`Assumptions.bundle`). NOT modelled: attention BMMs (activation x activation), DRAM energy/bandwidth, FP32 tensor scale, norms/softmax.

Macro semantics (docs/spec/numerics.md): one op = one 32-element activation block x 32 columns = 1024 MACs.
One weight *set* = a 32 (K) x 32 (N) tile (4096 weight bits + 32 x 25 b scale RF). A macro holds M=16 sets and
switches between resident sets in ~10 cycles; a set is (re)written through a 256-bit port (16 cycles).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

MACS_PER_OP = 1024
TILE = 32
NG = 8  # OpenRAM stores per macro
SET_BITS = TILE * TILE * 4
SCALE_RF_BITS = TILE * 25
WRITE_PORT_BITS = 256
WRITE_CYCLES_PER_SET = SET_BITS // WRITE_PORT_BITS  # 16
WORDS_PER_SET = SET_BITS // 32  # 128 32-bit words over the 8 stores
READ_CYCLES_PER_SWITCH = 8  # numerics.md: 1 request + 8 reads + 1 capture = 10 cycles


@dataclass(frozen=True)
class Design:
    name: str
    e_logic_fj_mac: float  # measured logic dyn+leak (e22), real data, zero-delay, R=256 amortised
    e_wire_fj_mac: float  # [EST] boundary wires (e22)
    c_op: int  # steady cycles per op (derived from e21 cycles/op)
    ovh_visit: float  # non-op cycles per set visit (switch + fill/drain), derived from e21
    idle_logic_pj_cycle: float  # [EST] clock-driven logic energy in a non-op cycle (e21 seq-int + clock-group power)
    pr_factor: float  # e24 post-route / zero-delay total ratio (tile level) [EST at macro]
    fmax_pr_mhz: float  # e24 post-route fmax lower bound (tile) [EST at macro]
    fmt: str = "mx"  # activation/weight format of the macro mode: mx | nv (selects the e28 quantizer)


@dataclass(frozen=True)
class Assumptions:
    n_macros: int = 64
    f_mhz: float = 40.0
    sram_corner: str = "nominal"  # low | nominal | high ([EST])
    sram_clock: str = "as_built"  # as_built (free-running) | gated [EST design change]
    post_route: bool = False  # scale logic+wire energy by e24 factor [EST]
    idle_logic: bool = False  # charge idle-cycle logic clock energy [EST]
    wires: bool = True
    overlap_reload: bool = True  # next set written while current set computes (write-while-compute)
    include_lm_head: bool = False
    # e28 (T2.3) online quantizer + anchor + tile-buffer + HIER-reducer bundle:
    #   none | per_token (each activation block quantised once per token row and reused over all N/32 column groups;
    #   HIER per useful MAC) | per_op (quantizer/anchor/buffer re-run for every macro op: e28 per-MAC table, conservative)
    bundle: str = "none"


@dataclass
class Inputs:
    designs: dict[str, Design]
    sram: dict[str, dict[str, float]]
    provenance: dict[str, str] = field(default_factory=dict)
    bundle: dict[str, dict[str, float]] = field(default_factory=dict)  # fmt -> e28 per-block / per-MAC fJ


def load_bundle(root: Path, run: str | None = None) -> tuple[dict[str, dict[str, float]], str]:
    """e28 summary (T2.3): per 32-element activation block (= per macro op) fJ for quantizer, anchor and gated tile
    buffer [EST block], and HIER reducer fJ per MAC [EST block]. The E2M1->INT5 conversion is already inside the INT5
    macro energy (e28 caveat 5) and is not added."""
    base = root / "results/e28_quant_summary"
    p = (base / run) if run else (base / "latest").resolve()
    m = json.loads((p / "metrics.json").read_text())
    po, pm = m["per_op_fj"], m["per_mac_fj"]
    out = {f: {"quant_block_fj": po[f"quant_{f.upper()}"], "anchor_block_fj": po["anchor"], "tilebuf_block_fj": po["tilebuf"],
               "hier_fj_mac": pm["hier"]} for f in ("mx", "nv")}
    return out, str(p.relative_to(root))


def _num(x):
    return float(x)


def load_inputs(root: Path, nv: bool = False, bundle_run: str | None = None, with_bundle: bool = False) -> Inputs:
    R = root / "results"
    e22p = (R / "e22_macro_compare/latest").resolve()
    e22 = json.loads((e22p / "metrics.json").read_text())
    e22cfg = yaml.safe_load((e22p / "config.yaml").read_text()) if (e22p / "config.yaml").exists() else {}
    # the e21 runs the e22 compare used (pre-T2.7 runs did not differ from the hard-coded pair)
    e21_dirs = list(e22cfg.get("e21_runs") or ["20261110T050824693997Z_f25a5c5", "20261110T151121726136Z_58882a0"])
    hier = float(e22cfg.get("flat_eq_hier_penalty") or 0.0)
    rows = []
    for d in e21_dirs:
        with (R / "e21_macro_power" / d / "power.csv").open() as f:
            rows += list(csv.DictReader(f))
    sram = yaml.safe_load((root / "docs/spec/sram_energy.yaml").read_text())["corners"]
    e24 = json.loads((R / "e24_compare/latest/metrics.json").read_text())["per_design"]
    pr = {}
    for d, key in (("v2", "e24_pnr_v2"), ("bs", "e24_pnr_bs_int5"), ("dpu", "e24_pnr_dpu")):
        pr[d] = json.loads((R / key / "latest/metrics.json").read_text())["postroute_over_e13_zero_total"]
    fmax = {"v2": e24["v2"]["fmax_lb_mhz"], "bs": e24["int5"]["fmax_lb_mhz"], "dpu": e24["dpu"]["fmax_lb_mhz"]}
    designs = {}
    pr["dpufeq"], fmax["dpufeq"] = pr["dpu"], fmax["dpu"]
    pr["dpuflat"], fmax["dpuflat"] = pr["dpu"], fmax["dpu"]  # T2.11 measured flat DPU: same tile post-route factor
    plan = [("v2", "v2_macro_mx", "v2"), ("bs", "bs_macro_mx", "INT5 bit-serial"), ("dpu", "dpu_macro_mx", "DPU")]
    if "dpufeq_macro_mx_primary" in e22 and hier:  # [DERIVED] flat-equivalent DPU: logic / measured tile hier penalty
        plan.append(("dpufeq", "dpufeq_macro_mx", "DPU flat-eq. [DERIVED]"))
    have = {r["design"] for r in rows}
    if "dpuflat_macro_mx_primary" in e22 and "dpuflat_macro_mx" in have:  # T2.11: measured flat-Yosys DPU macro
        plan.append(("dpuflat", "dpuflat_macro_mx", "DPU flat [MEASURED]"))
    if nv:  # NV modes (INT5 is an MX-only control)
        for short, macro, label in (("v2nv", "v2_macro_nv", "v2 NV"), ("dpunv", "dpu_macro_nv", "DPU NV"),
                                    ("dpufeqnv", "dpufeq_macro_nv", "DPU NV flat-eq. [DERIVED]"),
                                    ("dpuflatnv", "dpuflat_macro_nv", "DPU NV flat [MEASURED]")):
            if f"{macro}_primary" in e22 and macro.replace("dpufeq_", "dpu_") in have and (hier or "feq" not in short):
                plan.append((short, macro, label))
        for k in ("v2nv", "dpunv", "dpufeqnv", "dpuflatnv"):
            base_k = k[:-2]
            pr[k], fmax[k] = pr[base_k], fmax[base_k]
    for short, macro, label in plan:
        m = e22[f"{macro}_primary"]
        src = macro.replace("dpufeq_", "dpu_")
        div = hier if short.startswith("dpufeq") else 1.0
        rr = [r for r in rows if r["design"] == src and r["kind"] == "real"]
        n_ops, cyc = _num(rr[0]["n_ops"]), _num(rr[0]["cycles"])
        sw = _num(rr[0]["set_switches"])
        c_op = round((cyc - 26) / n_ops)  # steady cycles/op: 1 (v2, DPU), 6 (INT5)
        ovh = (cyc - n_ops * c_op) / sw
        per_cycle_ns = sum(_num(r["window_ns"]) / _num(r["cycles"]) for r in rr) / len(rr)
        p_idle = sum(_num(r["p_sequential_int_w"]) + _num(r["p_clock_w"]) for r in rr) / len(rr)
        designs[short] = Design(
            label, m["e_logic_dyn_fj_per_mac"] + m["e_logic_leak_fj_per_mac"], m["e_wire_fj_per_mac"], c_op, ovh,
            p_idle / div * per_cycle_ns * 1e3, pr[short], fmax[short], "nv" if macro.endswith("_nv") else "mx")
    prov = {"e22": str(e22p.relative_to(root)), "e21": ",".join(e21_dirs), "sram": "docs/spec/sram_energy.yaml",
            "e24_compare": str((R / "e24_compare/latest").resolve().relative_to(root)),
            "e24_postroute_factors": "results/e24_pnr_{v2,bs_int5,dpu}/latest/metrics.json"}
    bundle = {}
    if with_bundle:
        bundle, prov["e28_bundle"] = load_bundle(root, bundle_run)
    return Inputs(designs, sram, prov, bundle)


# ---------------------------------------------------------------------------------------------------------
# LLM layer shapes
# ---------------------------------------------------------------------------------------------------------
def layer_groups(cfg: dict, include_lm_head: bool = False):
    """Return (per-layer groups, n_layers, tail groups). A group = GEMMs that may share the macro pool in
    parallel; groups are sequential. Each GEMM is (name, K, N)."""
    h, i = cfg["hidden_size"], cfg["intermediate_size"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim", h // nh)
    layer = [[("q", h, nh * hd), ("k", h, nkv * hd), ("v", h, nkv * hd)], [("o", nh * hd, h)],
             [("gate", h, i), ("up", h, i)], [("down", i, h)]]
    tail = [[("lm_head", h, cfg["vocab_size"])]] if include_lm_head else []
    return layer, cfg["num_hidden_layers"], tail


def _tiles(k: int, n: int) -> int:
    return math.ceil(k / TILE) * math.ceil(n / TILE)


def group_stats(group, m: int):
    t = sum(_tiles(k, n) for _, k, n in group)
    useful = sum(m * k * n for _, k, n in group)
    return t, useful


def evaluate(cfg: dict, rows: int, d: Design, sram: dict, a: Assumptions, bundle: dict | None = None) -> dict:
    """One forward pass over all weight GEMMs with `rows` activation rows per weight set (prefill: seq tokens;
    decode: batch). Returns totals plus a per-layer-type breakdown."""
    layer, n_layers, tail = layer_groups(cfg, a.include_lm_head)
    sc = sram[a.sram_corner]
    e_logic_op = MACS_PER_OP * (d.e_logic_fj_mac + (d.e_wire_fj_mac if a.wires else 0.0)) * 1e-15
    if a.post_route:
        e_logic_op *= d.pr_factor
    e_sw = NG * sc["de_set_switch_per_macro_pj"] * 1e-12  # J per set visit (read into latches)
    e_wr = WORDS_PER_SET * sc["de_write_flip_half_pj"] * 1e-12  # J per set rewrite (data bits only)
    e_idle_cyc = NG * sc["e_idle_clock_cycle_pj"] * 1e-12  # J per macro-cycle, 8 stores
    leak_w = NG * sc["leakage_uw"] * 1e-6
    f = a.f_mhz * 1e6
    tile_cycles = d.ovh_visit + rows * d.c_op
    tile_cycles = max(tile_cycles, WRITE_CYCLES_PER_SET) if a.overlap_reload else tile_cycles + WRITE_CYCLES_PER_SET
    cycles = tiles = useful = act_blocks = 0.0
    breakdown = []
    for mult, groups in ((n_layers, layer), (1, tail)):
        for g in groups:
            t, u = group_stats(g, rows)
            span = math.ceil(t / a.n_macros) * tile_cycles
            breakdown.append({"group": "+".join(x[0] for x in g), "mult": mult, "tiles": t, "R": rows,
                              "makespan_cycles": span, "busy_macro_cycles": t * rows * d.c_op,
                              "pool_util": t * rows * d.c_op / (a.n_macros * span),
                              "tile_fill": t / (math.ceil(t / a.n_macros) * a.n_macros)})
            act_blocks += mult * rows * math.ceil(g[0][1] / TILE)  # GEMMs of a group share their input (same K)
            cycles += mult * span
            tiles += mult * t
            useful += mult * u
    ops = tiles * rows
    active = ops * d.c_op
    macro_cycles = a.n_macros * cycles
    lane_util = useful / (ops * MACS_PER_OP)
    e = {"logic": ops * e_logic_op, "sram_set_switch": tiles * e_sw, "sram_write": tiles * e_wr}
    clocked = macro_cycles if a.sram_clock == "as_built" else min(macro_cycles, tiles * (READ_CYCLES_PER_SWITCH + WRITE_CYCLES_PER_SET))
    e["sram_idle_clock"] = clocked * e_idle_cyc
    e["sram_leak"] = a.n_macros * leak_w * cycles / f
    e["idle_logic"] = (macro_cycles - active) * d.idle_logic_pj_cycle * 1e-12 * (d.pr_factor if a.post_route else 1.0) \
        if a.idle_logic else 0.0
    if a.bundle != "none":
        if bundle is None:
            raise ValueError("Assumptions.bundle set but no e28 bundle passed")
        b = bundle[d.fmt]
        blk = b["quant_block_fj"] + b["anchor_block_fj"] + b["tilebuf_block_fj"]
        if a.bundle == "per_token":
            e["quant_anchor_buf"] = act_blocks * blk * 1e-15
            e["hier"] = useful * b["hier_fj_mac"] * 1e-15
        elif a.bundle == "per_op":
            e["quant_anchor_buf"] = ops * blk * 1e-15
            e["hier"] = ops * MACS_PER_OP * b["hier_fj_mac"] * 1e-15
        else:
            raise ValueError(a.bundle)
    total = sum(e.values())
    for b in breakdown:
        b["R"] = rows
    return {"energy_j": total, "parts_j": e, "cycles": cycles, "latency_s": cycles / f, "tiles": tiles, "ops": ops,
            "useful_macs": useful, "lane_util": lane_util, "time_util": active / macro_cycles,
            "overhead_share": (macro_cycles - active) / macro_cycles, "tops_per_w": 2 * useful / total / 1e12,
            "set_visits": tiles, "dram_bytes": tiles * (SET_BITS + SCALE_RF_BITS) / 8,
            "weight_reuse_R": rows, "act_blocks": act_blocks, "breakdown": breakdown}


def breakeven_rows(d: Design, sram: dict, a: Assumptions) -> float:
    """R at which per-visit (switch + write + overhead idle-clock) energy equals the per-set compute energy
    (R ops x 1024 MAC x e_logic). SRAM idle clock charged for the non-op cycles of one visit (as-built)."""
    sc = sram[a.sram_corner]
    e_op = MACS_PER_OP * (d.e_logic_fj_mac + (d.e_wire_fj_mac if a.wires else 0.0)) * 1e-15
    e_visit = NG * sc["de_set_switch_per_macro_pj"] * 1e-12 + WORDS_PER_SET * sc["de_write_flip_half_pj"] * 1e-12
    e_visit += NG * sc["e_idle_clock_cycle_pj"] * 1e-12 * d.ovh_visit
    return e_visit / e_op


def with_(a: Assumptions, **kw) -> Assumptions:
    return replace(a, **kw)
