#!/usr/bin/env python3
"""T2.4: generate docs/spec/hw_cost_table.yaml (v0.2) from pinned results and check docs/spec/hw_cost_table.md.

Every measured number is read from results/ (run dirs pinned in experiments/configs/paper_runs.yaml, e22 + e30 flat
macro synthesis); nothing is typed here except labels and the not-measured/static entries marked below.
Usage: python scripts/gen_hw_cost_table.py [--write | --check]   (default: print YAML to stdout)
--check: regenerate in memory, require it to equal the YAML file and every number to appear in the MD.
Light job (CSV/JSON only)."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_pins  # noqa: E402

E22 = paper_pins.run_dir("e22_macro_compare")
SDF = paper_pins.run_dir("e13_sdf_td", "numbers")
FLAT = {d: paper_pins.run_dir(f"e30_td_macro_{d}_macro", "numbers") for d in ("v2", "bs", "dpu")}
HIER = {d: paper_pins.run_dir(f"e30_td_macro_{d}_macro_hier", "numbers") for d in ("v2", "bs", "dpu")}
SRAM = ROOT / "docs/spec/sram_energy.yaml"
rel = lambda p: str(Path(p).resolve().relative_to(ROOT))
r2 = lambda x, n=2: round(float(x), n)

m = json.load(open(E22 / "metrics.json"))
grid = list(csv.DictReader(open(E22 / "grid.csv")))
ci = {(r["pair"], r["kind"]): r for r in csv.DictReader(open(E22 / "ratios_ci.csv"))}
fm = {d: json.load(open(FLAT[d] / "metrics.json")) for d in FLAT}
hm = {d: json.load(open(HIER[d] / "metrics.json")) for d in HIER}
sdf = json.load(open(SDF / "metrics.json"))


def row(design, sram, delay="zero", wire="nom", clock="common_40MHz", sc="as_built"):
    for r in grid:
        if (r["design"], r["kind"], r["sram"], r["delay"], r["wire"], r["clock"], r["sram_clock"]) == \
           (design, "real", sram, delay, wire, clock, sc):
            return r
    raise KeyError((design, sram, delay, wire, clock, sc))


def en(design, sram):
    r = row(design, sram)
    logic = float(r["e_logic_dyn_fj_per_mac"]) + float(r["e_logic_leak_fj_per_mac"])
    sr = sum(float(r[k]) for k in ("e_sram_access_fj_per_mac", "e_sram_idle_clock_fj_per_mac", "e_sram_leak_fj_per_mac"))
    w = float(r["e_wire_fj_per_mac"]); tot = float(r["e_total_fj_per_mac"])
    assert abs(logic + sr + w - tot) < 1e-6
    return {"logic": r2(logic), "sram": r2(sr), "wires_est": r2(w), "total": r2(tot)}


def unit(d):
    return {k: r2(v * 32 / 1000) for k, v in d.items()}


E = {f: {s: en(f"v2_macro_{k}", s) for s in ("low", "nominal", "high")} for f, k in (("MXFP4", "mx"), ("NVFP4", "nv"))}
EU = {f: {s: unit(v) for s, v in E[f].items()} for f in E}
G = {s: r2(row("v2_macro_mx", s, "sdf_est")["e_total_fj_per_mac"], 1) for s in ("low", "nominal", "high")}
rc = ci[("ee_v2_nv_over_v2_mx", "real")] if ("ee_v2_nv_over_v2_mx", "real") in ci else None
nvmx_ci = m["v2_macro_nv_over_v2_macro_mx_energy_ci"]
v2tot = fm["v2"]["total_area_um2"] / 1e6; v2logic = fm["v2"]["logic_area_um2"] / 1e6
reload = {k: m["reload_fj_per_set_switch_nominal"][k] / 1e6 for k in ("v2_macro_mx", "v2_macro_nv")}
sram_y = yaml.safe_load(open(SRAM))


def design_block(tag, key_mx, key_nv=None, flat=True):
    d = {"area_mm2_flat_td": r2(fm[tag]["total_mm2"], 4), "r2r_fmax_synth_sta_mhz": r2(fm[tag]["r2r_fmax_mhz"], 1),
         "cycles_per_op": r2(m[f"{key_mx}_cycles_per_op_real"], 4),
         "mx_fj_per_mac_nominal": en(key_mx, "nominal")["total"],
         "gflops_per_mm2_40mhz": r2(m[f"{key_mx}_common_40MHz_gflops_per_mm2"]),
         "gmac_per_s_40mhz": r2(m[f"{key_mx}_common_40MHz_gmac_per_s"])}
    if key_nv:
        d["nv_fj_per_mac_nominal"] = en(key_nv, "nominal")["total"]
    return d


out = {
    "version": "0.2", "status": "draft", "date": "2026-11-15",
    "git_sha": "see sources (each run dir carries git.txt)",
    "conditions": {
        "pdk": "sky130_fd_sc_hd tt 1.8 V (logic 025C); OpenRAM TT_1p8V_25C; SRAM SPICE 27 C",
        "clock_mhz": 40, "clock_note": "34 MHz = silicon-verified SRAM clock: energy/op unchanged, throughput x0.85; 40 MHz is [SIM]",
        "delay_model": "zero-delay GLS + OpenSTA power unless a field says glitch [EST]",
        "netlist": "T2.6 td recipe, flat Yosys for v2, INT5 and DPU (docs/spec/synth_recipe.md; DPU flat measured T2.11); macro port nets unbuffered",
        "boundary": "full 32x32 v2 macro_top (32 columns, 8 OpenRAM 1 KB stores, scale RFs, write port, set-switch control), NO activation quantizer, NO host/accumulator beyond the column windows",
        "post_route": False,
        "workload": "9 real prefill windows (4 Qwen2.5-0.5B + 4 SmolLM2-360M RTN captures + 1 GPTQ-weight window), MX and NV on the same windows; R=256 ops/window, M=16 sets, 2 set switches/window; 1024 MAC/op",
        "not_ours": "partner-reported scale_dcim numbers are excluded",
    },
    "decision_unit": {"elements": 32, "mxfp4": "1 block x 32 elements (UE8M0 scale)",
                      "nvfp4": "2 blocks x 16 elements (E4M3 scales + one FP32 tensor scale per tensor)",
                      "macs_per_unit": 32, "units_per_macro_op": 32, "conversion": "unit value = fJ/MAC x 32"},
    "sources": {
        "e22_pinned": rel(E22 / "metrics.json") + " (grid.csv, ratios_ci.csv, nv_row.csv, T2.11 rerun with measured flat DPU)",
        "e30_flat_area_sta": {d: rel(FLAT[d] / "metrics.json") for d in FLAT},
        "e30_hier_reference_only": {d: rel(HIER[d] / "metrics.json") for d in HIER},
        "e13_sdf_td": rel(SDF / "metrics.json"),
        "e20_sram": "docs/spec/sram_energy.yaml",
        "e23_idle_nv": "results/e23_reconcile_analysis/ (td rerun 2026-11-15; e23 value 160 fJ/MAC is pre-T2.6, not re-measured at macro with td)",
        "pins": "experiments/configs/paper_runs.yaml",
        "generator": "scripts/gen_hw_cost_table.py",
        "spec": ["docs/spec/numerics.md", "docs/spec/rounding_points.md", "docs/spec/synth_recipe.md"],
    },
    "energy_fj_per_mac": E,
    "energy_pj_per_32el_unit": EU,
    "glitch_corrected_est_fj_per_mac": {"MXFP4": G, "NVFP4": None,
        "factor_v2_sdf_over_zero": r2(sdf["v2_sram_m16_sdf_over_zero"], 3), "label": "[EST]; tile factor applied to macro logic dynamic; NV factor not measured"},
    "nv_minus_mx_fj_per_mac": {"logic": r2(E["NVFP4"]["nominal"]["logic"] - E["MXFP4"]["nominal"]["logic"]),
        "total_nominal": r2(E["NVFP4"]["nominal"]["total"] - E["MXFP4"]["nominal"]["total"]),
        "ratio_nominal": r2(m["v2_macro_nv_over_v2_macro_mx_energy"], 3),
        "ratio_ci95_paired_windows": [r2(nvmx_ci[0], 3), r2(nvmx_ci[1], 3)]},
    # static, not re-measured (labelled)
    "idle_nv_overhead_in_mx_mode_fj_per_mac": {"value": 160, "basis": "e23 L4 keepcol-vs-folded tile, pre-T2.6 netlists; td rerun 2026-11-15 gives v2 +168 fJ (e23 td)", "label": "DERIVED, not re-measured at macro in T2.7",
        "tile_td_nv_gate": {"nvg0": 0.229, "nvg1": 0.082}},
    "timing": {"cycles_per_op_amortized": {"MXFP4": r2(m["v2_macro_mx_cycles_per_op_real"], 3), "NVFP4": r2(m["v2_macro_nv_cycles_per_op_real"], 3)},
        "steady_state_cycles_per_unit_op": 1, "set_switch_cycles": 10,
        "throughput_gmac_per_s": {"at_40mhz": r2(m["v2_macro_mx_common_40MHz_gmac_per_s"]), "at_34mhz": r2(m["v2_macro_mx_common_34MHz_gmac_per_s"])},
        "r2r_fmax_synthesis_sta_mhz": r2(fm["v2"]["r2r_fmax_mhz"], 1),
        "fmax_note": "td flat v2 reg-to-reg, ideal clock, no wires (metrics.json r2r_fmax_mhz); the SRAM limit (34 MHz verified, 40 MHz [SIM]) binds. No other fmax entry is quoted; the former 69.7 MHz value is removed",
        "op_latency_end_to_end_cycles": None},
    "area_mm2": {"source": "flat td v2 macro: " + rel(FLAT["v2"] / "metrics.json") + " (hier area not used)",
        "macro_total_flat": r2(v2tot, 4),
        "sram": {"value": r2(v2tot - v2logic, 4), "share": r2((v2tot - v2logic) / v2tot, 3), "derivation": "total_area_um2 - logic_area_um2 (8 OpenRAM 1 KB stores)"},
        "logic_incl_seq": {"value": r2(v2logic, 4), "share": r2(v2logic / v2tot, 3), "seq_share": r2(fm["v2"]["seq_area_um2"] / 1e6 / v2tot, 3)},
        "per_unit_slot": r2(v2tot / 32, 4),
        "gflops_per_mm2": {"at_40mhz": r2(m["v2_macro_mx_common_40MHz_gflops_per_mm2"]), "at_34mhz": r2(m["v2_macro_mx_common_34MHz_gflops_per_mm2"])},
        "hier_reference_only_not_used": {d: r2(hm[d]["total_mm2"], 4) for d in hm}},
    "comparison_designs_flat_td": {
        "label": "same boundary, same recipe, all flat; zero-delay, as-built, nominal SRAM+wires, 40 MHz, real windows",
        "v2": design_block("v2", "v2_macro_mx", "v2_macro_nv"),
        "int5_bit_serial_mx_only": design_block("bs", "bs_macro_mx"),
        "dpu_generic_leaf_measured_flat": design_block("dpu", "dpuflat_macro_mx", "dpuflat_macro_nv"),
        "ratios_v2_over": {
            "int5_mx": {"value": r2(m["ee_v2_mx_over_bs_mx_real_primary"]), "ci95": [r2(ci[("ee_v2_mx_over_bs_mx", "real")]["ci_lo"]), r2(ci[("ee_v2_mx_over_bs_mx", "real")]["ci_hi"])], "glitch_est": r2(ci[("ee_v2_mx_over_bs_mx", "real")]["glitch_est"])},
            "dpu_flat_mx": {"value": r2(m["ee_v2_mx_over_dpuflat_mx_real_primary"]), "ci95": [r2(x) for x in m["ee_v2_mx_over_dpuflat_mx_real_primary_ci"]], "glitch_est": r2(m["ee_v2_mx_over_dpuflat_mx_real_glitch_est"])},
            "dpu_flat_nv": {"value": r2(m["ee_v2_nv_over_dpuflat_nv_real_primary"]), "ci95": [r2(x) for x in m["ee_v2_nv_over_dpuflat_nv_real_primary_ci"]], "glitch_est": r2(m["ee_v2_nv_over_dpuflat_nv_real_glitch_est"])},
            "density_vs_int5_40mhz": r2(m["density_v2_macro_mx_over_bs_macro_mx_common_40MHz"]),
            "density_vs_dpu_flat_40mhz": r2(m["v2_macro_mx_common_40MHz_gflops_per_mm2"] / m["dpuflat_macro_mx_common_40MHz_gflops_per_mm2"]),
            "throughput_vs_int5": r2(m["primary_throughput_ratio_v2_over_bs"]),
        },
        "dpu_hier_over_flat_logic_penalty": {"mx": r2(m["dpu_hier_over_flat_logic_mx_mean_ratio"], 3), "nv": r2(m["dpu_hier_over_flat_logic_nv_mean_ratio"], 3)},
        "note": "hier DPU and flat-equivalent [DERIVED] 1.191 rows are superseded by the measured flat DPU (T2.11); not quoted here",
    },
    "metadata": {  # spec constants (numerics.md), not measurements
        "mxfp4": {"scale_bits_per_unit": 8, "bits_per_element": 4.25, "scale_format": "UE8M0", "tensor_scale": "none"},
        "nvfp4": {"scale_bits_per_unit": 16, "bits_per_element": 4.5, "scale_format": "E4M3 x2", "tensor_scale": "FP32 x1 per tensor, applied outside macro"},
        "scale_rf_bits_provisioned_per_set_column": 25, "scale_rf_bits_per_macro": 12800},
    "format_switch": {"per_op_op_nv_toggle_cost": None,
        "set_switch": {"cycles": 10, "sram_only_nj_per_macro": {k: r2(8 * sram_y["corners"][k]["de_set_switch_per_macro_pj"] / 1000, 2) for k in ("low", "nominal", "high")},
                       "total_nj_per_macro_nominal": {"MXFP4": r2(reload["v2_macro_mx"], 2), "NVFP4": r2(reload["v2_macro_nv"], 2)}}},
    "gaps": [
        "activation quantizer (act_quant_mx/nv) not in the macro boundary (e28 reports per-MAC cost separately)",
        "FP32 tensor-scale cost not measured (outside macro)",
        "NV glitch factor, mixed-format op streams, per-op format toggling, end-to-end pipeline latency not measured",
        "isolated E8M0-vs-E4M3 scale-path energy not measured (only NV-mode minus MX-mode total)",
        "post-route / post-SDF macro power not available; macro port nets unbuffered",
        "flat per-block area split (scale RF share) and MX-only (NV-stripped) macro area not available",
        "idle-NV overhead at macro level with td netlists / NV_GATE=1 (T4.2)",
    ],
}
text = ("# Hardware-cost table v0.2 DRAFT, GENERATED by scripts/gen_hw_cost_table.py from results/ (do not hand-edit).\n"
        "# Regenerate: python scripts/gen_hw_cost_table.py --write ; verify against the MD: --check.\n"
        "# Not yet sent. Partner values excluded.\n" +
        yaml.safe_dump(out, sort_keys=False, width=200))

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    y = ROOT / "docs/spec/hw_cost_table.yaml"; md = (ROOT / "docs/spec/hw_cost_table.md").read_text()
    if arg == "--write":
        y.write_text(text); print("wrote", y)
    elif arg == "--check":
        bad = []
        if y.read_text() != text: bad.append("YAML differs from regenerated")
        def need(s):
            if s not in md: bad.append("MD missing " + s)
        def num(x, nd=1):  # MD may round half-up where Python rounds half-even
            for v in (x, x + 1e-9, x - 1e-9):
                if f"{v:.{nd}f}" in md: return
            if f"{x:.2f}" in md: return
            bad.append(f"MD missing {x}")
        for f in E:
            for s in E[f]:
                for k in ("total", "logic"): num(E[f][s][k])
        for s in G.values(): need(f"{s:.0f}")
        a = out["area_mm2"]; need(f"{a['macro_total_flat']:.4f}"); need(f"{a['sram']['value']:.4f}"); need(f"{a['logic_incl_seq']['value']:.4f}")
        need(f"{a['per_unit_slot']:.4f}")
        c = out["comparison_designs_flat_td"]
        for dd in ("v2", "int5_bit_serial_mx_only", "dpu_generic_leaf_measured_flat"):
            need(f"{c[dd]['area_mm2_flat_td']:.4f}"); num(c[dd]['mx_fj_per_mac_nominal'])
        num(c['dpu_generic_leaf_measured_flat']['nv_fj_per_mac_nominal'])
        for k, v in c["ratios_v2_over"].items():
            if isinstance(v, dict): need(f"{v['value']:.2f}")
        print("\n".join(bad) or "OK: YAML == regenerated, MD numbers present"); sys.exit(1 if bad else 0)
    else:
        print(text)
