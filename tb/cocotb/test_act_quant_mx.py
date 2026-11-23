"""Online MXFP4 activation quantizer RTL vs cimfp4.formats.mx_quant (WP-33): corners + random bf16."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from cimfp4.formats.mx_quant import quantize_mxfp4
from cimfp4.utils.provenance import REPO_ROOT


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float -> bf16 bit patterns by truncation (any bf16 value set works; the golden sees the exact value)."""
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def bf16_to_f64(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def corner_blocks() -> np.ndarray:
    grid_mid = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    blocks = []
    b = np.zeros(32)
    b[0] = 6.0
    b[1:8] = grid_mid  # every tie with scale 2^0 (amax 6 -> X = 0)
    blocks.append(b)
    blocks.append(np.where(np.arange(32) % 2, -1, 1) * np.linspace(0.01, 7.9, 32))
    for amax in (7.0, 6.99, 3.5, 3.0, 6.0, 6.01, 1.75 * 2**10, 1.5 * 2**-20):  # expAdj / RCEIL boundaries
        c = np.full(32, amax / 3)
        c[5] = amax
        blocks.append(c)
    blocks.append(np.zeros(32))  # all zero
    sub = np.zeros(32)
    sub[:4] = [2.0**-130, 2.0**-133, 3 * 2.0**-131, -(2.0**-127)]  # bf16 subnormals (min normal 2^-126)
    blocks.append(sub)
    blocks.append(np.full(32, 1e30))  # saturation / large exponent
    blocks.append(np.concatenate([[100.0], np.full(31, 1e-6)]))  # tiny elements -> 0
    return np.array(blocks)


try:
    import cocotb
    from cocotb.triggers import Timer

    @cocotb.test()
    async def act_quant_matches(dut):
        rng = np.random.default_rng(int(os.environ.get("AQ_SEED", "0")))
        n = int(os.environ.get("AQ_NBLOCKS", "10000"))
        blocks = [to_bf16_bits(b) for b in corner_blocks()]
        x = rng.standard_normal((n, 32)) * np.exp(rng.normal(0, 4, (n, 1)))
        x[:, 0] *= rng.choice([1, 30], n)
        blocks += list(to_bf16_bits(x))
        nanb = to_bf16_bits(np.ones(32))
        nanb[3] = 0x7FC0  # NaN
        infb = to_bf16_bits(np.ones(32))
        infb[9] = 0xFF80  # -Inf
        bad = 0
        for mode, rule in ((0, "ocp_v11"), (1, "rceil")):
            dut.mode_rceil.value = mode
            for bb in blocks + [nanb, infb]:
                dut.x_bf16.value = sum(int(v) << (16 * i) for i, v in enumerate(bb))
                await Timer(1, unit="ns")
                vals = bf16_to_f64(bb)
                q = quantize_mxfp4(vals, rule)
                want_codes = sum(int(c) << (4 * i) for i, c in enumerate(q.codes))
                want_scale = 0xFF if q.scale_nan[0] else int(q.scale_exp[0]) + 127
                if int(dut.codes.value) != want_codes or int(dut.scale_code.value) != want_scale:
                    bad += 1
                    if bad <= 3:
                        dut._log.error(
                            f"mismatch rule={rule} vals={vals[:4]} scale rtl={int(dut.scale_code.value)} "
                            f"want={want_scale}"
                        )
        assert bad == 0, f"{bad} mismatching blocks"
        dut._log.info(f"{2 * (len(blocks) + 2)} blocks (both rules), 0 mismatches")

except ImportError:
    pass


def _run(tmp_path, n):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    runner = get_runner("verilator")
    runner.build(
        sources=[REPO_ROOT / "rtl/src/act_quant_mx.sv"],
        hdl_toplevel="act_quant_mx",
        build_dir=tmp_path / "b",
        always=True,
        build_args=["-Wno-fatal"],
    )
    xml = runner.test(
        hdl_toplevel="act_quant_mx",
        test_module="test_act_quant_mx",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env={"PYTHONPATH": str(REPO_ROOT / "src"), "AQ_NBLOCKS": str(n)},
    )
    assert get_results(xml) == (1, 0)


needs = pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")


@needs
def test_act_quant_mx(tmp_path):
    _run(tmp_path, 10_000)


@needs
@pytest.mark.slow
def test_act_quant_mx_1e6(tmp_path):
    _run(tmp_path, 500_000)  # x 2 rules = 1e6 blocks
