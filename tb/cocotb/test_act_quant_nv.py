"""Online NVFP4 activation quantizer RTL vs cimfp4.formats.nv_quant ("hw" convention) (WP-68)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from cimfp4.formats.nv_quant import nv_tensor_scale, quantize_nvfp4
from cimfp4.utils.provenance import REPO_ROOT


def bf16(x):
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def bf16_f64(b):
    return (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def f32_bits(v) -> int:
    return int(np.float32(v).view(np.uint32))


def corner_cases():
    mids = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    cases = []
    b = np.zeros(16)
    b[0], b[1:8] = 6.0, mids
    cases.append((b, 2688.0))  # s_dec = 1, scale 1.0 -> exact ties
    cases.append((np.linspace(-6, 6, 16), 2688.0))
    cases.append((np.full(16, 6 * 2.0**-8), 2688.0))  # E4M3 subnormal block scale
    cases.append((np.full(16, 6 * 2.0**-12), 2688.0))  # scale underflows to 0
    cases.append((np.full(16, 100.0), 1.0))  # static amax too small: satfinite 448
    cases.append((np.zeros(16), 2688.0))  # all zero
    return cases


try:
    import cocotb
    from cocotb.triggers import Timer

    @cocotb.test()
    async def nv_quant_matches(dut):
        rng = np.random.default_rng(int(os.environ.get("NQ_SEED", "0")))
        n = int(os.environ.get("NQ_NBLOCKS", "10000"))
        cases = corner_cases()
        for _ in range(n):
            x = rng.standard_normal(16) * np.exp(rng.normal(0, 2))
            cases.append((x, float(np.abs(x).max()) * float(np.exp(rng.normal(0.5, 1.5)))))
        bad = 0
        for x, t_amax in cases:
            bits = bf16(x)
            vals = bf16_f64(bits)
            s_dec = nv_tensor_scale(t_amax)
            s_enc = np.float32(np.float32(1.0) / s_dec)
            c = np.float32(s_enc / np.float32(6.0))
            q = quantize_nvfp4(vals[None], "hw", tensor_amax=t_amax)
            dut.x_bf16.value = sum(int(v) << (16 * i) for i, v in enumerate(bits))
            dut.c_fp32.value, dut.sdec_fp32.value = f32_bits(c), f32_bits(s_dec)
            await Timer(1, unit="ns")
            want_codes = sum(int(cd) << (4 * i) for i, cd in enumerate(q.codes[0]))
            if int(dut.codes.value) != want_codes or int(dut.scale_e4m3.value) != int(q.scale_e4m3[0, 0]):
                bad += 1
                if bad <= 3:
                    dut._log.error(
                        f"scale rtl {int(dut.scale_e4m3.value):#x} want {int(q.scale_e4m3[0, 0]):#x}"
                    )
        assert bad == 0, f"{bad} mismatching blocks of {len(cases)}"
        dut._log.info(f"{len(cases)} NV blocks, 0 mismatches")

except ImportError:
    pass


def _run(tmp_path, n):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    runner = get_runner("verilator")
    runner.build(
        sources=[REPO_ROOT / "rtl/src/act_quant_nv.sv"],
        hdl_toplevel="act_quant_nv",
        build_dir=tmp_path / "b",
        always=True,
        build_args=["-Wno-fatal"],
    )
    xml = runner.test(
        hdl_toplevel="act_quant_nv",
        test_module="test_act_quant_nv",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env={"PYTHONPATH": str(REPO_ROOT / "src"), "NQ_NBLOCKS": str(n)},
    )
    assert get_results(xml) == (1, 0)


needs = pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")


@needs
def test_act_quant_nv(tmp_path):
    _run(tmp_path, 10_000)


@needs
@pytest.mark.slow
def test_act_quant_nv_1e6(tmp_path):
    _run(tmp_path, 1_000_000)
