# FP4 DCIM: a shared MXFP4/NVFP4 digital SRAM macro

![SystemVerilog](https://img.shields.io/badge/RTL-SystemVerilog-1f6feb)
![Python](https://img.shields.io/badge/python-3.12-3776ab?logo=python&logoColor=white)
![PDK](https://img.shields.io/badge/PDK-SkyWater%20130nm-orange)
![Flow](https://img.shields.io/badge/flow-Yosys%20%7C%20OpenLane2%20%7C%20OpenSTA-6e40c9)
![Sim](https://img.shields.io/badge/sim-Verilator%20%7C%20Icarus%20%7C%20cocotb-2ea44f)
![Formats](https://img.shields.io/badge/formats-MXFP4%20%7C%20NVFP4-ff69b4)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

One digital compute-in-memory macro that runs both OCP MXFP4 and NVIDIA NVFP4
on the same datapath. Every 4-bit (E2M1) product is computed exactly, and a
whole block is summed in integer form before any shift, so no per-element
mantissa alignment is needed.

## Highlights

| | v2 vs exact INT5 bit-serial | v2 vs digital MAC |
|---|---|---|
| Energy efficiency (macro, free-running clock) | 2.61x | 1.48x |
| Energy efficiency (routed tile) | 2.06x | 1.76x |

All numbers are sky130, typical corner, measured with the same flow and the
same SRAM for every design.

## What is inside

```
rtl/
  src/          proposed macro (top: macro_top.sv)
  baselines/    bit-serial INT5, bit-serial INT4-SM, generic digital MAC
  gen/          compressor trees      (generated, do not edit)
  include/      width package         (generated, do not edit)
src/cimfp4/     Python: formats, golden models, RTL generator, power and system models
tests/          pytest suite
tb/             cocotb testbenches and formal scripts
flows/          synthesis, place and route, power, PDK and SRAM configs
experiments/    experiment drivers
docs/spec/      numerics, rounding points, SRAM energy, cost tables
scripts/        utilities, including the PDK shim builder
```

## Quick start

```bash
conda env create -f env/environment-cim.yml
conda activate cim
pip install -e .

mkdir -p build/pytest
pytest -q tests        # python and generator tests
make rtl-lint          # Verilator lint of all RTL
```

## Physical flow setup

```bash
pip install ciel==2.4.1
ciel enable --pdk-family sky130 1c8019c4421c99348a0ec78f4e969eed6fb5c647
scripts/make_pdk_shim.sh                # PDK shim for OpenLane2 2.3.10
docker pull ghcr.io/efabless/openlane2:2.3.10
```

Static timing uses OpenSTA 3.1.0. A prebuilt Linux bundle is attached to the
v0.1 release:

```bash
tar -xzf cim_sta_bundle_opensta-3.1.0_linux-x86_64.tar.gz -C ~/opt
export CIM_STA_BIN=~/opt/cim_sta_bundle/sta_wrap
```

## Toolchain

| Tool | Version |
|---|---|
| Yosys | 0.69 (9f09efcdb) |
| OpenLane2 | 2.3.10 |
| OpenSTA | 3.1.0 |
| Icarus Verilog | 13.0 |
| Verilator | 5.052 |
| cocotb | 2.1.0 |
| sky130 (open_pdks) | 1c8019c4 |

## Regenerating RTL

Files in `rtl/gen` and `rtl/include` come from the Python generator. Change
the generator, then run:

```bash
make rtl-gen
```

## Status

Research code under active development. A paper is in preparation. Thank you Pillai.

## License

Apache License 2.0. See [LICENSE](LICENSE).
