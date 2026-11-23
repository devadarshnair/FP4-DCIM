# Top-level entry points. Run inside the `cim` conda env:  conda activate cim
PY      ?= python
PYTEST  ?= $(PY) -m pytest
SRAM_V_DIR ?= $(HOME)/.ciel/sky130A/libs.ref/sky130_sram_macros/verilog
RTL_SRC := $(wildcard rtl/src/*.sv) $(wildcard rtl/gen/*.sv) $(wildcard rtl/baselines/*/*.sv)

.PHONY: help install test test-fast lint fmt rtl-gen rtl-lint rtl-test formal synth power clean

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-12s %s\n", $$1, $$2}'

install: ## editable install with dev extras
	$(PY) -m pip install -e ".[dev]"

test: ## full python test suite
	$(PYTEST)

test-fast: ## skip slow / gpu / llm tests, run in parallel
	$(PYTEST) -n auto -m "not slow and not gpu and not llm"

lint: ## ruff + verilator lint
	$(PY) -m ruff check src tests scripts
	$(MAKE) rtl-lint

fmt: ## auto-format python
	$(PY) -m ruff format src tests scripts
	$(PY) -m ruff check --fix src tests scripts

rtl-gen: ## regenerate rtl/include + rtl/gen from python generators (WP-22/24)
	$(PY) -m cimfp4.rtlgen

rtl-lint: ## verilator lint of all RTL
	verilator --lint-only -Wall rtl/lint/waivers.vlt rtl/include/cim_pkg.sv
	@set -e; for f in $(RTL_SRC); do echo "lint $$f"; verilator --lint-only -Wall -Irtl/include -Irtl/gen -y rtl/src -y rtl/gen -y rtl/baselines/bitserial_int5 -y rtl/baselines/bitserial_int4sm -y rtl/baselines/digital_mac -y $(SRAM_V_DIR) --timing rtl/lint/waivers.vlt rtl/include/cim_pkg.sv $$f; done

rtl-test: ## cocotb testbenches (Verilator; PATH must include the cim env bin)
	$(PYTEST) tb/cocotb

formal: ## SymbiYosys / EQY proofs
	@for f in tb/formal/*.sby; do [ -e "$$f" ] && sby -f $$f || true; done

synth: ## yosys synthesis sweep (WP-27)
	$(MAKE) -C flows/synth

power: ## power flow (WP-28)
	$(MAKE) -C flows/power

clean:
	rm -rf build .pytest_cache .ruff_cache sim_build **/__pycache__
