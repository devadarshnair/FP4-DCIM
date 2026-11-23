"""cimfp4 - alignment-free MXFP4/NVFP4 digital SRAM compute-in-memory research package.

Module docstrings state the numerical facts each module relies on; the hardware contract is in
docs/spec/numerics.md.
"""

__version__ = "0.0.1"

SUBPACKAGES = (
    "formats",
    "golden",
    "kernels",
    "rtlgen",
    "llm",
    "stimulus",
    "power",
    "sysmodel",
    "utils",
)
