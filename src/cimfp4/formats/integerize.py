"""Cost of lossless FP->integer conversion per element format.

Every finite value of a minifloat format is an integer multiple of its smallest subnormal. Expressing
elements as such integers makes a block dot product exact integer arithmetic with *no alignment*.
The price is the integer width. For E2M1 it is +1 bit (4 -> 5 signed), which is why alignment hardware
(MIDAS-style) is pure overhead for FP4 but pays off for FP8/BF16.

All quantities are derived by enumerating every code of the format (no closed-form shortcuts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class MiniFloat:
    name: str
    ebits: int
    mbits: int
    bias: int
    #: how the top exponent field is used: "finite" (all codes finite, e.g. E2M1/E2M3/E3M2),
    #: "fn" (only S.1111.111 is NaN, e.g. OCP E4M3), "ieee" (top exponent = inf/NaN, e.g. E5M2, BF16)
    top: str = "finite"

    @property
    def width(self) -> int:
        return 1 + self.ebits + self.mbits

    def magnitudes(self) -> list[Fraction]:
        """All finite non-negative values, exactly."""
        out = []
        emax_field = (1 << self.ebits) - 1
        for e in range(emax_field + 1):
            for m in range(1 << self.mbits):
                if self.top == "ieee" and e == emax_field:
                    continue
                if self.top == "fn" and e == emax_field and m == (1 << self.mbits) - 1:
                    continue
                if e == 0:
                    v = Fraction(m, 1 << self.mbits) * Fraction(2) ** (1 - self.bias)
                else:
                    v = (1 + Fraction(m, 1 << self.mbits)) * Fraction(2) ** (e - self.bias)
                out.append(v)
        return sorted(set(out))


FORMATS: dict[str, MiniFloat] = {
    "E2M1": MiniFloat("E2M1", 2, 1, 1),  # FP4 (MXFP4 / NVFP4 element)
    "E2M3": MiniFloat("E2M3", 2, 3, 1),  # MXFP6 variant
    "E3M2": MiniFloat("E3M2", 3, 2, 3),  # MXFP6 variant (MIDAS's MXFP6)
    "E4M3": MiniFloat("E4M3", 4, 3, 7, "fn"),  # MXFP8 / NVFP4 scale
    "E5M2": MiniFloat("E5M2", 5, 2, 15, "ieee"),  # MXFP8 variant
    "BF16": MiniFloat("BF16", 8, 7, 127, "ieee"),
}


@dataclass(frozen=True)
class IntegerizeResult:
    name: str
    width: int  # bits of the FP code
    lsb: Fraction  # value of one integer unit (min subnormal)
    max_int: int  # largest magnitude in LSB units
    int_bits_signed: int  # sign + magnitude bits needed for the lossless integer
    extra_bits: int  # int_bits_signed - width
    product_bits: int  # unsigned magnitude bits of an element×element product
    block_sum_unsigned: int  # unsigned bits of an exact sum of `block` products (per polarity)
    block_sum_signed: int  # signed bits of the exact block sum
    block: int


def integerize(fmt: MiniFloat | str, block: int = 32) -> IntegerizeResult:
    f = FORMATS[fmt] if isinstance(fmt, str) else fmt
    mags = f.magnitudes()
    lsb = min(v for v in mags if v > 0)
    ints = [v / lsb for v in mags]
    assert all(i.denominator == 1 for i in ints), "every value must be an integer multiple of the LSB"
    max_int = int(max(ints))
    mag_bits = max_int.bit_length()
    prod_max = max_int * max_int
    sum_max = block * prod_max
    return IntegerizeResult(
        name=f.name,
        width=f.width,
        lsb=lsb,
        max_int=max_int,
        int_bits_signed=mag_bits + 1,
        extra_bits=mag_bits + 1 - f.width,
        product_bits=prod_max.bit_length(),
        block_sum_unsigned=sum_max.bit_length(),
        block_sum_signed=sum_max.bit_length() + 1,
        block=block,
    )


def lsb_log2(r: IntegerizeResult) -> int:
    return int(math.log2(r.lsb))
