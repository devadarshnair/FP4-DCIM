import pytest

from cimfp4.formats.integerize import FORMATS, integerize


def test_e2m1_costs_one_bit():
    r = integerize("E2M1", block=32)
    assert (r.max_int, r.int_bits_signed, r.extra_bits) == (12, 5, 1)
    assert r.product_bits == 8
    assert (r.block_sum_unsigned, r.block_sum_signed) == (13, 14)  # MX
    r16 = integerize("E2M1", block=16)
    assert (r16.block_sum_unsigned, r16.block_sum_signed) == (12, 13)  # NVFP4


def test_e2m3_costs_one_bit():
    r = integerize("E2M3")
    assert (r.int_bits_signed, r.extra_bits) == (7, 1)


@pytest.mark.parametrize("name, extra", [("E3M2", 4), ("E4M3", 11), ("E5M2", 25)])
def test_wider_formats_cost_four_to_twentyfive_bits(name, extra):
    assert integerize(name).extra_bits == extra  # range +4...+25


def test_value_counts_and_maxima():
    mags = {n: f.magnitudes() for n, f in FORMATS.items()}
    assert float(max(mags["E2M1"])) == 6.0 and len(mags["E2M1"]) == 8
    assert float(max(mags["E3M2"])) == 28.0
    assert float(max(mags["E4M3"])) == 448.0
    assert float(max(mags["E5M2"])) == 57344.0
    assert integerize("BF16").extra_bits > 200
