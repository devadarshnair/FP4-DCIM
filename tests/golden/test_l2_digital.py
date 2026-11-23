import numpy as np

from cimfp4.golden.l2_digital import WORDS_PER_SET, WeightBus


def test_bus_counts_beats_and_toggles():
    bus = WeightBus()
    bus.transfer_set(np.zeros(WORDS_PER_SET, np.uint64))
    assert bus.beats == 16 and bus.toggles == 0
    bus.transfer_set(np.array([0xFFFFFFFF] + [0] * 15, np.uint64))
    assert bus.beats == 32 and bus.toggles == 64  # 0->all ones->0


def test_bus_energy_scales_with_length():
    bus = WeightBus()
    bus.transfer_set(np.arange(16, dtype=np.uint64))
    e1, e2 = bus.energy_fj(100.0), bus.energy_fj(1000.0)
    assert e2 > e1 > 0 and np.isclose(e2 / e1, 10.0)
