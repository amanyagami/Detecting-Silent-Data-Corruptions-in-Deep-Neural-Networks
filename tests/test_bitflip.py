"""Unit tests for the IEEE-754 bit manipulation in src/utils/customFI_methods.py.
No model, dataset, or GPU required -- these run in well under a second and
are the fast, deterministic first line of defense in CI."""
import struct

import torch

from src.utils.customFI_methods import single_bit_flip_func


def _make_injector():
    # bit_pos/bits are required by the constructor but unused by the
    # _flip_bit(s)_signed_Aman helpers directly under test here.
    return single_bit_flip_func(model=torch.nn.Identity(), batch_size=1, input_shape=[3, 4, 4], bit_pos=0)


def test_flip_sign_bit_negates_value():
    pfi = _make_injector()
    value = torch.tensor(1.5, dtype=torch.float32)
    pfi.bit_pos = 0  # IEEE-754 float32 sign bit
    flipped = pfi._flip_bit_signed_Aman(value, max_value=None, bit_pos=0)
    assert flipped.item() == -1.5


def test_flip_is_its_own_inverse():
    """Flipping the same bit twice must round-trip back to the original value."""
    pfi = _make_injector()
    value = torch.tensor(3.25, dtype=torch.float32)
    for bit_pos in (0, 1, 8, 16, 31):
        pfi.bit_pos = bit_pos
        once = pfi._flip_bit_signed_Aman(value, max_value=None, bit_pos=bit_pos)
        twice = pfi._flip_bit_signed_Aman(once, max_value=None, bit_pos=bit_pos)
        assert twice.item() == value.item(), f"bit {bit_pos} did not round-trip"


def test_flip_actually_changes_the_bit_pattern():
    pfi = _make_injector()
    value = torch.tensor(0.1, dtype=torch.float32)
    pfi.bit_pos = 10
    flipped = pfi._flip_bit_signed_Aman(value, max_value=None, bit_pos=10)
    orig_bits = bin(int.from_bytes(struct.pack("!f", value.item()), "big"))[2:].zfill(32)
    new_bits = bin(int.from_bytes(struct.pack("!f", flipped.item()), "big"))[2:].zfill(32)
    assert orig_bits != new_bits
    assert sum(a != b for a, b in zip(orig_bits, new_bits)) == 1


def test_multi_bit_flip_touches_exactly_the_requested_bits():
    pfi = _make_injector()
    value = torch.tensor(2.0, dtype=torch.float32)
    positions = [1, 9, 20]
    flipped = pfi._flip_bits_signed_Aman(value, positions)
    orig_bits = bin(int.from_bytes(struct.pack("!f", value.item()), "big"))[2:].zfill(32)
    new_bits = bin(int.from_bytes(struct.pack("!f", flipped.item()), "big"))[2:].zfill(32)
    changed = [i for i, (a, b) in enumerate(zip(orig_bits, new_bits)) if a != b]
    assert sorted(changed) == sorted(positions)


def test_multi_bit_flip_is_its_own_inverse():
    pfi = _make_injector()
    value = torch.tensor(7.75, dtype=torch.float32)
    positions = [2, 5, 30]
    once = pfi._flip_bits_signed_Aman(value, positions)
    twice = pfi._flip_bits_signed_Aman(once, positions)
    assert twice.item() == value.item()
