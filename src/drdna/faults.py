"""Single-bit-flip neuron fault injection."""

from __future__ import annotations

from types import TracebackType

import torch
from torch import nn

SIGN_BIT = 31


def flip_bit(x: torch.Tensor, bit: int) -> torch.Tensor:
    """Flip one bit of the IEEE-754 float32 representation of every element.

    Args:
        x: float32 tensor.
        bit: Bit position, 0 (least significant mantissa bit) to 31 (sign).

    Returns:
        A new tensor with the bit flipped.
    """
    if x.dtype != torch.float32:
        raise TypeError("flip_bit expects a float32 tensor")
    if not 0 <= bit <= 31:
        raise ValueError("bit must be in [0, 31]")
    mask = -(1 << 31) if bit == SIGN_BIT else (1 << bit)
    return (x.contiguous().view(torch.int32) ^ mask).view(torch.float32)


class NeuronFault:
    """Context manager that flips one bit of one neuron's output in a module.

    The fault is applied to every sample of the batch at the same location. Register it
    *before* attaching a detector so the detector observes the corrupted value.

    Args:
        model: Model containing the module.
        layer: Dotted module name, e.g. ``"layer2.0.conv1"``.
        channel: Channel index (``0`` for flat outputs).
        row: Row index (ignored for outputs without spatial dimensions).
        col: Column index (ignored for outputs without spatial dimensions).
        bit: Bit position to flip, 0 to 31.
    """

    def __init__(
        self, model: nn.Module, layer: str, channel: int, row: int, col: int, bit: int
    ) -> None:
        self.module = dict(model.named_modules())[layer]
        self.index = (channel, row, col)
        self.bit = bit
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(self, _m: nn.Module, _inp: tuple, out: torch.Tensor) -> torch.Tensor:
        out = out.clone()
        if out.dim() == 4:
            c, h, w = self.index
            out[:, c, h, w] = flip_bit(out[:, c, h, w].contiguous(), self.bit)
        elif out.dim() == 2:
            out[:, self.index[0]] = flip_bit(out[:, self.index[0]].contiguous(), self.bit)
        else:
            raise ValueError(f"unsupported output rank {out.dim()}")
        return out

    def __enter__(self) -> NeuronFault:
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            self._handle.remove()
