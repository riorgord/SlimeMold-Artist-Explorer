"""Minimal ops replacement for ComfyUI clip models.
Replaces comfy.ops with plain torch.nn — no memory management needed for inference.
"""
import torch


class manual_cast:
    """Default operations class. Just wraps torch.nn with no casting behavior."""

    class Linear(torch.nn.Linear):
        pass

    class LayerNorm(torch.nn.LayerNorm):
        pass

    class Conv2d(torch.nn.Conv2d):
        pass

    class Embedding(torch.nn.Embedding):
        def forward(self, input, out_dtype=None):
            x = super().forward(input)
            if out_dtype is not None:
                x = x.to(dtype=out_dtype)
            return x


def cast_to(tensor, dtype, device, **kwargs):
    return tensor.to(dtype=dtype, device=device)


def cast_to_input(weight, input, non_blocking=False, copy=True):
    return weight.to(dtype=input.dtype, device=input.device)