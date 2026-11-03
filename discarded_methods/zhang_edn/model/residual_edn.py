"""Residual EDN: x' = x_lsqr + EDN(x_hat). The network learns the CORRECTION over LSQR."""

import torch
import torch.nn as nn
from .encoder_decoder import EncoderDecoderNet


class ResidualEDN(nn.Module):
    """EDN with zero-init output layer so that EDN(x) = 0 at initialization.

    The full Strategy-A pipeline applies the residual outside this module:
        x' = x_lsqr + ResidualEDN(x_hat)

    Zero-initialising the final 1x1 conv guarantees the network starts from
    the LSQR baseline (loss = ||y - H x_lsqr||^2) instead of a random delta.
    """

    def __init__(self, norm_type: str = "groupnorm"):
        super().__init__()
        self.edn = EncoderDecoderNet(norm_type=norm_type)

        last_conv = None
        for module in self.edn.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise RuntimeError("No Conv2d found in EncoderDecoderNet to zero-init")
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

    def forward(self, x):
        return self.edn(x)
