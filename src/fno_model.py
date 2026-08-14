import torch
import torch.nn as nn
from neuralop.models import TFNO


class TFNOFlood(nn.Module):
    """
    TFNO wrapper for flood depth prediction.

    Parameters
    ----------
    in_channels     : C in the input tensor (5 + n_rain_history, default 17)
    out_channels    : output channels (1 = water depth increment)
    hidden_dim      : FNO hidden channel width
    n_layers        : number of Fourier layers
    modes1          : Fourier truncation modes along dim 0
    modes2          : Fourier truncation modes along dim 1
    rank            : Tucker decomposition rank fraction (default 0.42)
    domain_padding  : fractional zero-padding to reduce boundary aliasing (default 0.1)
    """

    def __init__(
        self,
        in_channels:    int   = 7,
        out_channels:   int   = 1,
        hidden_dim:     int   = 64,
        n_layers:       int   = 4,
        modes1:         int   = 16,
        modes2:         int   = 16,
        rank:           float = 0.42,
        domain_padding: float = 0.1,
        # kept for CLI / checkpoint compatibility — unused
        latent_grid_size: int   = 32,
        gno_radius:       float = 0.05,
    ):
        super().__init__()
        self.out_channels = out_channels

        self.model = TFNO(
            n_modes          = (modes1, modes2),
            in_channels      = in_channels,
            out_channels     = out_channels,
            hidden_channels  = hidden_dim,
            n_layers         = n_layers,
            factorization    = 'tucker',
            rank             = rank,
            domain_padding   = domain_padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, H, W)  →  (B, out_channels, H, W)"""
        # rfftn does not support bf16/fp16 — run in fp32 and cast output back
        with torch.amp.autocast('cuda', enabled=False):
            return self.model(x.float()).to(x.dtype)
        