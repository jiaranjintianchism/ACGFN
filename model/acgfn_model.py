import torch
from torch import nn

from model.config import Config
from model.acgfn_encoder import ACGFNEncoder
from model.acgfn_subsampling import ConvSubsampling


class ACGFNModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.subsampling_conv = ConvSubsampling(config)
        self.encoder = ACGFNEncoder(config)

    def forward(self, input_values: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_values (torch.Tensor): with shape `(B, T, D1)`
            input_lengths (torch.Tensor): with shape `(B)`

        Returns:
            tuple(
            torch.Tensor with shape `(B, L, D)`
            torch.Tensor with shape `(B)`
            )
        """
        hidden_states, input_lengths = self.subsampling_conv(input_values, input_lengths)

        hidden_states = self.encoder(hidden_states)

        return hidden_states, input_lengths
