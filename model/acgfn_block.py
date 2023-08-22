from typing import Optional

import torch
from torch import nn

from model.config import Config
from model.self_attention import SelfAttentionModule


class ACGFNBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.self_attn = SelfAttentionModule(config)
        self.conv_module = ConvolutionModule(config)
        self.glu_ffn = GateFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D)`
            position_embeddings (torch.Tensor): with shape `(1, L, D)`
            attention_mask: (torch.Tensor): with shape `(B, L)`

        Returns:
            torch.Tensor with shape `(B, L, D)`
        """

        # Convolution Module
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings

        )
        hidden_states = hidden_states + residual

        # Self-attention Module
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states)
        hidden_states = residual + hidden_states

        # Gate Feed forward
        residual = hidden_states
        hidden_states = self.glu_ffn(hidden_states)
        hidden_states = hidden_states + residual

        out = self.final_layer_norm(hidden_states)

        return out


class ConvolutionModule(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.point_wise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.glu = nn.GLU(dim=1)

        self.depth_wise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depth_wise_kernel_size,
            stride=1,
            padding=(config.conv_depth_wise_kernel_size - 1) // 2,
            groups=config.hidden_size,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.activation = nn.SiLU()

        self.point_wise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D)`

        Returns:
            torch.Tensor with shape `(B, L, D)`
        """

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.point_wise_conv1(hidden_states)
        hidden_states = self.glu(hidden_states)

        hidden_states = self.depth_wise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.point_wise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        return hidden_states


class GateFeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.intermediate_dense_w = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_dense_v = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.intermediate_dropout = nn.Dropout(p=0.1)

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(p=0.1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D)`

        Returns:
            torch.Tensor with shape`(B, L, D)`
        """

        hidden_states = self.layer_norm(hidden_states)
        hidden_glue = self.intermediate_act_fn(self.intermediate_dense_w(hidden_states))
        hidden_linear = self.intermediate_dense_v(hidden_states)
        hidden_states = hidden_glue*hidden_linear
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        return hidden_states

