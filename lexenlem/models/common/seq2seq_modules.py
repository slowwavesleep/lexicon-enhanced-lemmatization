"""
Pytorch implementation of basic sequence to Sequence modules.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

import lexenlem.models.common.seq2seq_constant as constant


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim: int):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(
        self,
        input: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propogate input through the network.

        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL

        if mask is not None:
            # sett the padding attention logits to -inf
            assert mask.size() == attn.size(), "Mask size must match the attention size!"
            attn = attn.masked_fill(mask, -constant.INFINITY_NUMBER)

        attn = self.softmax(attn)

        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class LSTMDoubleAttention(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        batch_first: bool = True,
    ):
        """Initialize params."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        input: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        src_ctx: torch.Tensor,
        lex_ctx: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
        lex_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Propogate input through the network."""
        if self.batch_first:
            input = input.transpose(0, 1)

        # print('dec_input:', input.size())
        # print('dec_hidden:', hidden[0].size(), hidden[1].size())

        output: List[torch.Tensor] = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.lstm_cell(input[i], hidden)
            hy, _ = hidden
            h_tilde0, _ = self.attention_layer(hy, src_ctx, mask=ctx_mask)
            h_tilde1, _ = self.attention_layer(hy, lex_ctx, mask=lex_mask)
            h_tilde = torch.cat((h_tilde0, h_tilde1), 1)
            h_tilde = self.linear(h_tilde)
            output.append(h_tilde)
        output = torch.cat(output, 0).view(input.size(0), output[0].size(0), -1)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden
