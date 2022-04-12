"""
Pytorch implementation of basic sequence to Sequence modules.
"""

import torch
import torch.nn as nn

import lexenlem.models.common.seq2seq_constant as constant


class BasicAttention(nn.Module):
    """
    A basic MLP attention layer.
    """

    def __init__(self, dim):
        super(BasicAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_c = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=1)

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        target = self.linear_in(input)  # batch x dim
        source = self.linear_c(context.contiguous().view(-1, dim)).view(batch_size, source_len, dim)
        attn = target.unsqueeze(1).expand_as(context) + source
        attn = self.tanh(attn)  # batch x sourceL x dim
        attn = self.linear_v(attn.view(-1, dim)).view(batch_size, source_len)

        if mask is not None:
            attn.masked_fill_(mask, -constant.INFINITY_NUMBER)

        attn = self.sm(attn)
        if attn_only:
            return attn

        weighted_context = torch.bmm(attn.unsqueeze(1), context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """Propagate input through the network.

        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL

        if mask is not None:
            # sett the padding attention logits to -inf
            assert mask.size() == attn.size(), "Mask size must match the attention size!"
            attn.masked_fill_(mask, -constant.INFINITY_NUMBER)

        attn = self.sm(attn)
        if attn_only:
            return attn

        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn

# unused
# class MultiHeadAttention(nn.Module):
#     """ Multi-head Attention
#
#     Ref: https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention"""
#
#     def __init__(self, dim, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#
#         # assert d_model % self.num_heads == 0
#         #
#         # self.depth = d_model
#
#         self.wq = nn.Linear(dim, dim)
#         self.wk = nn.Linear(dim, dim)
#         self.wv = nn.Linear(dim, dim)
#
#         self.linear = nn.Linear(dim, dim)
#
#     def forward(self, input, context, mask=None, attn_only=False):
#         target = self.linear(input)


class LinearAttention(nn.Module):
    """ A linear attention form, inspired by BiDAF:
        a = W (u; v; u o v)
    """

    def __init__(self, dim):
        super(LinearAttention, self).__init__()
        self.linear = nn.Linear(dim * 3, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        u = input.unsqueeze(1).expand_as(context).contiguous().view(-1, dim)  # batch*sourceL x dim
        v = context.contiguous().view(-1, dim)
        attn_in = torch.cat((u, v, u.mul(v)), 1)
        attn = self.linear(attn_in).view(batch_size, source_len)

        if mask is not None:
            # sett the padding attention logits to -inf
            assert mask.size() == attn.size(), "Mask size must match the attention size!"
            attn.masked_fill_(mask, -constant.INFINITY_NUMBER)

        attn = self.sm(attn)
        if attn_only:
            return attn

        attn3 = attn.view(batch_size, 1, source_len)  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class DeepAttention(nn.Module):
    """ A deep attention form, invented by Robert:
        u = ReLU(Wx)
        v = ReLU(Wy)
        a = V.(u o v)
    """

    def __init__(self, dim):
        super(DeepAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.relu = nn.ReLU()
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        u = input.unsqueeze(1).expand_as(context).contiguous().view(-1, dim)  # batch*sourceL x dim
        u = self.relu(self.linear_in(u))
        v = self.relu(self.linear_in(context.contiguous().view(-1, dim)))
        attn = self.linear_v(u.mul(v)).view(batch_size, source_len)

        if mask is not None:
            # sett the padding attention logits to -inf
            assert mask.size() == attn.size(), "Mask size must match the attention size!"
            attn.masked_fill_(mask, -constant.INFINITY_NUMBER)

        attn = self.sm(attn)
        if attn_only:
            return attn

        attn3 = attn.view(batch_size, 1, source_len)  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class LSTMAttention(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True, attn_type='soft'):
        """Initialize params."""
        super(LSTMAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        if attn_type == 'soft':
            self.attention_layer = SoftDotAttention(hidden_size)
        elif attn_type == 'mlp':
            self.attention_layer = BasicAttention(hidden_size)
        elif attn_type == 'linear':
            self.attention_layer = LinearAttention(hidden_size)
        elif attn_type == 'deep':
            self.attention_layer = DeepAttention(hidden_size)
        else:
            raise NotImplementedError("Unsupported LSTM attention type: {}".format(attn_type))
        print("Using {} attention for LSTM.".format(attn_type))

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propagate input through the network."""
        if self.batch_first:
            input = input.transpose(0, 1)

        # print('dec_input:', input.size())
        # print('dec_hidden:', hidden[0].size(), hidden[1].size())

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.lstm_cell(input[i], hidden)
            hy, cy = hidden
            h_tilde, alpha = self.attention_layer(hy, ctx, mask=ctx_mask)
            output.append(h_tilde)
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden


class LSTMDoubleAttention(LSTMAttention):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True, attn_type='soft'):
        """Initialize params."""
        super().__init__(input_size, hidden_size, batch_first, attn_type)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input, hidden, src_ctx, lex_ctx, ctx_mask=None, lex_mask=None):
        """Propogate input through the network."""
        if self.batch_first:
            input = input.transpose(0, 1)

        # print('dec_input:', input.size())
        # print('dec_hidden:', hidden[0].size(), hidden[1].size())

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.lstm_cell(input[i], hidden)
            hy, _ = hidden
            h_tilde0, attn0 = self.attention_layer(hy, src_ctx, mask=ctx_mask)
            h_tilde1, attn1 = self.attention_layer(hy, lex_ctx, mask=lex_mask)
            h_tilde = torch.cat((h_tilde0, h_tilde1), 1)
            h_tilde = self.linear(h_tilde)
            output.append(h_tilde)
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        attn = (attn0, attn1)

        return output, hidden, attn
