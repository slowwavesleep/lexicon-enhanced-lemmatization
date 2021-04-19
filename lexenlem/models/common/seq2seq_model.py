"""
The full encoder-decoder model, built on top of the base seq2seq modules.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import lexenlem.models.common.seq2seq_constant as constant
from lexenlem.models.common import utils
from lexenlem.models.common.seq2seq_modules import LSTMDoubleAttention
from lexenlem.models.common.vocab import BaseVocab

from typing import Dict, Any, Union, List, Tuple, Optional


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        nlayers: int,
        bidirectional: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.dropout = dropout

        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.nlayers,
            bidirectional=self.bidirectional,
            batch_first=self.batch_first,
            dropout=self.dropout if self.nlayers > 1 else 0,
        )

    def forward(
        self, enc_inputs: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode source sequence.

        Args:
            encoder_id (int): Encoder model.
            enc_inputs (torch.Tensor): Tensor of type `torch.float` and shape
                `(batch_size, seq_len, emb_dim)`, containing the encoded
                input representations.
            lens (torch.Tensor): Tensor of type `torch.long` and shape
                `(batch_size)`, containing the original lengths (before
                padding) of each input. Used for creating a packed sequence
                (see `torch.nn.utils.rnn.pack_padded_sequence`).
                **Must be a CPU tensor**.

        Returns:
            h_in (torch.Tensor): Tensor of type `torch.float` and shape
                `(batch_size, seq_len, num_directions * enc_hidden_dim)`,
                containing the output features from the last layer of the LSTM.
            hn, cn (torch.Tensor): Tensors of type `torch.float` and shape
                `(batch_size, num_directions * enc_hidden_dim`),
                containing the concatenated last hidden states (hn) and cell
                states (cn) of the last layers of both BiLSTM directions.
        """
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            enc_inputs, lens, batch_first=self.batch_first, enforce_sorted=False
        )
        packed_h_in, (hn, cn) = self.lstm(packed_inputs)
        h_in, _ = nn.utils.rnn.pad_packed_sequence(packed_h_in, batch_first=self.batch_first)
        hn = torch.cat((hn[-1], hn[-2]), 1)
        cn = torch.cat((cn[-1], cn[-2]), 1)
        return h_in, (hn, cn)


class Seq2SeqModel(nn.Module):
    """
    A complete encoder-decoder model, with optional attention.

    Args:
        args (Dict[str, Any]): Dict containing the arguments for the model
            initialization.
        vocab (BaseVocab): Vocabulary with the char-index mappings.
        emb_matrix: (Union[np.array, torch.Tensor], optional): Pretrained
            character embeddings.
        use_cuda: (bool, optional): Whether or not use CUDA. Defaults to False.

    Attributes:
        vocab_size (int): Character vocabulary size. Specified by the
            "vocab_size" argument.
        emb_dim (int): Character embeddings dimension. Specified by the
            "emb_dim" argument.
        hidden_dim (int): Hidden dimension for the encoder and decoder.
            Specified by the "hidden_dim" argument.
        nlayers (int): Number of encoder and decoder hidden layers.
            Specified by the "num_layers" argument.
        emb_dropout (float): Dropout rate applied to the input embeddings.
            Specified by the "emb_dropout" argument. If not specified,
            defaults to `0` (no dropout).
        dropout (float): Dropout rate applied after each hidden layer in the
            encoder and decoder. Specified by the "dropout" argument.
            Equals to `0` if `nlayers` is `1`.
        is_lexicon (bool): `True` if "lemmatizer" argument contains external
            lemmatizer, `False` otherwise.
        lexicon_dropout (float): Dropout rate applied to the candidates inputs.
            Specified by the "lexicon_dropout" argument.
        pad_token (int): Padding token id. See `constant.PAD_ID`.
        max_dec_len (int): Maximum decoded length. Specified by the
            "max_dec_len" argument.
        device (torch.device): Device to use in the model. Depends of `use_cuda`.
        top (int): Number of char embeddings to finetune. Specified by the "top"
            argument. If not specified, defaults to `1e10`.
        args (Dict[str, Any]): Dict containing the arguments for the model
            initialization.
        emb_matrix: (Union[np.array, torch.Tensor], optional): Pretrained
            character embeddings.
        vocab (BaseVocab): Vocabulary with the char-index mappings.
        log_attn (bool): Whether or not to log attention scores. Specified by
            the "log_attn" argument.
    """

    def __init__(
        self,
        args: Dict[str, Any],
        vocab: BaseVocab,
        emb_matrix: Optional[Union[np.array, torch.Tensor]] = None,
        use_cuda: bool = False,
    ):
        super().__init__()
        self.vocab_size = args["vocab_size"]
        self.emb_dim = args["emb_dim"]
        self.hidden_dim = args["hidden_dim"]
        self.nlayers = args["num_layers"]  # encoder layers, decoder layers = 1
        self.emb_dropout = args.get("emb_dropout", 0.0)
        self.dropout = args["dropout"]
        self.is_lexicon = False if args.get("lemmatizer", None) is None else True
        self.lexicon_dropout = args["lexicon_dropout"]
        self.pad_token = constant.PAD_ID
        self.max_dec_len = args["max_dec_len"]
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        self.top = args.get("top", 1e10)
        self.args = args
        self.emb_matrix = emb_matrix
        self.vocab = vocab
        self.log_attn = args["log_attn"]

        print("Building an attentional Seq2Seq model...")
        print("Using a Bi-LSTM encoder")
        print("Using a lexicon:", self.is_lexicon)
        print("Lexicon dropout:", self.lexicon_dropout)
        self.num_directions = 2
        self.enc_hidden_dim = self.hidden_dim // 2
        self.dec_hidden_dim = self.hidden_dim

        self.emb_drop = nn.Dropout(self.emb_dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, self.pad_token)

        self.encoder = LSTMEncoder(
            self.emb_dim,
            self.enc_hidden_dim,
            self.nlayers,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout if self.nlayers > 1 else 0,
        )

        self.lexicon_encoder = LSTMEncoder(
            self.emb_dim,
            self.enc_hidden_dim,
            self.nlayers,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout if self.nlayers > 1 else 0,
        )

        self.decoder = LSTMDoubleAttention(
            self.emb_dim,
            self.dec_hidden_dim,
            batch_first=True,
        )

        self.hn_linear = nn.Linear(self.enc_hidden_dim * 4, self.enc_hidden_dim * 2)
        self.cn_linear = nn.Linear(self.enc_hidden_dim * 4, self.enc_hidden_dim * 2)
        self.h_in_linear = nn.Linear(self.enc_hidden_dim * 2, self.enc_hidden_dim)
        self.dec2vocab = nn.Linear(self.dec_hidden_dim, self.vocab_size)

        self.SOS_tensor = torch.tensor([constant.SOS_ID], dtype=torch.long, device=self.device)

        self.h0: torch.Tensor = torch.empty(0)
        self.c0: torch.Tensor = torch.empty(0)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize embeddings and decide finetuning."""
        # initialize embeddings
        init_range = constant.EMB_INIT_RANGE
        if self.emb_matrix is not None:
            if isinstance(self.emb_matrix, np.ndarray):
                self.emb_matrix = torch.from_numpy(self.emb_matrix)
            assert self.emb_matrix.size() == (
                self.vocab_size,
                self.emb_dim,
            ), "Input embedding matrix must match size: {} x {}".format(
                self.vocab_size, self.emb_dim
            )
            self.embedding.weight.data.copy_(self.emb_matrix)
        else:
            self.embedding.weight.data.uniform_(-init_range, init_range)
        # decide finetuning
        if self.top <= 0:
            print("Do not finetune embedding layer.")
            self.embedding.weight.requires_grad = False
        elif self.top < self.vocab_size:
            print("Finetune top {} embeddings.".format(self.top))
            self.embedding.weight.register_hook(lambda x: utils.keep_partial_grad(x, self.top))
        else:
            print("Finetune all embeddings.")

    def cuda(self):
        super().cuda()
        self.device = torch.device("cuda")

    def cpu(self):
        super().cpu()
        self.device = torch.device("cpu")

    def decode(
        self,
        dec_inputs: torch.Tensor,
        hn: torch.Tensor,
        cn: torch.Tensor,
        src_ctx: torch.Tensor,
        lex_ctx: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
        lex_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode a step, based on context encoding and source context states.

        Args:
            dec_inputs (torch.Tensor): Tensor of type `torch.long` and shape
                `(batch_size, seq_len, emb_hidden_dim)`, containing the
                encoded output representations. During training contains
                output sequence and during inference the generated
                sequence on each step.
            hn (torch.Tensor): Tensor of type `torch.float` and shape
                `(batch_size, num_directions * enc_hidden_dim)`, containing
                the last hidden state from the encoder.
            cn (torch.Tensor): Tensor of type `torch.float` and shape
                `(batch_size, num_directions * enc_hidden_dim)`, containing
                the last cell state from the encoder.
            src_ctx (torch.Tensor): Tensor of type `torch.long` and shape
                `(batch_size, seq_len, emb_hidden_dim)`, containing the
                encoded input representations.
            lex_ctx (Optional[torch.Tensor], optional): Tensor of type
                `torch.long` and shape `(batch_size, seq_len, emb_hidden_dim)`,
                containing the encoded candidates representations.
            ctx_mask (Optional[torch.Tensor], optional): Tensor of type
                `torch.bool` and shape `(batch_size, seq_len)`, containing
                `True` in the positions of padding symbols for the input
                sequence and `False` in all others.
            lex_mask (Optional[torch.Tensor], optional): Tensor of type
                `torch.bool` and shape `(batch_size, seq_len)`, containing
                `True` in the positions of padding symbols for the candidates
                sequence and `False` in all others.

        Returns:
            log_probs (torch.Tensor): Tensor of type `torch.float` and shape
                `(batch_size, seq_len, vocab_size)`, containing the softmax
                probabilities of each symbol in the vocab to be the next
                symbol in the decoded sequence.
            dec_hidden (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors of
                type `torch.float` and shape
                `(batch_size, num_directions * dec_hidden_dim)`, containing
                the last hidden and cell states for the decoder BiLSTM.
            attn (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors of type
                `torch.float` and shape `(batch_size, seq_len)`, containing
                the attention scores for each input and candidate sequence item
                correspondingliy.
        """
        dec_hidden = (hn, cn)
        h_out, dec_hidden = self.decoder(
            dec_inputs, dec_hidden, src_ctx, lex_ctx, ctx_mask, lex_mask
        )

        h_out_reshape = h_out.contiguous().view(h_out.size(0) * h_out.size(1), -1)
        decoder_logits = self.dec2vocab(h_out_reshape)
        decoder_logits = decoder_logits.view(h_out.size(0), h_out.size(1), -1)
        log_probs = self.get_log_prob(decoder_logits)
        return log_probs, dec_hidden

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_in: torch.Tensor,
        lem: torch.Tensor,
        lem_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Perform the forward pass of the model.

        Args:
            src (torch.Tensor): Tensor of type `torch.long` and shape
                `(batch_size, seq_len)`, containing the padded indices of each
                character of the input. `seq_len` is the maximum sequence
                length of the source input in the batch.
            src_mask (torch.Tensor): Tensor of type `torch.bool` and shape
                `(batch_size, seq_len)`, containing `True` in the positions
                of padding symbols and `False` in all others.
            tgt_in (torch.Tensor): Tensor of type `torch.long` and shape
                `(batch_size, seq_len)`, containing the padded indices of each
                character of the output. `seq_len` is the maximum sequence
                length of the output in the batch.
            lem (Optional[torch.Tensor], optional): Tensor of type `torch.long`
                and shape `(batch_size, seq_len)`, containing the padded
                indices of each character of the candidates input. `seq_len`
                is the maximum sequence length of the candidate input in the
                batch. Defaults to None.
            lem_mask (Optional[torch.Tensor], optional): Tensor of type
                `torch.bool` and shape `(batch_size, seq_len)`, containing
                `True` in the positions of padding symbols and `False`
                otherwise. Defaults to None.

        Returns:
            log_probs (torch.Tensor): Tensor of type `torch.float` and
                shape `(batch_size, seq_len, vocab_size)`, containing the
                softmax probabilities of each symbol in the vocab to be the
                next symbol in the decoded sequence.
        """
        # prepare for encoder/decoder
        enc_inputs = self.emb_drop(self.embedding(src))
        dec_inputs = self.emb_drop(self.embedding(tgt_in))
        src_lens = torch.sum(torch.eq(src_mask, 0), dim=1)
        if self.device.type == "cuda":
            src_lens = src_lens.detach().cpu()

        h_in, (hn, cn) = self.encoder(enc_inputs, src_lens)

        # Replace the word from the lexicon with UNK with the probability of
        # lexicon_dropout
        if (
            self.is_lexicon
            and self.lexicon_dropout > 0
            and lem is not None
            and lem_mask is not None
        ):
            lem_stump = torch.tensor(
                [constant.SOS_ID, constant.EOS_ID] + [constant.PAD_ID] * (lem.size(1) - 2),
                device=self.device,
            )
            lem_mask_stump = torch.tensor(
                [0] * 3 + [1] * (lem.size(1) - 3), dtype=lem_mask.dtype, device=self.device
            )
            lem_hide = (
                torch.tensor(lem.size(0), dtype=torch.float, device=self.device).uniform_()
                < self.lexicon_dropout
            )
            if lem_hide.any():
                lem[lem_hide] = lem_stump.repeat(lem[lem_hide].size(0), 1)
                lem_mask[lem_hide] = lem_mask_stump.repeat(lem_mask[lem_hide].size(0), 1)

        lem_inputs = self.emb_drop(self.embedding(lem))
        lem_lens = torch.sum(torch.eq(lem_mask, 0), dim=1)
        if self.device.type == "cuda":
            lem_lens = lem_lens.detach().cpu()

        # Make the mask elements have the same size as encoder outputs
        if lem_mask.size(1) != torch.max(lem_lens).item():
            lem_mask = lem_mask.narrow(1, 0, torch.max(lem_lens).item())

        h_in1, (hn1, cn1) = self.lexicon_encoder(lem_inputs, lem_lens)

        hn = torch.cat((hn, hn1), 1)
        cn = torch.cat((cn, cn1), 1)

        hn = self.hn_linear(hn)
        cn = self.cn_linear(cn)

        log_probs, _ = self.decode(dec_inputs, hn, cn, h_in, h_in1, src_mask, lem_mask)

        return log_probs

    def predict(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        lem: torch.Tensor,
        lem_mask: torch.Tensor,
    ) -> Tuple[List[List[int]]]:
        """Predict with greedy decoding.

        Args:
            src (torch.Tensor): [description]
            src_mask (torch.Tensor): [description]
            lem (torch.Tensor): [description]
            lem_mask (torch.Tensor): [description]

        Returns:
            Tuple[List[List[int]]: [description]
        """
        enc_inputs = self.embedding(src)
        batch_size = enc_inputs.size(0)
        src_lens = torch.sum(torch.eq(src_mask, 0), dim=1)
        if self.device.type == "cuda":
            src_lens = src_lens.detach().cpu()

        # encode source
        h_in, (hn, cn) = self.encoder(enc_inputs, src_lens)

        lem_inputs = self.emb_drop(self.embedding(lem))
        lem_lens = torch.sum(torch.eq(lem_mask, 0), dim=1)
        if self.device.type == "cuda":
            lem_lens = lem_lens.detach().cpu()
        h_in1, (hn1, cn1) = self.lexicon_encoder(lem_inputs, lem_lens)

        hn = torch.cat((hn, hn1), 1)
        cn = torch.cat((cn, cn1), 1)

        hn = self.hn_linear(hn)
        cn = self.cn_linear(cn)

        # greedy decode by step
        dec_inputs = self.embedding(self.SOS_tensor)
        dec_inputs = dec_inputs.expand(batch_size, dec_inputs.size(0), dec_inputs.size(1))

        done = [False for _ in range(batch_size)]
        total_done = 0
        max_len = 0
        output_seqs = [[] for _ in range(batch_size)]

        while total_done < batch_size and max_len < self.max_dec_len:
            log_probs, (hn, cn) = self.decode(dec_inputs, hn, cn, h_in, h_in1, src_mask, lem_mask)
            assert log_probs.size(1) == 1, "Output must have 1-step of output."
            _, preds = log_probs.squeeze(1).max(1, keepdim=True)
            dec_inputs = self.embedding(preds)  # update decoder inputs
            max_len += 1
            for i in range(batch_size):
                if not done[i]:
                    token = preds.data[i][0].item()
                    if token == constant.EOS_ID:
                        done[i] = True
                        total_done += 1
                    else:
                        output_seqs[i].append(token)

        return output_seqs

    def get_log_prob(self, logits: torch.Tensor) -> torch.Tensor:
        logits_reshape = logits.view(-1, self.vocab_size)
        log_probs = F.log_softmax(logits_reshape, dim=1)
        if logits.dim() == 2:
            return log_probs
        return log_probs.view(logits.size(0), logits.size(1), logits.size(2))
