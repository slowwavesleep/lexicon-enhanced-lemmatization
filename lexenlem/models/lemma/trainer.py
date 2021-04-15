"""
A trainer class to handle training and testing of models.
"""

import sys
import numpy as np
from collections import Counter
import torch
from typing import Dict, Any, Union, Optional, List, Tuple

import lexenlem.models.common.seq2seq_constant as constant
from lexenlem.models.common.seq2seq_model import Seq2SeqModel
from lexenlem.models.common import utils, loss
from lexenlem.models.lemma.vocab import MultiVocab, Vocab
from lexenlem.models.lemma.data import BatchItem


class Trainer:
    """ A trainer for training models. """

    def __init__(
        self,
        args: Dict[str, Any] = None,
        vocab: Vocab = None,
        emb_matrix: Union[np.array, torch.Tensor] = None,
        model_file: Optional[str] = None,
        use_cuda: bool = False,
    ):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load everything from file
            self.load(model_file, use_cuda)
        else:
            # build model from scratch
            self.args = args
            self.model = (
                None
                if args["dict_only"]
                else Seq2SeqModel(args, vocab, emb_matrix=emb_matrix, use_cuda=use_cuda)
            )
            self.vocab = vocab
            # dict-based components
            self.word_dict = dict()
            self.composite_dict = dict()
            # lexicon
            self.lexicon = None
        if not self.args["dict_only"]:
            if self.args.get("edit", False):
                self.crit = loss.MixLoss(self.vocab["combined"].size, self.args["alpha"])
                print("[Running seq2seq lemmatizer with edit classifier]")
            else:
                self.crit = loss.SequenceLoss(self.vocab["combined"].size)
            self.parameters = [p for p in self.model.parameters() if p.requires_grad]
            if use_cuda:
                self.model.cuda()
                self.crit.cuda()
            else:
                self.model.cpu()
                self.crit.cpu()
            self.optimizer = utils.get_optimizer(
                self.args["optim"], self.parameters, self.args["lr"]
            )

    def update(self, batch: BatchItem, eval: bool = False) -> float:
        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        log_probs = self.model(batch.src, batch.src_mask, batch.tgt_in, batch.lem, batch.lem_mask)
        loss = self.crit(log_probs.view(-1, self.vocab["combined"].size), batch.tgt_out.view(-1))
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args["max_grad_norm"])
        self.optimizer.step()
        return loss_val

    def predict(
        self, batch: BatchItem, beam_size: int = 1, log_attn: bool = False
    ) -> Tuple[List[str], Union[np.array, None]]:
        self.model.eval()
        preds, log_attns = self.model.predict(
            batch.src,
            batch.src_mask,
            batch.lem,
            batch.lem_mask,
            beam_size=beam_size,
            log_attn=log_attn,
        )
        pred_seqs = [self.vocab["combined"].unmap(ids) for ids in preds]  # unmap to tokens
        pred_seqs = utils.prune_decoded_seqs(pred_seqs)
        pred_tokens = ["".join(seq) for seq in pred_seqs]  # join chars to be tokens
        return pred_tokens, log_attns

    def postprocess(self, words, preds):
        """ Postprocess, mainly for handing edits. """
        assert len(words) == len(preds), "Lemma predictions must have same length as words."
        final = []
        for lemma, word in zip(preds, words):
            if len(lemma) == 0 or constant.UNK in lemma:
                final += [word]  # invalid prediction, fall back to word
            else:
                final += [lemma]
        return final

    def update_lr(self, new_lr):
        utils.change_lr(self.optimizer, new_lr)

    def train_dict(self, triples):
        """Train a dict lemmatizer given training (word, pos, lemma) triples."""
        # accumulate counter
        ctr = Counter()
        ctr.update([(p[0], p[1], p[2]) for p in triples])
        # find the most frequent mappings
        for p, _ in ctr.most_common():
            word, pos, lemma = p
            if (word, pos) not in self.composite_dict:
                self.composite_dict[(word, pos)] = lemma
            if word not in self.word_dict:
                self.word_dict[word] = lemma
        return

    def predict_dict(self, pairs, ignore_empty=False):
        """ Predict a list of lemmas using the dict model given (word, pos) pairs. """
        lemmas = []
        for p in pairs:
            w, pos = p
            if (w, pos) in self.composite_dict:
                lemmas += [self.composite_dict[(w, pos)]]
            elif w in self.word_dict:
                lemmas += [self.word_dict[w]]
            else:
                if ignore_empty:
                    lemmas += [""]
                else:
                    lemmas += [w]
        return lemmas

    def skip_seq2seq(self, pairs):
        """ Determine if we can skip the seq2seq module when ensembling with the frequency lexicon. """

        skip = []
        for p in pairs:
            w, pos = p
            if (w, pos) in self.composite_dict:
                skip.append(True)
            elif w in self.word_dict:
                skip.append(True)
            else:
                skip.append(False)
        return skip

    def ensemble(self, pairs, other_preds):
        """ Ensemble the dict with statitical model predictions. """
        lemmas = []
        assert len(pairs) == len(other_preds)
        for p, pred in zip(pairs, other_preds):
            w, pos = p
            if (w, pos) in self.composite_dict:
                lemmas += [self.composite_dict[(w, pos)]]
            elif w in self.word_dict:
                lemmas += [self.word_dict[w]]
            else:
                lemmas += [pred]
        return lemmas

    def save(self, filename):
        params = {
            "model": self.model.state_dict() if self.model is not None else None,
            "dicts": (self.word_dict, self.composite_dict),
            "vocab": self.vocab.state_dict(),
            "config": self.args,
            "lexicon": self.lexicon,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename, use_cuda=False):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            sys.exit(1)
        self.args = checkpoint["config"]
        self.word_dict, self.composite_dict = checkpoint["dicts"]
        self.vocab = MultiVocab.load_state_dict(checkpoint["vocab"])
        self.lexicon = checkpoint["lexicon"]
        if not self.args["dict_only"]:
            self.model = Seq2SeqModel(self.args, self.vocab, use_cuda=use_cuda)
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model = None
