"""
A trainer class to handle training and testing of models.
"""

import sys
from collections import Counter

import numpy as np
import torch

import lexenlem.models.common.seq2seq_constant as constant
from lexenlem.models.common.seq2seq_model import Seq2SeqModel, Seq2SeqModelCombined
from lexenlem.models.common import utils, loss
from lexenlem.models.lemma import edit
from lexenlem.models.lemma.vocab import MultiVocab
from lexenlem.preprocessing.vabamorf_pipeline import AdHocModelInput


def unpack_batch(batch, use_cuda: bool):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:9]]
    else:
        inputs = [b if b is not None else None for b in batch[:9]]
    orig_idx = batch[9]
    return inputs, orig_idx


def unpack_batch_combined(batch, use_cuda: bool):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:7]]
    else:
        inputs = [b if b is not None else None for b in batch[:7]]
    orig_idx = batch[7]
    return inputs, orig_idx


class Trainer:
    """ A trainer for training models. """
    def __init__(self, args=None, vocab=None, emb_matrix=None, model_file=None, use_cuda=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load everything from file
            self.load(model_file, use_cuda)
        else:
            # build model from scratch
            self.args = args
            self.model = None if args['dict_only'] else Seq2SeqModel(args, vocab, emb_matrix=emb_matrix, use_cuda=use_cuda)
            self.vocab = vocab
            # dict-based components
            self.word_dict = dict()
            self.composite_dict = dict()
        if not self.args['dict_only']:
            if self.args.get('edit', False):
                self.crit = loss.MixLoss(self.vocab['char'].size, self.args['alpha'])
                print("[Running seq2seq lemmatizer with edit classifier]")
            else:
                self.crit = loss.SequenceLoss(self.vocab['char'].size)
            self.parameters = [p for p in self.model.parameters() if p.requires_grad]
            if use_cuda:
                self.model.cuda()
                self.crit.cuda()
            else:
                self.model.cpu()
                self.crit.cpu()
            self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'])

    def update(self, batch, eval=False):
        inputs, orig_idx = unpack_batch(batch, self.use_cuda)
        src, src_mask, tgt_in, tgt_out, pos, feats, lem, lem_mask, edits = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        log_probs, edit_logits = self.model(src, src_mask, tgt_in, pos, feats, lem, lem_mask)
        if self.args.get('edit', False):
            assert edit_logits is not None
            loss = self.crit(
                log_probs.view(-1, self.vocab['char'].size), tgt_out.view(-1), edit_logits, edits
            )
        else:
            loss = self.crit(log_probs.view(-1, self.vocab['char'].size), tgt_out.view(-1))
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, beam_size=1):
        inputs, orig_idx = unpack_batch(batch, self.use_cuda)
        src, src_mask, tgt_in, tgt_out, pos, feats, lem, lem_mask, edits = inputs

        self.model.eval()
        batch_size = src.size(0)
        preds, edit_logits = self.model.predict(src, src_mask, pos=pos, feats=feats, lem=lem, lem_mask=lem_mask, beam_size=beam_size)
        pred_seqs = [self.vocab['char'].unmap(ids) for ids in preds]  # unmap to tokens
        pred_seqs = utils.prune_decoded_seqs(pred_seqs)
        pred_tokens = ["".join(seq) for seq in pred_seqs]  # join chars to be tokens
        pred_tokens = utils.unsort(pred_tokens, orig_idx)
        if self.args.get('edit', False):
            assert edit_logits is not None
            edits = np.argmax(edit_logits.data.cpu().numpy(), axis=1).reshape([batch_size]).tolist()
            edits = utils.unsort(edits, orig_idx)
        else:
            edits = None
        return pred_tokens, edits

    def postprocess(self, words, preds, edits=None):
        """ Postprocess, mainly for handing edits. """
        assert len(words) == len(preds), "Lemma predictions must have same length as words."
        edited = []
        if self.args.get('edit', False):
            assert edits is not None and len(words) == len(edits)
            for w, p, e in zip(words, preds, edits):
                lem = edit.edit_word(w, p, e)
                edited += [lem]
        else:
            edited = preds  # do not edit
        # final sanity check
        assert len(edited) == len(words)
        final = []
        for lem, w in zip(edited, words):
            if len(lem) == 0 or constant.UNK in lem:
                final += [w]  # invalid prediction, fall back to word
            else:
                final += [lem]
        return final

    def update_lr(self, new_lr):
        utils.change_lr(self.optimizer, new_lr)

    def train_dict(self, triples):
        """ Train a dict lemmatizer given training (word, pos, lemma) triples. """
        # accumulate counter
        ctr = Counter()
        ctr.update([(p[0], p[1], p[2]) for p in triples])
        # find the most frequent mappings
        for p, _ in ctr.most_common():
            w, pos, l = p
            if (w,pos) not in self.composite_dict:
                self.composite_dict[(w,pos)] = l
            if w not in self.word_dict:
                self.word_dict[w] = l
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
                    lemmas += ['']
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
        """ Ensemble the dict with statistical model predictions. """
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
                'model': self.model.state_dict() if self.model is not None else None,
                'dicts': (self.word_dict, self.composite_dict),
                'vocab': self.vocab.state_dict(),
                'config': self.args
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
        self.args = checkpoint['config']
        self.word_dict, self.composite_dict = checkpoint['dicts']
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        if not self.args['dict_only']:
            self.model = Seq2SeqModelCombined(self.args, self.vocab, use_cuda=use_cuda)
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model = None


class TrainerCombined(Trainer):
    """ A trainer for training models. """
    def __init__(self, args: dict = None, vocab=None, emb_matrix=None, model_file: str = None, use_cuda: bool = False):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load everything from file
            self.load(model_file, use_cuda)
        else:
            # build model from scratch
            self.args = args
            self.model = None if args['dict_only'] else Seq2SeqModelCombined(args, vocab, emb_matrix=emb_matrix, use_cuda=use_cuda)
            self.vocab = vocab
            # dict-based components
            self.word_dict = dict()
            self.composite_dict = dict()
            # lexicon
            self.lexicon = None
        if not self.args['dict_only']:
            if self.args.get('edit', False):
                self.crit = loss.MixLoss(self.vocab['combined'].size, self.args['alpha'])
                print("[Running seq2seq lemmatizer with edit classifier]")
            else:
                self.crit = loss.SequenceLoss(self.vocab['combined'].size)
            self.parameters = [p for p in self.model.parameters() if p.requires_grad]
            if use_cuda:
                self.model.cuda()
                self.crit.cuda()
            else:
                self.model.cpu()
                self.crit.cpu()
            self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'])

    def update(self, batch, eval: bool = False):
        inputs, _ = unpack_batch_combined(batch, self.use_cuda)
        src, src_mask, lem, lem_mask, tgt_in, tgt_out, edits = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        log_probs, edit_logits = self.model(src, src_mask, tgt_in, lem, lem_mask)
        if self.args.get('edit', False):
            assert edit_logits is not None
            loss = self.crit(log_probs.view(-1, self.vocab['combined'].size), tgt_out.view(-1), \
                    edit_logits, edits)
        else:
            loss = self.crit(log_probs.view(-1, self.vocab['combined'].size), tgt_out.view(-1))
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, beam_size: int = 1, log_attn: bool = False):
        inputs, orig_idx = unpack_batch_combined(batch, self.use_cuda)
        src, src_mask, lem, lem_mask, _, _, edits = inputs

        self.model.eval()
        batch_size = src.size(0)
        # not all predicts match this
        preds, edit_logits, log_attns = self.model.predict(src, src_mask, lem, lem_mask, beam_size=beam_size, log_attn=log_attn)
        pred_seqs = [self.vocab['combined'].unmap(ids) for ids in preds]  # unmap to tokens
        pred_seqs = utils.prune_decoded_seqs(pred_seqs)
        pred_tokens = ["".join(seq) for seq in pred_seqs]  # join chars to be tokens
        pred_tokens = utils.unsort(pred_tokens, orig_idx)
        if self.args.get('edit', False):
            assert edit_logits is not None
            edits = np.argmax(edit_logits.data.cpu().numpy(), axis=1).reshape([batch_size]).tolist()
            edits = utils.unsort(edits, orig_idx)
        else:
            edits = None
        return pred_tokens, edits, log_attns

    def save(self, filename: str):
        params = {
                'model': self.model.state_dict() if self.model is not None else None,
                'dicts': (self.word_dict, self.composite_dict),
                'vocab': self.vocab.state_dict(),
                'config': self.args,
                'lexicon': self.lexicon
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename: str, use_cuda: str = False):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            sys.exit(1)
        self.args = checkpoint['config']
        self.word_dict, self.composite_dict = checkpoint['dicts']
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        self.lexicon = checkpoint['lexicon']
        if not self.args['dict_only']:
            self.model = Seq2SeqModelCombined(self.args, self.vocab, use_cuda=use_cuda)
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model = None


class TrainerVb(Trainer):
    def __init__(self, args: dict = None, vocab=None, emb_matrix=None, model_file: str = None, use_cuda: bool = False):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load everything from file
            self.load(model_file, use_cuda)
        else:
            # build model from scratch
            self.args = args
            self.model = Seq2SeqModelCombined(args, vocab, emb_matrix=emb_matrix, use_cuda=use_cuda)
            self.vocab = vocab
            # dict-based components
            self.word_dict = dict()
            self.composite_dict = dict()
        self.crit = loss.SequenceLoss(self.vocab['combined'].size)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if use_cuda:
            print("Using CUDA...")
            self.model.cuda()
            self.crit.cuda()
        else:
            print("Using CPU...")
            self.model.cpu()
            self.crit.cpu()
        self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'])

    def update(self, batch: AdHocModelInput, evaluate: bool = False):

        if self.use_cuda:
            batch.cuda()
        # inputs, _ = unpack_batch_combined(batch, self.use_cuda)
        # src, src_mask, lem, lem_mask, tgt_in, tgt_out = inputs

        if evaluate:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()

        log_probs, edit_logits = self.model(
            src=batch.src, src_mask=batch.src_mask, tgt_in=batch.tgt_in, lem=batch.lem, lem_mask=batch.lem_mask
        )
        loss = self.crit(log_probs.view(-1, self.vocab['combined'].size), batch.tgt_out.view(-1))
        loss_val = loss.data.item()
        if evaluate:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch: AdHocModelInput, beam_size: int = 1, log_attn: bool = False):
        if self.use_cuda:
            batch.cuda()

        self.model.eval()
        batch_size = batch.src.size(0)
        # not all predicts match this
        preds, _, log_attns = self.model.predict(
            src=batch.src,
            src_mask=batch.src_mask,
            lem=batch.lem,
            lem_mask=batch.lem_mask,
            beam_size=beam_size,
            log_attn=log_attn,
        )
        pred_seqs = [self.vocab['combined'].unmap(ids) for ids in preds]  # unmap to tokens
        pred_seqs = utils.prune_decoded_seqs(pred_seqs)
        pred_tokens = ["".join(seq) for seq in pred_seqs]  # join chars to be tokens
        pred_tokens = utils.unsort(pred_tokens, batch.orig_idx)
        return pred_tokens, log_attns

    def save(self, filename: str):
        params = {
                'model': self.model.state_dict() if self.model is not None else None,
                'dicts': (self.word_dict, self.composite_dict),
                'vocab': self.vocab.state_dict(),
                'config': self.args,
                }
        torch.save(params, filename)
        print("model saved to {}".format(filename))

    def load(self, filename: str, use_cuda: str = False):
        checkpoint = torch.load(filename, lambda storage, loc: storage)
        self.args = checkpoint['config']
        self.word_dict, self.composite_dict = checkpoint['dicts']
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        self.model = Seq2SeqModelCombined(self.args, self.vocab, use_cuda=use_cuda)
        self.model.load_state_dict(checkpoint['model'])

    def postprocess(self, words, preds):
        if len(words) != len(preds):
            print(len(words), len(preds))
            raise RuntimeError("Lemma predictions must have same length as words.")
        final = []
        for lem, w in zip(preds, words):
            if len(lem) == 0 or constant.UNK in lem:
                final += [w]  # invalid prediction, fall back to word
            else:
                final += [lem]
        return final
