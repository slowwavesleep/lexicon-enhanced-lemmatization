import random
import numpy as np
import os
from collections import Counter
import torch
import sys

import stanfordnlp.models.common.seq2seq_constant as constant
from stanfordnlp.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanfordnlp.models.common import conll
from stanfordnlp.models.lemma.vocab import Vocab, MultiVocab
from stanfordnlp.models.lemma import edit
from stanfordnlp.models.pos.vocab import FeatureVocab
from stanfordnlp.pipeline.doc import Document
from estnltk import Text
import json


class DataLoader:
    def __init__(self, input_src, batch_size, args, vocab=None, evaluation=False, conll_only=False, skip=None):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.text = Text

        # check if input source is a file or a Document object
        if isinstance(input_src, str):
            filename = input_src
            assert filename.endswith('conllu'), "Loaded file must be conllu file."
            self.conll, data = self.load_file(filename)
        elif isinstance(input_src, Document):
            filename = None
            doc = input_src
            self.conll, data = self.load_doc(doc)

        if conll_only: # only load conll file
            return

        if skip is not None:
            assert len(data) == len(skip)
            data = [x for x, y in zip(data, skip) if not y]

        # handle vocab
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = dict()
            char_vocab, pos_vocab, feats_vocab = self.init_vocab(data)
            self.vocab = MultiVocab({'char': char_vocab, 'pos': pos_vocab, 'feats': feats_vocab})

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            print("Subsample training set with rate {:g}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab['char'], self.vocab['pos'], self.vocab['feats'], args)
        # shuffle for training
        if self.shuffled:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def make_feats_data(self, data, feats_idx=2):
        feats_data = []
        for d in data:
            feats = d[feats_idx]
            if '|' in feats:
                feats_data.extend(feats.split('|'))
            else:
                feats_data.append(feats)
        return feats_data

    def init_vocab(self, data):
        assert self.eval is False, "Vocab file must exist for evaluation"
        char_data = list("".join(d[0] + d[2] for d in data))
        pos_data = [d[1] for d in data]
        feats_data = self.make_feats_data(data)
        char_vocab = Vocab(char_data, self.args['lang'])
        pos_vocab = Vocab(pos_data, self.args['lang'])
        feats_vocab = FeatureVocab([data], self.args['lang'], idx=3)
        return char_vocab, pos_vocab, feats_vocab

    def preprocess(self, data, char_vocab, pos_vocab, feats_vocab, args):
        processed = []
        for d in data:
            edit_type = edit.EDIT_TO_ID[edit.get_edit_type(d[0], d[2])]
            src = list(d[0])
            src = [constant.SOS] + src + [constant.EOS]
            src = char_vocab.map(src)
            processed_sent = [src]
            pos = d[1]
            pos = pos_vocab.unit2id(pos)
            tgt = list(d[2])
            tgt_in = char_vocab.map([constant.SOS] + tgt)
            tgt_out = char_vocab.map(tgt + [constant.EOS])
            processed_sent += [tgt_in]
            processed_sent += [tgt_out]
            processed_sent += [pos]
            vabamorf_lemmas = self.text(d[0]).lemmas
            vabamorf_lemmas = ''.join([''.join(list(l.replace('|', ''))) for l in vabamorf_lemmas])
            vbf = list(vabamorf_lemmas)
            vbf = [constant.SOS] + vbf + [constant.EOS]
            vbf = char_vocab.map(vbf)
            processed_sent += [vbf]
            feats = d[3]
            feats = feats_vocab.map([feats])
            processed_sent += [feats]
            processed_sent += [edit_type]
            processed.append(processed_sent)
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 7

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        src = batch[0]
        src = get_long_tensor(src, batch_size)
        src_mask = torch.eq(src, constant.PAD_ID)
        tgt_in = get_long_tensor(batch[1], batch_size)
        tgt_out = get_long_tensor(batch[2], batch_size)
        pos = torch.LongTensor(batch[3])
        vbf = batch[4]
        vbf = get_long_tensor(vbf, batch_size)
        vbf_mask = torch.eq(vbf, constant.PAD_ID)
        feats = torch.LongTensor(batch[5])
        edits = torch.LongTensor(batch[6])
        assert tgt_in.size(1) == tgt_out.size(1), "Target input and output sequence sizes do not match."
        return src, src_mask, tgt_in, tgt_out, pos, feats, vbf, vbf_mask, edits, orig_idx

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_file(self, filename):
        conll_file = conll.CoNLLFile(filename)
        data = conll_file.get(['word', 'upos', 'lemma', 'feats'])
        return conll_file, data

    def load_doc(self, doc):
        data = doc.conll_file.get(['word', 'upos', 'lemma', 'feats'])
        return doc.conll_file, data


class DataLoaderCombined(DataLoader):
    def __init__(self, input_src, batch_size, args, vocab=None, evaluation=False, conll_only=False, skip=None):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.text = Text

        # check if input source is a file or a Document object
        if isinstance(input_src, str):
            filename = input_src
            assert filename.endswith('conllu'), "Loaded file must be conllu file."
            self.conll, data = self.load_file(filename)
        elif isinstance(input_src, Document):
            filename = None
            doc = input_src
            self.conll, data = self.load_doc(doc)

        if conll_only: # only load conll file
            return

        if skip is not None:
            assert len(data) == len(skip)
            data = [x for x, y in zip(data, skip) if not y]

        # handle vocab
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = dict()
            combined_vocab = self.init_vocab(data)
            self.vocab = MultiVocab({'combined': combined_vocab})

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            print("Subsample training set with rate {:g}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab['combined'], args)
        # shuffle for training
        if self.shuffled:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def make_feats_data(self, data, feats_idx=3):
        feats_data = []
        for d in data:
            feats = d[feats_idx]
            if '|' in feats:
                feats_data.extend(feats.split('|'))
            else:
                feats_data.append(feats)
        return feats_data

    def init_vocab(self, data):
        assert self.eval is False, "Vocab file must exist for evaluation"
        char_data = list("".join(d[0] + d[2] for d in data))
        pos_data = [d[1] for d in data]
        feats_data = self.make_feats_data(data)
        combined_data = char_data + pos_data + feats_data
        combined_vocab = Vocab(combined_data, self.args['lang'])
        return combined_vocab

    def preprocess(self, data, combined_vocab, args):
        processed = []
        for d in data:
            edit_type = edit.EDIT_TO_ID[edit.get_edit_type(d[0], d[2])]
            src = list(d[0])
            vabamorf_lemmas = self.text(d[0]).lemmas
            vabamorf_lemmas = ''.join([''.join(list(l.replace('|', ''))) for l in vabamorf_lemmas])
            vbf = list(vabamorf_lemmas)
            vbf = [constant.SOS] + vbf + [constant.EOS]
            src = [constant.SOS] + src + [constant.EOS]
            pos = [d[1]]
            feats = []
            if '|' in d[3]:
                feats.extend(d[3].split('|'))
            else:
                feats.append(d[3])
            inp = src + pos + feats

            #Make source and vabamorf inputs to be equal length
            #max_len = max(len(inp), len(vbf))
            #for i, row in enumerate([inp, vbf]):
            #    if len(row) < max_len:
            #        row.extend([constant.FILL] * (max_len - len(row)))

            inp = combined_vocab.map(inp)
            processed_sent = [inp]
            vbf = combined_vocab.map(vbf)
            #assert len(vbf) == len(inp), f"{len(vbf)}, {len(inp)}"
            processed_sent += [vbf]
            tgt = list(d[2])
            tgt_in = combined_vocab.map([constant.SOS] + tgt)
            tgt_out = combined_vocab.map(tgt + [constant.EOS])
            processed_sent += [tgt_in]
            processed_sent += [tgt_out]
            processed_sent += [edit_type]
            processed.append(processed_sent)
        return processed

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 5

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        src = batch[0]
        src = get_long_tensor(src, batch_size)
        src_mask = torch.eq(src, constant.PAD_ID)
        vbf = batch[1]
        vbf = get_long_tensor(vbf, batch_size)
        vbf_mask = torch.eq(vbf, constant.PAD_ID)
        tgt_in = get_long_tensor(batch[2], batch_size)
        tgt_out = get_long_tensor(batch[3], batch_size)
        edits = torch.LongTensor(batch[4])
        assert tgt_in.size(1) == tgt_out.size(1), "Target input and output sequence sizes do not match."
        return src, src_mask, vbf, vbf_mask, tgt_in, tgt_out, edits, orig_idx