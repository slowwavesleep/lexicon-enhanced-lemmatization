import random
import numpy as np
import os
from collections import Counter
import torch
import sys

import lexenlem.models.common.seq2seq_constant as constant
from lexenlem.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from lexenlem.models.common import conll
from lexenlem.models.lemma.vocab import Vocab, MultiVocab, FeatureVocab
from lexenlem.models.lemma import edit
from lexenlem.models.common.doc import Document
import json

class DataLoaderCombined:
    def __init__(self, input_src, batch_size, args, lemmatizer=None, vocab=None, evaluation=False, conll_only=False, skip=None):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.lemmatizer = lemmatizer

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
            if self.lemmatizer is None:
                lem = ['']
            else:
                lem = self.lemmatizer.lemmatize(d[0])
                lem = ''.join([''.join(list(l.replace('|', ''))) for l in lem])
                lem = list(lem)
                lem = [constant.SOS] + lem + [constant.EOS]
            src = [constant.SOS] + src + [constant.EOS]
            pos = [d[1]]
            feats = []
            if '|' in d[3]:
                feats.extend(d[3].split('|'))
            else:
                feats.append(d[3])
            inp = src + pos + feats
            inp = combined_vocab.map(inp)
            processed_sent = [inp]
            lem = combined_vocab.map(lem)
            processed_sent += [lem]
            tgt = list(d[2])
            tgt_in = combined_vocab.map([constant.SOS] + tgt)
            tgt_out = combined_vocab.map(tgt + [constant.EOS])
            processed_sent += [tgt_in]
            processed_sent += [tgt_out]
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
        assert len(batch) == 5

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        src = batch[0]
        src = get_long_tensor(src, batch_size)
        src_mask = torch.eq(src, constant.PAD_ID)
        lem = batch[1]
        lem = get_long_tensor(lem, batch_size)
        lem_mask = torch.eq(lem, constant.PAD_ID)
        tgt_in = get_long_tensor(batch[2], batch_size)
        tgt_out = get_long_tensor(batch[3], batch_size)
        edits = torch.LongTensor(batch[4])
        assert tgt_in.size(1) == tgt_out.size(1), "Target input and output sequence sizes do not match."
        return src, src_mask, lem, lem_mask, tgt_in, tgt_out, edits, orig_idx

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