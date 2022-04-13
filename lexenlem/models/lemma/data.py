import random
from typing import List, Union, Tuple

import torch
from tqdm.auto import tqdm

import lexenlem.models.common.seq2seq_constant as constant
from lexenlem.models.common.data import get_long_tensor, sort_all
from lexenlem.models.common import conll
from lexenlem.models.common.seq2seq_model import Seq2SeqModelCombined
from lexenlem.models.lemma.vocab import Vocab, MultiVocab
from lexenlem.models.lemma import edit
from lexenlem.models.common.doc import Document
from lexenlem.models.common.lexicon import Lexicon, ExtendedLexicon
from lexenlem.preprocessing.vabamorf import basic_preprocessing


def make_feats_data(data: List[List[str]], feats_idx: int = 3) -> List[str, List[str]]:
    feats_data = []
    for d in data:
        feats = d[feats_idx]
        if '|' in feats:
            feats_data.extend(feats.split('|'))
        else:
            feats_data.append(feats)
    return feats_data


def load_file(filename: str) -> Tuple[conll.CoNLLFile, List[List[str]]]:
    conll_file = conll.CoNLLFile(filename)
    data = conll_file.get(['word', 'upos', 'lemma', 'feats'])
    return conll_file, data


def load_doc(doc: Document) -> Tuple[conll.CoNLLFile, List[List[str]]]:
    data = doc.conll_file.get(['word', 'upos', 'lemma', 'feats'])
    return doc.conll_file, data


class DataLoaderCombined:
    def __init__(
            self,
            input_src: Union[str, Document],
            batch_size: int,
            args: dict,
            lemmatizer: Union[str, object] = None,
            vocab: MultiVocab = None,
            evaluation: bool = False,
            conll_only: bool = False,
            skip: List[bool] = None,
    ):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval

        self.lemmatizer = lemmatizer
        self.morph = args.get('morph', True)
        self.pos = args.get('pos', True)
        print('Using FEATS:', self.morph)
        print('Using POS:', self.pos)

        # check if input source is a file or a Document object
        if isinstance(input_src, str):
            filename = input_src
            assert filename.endswith('conllu'), "Loaded file must be conllu file."
            self.conll, data = load_file(filename)
        elif isinstance(input_src, Document):
            doc = input_src
            self.conll, data = load_doc(doc)
        else:
            raise TypeError("Incorrect input format.")

        if conll_only:  # only load conll file
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

        if lemmatizer == 'lexicon':
            print("Building the lexicon...")
            self.lemmatizer = Lexicon(
                unimorph=args['unimorph_dir'],
                use_pos=args.get('use_pos', False),
                use_word=args.get('use_word', False)
            )
            self.lemmatizer.init_lexicon(data)

        data = self.preprocess(data, self.vocab['combined'], args)
        # shuffle for training
        if self.shuffled:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def init_vocab(self, data: List[List[str]]) -> Vocab:
        assert self.eval is False, "Vocab file must exist for evaluation"
        char_data = list("".join(d[0] + d[2] for d in data))
        pos_data = ['POS=' + d[1] for d in data]
        feats_data = make_feats_data(data)
        combined_data = char_data + pos_data + feats_data
        combined_vocab = Vocab(combined_data, self.args['lang'])
        return combined_vocab

    def preprocess(self, data, combined_vocab, args) -> List[List[List[int], int]]:
        processed = []
        eos_after = args.get('eos_after', False)
        for d in tqdm(data, desc="Preprocessing data..."):
            edit_type = edit.EDIT_TO_ID[edit.get_edit_type(d[0], d[2])]
            src = list(d[0])
            if eos_after:
                src = [constant.SOS] + src
            else:
                src = [constant.SOS] + src + [constant.EOS]
            pos = ['POS=' + d[1]]
            feats = []
            if '|' in d[3]:
                feats.extend(d[3].split('|'))
            else:
                feats.append(d[3])
            inp = src
            if self.pos:
                inp += pos
            if self.morph:
                inp += feats
            if eos_after:
                inp += [constant.EOS]
            inp = combined_vocab.map(inp)
            processed_sent = [inp]
            if self.lemmatizer is None:
                lem = [constant.SOS, constant.EOS]
            else:
                if type(self.lemmatizer) in [Lexicon, ExtendedLexicon]:
                    lem = self.lemmatizer.lemmatize(d[0], d[1])
                elif args['lemmatizer'] == 'apertium':
                    lem = self.lemmatizer.lemmatize(d[0], args['lang'].split('_')[0])
                else:
                    lem = self.lemmatizer.lemmatize(d[0])
                lem = [constant.SOS] + lem + [constant.EOS]
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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key) -> Tuple:
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


class AdHocProcessor:

    def __init__(
            self,
            model: Seq2SeqModelCombined,
            vocab: MultiVocab,
            lemmatizer: object,
            use_pos: bool,
            use_feats: bool,
    ):
        self.model = model
        self.vocab = vocab
        self.lemmatizer = lemmatizer
        self.use_pos = use_pos
        self.use_feats = use_feats

    def lemmatize(self, input_str: str) -> List[str]:
        preprocessed = basic_preprocessing(input_str)  # token, pos, feats
        lemma_candidates: List[List[str]] = ...  # [[lemma1, lemma2, ...], [...], ...]
        lemma_candidates: List[str] = ["".join(lemma_list) for lemma_list in lemma_candidates]
        batch = []
        for (surface_form, pos, feats), lemma_candidate in zip(preprocessed, lemma_candidates):
            ...
        output_seqs, _, _ = self.model.predict_greedy(src=..., src_mask=..., lem=..., lem_mask=..., log_attn=False)
        return output_seqs


