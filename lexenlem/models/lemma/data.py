import random
import torch
from torch.utils.data import Dataset

import lexenlem.models.common.seq2seq_constant as constant
from lexenlem.models.common import conll
from lexenlem.models.lemma.vocab import Vocab, MultiVocab
from lexenlem.models.common.doc import Document
from lexenlem.models.common.lexicon import Lexicon, ExtendedLexicon
from typing import Union, Optional, List, NamedTuple
import argparse


class DataItem(NamedTuple):
    """Container for the dataset output.

    Attributes:
        src (torch.Tensor): Encoded source sequence.
        lem (torch.Tensor): Encoded candidate sequence.
        tgt_in (torch.Tensor): Encoded output sequence with BOS token.
        tgt_out (torch.Tensor): Encoded output sequence with EOS token.
    """

    src: torch.Tensor
    lem: torch.Tensor
    tgt_in: torch.Tensor
    tgt_out: torch.Tensor


class BatchItem(NamedTuple):
    src: torch.Tensor
    src_mask: torch.Tensor
    lem: torch.Tensor
    lem_mask: torch.Tensor
    tgt_in: torch.Tensor
    tgt_out: torch.Tensor


class CoNNLDataset(Dataset):
    def __init__(
        self,
        input_src: Union[str, Document],
        args: argparse.Namespace,
        lemmatizer: Optional[Union[str, Lexicon, ExtendedLexicon]] = None,
        vocab: Optional[Vocab] = None,
        evaluation: bool = False,
        conll_only: bool = False,
        skip: Optional[List] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.device = device

        self.lemmatizer = lemmatizer
        self.morph = args.get("morph", True)
        self.pos = args.get("pos", True)
        print("Using FEATS:", self.morph)
        print("Using POS:", self.pos)

        # check if input source is a file or a Document object
        if isinstance(input_src, str):
            filename = input_src
            assert filename.endswith("conllu"), "Loaded file must be conllu file."
            self.conll, data = self.load_file(filename)
        elif isinstance(input_src, Document):
            filename = None
            doc = input_src
            self.conll, data = self.load_doc(doc)

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
            self.vocab = MultiVocab({"combined": combined_vocab})

        # filter and sample data
        if args.get("sample_train", 1.0) < 1.0 and not self.eval:
            keep = int(args["sample_train"] * len(data))
            data = random.sample(data, keep)
            print("Subsample training set with rate {:g}".format(args["sample_train"]))

        if lemmatizer == "lexicon":
            print("Building the lexicon...")
            self.lemmatizer = Lexicon(
                unimorph=args["unimorph_dir"],
                use_pos=args.get("use_pos", False),
                use_word=args.get("use_word", False),
            )
            self.lemmatizer.init_lexicon(data)

        data = self.preprocess(data, self.vocab["combined"], args)
        # shuffle for training
        if self.shuffled:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)

        self.data = data

    def make_feats_data(self, data: List[List[str]], feats_idx: int = 3):
        feats_data = []
        for d in data:
            feats = d[feats_idx]
            if "|" in feats:
                feats_data.extend(feats.split("|"))
            else:
                feats_data.append(feats)
        return feats_data

    def init_vocab(self, data):
        assert self.eval is False, "Vocab file must exist for evaluation"
        char_data = list("".join(d[0] + d[2] for d in data))
        pos_data = ["POS=" + d[1] for d in data]
        feats_data = self.make_feats_data(data)
        combined_data = char_data + pos_data + feats_data
        combined_vocab = Vocab(combined_data, self.args["lang"])
        return combined_vocab

    def preprocess(self, data, combined_vocab, args):
        processed = []
        eos_after = args.get("eos_after", False)
        for d in data:
            src = list(d[0])
            if eos_after:
                src = [constant.SOS] + src
            else:
                src = [constant.SOS] + src + [constant.EOS]
            pos = ["POS=" + d[1]]
            feats = []
            if "|" in d[3]:
                feats.extend(d[3].split("|"))
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
                elif args["lemmatizer"] == "apertium":
                    lem = self.lemmatizer.lemmatize(d[0], args["lang"].split("_")[0])
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
            processed.append(processed_sent)
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key: int) -> DataItem:
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        items = self.data[key]

        # convert to tensors
        src = torch.tensor(items[0], dtype=torch.long, device=self.device)
        lem = torch.tensor(items[1], dtype=torch.long, device=self.device)
        tgt_in = torch.tensor(items[2], dtype=torch.long, device=self.device)
        tgt_out = torch.tensor(items[3], dtype=torch.long, device=self.device)
        assert tgt_in.size(0) == tgt_out.size(0), "Target input and output sequence sizes do not match."

        outputs = DataItem(src, lem, tgt_in, tgt_out)
        return outputs

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_file(self, filename: str):
        conll_file = conll.CoNLLFile(filename)
        data = conll_file.get(["word", "upos", "lemma", "feats"])
        return conll_file, data

    def load_doc(self, doc: Document):
        data = doc.conll_file.get(["word", "upos", "lemma", "feats"])
        return doc.conll_file, data
