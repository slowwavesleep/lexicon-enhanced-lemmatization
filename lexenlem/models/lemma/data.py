import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import dataclasses
from itertools import chain
from typing import List, Union, Tuple, Optional, Dict

import torch
from tqdm.auto import tqdm
from conllu import parse
from conllu.models import TokenList

import lexenlem.models.common.seq2seq_constant as constant
from lexenlem.models.common.data import get_long_tensor, sort_all
from lexenlem.models.common import conll
from lexenlem.models.lemma.vocab import Vocab, MultiVocab
from lexenlem.models.lemma import edit
from lexenlem.models.common.doc import Document
from lexenlem.models.common.lexicon import Lexicon, ExtendedLexicon
from lexenlem.preprocessing.vabamorf_pipeline import VbPipeline, VbTokenAnalysis, AdHocInput, AdHocModelInput


def make_feats_data(data: List[List[str]], feats_idx: int = 3) -> List[str]:
    feats_data = []
    for d in data:
        feats = d[feats_idx]
        if '|' in feats:
            feats_data.extend(feats.split('|'))
        else:
            feats_data.append(feats)
    # print(all([isinstance(feat, str) for feat in feats_data]))
    # raise Exception
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
        self.config = args
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
        combined_vocab = Vocab(combined_data, self.config['lang'])
        return combined_vocab

    def preprocess(self, data: List[List[str]], combined_vocab, args):
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
            inp: List[str] = src
            if self.pos:
                inp += pos
            if self.morph:
                inp += feats
            if eos_after:
                inp += [constant.EOS]
            inp: List[int] = combined_vocab.map(inp)
            processed_sent: List[List[int]] = [inp]
            if self.lemmatizer is None:
                lem = [constant.SOS, constant.EOS]
            else:
                # expected to return list of individual characters
                if type(self.lemmatizer) in [Lexicon, ExtendedLexicon]:
                    lem = self.lemmatizer.lemmatize(d[0], d[1])
                elif args['lemmatizer'] == 'apertium':
                    lem = self.lemmatizer.lemmatize(d[0], args['lang'].split('_')[0])
                else:
                    lem: List[str] = self.lemmatizer.lemmatize(d[0])  # original code
                    lem: List[str] = list("".join(list(dict.fromkeys(lem))))  # list of chars
                lem = [constant.SOS] + lem + [constant.EOS]
            lem: List[int] = combined_vocab.map(lem)
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

    def __getitem__(self, key: int) -> Tuple:
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


@dataclass(frozen=True)
class DataLoaderVbConfig:
    morph: bool = True
    pos: bool = True
    sample_train: float = 1.
    lang: str = "et"
    eos_after: bool = False
    split_feats: bool = False


class DataLoaderVb:

    def __init__(
            self,
            input_src: str,
            batch_size: int,
            config: Optional[DataLoaderVbConfig] = None,
            vocab: Optional[MultiVocab] = None,
            evaluation: bool = False,
            use_vb_lemmas: bool = True,
    ):
        self.batch_size = batch_size
        self.config = config
        self.eval = evaluation
        self.shuffled = not self.eval
        self.use_vb_lemmas = use_vb_lemmas

        if not config:
            self.config = DataLoaderVbConfig()

        self.analyzer = VbPipeline(
            use_context=True,
            use_proper_name_analysis=True,
            output_compound_separator=False,
            guess_unknown_words=True,
            output_phonetic_info=False,
            ignore_derivation_symbol=True,
        )
        self.morph = self.config.morph
        self.pos = self.config.pos
        print("Using Vabamorf morphological features:", self.morph)
        print("Using Vabamorf determined parts of speech:", self.pos)

        if isinstance(input_src, str):
            assert input_src.endswith("conllu"), "Loaded file must be conllu file."
            with open(input_src) as file:
                raw_data: str = file.read()
            self._parsed_data: Dict[str, TokenList] = {
                token_list.metadata["sent_id"]: token_list for token_list in parse(raw_data)
            }
        else:
            raise TypeError("Incorrect input format.")

        # filter and sample data
        if self.config.sample_train < 1.0 and not self.eval:
            keys = self._parsed_data.keys()
            keep = int(self.config.sample_train * len(keys))
            keys = random.sample(keys, keep)
            self._parsed_data = {key: value for key, value in self._parsed_data.items() if key in keys}
            print("Subsample training set with rate {:g}".format(self.config.sample_train))

        self._analyzed_data: Dict[str, List[VbTokenAnalysis]] = self._analyze(list(self._parsed_data.values()))
        # handle vocab
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = dict()
            combined_vocab = self._init_vocab(self._flat_analysis)
            self.vocab = MultiVocab({"combined": combined_vocab})

        # keys: 'id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']
        data: List[AdHocInput] = self._preprocess(self._flat_analysis, self.vocab["combined"])
        # shuffle for training
        if self.shuffled:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]

        self.num_examples = len(data)

        # chunk into batches
        data: List[List[AdHocInput]] = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def _analyze(self, data: List[TokenList]) -> Dict[str, List[VbTokenAnalysis]]:
        result: Dict[str, List[VbTokenAnalysis]] = defaultdict(lambda: [])
        for sentence in tqdm(data, desc="Vabamorf analyzing..."):
            tokens: List[str] = [token["form"] for token in sentence]
            true_lemmas: List[str] = [token["lemma"] for token in sentence]
            sent_id: str = sentence.metadata.get("sent_id")
            analyzed: List[VbTokenAnalysis] = self.analyzer.analyze(tokens)
            for token_analysis, true_lemma in zip(analyzed, true_lemmas):
                token_analysis.true_lemma = true_lemma
                token_analysis.sent_id = sent_id
                result[sent_id].append(token_analysis)
        if len(result) != len(data):
            raise RuntimeError("Number of sentences doesn't match")
        return result

    @property
    def _flat_analysis(self) -> List[VbTokenAnalysis]:
        return list(chain(*self._analyzed_data.values()))

    @property
    def original_tokens(self) -> List[str]:
        if self.eval:
            return [token.token for token in self._flat_analysis]
        else:
            raise RuntimeError("Not available in eval mode")

    def _process_predictions(self, predictions: List[str]):
        result: List[VbTokenAnalysis] = []
        for token_analysis, predicted_lemma in zip(self._flat_analysis, predictions):
            result.append(dataclasses.replace(token_analysis, predicted_lemma=predicted_lemma))
        return result

    def _pred_to_conll(self, predictions: List[str]) -> Dict[str, TokenList]:
        processed = self._process_predictions(predictions)
        conll_data: Dict[str, TokenList] = deepcopy(self._parsed_data)
        for token in processed:
            sent_id = token.sent_id
            token_index = token.index - 1
            conll_data[sent_id][token_index]["lemma"] = token.predicted_lemma
        return conll_data

    def write_to_conll(self, predictions: List[str], path: str):
        conll_data = self._pred_to_conll(predictions)
        with open(path, "w") as file:
            for sent in conll_data.values():
                file.write(sent.serialize())

    def _init_vocab(self, data: List[VbTokenAnalysis]) -> Vocab:
        assert self.eval is False, "Vocab file must exist for evaluation"
        char_data: List[str] = []
        pos_data: List[str] = []
        feats_data: List[str] = []
        for token in data:
            char_data.append(f"{token.token}{token.true_lemma}")
            pos_data.append(f"POS={token.part_of_speech}")
            if self.config.split_feats:
                feats_data.extend(token.features.split(" "))
            else:
                feats_data.append(token.features)
        char_data = list("".join(char_data))
        combined_data = char_data + pos_data + feats_data
        combined_vocab = Vocab(combined_data, self.config.lang)
        return combined_vocab

    def _preprocess(self, data: List[VbTokenAnalysis], combined_vocab) -> List[AdHocInput]:
        processed = []
        eos_after = self.config.eos_after
        for element in data:
            surface_form: List[str] = list(element.token)
            if eos_after:
                surface_form = [constant.SOS] + surface_form
            else:
                surface_form = [constant.SOS] + surface_form + [constant.EOS]
            part_of_speech = ['POS=' + element.part_of_speech]
            if self.config.split_feats:
                raise NotImplementedError
            else:
                feats = [element.features]
            if self.pos:
                surface_form += part_of_speech
            if self.morph:
                surface_form += feats
            if eos_after:
                surface_form += [constant.EOS]
            surface_form: List[int] = combined_vocab.map(surface_form)
            if not self.use_vb_lemmas:
                lemma_input: List[str] = [constant.SOS, constant.EOS]
            else:
                lemma_input: str = element.processed_lemma_candidates
                lemma_input: List[str] = list(lemma_input)  # list of chars
                lemma_input: List[str] = [constant.SOS] + lemma_input + [constant.EOS]
            lemma_input: List[int] = combined_vocab.map(lemma_input)
            target: List[str] = list(element.true_lemma)
            target_in: List[int] = combined_vocab.map([constant.SOS] + target)
            target_out: List[int] = combined_vocab.map(target + [constant.EOS])

            input_element = AdHocInput(
                src_input=surface_form,
                lemma_input=lemma_input,
                target_in=target_in,
                target_out=target_out,
            )
            processed.append(input_element)
        return processed

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: int) -> AdHocModelInput:
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise KeyError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch: List[AdHocInput] = self.data[key]
        batch_size: int = len(batch)
        # assert len(batch) == self.batch_size
        tmp_batch: List[Tuple[List[int], ...]] = []
        for element in batch:
            if element.target_in is not None and element.target_out is not None:
                tmp_batch.append(
                    (element.src_input, element.lemma_input, element.target_in, element.target_out)
                )
        batch: List[Tuple[List[int], ...]] = tmp_batch
        batch = list(zip(*batch))
        assert len(batch) == 4

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
        assert tgt_in.size(1) == tgt_out.size(1), "Target input and output sequence sizes do not match."
        return AdHocModelInput(
            src=src,
            src_mask=src_mask,
            lem=lem,
            lem_mask=lem_mask,
            tgt_in=tgt_in,
            tgt_out=tgt_out,
            orig_idx=orig_idx
        )

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
