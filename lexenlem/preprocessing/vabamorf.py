from dataclasses import dataclass
from typing import List, Union

import torch
from estnltk import Text
from estnltk.taggers import VabamorfTagger
from tqdm.auto import tqdm

import lexenlem.models.common.seq2seq_constant as constant
from lexenlem.models.common.seq2seq_model import Seq2SeqModelCombined
from lexenlem.models.common.utils import prune_decoded_seqs, unsort
from lexenlem.models.common.vabamorf2conll import neural_model_tags
from lexenlem.models.lemma import edit
from lexenlem.models.lemma.vocab import MultiVocab, Vocab
from lexenlem.models.common.data import get_long_tensor, sort_all
from lexenlem.preprocessing.vabamorf_pipeline import VbPipeline, VbTokenAnalysis


@dataclass
class AdHocInput:
    src_input: List[int]  # contains form itself + concatenated pos and features
    lemma_input: List[int]
    edit_type: int


@dataclass
class AdHocModelInput:
    src: torch.tensor
    src_mask: torch.tensor
    lem: torch.tensor
    lem_mask: torch.tensor
    orig_idx: List[int]

    def cuda(self):
        self.src.cuda()
        self.src_mask.cuda()
        self.lem.cuda()
        self.lem_mask.cuda()

    def cpu(self):
        self.src.cpu()
        self.src_mask.cpu()
        self.lem.cpu()
        self.lem_mask.cpu()


def prepare_batch(
        preprocessed_input: List[VbTokenAnalysis],
        eos_after: bool,
        use_pos: bool,
        use_feats: bool,
        skip_lemma: bool,
        vocab: Vocab,
) -> List[AdHocInput]:
    batch = []

    for element in preprocessed_input:

        # edit_type: int = edit.EDIT_TO_ID[edit.get_edit_type(word=element.token, lemma=element.lemma)]
        edit_type: int = edit.EDIT_TO_ID["none"]

        surface_form: List[str] = list(element.token)
        if eos_after:
            surface_form = [constant.SOS] + surface_form
        else:
            surface_form = [constant.SOS] + surface_form + [constant.EOS]

        part_of_speech: List[str] = [element.part_of_speech]

        feats: List[str] = [element.features]

        src_input = []
        src_input += surface_form
        if use_pos:
            src_input += part_of_speech
        if use_feats:
            src_input += feats
        if eos_after:
            src_input += [constant.EOS]

        if skip_lemma:
            lemma_input = [constant.SOS, constant.EOS]
        else:
            lemma_input = [constant.SOS, *list(element.processed_lemma_candidates), constant.EOS]

        input_element = AdHocInput(
            src_input=vocab.map(src_input), lemma_input=vocab.map(lemma_input), edit_type=edit_type
        )

        batch.append(input_element)
    return batch


def process_batch(raw_batch: List[AdHocInput]) -> AdHocModelInput:
    raw_batch = [[el.src_input, el.lemma_input] for el in raw_batch]

    batch_size = len(raw_batch)
    batch = list(zip(*raw_batch))
    assert len(batch) == 2

    lens = [len(x) for x in batch[0]]
    batch, orig_idx = sort_all(batch, lens)

    src = batch[0]
    src = get_long_tensor(src, batch_size)
    src_mask = torch.eq(src, constant.PAD_ID)

    lem = batch[1]
    lem = get_long_tensor(lem, batch_size)
    lem_mask = torch.eq(lem, constant.PAD_ID)

    return AdHocModelInput(src, src_mask, lem, lem_mask, orig_idx)


class VabamorfAdHocProcessor:

    def __init__(
            self,
            path: str,
            use_feats: bool = True,
            convert_to_conllu: bool = True,
            skip_lemma: bool = False,
            use_cuda: bool = False,
            output_compound_separator: bool = False,
            use_upos: bool = True,
            use_dict: bool = True,
    ):
        self.config = None
        self.word_dict = None
        self.composite_dict = None
        self.vocab = None
        self.lexicon = None
        self.output_compound_separator = output_compound_separator
        self.use_feats = use_feats
        self.skip_lemma = skip_lemma
        self.use_upos = use_upos
        self.use_dict = use_dict

        self.convert_to_conllu = convert_to_conllu

        self.use_cuda = use_cuda

        if self.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._init_model(path)

        self.use_pos = self.config["use_pos"]

        self.eos_after = self.config["eos_after"]

        self.analyzer = VbPipeline()

    def _init_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.word_dict, self.composite_dict = checkpoint['dicts']
        self.vocab: Vocab = MultiVocab.load_state_dict(checkpoint['vocab'])["combined"]
        self.lexicon = checkpoint['lexicon']
        if self.config['dict_only']:
            raise NotImplementedError
        self.model = Seq2SeqModelCombined(self.config, self.vocab, use_cuda=self.use_cuda)
        self.model.eval()
        self.model.load_state_dict(checkpoint['model'])

    def preprocess_text(self, raw_text: str) -> List[VbTokenAnalysis]:
        return self.analyzer(raw_text)[0]

    def lemmatize_dict(
            self, preprocessed: List[VbTokenAnalysis]
    ) -> List[Union[str, None]]:
        if self.use_pos:
            return [self.composite_dict.get((el.token, el.part_of_speech), None) for el in preprocessed]
        else:
            return [self.word_dict.get((el.token, el.part_of_speech), None) for el in preprocessed]

    def lemmatize(self, input_str: str):
        preprocessed: List[VbTokenAnalysis] = self.preprocess_text(input_str)

        raw_batch = prepare_batch(
            preprocessed,
            use_feats=self.use_feats,
            use_pos=self.use_pos,
            skip_lemma=self.skip_lemma,
            eos_after=self.eos_after,
            vocab=self.vocab
        )

        pt_batch = process_batch(raw_batch)

        if self.use_cuda:
            pt_batch.cuda()

        output_seqs, _, _ = self.model.predict_greedy(
            src=pt_batch.src, src_mask=pt_batch.src_mask, lem=pt_batch.lem, lem_mask=pt_batch.lem_mask, log_attn=False
        )

        output_seqs = [self.vocab.unmap(ids) for ids in output_seqs]

        output_seqs = prune_decoded_seqs(output_seqs)
        output_seqs = ["".join(seq) for seq in output_seqs]
        output_seqs = unsort(output_seqs, pt_batch.orig_idx)

        if self.use_dict and self.use_pos:
            tmp = []
            d_lemmatized = self.lemmatize_dict(preprocessed)
            for d_lemma, hypothesis in zip(d_lemmatized, output_seqs):
                if not d_lemma:
                    tmp.append(hypothesis)
                else:
                    tmp.append(d_lemma)
            output_seqs = tmp

        return output_seqs
