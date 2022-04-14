from dataclasses import dataclass
from typing import List, Union

import torch
from estnltk import Text
from estnltk.taggers import VabamorfTagger
from tqdm.auto import tqdm

import lexenlem.models.common.seq2seq_constant as constant
from lexenlem.models.common.seq2seq_model import Seq2SeqModelCombined
from lexenlem.models.common.vabamorf2conll import neural_model_tags
from lexenlem.models.lemma import edit
from lexenlem.models.lemma.data import DataLoaderCombined
from lexenlem.models.lemma.vocab import MultiVocab, Vocab
from lexenlem.models.common.data import get_long_tensor, sort_all

tagger = VabamorfTagger(compound=True, disambiguate=False, guess=False)


@dataclass
class BaseAnalysis:
    token: str
    lemma: str
    part_of_speech: str


@dataclass
class VabamorfAnalysis(BaseAnalysis):
    feats: str


@dataclass
class VabamorfAnalysisConll(BaseAnalysis):
    feats: str


@dataclass
class AdHocInput:
    src_input: List[int]
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


def get_vabamorf_analysis(token: str) -> List[VabamorfAnalysis]:
    text = Text(token)
    text.tag_layer(tagger.input_layers)
    tagger.tag(text)
    lemmas = text["morph_analysis"][0].root
    forms = text["morph_analysis"][0].form
    parts_of_speech = text["morph_analysis"][0].partofspeech
    return [
        VabamorfAnalysis(token=token, lemma=lemma, part_of_speech=part_of_speech, feats=form)
        for lemma, part_of_speech, form in zip(lemmas, parts_of_speech, forms)
    ]


def convert_vb_to_conll(vb_analysis: VabamorfAnalysis) -> List[VabamorfAnalysisConll]:
    # conversion from vb to conll may be ambiguous
    candidates: List[str] = neural_model_tags(vb_analysis.token, vb_analysis.part_of_speech, vb_analysis.feats)
    return [
        VabamorfAnalysisConll(
            token=vb_analysis.token,
            lemma=vb_analysis.lemma,
            part_of_speech=vb_analysis.part_of_speech,
            feats=feats_candidate,
        )
        for feats_candidate in candidates
    ]


def tokenize(raw_text: str) -> List[str]:
    text = Text(raw_text)
    text.tag_layer("tokens")
    tokenized = []
    for token in text.tokens:
        tokenized.append(token.text)
    return tokenized


def basic_vb_preprocessing(
        raw_text: str, convert_to_conll: bool = True
) -> List[Union[VabamorfAnalysis, VabamorfAnalysisConll]]:
    tokenized = tokenize(raw_text)
    analyzed = []
    for token in tokenized:
        token_analysis = get_vabamorf_analysis(token)[0]
        if convert_to_conll:
            token_analysis = convert_vb_to_conll(token_analysis)[0]
        analyzed.append(token_analysis)
    return analyzed


def prepare_batch(
        preprocessed_input: List[Union[VabamorfAnalysis, VabamorfAnalysisConll]],
        eos_after: bool,
        use_pos: bool,
        use_feats: bool,
        skip_lemma: bool,
        vocab: Vocab,
) -> List[AdHocInput]:

    batch = []

    for element in tqdm(preprocessed_input, desc="Preparing raw batch..."):

        if isinstance(element, VabamorfAnalysis):
            raise NotImplementedError("Vabamorf output format not supported at the moment.")

        edit_type: int = edit.EDIT_TO_ID[edit.get_edit_type(word=element.token, lemma=element.lemma)]

        surface_form: List[str] = list(element.token)
        if eos_after:
            surface_form = [constant.SOS] + surface_form
        else:
            surface_form = [constant.SOS] + surface_form + [constant.EOS]

        part_of_speech: List[str] = [f"POS={element.part_of_speech}"]

        feats: List[str] = element.feats.split("|")

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
            # TODO: account for a list of lemmas
            lemma_input = [constant.SOS, *list(element.lemma), constant.EOS]

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


def save_vocab(dataloader: DataLoaderCombined, filename: str):
    params = {"vocab": dataloader.vocab.state_dict()}
    torch.save(params, filename)


def load_vocab(filename: str) -> MultiVocab:
    return MultiVocab.load_state_dict(torch.load(filename)["vocab"])


class VabamorfAdHocProcessor:

    def __init__(
            self,
            model: Seq2SeqModelCombined,
            vocab: MultiVocab,
            use_pos: bool = True,
            use_feats: bool = True,
            eos_after: bool = False,
            convert_to_conll: bool = True,
            skip_lemma: bool = False,
            use_cuda: bool = False,
    ):
        self.model = model
        if use_cuda:
            self.model.cuda()
        self.vocab: Vocab = vocab["combined"]
        self.use_pos = use_pos
        self.use_feats = use_feats
        self.eos_after = eos_after
        self.convert_to_conll = convert_to_conll
        self.skip_lemma = skip_lemma
        self.use_cuda = use_cuda

    def lemmatize(self, input_str: str):
        preprocessed: List[Union[VabamorfAnalysis, VabamorfAnalysisConll]] = basic_vb_preprocessing(
            input_str, convert_to_conll=self.convert_to_conll
        )

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

        # self.vocab.id2unit

        # trainer.predict

        return output_seqs
