from dataclasses import dataclass
from typing import List, Union

from estnltk import Text
from estnltk.taggers import VabamorfTagger
from tqdm.auto import tqdm

import lexenlem.models.common.seq2seq_constant as constant
from lexenlem.models.common.seq2seq_model import Seq2SeqModelCombined
from lexenlem.models.common.vabamorf2conll import neural_model_tags
from lexenlem.models.lemma import edit
from lexenlem.models.lemma.vocab import MultiVocab, Vocab

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
    ):
        self.model = model
        self.vocab: Vocab = vocab["combined"]
        self.use_pos = use_pos
        self.use_feats = use_feats
        self.eos_after = eos_after
        self.convert_to_conll = convert_to_conll
        self.skip_lemma = skip_lemma

    def prepare_batch(
            self, preprocessed_input: List[Union[VabamorfAnalysis, VabamorfAnalysisConll]]
    ) -> List[AdHocInput]:

        batch = []

        for element in tqdm(preprocessed_input, desc="Preparing raw batch..."):

            if isinstance(element, VabamorfAnalysis):
                raise NotImplementedError("Vabamorf output format not supported at the moment.")

            edit_type: int = edit.EDIT_TO_ID[edit.get_edit_type(word=element.token, lemma=element.lemma)]

            surface_form: List[str] = list(element.token)
            if self.eos_after:
                surface_form = [constant.SOS] + surface_form
            else:
                surface_form = [constant.SOS] + surface_form + [constant.EOS]

            part_of_speech: List[str] = [f"POS={element.part_of_speech}"]

            feats: List[str] = element.feats.split("|")

            src_input = []
            src_input += surface_form
            if self.use_pos:
                src_input += part_of_speech
            if self.use_feats:
                src_input += feats
            if self.eos_after:
                src_input += [constant.EOS]

            if self.skip_lemma:
                lemma_input = [constant.SOS, constant.EOS]
            else:
                # TODO: account for a list of lemmas
                lemma_input = [f"{constant.SOS}{''.join(list(element.lemma))}{constant.EOS}"]

            input_element = AdHocInput(
                src_input=self.vocab.map(src_input), lemma_input=self.vocab.map(lemma_input), edit_type=edit_type
            )

            batch.append(input_element)
        return batch

    def process_batch(self, raw_batch: List[AdHocInput]):
        batch_size = len(raw_batch)
        return None

    def lemmatize(self, input_str: str) -> List[str]:
        preprocessed: List[Union[VabamorfAnalysis, VabamorfAnalysisConll]] = basic_vb_preprocessing(
            input_str, convert_to_conll=self.convert_to_conll
        )

        raw_batch = self.prepare_batch(preprocessed)
        pt_batch = self.process_batch(raw_batch)

        # lemma_candidates: List[List[str]] = ...  # [[lemma1, lemma2, ...], [...], ...]
        # lemma_candidates: List[str] = ["".join(lemma_list) for lemma_list in lemma_candidates]
        # batch = []
        # for (surface_form, pos, feats), lemma_candidate in zip(preprocessed, lemma_candidates):
        #     ...
        # output_seqs, _, _ = self.model.predict_greedy(src=..., src_mask=..., lem=..., lem_mask=..., log_attn=False)
        # return output_seqs
