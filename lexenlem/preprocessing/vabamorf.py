from dataclasses import dataclass
from typing import List, Union

from estnltk import Text
from estnltk.taggers import VabamorfTagger

from lexenlem.models.common.seq2seq_model import Seq2SeqModelCombined
from lexenlem.models.common.vabamorf2conll import neural_model_tags
from lexenlem.models.lemma.vocab import MultiVocab

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


# add lemmas?
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


def basic_preprocessing(
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
            lemmatizer: object,
            use_pos: bool,
            use_feats: bool,
            convert_to_conll: bool,
    ):
        self.model = model
        self.vocab = vocab
        self.lemmatizer = lemmatizer
        self.use_pos = use_pos
        self.use_feats = use_feats
        self.convert_to_conll = convert_to_conll

    def lemmatize(self, input_str: str) -> List[str]:
        # token, pos, feats
        preprocessed: List[Union[VabamorfAnalysis, VabamorfAnalysisConll]] = basic_preprocessing(
            input_str, convert_to_conll=self.convert_to_conll
        )

        # lemma_candidates: List[List[str]] = ...  # [[lemma1, lemma2, ...], [...], ...]
        # lemma_candidates: List[str] = ["".join(lemma_list) for lemma_list in lemma_candidates]
        # batch = []
        # for (surface_form, pos, feats), lemma_candidate in zip(preprocessed, lemma_candidates):
        #     ...
        # output_seqs, _, _ = self.model.predict_greedy(src=..., src_mask=..., lem=..., lem_mask=..., log_attn=False)
        # return output_seqs