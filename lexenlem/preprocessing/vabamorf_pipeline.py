from dataclasses import dataclass
from typing import List, Union, Dict

from estnltk import Text
from estnltk.taggers import VabamorfTagger

from lexenlem.models.common.vabamorf2conll import neural_model_tags

DEGREE_MAP = {
    "pos": "Pos",
    "comp": "Cmp",
    "super": "Sup",
}

CASE_MAP = {
    "abes": "Abe",
    "abl": "Abl",
    "adit": "Add",
    "ad": "Ade",
    "all": "All",
    "el": "Ela",
    "es": "Ess",
    "gen": "Gen",
    "ill": "Ill",
    "in": "Ine",
    "nom": "Nom",
    "part": "Par",
    "term": "Ter",
    "tr": "Tra",
    "kom": "Com",
}

PERSON_MAP = {
    "ps1": "1",
    "ps2": "2",
    "ps3": "3",
}

NUMBER_MAP = {
    "pl": "Plur",
    "sg": "Sing",
}

TENSE_MAP = {
    "impf": "Past",  # converb?
    "past": "Past",
    "pres": "Pres",
}

MOOD_MAP = {
    "cond": "Cnd",
    "imper": "Imp",
    "indic": "Ind",
    "quot": "Qot",
}

NUM_TYPE_MAP = {
    "card": "Card",
    "ord": "Ord",
}

VERB_FORM_MAP = {
    "ger": "Conv",  # ??
    "inf": "Inf",
    "partic": "Part",
    "sup": "Sup",
    # : "Fin" ??
}

SHARED_FEATS = {
    "VerbForm": VERB_FORM_MAP,
    "NumType": NUM_TYPE_MAP,
    "Mood": MOOD_MAP,
    "Case": CASE_MAP,
    "Number": NUMBER_MAP,
    "Person": PERSON_MAP,
    "Degree": DEGREE_MAP,
    "Tense": TENSE_MAP,
}


@dataclass
class VbTokenAnalysis:
    index: int  # 1-based numbering to avoid confusion with conll
    token: str
    disambiguated_lemma: str
    lemma_candidates: List[str]
    part_of_speech: str  # disambiguated only
    features: str  # disambiguated only

    @property
    def processed_lemma_candidates(self) -> str:
        return "".join(sorted(list(set(self.lemma_candidates))))


@dataclass
class VbTokenAnalysisConll(VbTokenAnalysis):
    conll_feature_candidates: List[Union[Dict[str, str], None]]


class VbPipeline:

    def __init__(self):
        self.amb_tagger = VabamorfTagger(compound=True, disambiguate=False, guess=False)
        self.disamb_tagger = VabamorfTagger(compound=True, disambiguate=True, guess=True)

    def __call__(self, data: Union[str, List[str]]) -> List[List[VbTokenAnalysis]]:
        if isinstance(data, str):
            return [self.analyze(data)]
        elif isinstance(data, list):
            return [self.analyze(el) for el in data]
        else:
            raise NotImplementedError

    def analyze(self, raw_text: str) -> List[VbTokenAnalysis]:

        if not isinstance(raw_text, str):
            raise ValueError("Input text should be of type `str`")

        ambiguous_analysis = self.ambiguous_analysis(raw_text)
        disambiguated_analysis = self.disambiguated_analysis(raw_text)

        # number of tokens must be the same in both analyses
        if len(disambiguated_analysis["morph_analysis"]) != len(ambiguous_analysis["morph_analysis"]):
            raise ValueError("Number of tokens mismatch")

        result = []
        tokens = disambiguated_analysis["morph_analysis"].text
        disambiguated_lemmas = disambiguated_analysis["morph_analysis"].root
        lemma_candidate_list = ambiguous_analysis["morph_analysis"].root
        features_list = disambiguated_analysis["morph_analysis"].form
        pos_list = disambiguated_analysis["morph_analysis"].partofspeech

        for index, (token, disambiguated_lemma, lemma_candidates, features, part_of_speech) in enumerate(
                zip(tokens, disambiguated_lemmas, lemma_candidate_list, features_list, pos_list)
        ):
            result.append(
                VbTokenAnalysis(
                    index=index + 1,  # 1-based
                    token=token,
                    disambiguated_lemma=disambiguated_lemma[0],
                    lemma_candidates=lemma_candidates,
                    features=features[0],
                    part_of_speech=part_of_speech[0],
                )
            )
        return result

    def ambiguous_analysis(self, raw_text: str) -> Text:
        text = Text(raw_text)
        text.tag_layer(self.amb_tagger.input_layers)
        self.amb_tagger.tag(text)
        return text

    def disambiguated_analysis(self, raw_text: str) -> Text:
        text = Text(raw_text)
        text.tag_layer(self.disamb_tagger.input_layers)
        self.disamb_tagger.tag(text)
        return text


def remove_pos_from_feats(feats: str) -> str:
    if "POS=" not in feats:
        return feats
    feats = feats.split("|")
    return "|".join([feat for feat in feats if "POS=" not in feat])


def convert_vb_to_conll(vb_analysis: VbTokenAnalysis) -> VbTokenAnalysisConll:
    # conversion from vb to conll may be ambiguous
    candidates: List[str] = neural_model_tags(vb_analysis.token, vb_analysis.part_of_speech, vb_analysis.features)
    return VbTokenAnalysisConll(
            index=vb_analysis.index,
            token=vb_analysis.token,
            disambiguated_lemma=vb_analysis.disambiguated_lemma,
            lemma_candidates=vb_analysis.lemma_candidates,
            part_of_speech=vb_analysis.part_of_speech,
            features=vb_analysis.features,
            conll_feature_candidates=[
                get_conll_shared_feats(remove_pos_from_feats(candidate))
                for candidate in candidates
            ],

        )


def str_tags_to_dict(str_tags: str) -> Union[Dict[str, str], None]:
    if str_tags:
        individual_features: List[str] = str_tags.split("|")
        features_dict = dict()
        for feat in individual_features:
            key, value = feat.split("=")
            features_dict[to_camel_case(key)] = value
        return features_dict
    else:
        return None


def to_camel_case(name: str) -> str:
    name = name.split("_")
    name = [el.capitalize() for el in name]
    return "".join(name)


def get_conll_shared_feats(hypothesis: str) -> Dict[str, str]:
    feats = str_tags_to_dict(hypothesis)
    if feats:
        shared_feats = dict()
        for key, value in feats.items():
            if key in SHARED_FEATS:
                shared_feats[key] = SHARED_FEATS[key][value]
        return shared_feats
