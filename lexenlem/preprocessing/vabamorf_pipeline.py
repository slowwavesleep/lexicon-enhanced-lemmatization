from dataclasses import dataclass
from typing import List, Union

from estnltk import Text
from estnltk.taggers import VabamorfTagger


@dataclass
class VbTokenAnalysis:
    token: str
    disambiguated_lemma: str
    lemma_candidates: List[str]
    part_of_speech: str  # disambiguated only
    features: str  # disambiguated only

    @property
    def processed_lemma_candidates(self) -> str:
        return "".join(sorted(list(set(self.lemma_candidates))))


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

        for token, disambiguated_lemma, lemma_candidates, features, part_of_speech in zip(
            tokens, disambiguated_lemmas, lemma_candidate_list, features_list, pos_list
        ):
            result.append(
                VbTokenAnalysis(
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

