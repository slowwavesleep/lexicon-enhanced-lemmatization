from dataclasses import dataclass
from typing import List, Union, Dict, Tuple

from estnltk import Text
from estnltk.taggers import VabamorfTagger, WhiteSpaceTokensTagger, PretokenizedTextCompoundTokensTagger

from lexenlem.models.common.vabamorf2conll import neural_model_tags


@dataclass(frozen=True)
class VbTokenAnalysis:
    index: int  # 1-based numbering to avoid confusion with conll
    token: str
    disambiguated_lemma: str
    lemma_candidates: Tuple[str]
    candidate_parts_of_speech: Tuple[str]
    part_of_speech: str  # disambiguated only
    features: str  # disambiguated only

    @property
    def processed_lemma_candidates(self) -> str:
        return "".join(sorted(list(set(self.lemma_candidates))))


@dataclass(frozen=True)
class VbTokenAnalysisConll(VbTokenAnalysis):
    conll_feature_candidates: List[Union[Dict[str, str], None]]


class VbPipeline:

    def __init__(
            self,
            use_context: bool = True,
            use_proper_name_analysis: bool = True,
            output_compound_separator: bool = False,
            guess_unknown_words: bool = True,
            output_phonetic_info: bool = False,
            restore_verb_ending: bool = True,
            ignore_derivation_symbol: bool = True,
    ) -> None:
        self.use_context = use_context
        self.use_proper_name_analysis = use_proper_name_analysis
        self.output_compound_separator = output_compound_separator  # doesn't do anything atm
        self.guess_unknown_words = guess_unknown_words
        self.output_phonetic_info = output_phonetic_info
        self.restore_verb_ending = restore_verb_ending
        self.ignore_derivation_symbol = ignore_derivation_symbol
        # ambiguous analyzer
        self._amb_morph_tagger = VabamorfTagger(
            compound=self.output_compound_separator,
            disambiguate=False,
            guess=self.guess_unknown_words,
            slang_lex=False,
            phonetic=self.output_phonetic_info,
            use_postanalysis=True,
            use_reorderer=True,
            propername=self.use_proper_name_analysis,
            predisambiguate=False,
            postdisambiguate=False,

        )
        # disambiguated analyzer
        self._disamb_morph_tagger = VabamorfTagger(
            compound=self.output_compound_separator,
            disambiguate=True,
            guess=self.guess_unknown_words,
            slang_lex=False,
            phonetic=self.output_phonetic_info,
            use_postanalysis=True,
            use_reorderer=True,
            propername=self.use_proper_name_analysis,
            predisambiguate=self.use_context,
            postdisambiguate=self.use_context,
        )
        # workaround taggers to be able with to work with pretokenized input as it's not accepted by `Text` class
        self._whitespace_tagger = WhiteSpaceTokensTagger()
        self._compound_token_tagger = PretokenizedTextCompoundTokensTagger()

    def __call__(
            self, data: Union[str, List[str]]
    ) -> List[VbTokenAnalysis]:
        return self.analyze(data)

    def analyze(self, input_text: Union[List[str], str]) -> List[VbTokenAnalysis]:
        if isinstance(input_text, str):
            ambiguous_analysis = self._ambiguous_str_analysis(input_text)
            disambiguated_analysis = self._disambiguated_str_analysis(input_text)
            # number of tokens must be the same in both analyses
            if len(disambiguated_analysis["morph_analysis"]) != len(ambiguous_analysis["morph_analysis"]):
                raise ValueError("Number of tokens mismatch")
        elif isinstance(input_text, list) and all((isinstance(el, str) and " " not in el for el in input_text)):
            ambiguous_analysis = self._ambiguous_pretokenized_analysis(input_text)
            disambiguated_analysis = self._disambiguated_pretokenized_analysis(input_text)
            # number of tokens must be the same in both analyses and match the number of original tokens
            if not (
                    len(disambiguated_analysis["morph_analysis"]) == len(ambiguous_analysis["morph_analysis"])
                    and len(disambiguated_analysis["morph_analysis"]) == len(input_text)
            ):
                raise ValueError("Number of tokens mismatch")
        else:
            raise ValueError(
                "Input text should may be raw `str` or pretokenized `List[str]` with no whitespaces in each token"
            )

        result = []
        tokens = disambiguated_analysis["morph_analysis"].text
        if self.output_compound_separator:
            disambiguated_lemmas = disambiguated_analysis["morph_analysis"].root
            lemma_candidate_list = ambiguous_analysis["morph_analysis"].root
        else:
            disambiguated_lemmas = disambiguated_analysis["morph_analysis"].lemma
            lemma_candidate_list = ambiguous_analysis["morph_analysis"].lemma
        features_list = disambiguated_analysis["morph_analysis"].form
        pos_list = disambiguated_analysis["morph_analysis"].partofspeech
        ambiguous_pos_list = ambiguous_analysis["morph_analysis"].partofspeech

        for index, (
                token, disambiguated_lemma, lemma_candidates, features, part_of_speech, part_of_speech_candidates
        ) in enumerate(
                zip(tokens, disambiguated_lemmas, lemma_candidate_list, features_list, pos_list, ambiguous_pos_list)
        ):
            disambiguated_lemma: str = disambiguated_lemma[0]
            if self.ignore_derivation_symbol:
                disambiguated_lemma = disambiguated_lemma.replace("=", "")
                lemma_candidates = [lemma_candidate.replace("=", "") for lemma_candidate in lemma_candidates]
            part_of_speech: str = part_of_speech[0]

            # add `ma` back to verbs
            if self.restore_verb_ending and self.output_compound_separator:
                print("bb")
                lemma_candidates = [
                    f"{lemma_candidate}ma" if pos == "V" else lemma_candidate
                    for lemma_candidate, pos in zip(lemma_candidates, part_of_speech_candidates)
                ]
                if part_of_speech == "V":
                    disambiguated_lemma = f"{disambiguated_lemma}ma"

            lemma_candidates: List[str] = sorted(lemma_candidates)
            result.append(
                VbTokenAnalysis(
                    index=index + 1,  # 1-based
                    token=token,
                    disambiguated_lemma=disambiguated_lemma,
                    lemma_candidates=tuple(lemma_candidates),
                    features=features[0],
                    part_of_speech=part_of_speech[0],
                    candidate_parts_of_speech=tuple(part_of_speech_candidates),
                )
            )
        return result

    def _ambiguous_str_analysis(self, raw_text: str) -> Text:
        text = Text(raw_text)
        text.tag_layer(self._amb_morph_tagger.input_layers)
        self._amb_morph_tagger.tag(text)
        return text

    def _disambiguated_str_analysis(self, raw_text: str) -> Text:
        text = Text(raw_text)
        text.tag_layer(self._disamb_morph_tagger.input_layers)
        self._disamb_morph_tagger.tag(text)
        return text

    def _ambiguous_pretokenized_analysis(self, tokens: List[str]) -> Text:
        joined_text = " ".join(tokens)
        text = Text(joined_text)
        self._whitespace_tagger.tag(text)
        self._compound_token_tagger(text)
        text.tag_layer(self._amb_morph_tagger.input_layers)
        self._amb_morph_tagger.tag(text)
        return text

    def _disambiguated_pretokenized_analysis(self, tokens: List[str]) -> Text:
        joined_text = " ".join(tokens)
        text = Text(joined_text)
        self._whitespace_tagger.tag(text)
        self._compound_token_tagger(text)
        text.tag_layer(self._amb_morph_tagger.input_layers)
        self._disamb_morph_tagger.tag(text)
        return text
