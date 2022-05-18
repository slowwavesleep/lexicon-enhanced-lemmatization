import dataclasses
from typing import Union, List

import stanza

from lexenlem.preprocessing.vabamorf_pipeline import VbTokenAnalysis


class StanzaPretokenizedAnalyzer:

    def __init__(self):
        stanza.download("et")
        self.nlp = stanza.Pipeline(
            lang="et", processors="tokenize,pos", tokenize_pretokenized=True, pos_batch_size=256, verbose=False
        )

    def _analyze(self, tokens: Union[List[List[str]], List[str]]) -> stanza.Document:
        processed = self.nlp(tokens)
        return processed

    def __call__(self, token_analyses: List[List[VbTokenAnalysis]]) -> List[List[VbTokenAnalysis]]:

        if not token_analyses:
            raise RuntimeError("Passed an empty list")

        if len(token_analyses) == 1:
            tokens_to_analyze: List[str] = [el.token for el in token_analyses[0]]
        else:
            tokens_to_analyze: List[List[str]] = []
            for sentence in token_analyses:
                tokens_to_analyze.append([el.token for el in sentence])

        processed = self._analyze(tokens_to_analyze)

        if len(token_analyses) != len(processed.sentences):
            raise RuntimeError("Lengths mismatch")

        reanalyzed_tokens: List[List[VbTokenAnalysis]] = []

        for vb_sentence, stanza_sentence in zip(token_analyses, processed.sentences):
            cur_sentence: List[VbTokenAnalysis] = []
            for vb_token, stanza_token in zip(vb_sentence, stanza_sentence.words):
                cur_sentence.append(
                    dataclasses.replace(
                        vb_token,
                        part_of_speech=stanza_token.upos,
                        features=stanza_token.feats
                    )
                )
            reanalyzed_tokens.append(cur_sentence)

        if len(token_analyses) != len(reanalyzed_tokens):
            raise RuntimeError("Lengths mismatch")

        return reanalyzed_tokens
