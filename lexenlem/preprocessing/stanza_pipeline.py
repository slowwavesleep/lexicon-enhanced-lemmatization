import dataclasses
from typing import List

import stanza

from lexenlem.preprocessing.vabamorf_pipeline import VbTokenAnalysis


class StanzaPretokenizedAnalyzer:

    def __init__(self):
        stanza.download("et")
        self.nlp = stanza.Pipeline(
            lang="et", processors="tokenize,pos", tokenize_pretokenized=True, verbose=False
        )

    def _analyze(self, tokens: List[str]) -> stanza.Document:
        processed = self.nlp([tokens])
        return processed

    def __call__(self, token_analyses: List[VbTokenAnalysis]) -> List[VbTokenAnalysis]:

        if not token_analyses:
            raise RuntimeError("Passed an empty list")

        tokens_to_analyze: List[str] = [el.token for el in token_analyses]

        processed = self._analyze(tokens_to_analyze)

        reanalyzed_tokens: List[VbTokenAnalysis] = []
        for vb_token, st_token in zip(token_analyses, processed.sentences[0].tokens):
            reanalyzed_tokens.append(
                dataclasses.replace(
                    vb_token,
                    part_of_speech=st_token.words[0].upos,
                    features=st_token.words[0].feats,
                )
            )

        return reanalyzed_tokens
