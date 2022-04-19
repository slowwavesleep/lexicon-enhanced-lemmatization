from typing import List

from estnltk import Text
from estnltk.taggers import VabamorfTagger

tagger = VabamorfTagger(compound=True, disambiguate=False, guess=False)


# estnltk 1.7.1
def lemmatize(token: str) -> List[str]:
    text = Text(token)
    text.tag_layer(tagger.input_layers)
    tagger.tag(text)
    lemmas = list(text["morph_analysis"][0].root)
    return lemmas

# estnltk 1.4.1
# def lemmatize(token):
#     return Text(token).lemmas[0].split('|')


class CombinedAnalyzer:

    def __init__(self, output_compound_separator: bool = False):

        self.output_compound_separator = output_compound_separator

        self.tagger_lemmas = VabamorfTagger(compound=True, disambiguate=False, guess=False)
        self.tagger_morph = VabamorfTagger(compound=True, disambiguate=True, guess=True)

    def analyze(self, token: str):

        text_lemmas = Text(token)
        text_lemmas.tag_layer(self.tagger_lemmas.input_layers)
        self.tagger_lemmas.tag(text_lemmas)
        if self.output_compound_separator:
            lemmas = list(text_lemmas["morph_analysis"][0].root)
        else:
            lemmas = list(text_lemmas["morph_analysis"][0].lemma)
        lemmas = "".join(lemmas)

        text_morph = Text(token)
        text_morph.tag_layer(self.tagger_morph.input_layers)
        self.tagger_morph.tag(text_morph)
        form = text_morph["morph_analysis"][0].form
        pos = text_morph["morph_analysis"][0].partofspeech

        return lemmas, form, pos
