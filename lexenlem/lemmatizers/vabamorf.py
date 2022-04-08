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
