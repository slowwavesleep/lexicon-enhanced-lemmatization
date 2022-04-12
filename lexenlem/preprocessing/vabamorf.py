from typing import List, Tuple

from estnltk import Text
from estnltk.taggers import VabamorfTagger

from lexenlem.models.common.vabamorf2conll import neural_model_tags

tagger = VabamorfTagger(compound=True, disambiguate=False, guess=False)


# add lemmas?
def get_vabamorf_tags(token: str) -> List[Tuple[str, str]]:
    text = Text(token)
    text.tag_layer(tagger.input_layers)
    tagger.tag(text)
    forms = text["morph_analysis"][0].form
    parts_of_speech = text["morph_analysis"][0].partofspeech
    return list(zip(parts_of_speech, forms))


def get_conll_tags(token: str) -> List[List[str]]:
    vb_tags: List[Tuple[str, str]] = get_vabamorf_tags(token)
    results = []
    for pos, form in vb_tags:
        results.append(neural_model_tags(token, pos, form))
    return results


def tokenize(raw_text: str) -> List[str]:
    text = Text(raw_text)
    text.tag_layer("tokens")
    tokenized = []
    for token in text.tokens:
        tokenized.append(token.text)
    return tokenized


def basic_preprocessing(raw_text: str):
    tokenized = tokenize(raw_text)
    result = []
    for token in tokenized:
        result.append((token, get_conll_tags(token)[0][0]))
    return result

