from estnltk import Text


def lemmatize(token: str) -> str:
    text = Text(token)
    text.tag_layer("morph_analysis")
    lemma: str = text["morph_analysis"][0].lemma[0]
    return lemma

# def lemmatize(token):
#     return Text(token).lemmas[0].split('|')
