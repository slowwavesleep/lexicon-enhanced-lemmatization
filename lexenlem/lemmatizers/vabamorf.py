from estnltk import Text


# estnltk 1.7.1
def lemmatize(token: str) -> str:
    text = Text(token)
    text.tag_layer("morph_analysis")
    lemma: str = text["morph_analysis"][0].lemma[0]
    return lemma

# estnltk 1.4.1
# def lemmatize(token):
#     return Text(token).lemmas[0].split('|')
