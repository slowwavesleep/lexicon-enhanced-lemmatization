from estnltk import Text

def lemmatize(token):
    return Text(token).lemmas[0].split('|')