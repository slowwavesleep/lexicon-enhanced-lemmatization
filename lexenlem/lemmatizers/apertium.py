import apertium
from collections import OrderedDict


def lemmatize(word, lang):
    analysis = apertium.analyze(lang, word)
    if analysis:
        lemmas = list(OrderedDict.fromkeys([x.split('<')[0] for x in str(analysis[0]).split('/')[1:]]))
    else:
        lemmas = []
    return list(''.join(lemmas))
