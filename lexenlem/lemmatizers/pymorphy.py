import pymorphy2

morph = pymorphy2.MorphAnalyzer()

def lemmatize(token):
    normal_forms = set()
    for p in morph.parse(token):
        normal_forms.add(p.normal_form)
    return list(normal_forms)