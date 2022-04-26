from conllu import parse
from estnltk.taggers import VabamorfTagger
from estnltk import Text
from tqdm.auto import tqdm

from lexenlem.preprocessing.vabamorf import VabamorfAdHocProcessor, tokenize


def sanity_check(use_pretokenized: bool = True, do_lower_case: bool = False):
    v = VabamorfAdHocProcessor("models/et_edt_lemmatizer.pt")
    with open("./data/et_edt-ud-test.conllu") as file:
        data = file.read()
    parsed = parse(data)
    total = 0
    correct = 0
    for sentence in tqdm(parsed):
        text = sentence.metadata["text"]
        tokens = [el["form"] for el in sentence]
        gold_lemmas = [el["lemma"] for el in sentence]
        if use_pretokenized:
            hypotheses = [v.lemmatize(token)[0] for token in tokens]
        else:
            hypotheses = v.lemmatize(text)
        if len(hypotheses) == len(gold_lemmas):
            tmp = [g.replace("_", "") == h.replace("_", "") for g, h in zip(gold_lemmas, hypotheses)]
            total += len(tmp)
            correct += sum(tmp)
        else:
            tmp = [g.replace("_", "") == h.replace("_", "") for g, h in zip(gold_lemmas, hypotheses)]
            total += len(tmp)
        print(correct/total)

    print(correct/total)

# return form as lemma
def identity_baseline(use_pretokenized: bool = True, do_lower_case: bool = False):
    with open("./data/et_edt-ud-test.conllu") as file:
        data = file.read()
    parsed = parse(data)
    total = 0
    correct = 0
    for sentence in tqdm(parsed):
        text = sentence.metadata["text"]
        tokens = [el["form"] for el in sentence]
        gold_lemmas = [el["lemma"] for el in sentence]
        if use_pretokenized:
            hypotheses = tokens
            if do_lower_case:
                hypotheses = [token.lower() for token in tokens]
        else:
            raise NotImplementedError
        if len(hypotheses) == len(gold_lemmas):
            tmp = [g.replace("_", "") == h.replace("_", "") for g, h in zip(gold_lemmas, hypotheses)]
            total += len(tmp)
            correct += sum(tmp)
        print(correct/total)

    print(correct / total)

# return disambiguated vabamorf candidate
def vabamorf_baseline(use_pretokenized: bool = False, do_lower_case: bool = False):
    with open("./data/et_edt-ud-test.conllu") as file:
        data = file.read()
    parsed = parse(data)
    total = 0
    correct = 0
    progress_bar = tqdm(total=len(parsed))
    for sentence in parsed:
        text = sentence.metadata["text"]
        tokens = [el["form"] for el in sentence]
        gold_lemmas = [el["lemma"] for el in sentence]
        if not use_pretokenized:
            tokens = tokenize(text)
        tagger = VabamorfTagger(compound=True, disambiguate=True, guess=True)
        hypotheses = []
        for token in tokens:
            text = Text(token)
            text.tag_layer(tagger.input_layers)
            tagger.tag(text)
            lemma = text["morph_analysis"][0].root[0]
            hypotheses.append(lemma)
        if len(hypotheses) == len(gold_lemmas):
            tmp = [g.replace("_", "") == h.replace("_", "") for g, h in zip(gold_lemmas, hypotheses)]
            total += len(tmp)
            correct += sum(tmp)
        else:
            tmp = [g.replace("_", "") == h.replace("_", "") for g, h in zip(gold_lemmas, hypotheses)]
            total += len(tmp)
        progress_bar.set_postfix(accuracy=correct / total)
        progress_bar.update()
    progress_bar.close()
    print(correct / total)
