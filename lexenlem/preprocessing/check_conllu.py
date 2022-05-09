import dataclasses
import json
from collections import namedtuple
from itertools import zip_longest
from pprint import pprint
from hashlib import md5


from conllu import parse
from tqdm.auto import tqdm
from estnltk import Text
from estnltk.taggers import VabamorfTagger, WhiteSpaceTokensTagger, PretokenizedTextCompoundTokensTagger

from lexenlem.preprocessing.vabamorf_pipeline import VbPipeline


def check_len():
    with open("./data/et_edt-ud-test.conllu") as file:
        data = file.read()

    parsed = parse(data)

    # return parsed

    mismatch = []
    Mismatched = namedtuple("Mismatched", "original conll_tokens vb_tokens")

    for sentence in tqdm(parsed):
        text = sentence.metadata["text"]
        t = Text(text)
        t.tag_layer()
        morph = t["morph_analysis"]

        if len(sentence) != len(morph):
            mismatch.append(
                Mismatched(
                    text, [el["form"] for el in sentence], [el["normalized_text"][0] for el in morph]
                )
            )
    return mismatch


def check_tokenization():
    with open("./data/et_edt-ud-test.conllu") as file:
        data = file.read()

    parsed = parse(data)

    m_tagger = VabamorfTagger(compound=True, disambiguate=True, guess=True)
    w_tagger = WhiteSpaceTokensTagger()
    c_tagger = PretokenizedTextCompoundTokensTagger()

    for sentence in tqdm(parsed):
        text = " ".join([el["form"] for el in sentence])
        t = Text(text)
        w_tagger.tag(t)
        c_tagger(t)
        t.tag_layer(m_tagger.input_layers)
        m_tagger.tag(t)
        if len(sentence) != len(t["morph_analysis"].normalized_text):
            print(text)
            print([el["form"] for el in sentence])
            print(text.split(" "))
            print(t["morph_analysis"].normalized_text)
            print(text.split(" "))
            pprint(list(zip_longest([el["form"] for el in sentence], [el[0] for el in t["morph_analysis"].normalized_text])))


def check_tokenization_vb():
    with open("./data/et_edt-ud-test.conllu") as file:
        data = file.read()

    parsed = parse(data)
    parsed = [[el["form"] for el in sentence] for sentence in parsed]
    pipe = VbPipeline()
    for sentence in tqdm(parsed):
        analyzed = pipe(sentence, True)
        if len(analyzed[0]) != len(sentence):
            print(analyzed[0])
            print(sentence)
            break


def check_vb():
    with open("./data/et_edt-ud-test.conllu") as file:
        data = file.read()

    parsed = parse(data)
    pipe = VbPipeline()
    total = 0
    correct = 0

    for sentence in tqdm(parsed):
        forms = [el["form"] for el in sentence]
        lemmas = [el["lemma"] for el in sentence]
        parts_of_speech = [el["upos"] for el in sentence]
        analyzed = pipe(forms)
        for vb_analysis, lemma, form, pos in zip(analyzed, lemmas, forms, parts_of_speech):
            total += 1
            lemma = lemma.replace("=", "").replace("_", "")
            if vb_analysis.disambiguated_lemma == lemma:
                correct += 1
            elif vb_analysis.part_of_speech == "V":
                # print(f"Restored verb lemma: {vb_analysis.disambiguated_lemma}, UD lemma: {lemma}")
                # if vb_analysis.disambiguated_lemma + "ma" == lemma:
                #     correct += 1
                #     hits.update((vb_analysis.features,))
                # else:
                if lemma not in set(vb_analysis.lemma_candidates):
                    print(
                        f"Vabamorf: {vb_analysis.disambiguated_lemma},"
                        f" UD lemma: {lemma},"
                        f" UD form: {form},"
                        f" UPOS: {pos},"
                        f" predicted feats: {vb_analysis.features},"
                        f" lemma_candidates: {vb_analysis.lemma_candidates}"
                        f" parts of speech: {vb_analysis.candidate_parts_of_speech}"
                    )
    # ~0.96
    return correct/total


def check_hash():
    with open("./data/et_edt-ud-test.conllu") as file:
        data = file.read()

    parsed = parse(data)
    parsed = {item.metadata["sent_id"]: item for item in parsed}

    return parsed

    # pipe = VbPipeline()
    #
    # result = {}
    # for key, value in tqdm(parsed.items()):
    #     forms = [el["form"] for el in value]
    #     result[key] = pipe(forms)
    # result = dict(sorted(result.items()))
    # new_result = []
    # for key, data in result.items():
    #     serialized = json.dumps(
    #         {key: [dataclasses.asdict(element) for element in data]},
    #         ensure_ascii=False,
    #     )
    #     new_result.append(serialized)
    #
    # hashed = md5("\n".join(new_result).encode("utf-8")).hexdigest()
    #
    # return hashed

