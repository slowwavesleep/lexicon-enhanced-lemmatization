from collections import namedtuple, defaultdict, Counter
from itertools import zip_longest
from pprint import pprint


from conllu import parse
from tqdm.auto import tqdm
from estnltk import Text
from estnltk.taggers import VabamorfTagger, WhiteSpaceTokensTagger, PretokenizedTextCompoundTokensTagger

from lexenlem.preprocessing.vabamorf_pipeline import VbPipeline, convert_vb_to_conll, str_tags_to_dict


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


def check_converted_tags():
    with open("./data/et_edt-ud-test.conllu") as file:
        data = file.read()

    parsed = parse(data)

    pipeline = VbPipeline()

    new_parsed = []

    for sentence in tqdm(parsed):
        text = sentence.metadata["text"]
        processed = pipeline(text)[0]
        processed = [convert_vb_to_conll(el) for el in processed]
        if len(processed) == len(sentence):
            for vb_token, c_token in zip(processed, sentence):
                vb_token_tag_candidates = vb_token.conll_feature_candidates

                c_token_tags = c_token["feats"]
                if len(vb_token_tag_candidates) > 1 and len(set(vb_token.lemma_candidates)) > 1:
                    print(f"Tag candidates: {vb_token_tag_candidates}")
                    for candidate in vb_token_tag_candidates:
                        # print(vb_token_tag_candidates)
                        print(f"Token: {vb_token.token}, True lemma: {c_token['lemma']}")
                        print(f"Disambiguated: {vb_token.disambiguated_lemma}")
                        print(f"Lemma candidates: {vb_token.lemma_candidates}")
                        print(f"Current candidate: {candidate}, candidated feats: {vb_token.features}")
                        print(f"True tags: {c_token_tags}")
                        print("shared tags:")
                        print({x: candidate[x] for x in candidate if x in c_token_tags and candidate[x] == c_token_tags[x]})
                        # print(dict(candidate.items() & c_token_tags.items()))
                        print()
                        print()


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
        analyzed = pipe(forms, True)[0]
        for vb_analysis, lemma, form, pos in zip(analyzed, lemmas, forms, parts_of_speech):
            total += 1
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
                        f" lemma_candidates: {set(vb_analysis.lemma_candidates)}"
                    )
    # 0.97 ~0.96
    return correct/total
