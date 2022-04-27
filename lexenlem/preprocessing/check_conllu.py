from collections import namedtuple

from conllu import parse
from tqdm.auto import tqdm
from estnltk import Text

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

    vb_set = set()
    conll_set = set()

    for sentence in tqdm(parsed):
        text = sentence.metadata["text"]
        processed = pipeline(text)[0]
        processed = [convert_vb_to_conll(el) for el in processed]
        if len(processed) == len(sentence):
            for x, y in zip(processed, sentence):
                vb = [str_tags_to_dict(feat) for feat in x.conll_feature_candidates]
                for candidate in vb:
                    if candidate:
                        for key, value in candidate.items():
                            vb_set.add((key, value))
                conll = y["feats"]
                if conll:
                    for key, value in conll.items():
                        conll_set.add((key, value))

    return vb_set, conll_set
