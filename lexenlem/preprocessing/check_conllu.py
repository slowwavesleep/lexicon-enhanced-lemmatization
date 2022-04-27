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
                    for candidate in vb_token_tag_candidates:
                        # print(vb_token_tag_candidates)
                        print(vb_token.token, len(vb_token_tag_candidates))
                        print(vb_token.lemma_candidates)
                        print(candidate)
                        print(c_token_tags)
                        print("shared:")
                        print({x: candidate[x] for x in candidate if x in c_token_tags and candidate[x] == c_token_tags[x]})
                        # print(dict(candidate.items() & c_token_tags.items()))
                        print(vb_token.features)
                        print()
                else:
                    pass


