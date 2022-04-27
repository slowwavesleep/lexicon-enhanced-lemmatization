from collections import namedtuple

from conllu import parse
from tqdm.auto import tqdm
from estnltk import Text


def check():
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

