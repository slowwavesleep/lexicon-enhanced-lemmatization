from lexenlem.models.lemma.scorer import score
from lexenlem.utils import conll18_ud_eval as ud_eval
import conllu

def check():
    # print(score("./data/output.conllu", "./data/et_edt-ud-test.conllu"))
    with open("./data/output.conllu") as file:
        data = file.read()
    parsed = conllu.parse(data)

    with open("./data/et_edt-ud-test.conllu") as file:
        data = file.read()
    parsed_orig = conllu.parse(data)

    return ud_eval.load_conllu_file("./data/output.conllu"), parsed, parsed_orig
