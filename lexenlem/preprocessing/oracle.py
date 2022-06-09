from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import conllu
from tqdm.auto import tqdm
import stanza

from lexenlem.preprocessing.vabamorf_pipeline import VbPipeline


class Mode(Enum):
    STANZA = "STANZA"
    VABAMORF = "VABAMORF"
    COMPLEMENTARY = "COMPLEMENTARY"


class StanzaMode(Enum):
    EDT = "edt"
    EWT = "ewt"


@dataclass
class VbConfig:
    context: bool
    compound: bool
    derivation: bool
    guess_unknown: bool
    proper_name: bool


@dataclass
class OracleConfig:
    path: str
    mode: Mode
    vb_config: VbConfig
    st_mode: Optional[StanzaMode]


base_vb_config = VbConfig(
    context=True,
    proper_name=True,
    compound=False,
    derivation=False,
    guess_unknown=True,
)


edt_vabamorf_base = OracleConfig(
    path="./data/et_edt-ud-test.conllu",
    mode=Mode.VABAMORF,
    vb_config=base_vb_config,
    st_mode=StanzaMode.EDT,
)

edt_stanza_edt = OracleConfig(
    path="./data/et_edt-ud-test.conllu",
    mode=Mode.STANZA,
    vb_config=base_vb_config,
    st_mode=StanzaMode.EDT
)

edt_stanza_ewt = OracleConfig(
    path="./data/et_edt-ud-test.conllu",
    mode=Mode.STANZA,
    vb_config=base_vb_config,
    st_mode=StanzaMode.EWT
)

edt_complementary_edt = OracleConfig(
    path="./data/et_edt-ud-test.conllu",
    mode=Mode.COMPLEMENTARY,
    vb_config=base_vb_config,
    st_mode=StanzaMode.EDT
)

edt_complementary_ewt = OracleConfig(
    path="./data/et_edt-ud-test.conllu",
    mode=Mode.COMPLEMENTARY,
    vb_config=base_vb_config,
    st_mode=StanzaMode.EWT
)

compound_vb_config = VbConfig(
    context=True,
    proper_name=True,
    compound=True,
    derivation=False,
    guess_unknown=True,
)

edt_vabamorf_base_compound = OracleConfig(
    path="./data/et_edt-ud-test.conllu",
    mode=Mode.VABAMORF,
    vb_config=compound_vb_config,
    st_mode=StanzaMode.EDT,
)

edt_stanza_edt_compound = OracleConfig(
    path="./data/et_edt-ud-test.conllu",
    mode=Mode.STANZA,
    vb_config=compound_vb_config,
    st_mode=StanzaMode.EDT
)

edt_stanza_ewt_compound = OracleConfig(
    path="./data/et_edt-ud-test.conllu",
    mode=Mode.STANZA,
    vb_config=compound_vb_config,
    st_mode=StanzaMode.EWT
)

edt_complementary_edt_compound = OracleConfig(
    path="./data/et_edt-ud-test.conllu",
    mode=Mode.COMPLEMENTARY,
    vb_config=compound_vb_config,
    st_mode=StanzaMode.EDT
)

edt_complementary_ewt_compound = OracleConfig(
    path="./data/et_edt-ud-test.conllu",
    mode=Mode.COMPLEMENTARY,
    vb_config=compound_vb_config,
    st_mode=StanzaMode.EWT
)

symbols_vb_config = VbConfig(
    context=True,
    proper_name=True,
    compound=True,
    derivation=True,
    guess_unknown=True,
)

edt_vabamorf_base_symbols = OracleConfig(
    path="./data/et_edt-ud-test.conllu",
    mode=Mode.VABAMORF,
    vb_config=symbols_vb_config,
    st_mode=StanzaMode.EDT,
)

edt_stanza_edt_symbols = OracleConfig(
    path="./data/et_edt-ud-test.conllu",
    mode=Mode.STANZA,
    vb_config=symbols_vb_config,
    st_mode=StanzaMode.EDT
)

edt_stanza_ewt_symbols = OracleConfig(
    path="./data/et_edt-ud-test.conllu",
    mode=Mode.STANZA,
    vb_config=symbols_vb_config,
    st_mode=StanzaMode.EWT
)

edt_complementary_edt_symbols = OracleConfig(
    path="./data/et_edt-ud-test.conllu",
    mode=Mode.COMPLEMENTARY,
    vb_config=compound_vb_config,
    st_mode=StanzaMode.EDT
)

edt_complementary_ewt_symbols = OracleConfig(
    path="./data/et_edt-ud-test.conllu",
    mode=Mode.COMPLEMENTARY,
    vb_config=compound_vb_config,
    st_mode=StanzaMode.EWT
)

ewt_vabamorf_base = OracleConfig(
    path="./data/et_ewt-ud-dev.conllu",
    mode=Mode.VABAMORF,
    vb_config=base_vb_config,
    st_mode=StanzaMode.EDT,
)

ewt_stanza_edt = OracleConfig(
    path="./data/et_ewt-ud-test.conllu",
    mode=Mode.STANZA,
    vb_config=base_vb_config,
    st_mode=StanzaMode.EDT
)

ewt_stanza_ewt = OracleConfig(
    path="./data/et_ewt-ud-test.conllu",
    mode=Mode.STANZA,
    vb_config=base_vb_config,
    st_mode=StanzaMode.EWT
)

ewt_complementary_edt = OracleConfig(
    path="./data/et_ewt-ud-test.conllu",
    mode=Mode.COMPLEMENTARY,
    vb_config=base_vb_config,
    st_mode=StanzaMode.EDT
)

ewt_complementary_ewt = OracleConfig(
    path="./data/et_ewt-ud-test.conllu",
    mode=Mode.COMPLEMENTARY,
    vb_config=base_vb_config,
    st_mode=StanzaMode.EWT
)


ewt_vabamorf_base_compound = OracleConfig(
    path="./data/et_ewt-ud-dev.conllu",
    mode=Mode.VABAMORF,
    vb_config=compound_vb_config,
    st_mode=StanzaMode.EDT,
)

ewt_stanza_edt_compound = OracleConfig(
    path="./data/et_ewt-ud-test.conllu",
    mode=Mode.STANZA,
    vb_config=compound_vb_config,
    st_mode=StanzaMode.EDT
)

ewt_stanza_ewt_compound = OracleConfig(
    path="./data/et_ewt-ud-test.conllu",
    mode=Mode.STANZA,
    vb_config=compound_vb_config,
    st_mode=StanzaMode.EWT
)

ewt_complementary_edt_compound = OracleConfig(
    path="./data/et_ewt-ud-test.conllu",
    mode=Mode.COMPLEMENTARY,
    vb_config=compound_vb_config,
    st_mode=StanzaMode.EDT
)

ewt_complementary_ewt_compound = OracleConfig(
    path="./data/et_ewt-ud-test.conllu",
    mode=Mode.COMPLEMENTARY,
    vb_config=compound_vb_config,
    st_mode=StanzaMode.EWT
)

ewt_vabamorf_base_symbols = OracleConfig(
    path="./data/et_ewt-ud-dev.conllu",
    mode=Mode.VABAMORF,
    vb_config=symbols_vb_config,
    st_mode=StanzaMode.EDT,
)

ewt_stanza_edt_symbols = OracleConfig(
    path="./data/et_ewt-ud-test.conllu",
    mode=Mode.STANZA,
    vb_config=symbols_vb_config,
    st_mode=StanzaMode.EDT
)

ewt_stanza_ewt_symbols = OracleConfig(
    path="./data/et_ewt-ud-test.conllu",
    mode=Mode.STANZA,
    vb_config=symbols_vb_config,
    st_mode=StanzaMode.EWT
)

ewt_complementary_edt_symbols = OracleConfig(
    path="./data/et_ewt-ud-test.conllu",
    mode=Mode.COMPLEMENTARY,
    vb_config=symbols_vb_config,
    st_mode=StanzaMode.EDT
)

ewt_complementary_ewt_symbols = OracleConfig(
    path="./data/et_ewt-ud-test.conllu",
    mode=Mode.COMPLEMENTARY,
    vb_config=symbols_vb_config,
    st_mode=StanzaMode.EWT
)




edt_stanza_edt_symbols_dev = OracleConfig(
    path="./data/et_edt-ud-dev.conllu",
    mode=Mode.STANZA,
    vb_config=symbols_vb_config,
    st_mode=StanzaMode.EDT
)

edt_stanza_edt_compound_dev = OracleConfig(
    path="./data/et_edt-ud-dev.conllu",
    mode=Mode.STANZA,
    vb_config=compound_vb_config,
    st_mode=StanzaMode.EDT
)

edt_stanza_ewt_base_dev = OracleConfig(
    path="./data/et_edt-ud-dev.conllu",
    mode=Mode.STANZA,
    vb_config=base_vb_config,
    st_mode=StanzaMode.EDT
)

def preprocess_lemma(lemma: str, config: OracleConfig) -> str:
    if not config.vb_config.derivation:
        lemma = lemma.replace("=", "")
    if not config.vb_config.compound:
        lemma = lemma.replace("_", "")
    return lemma


def run_oracle(config: OracleConfig):
    vb_pipe = None
    st_pipe = None
    correct = 0
    total = 0
    if config.mode != Mode.STANZA:
        vb_pipe = VbPipeline(
            use_context=config.vb_config.context,
            use_proper_name_analysis=config.vb_config.proper_name,
            output_compound_separator=config.vb_config.compound,
            guess_unknown_words=config.vb_config.guess_unknown,
            output_phonetic_info=False,
            restore_verb_ending=True,
            ignore_derivation_symbol=not config.vb_config.derivation,
        )

    if config.mode in (Mode.STANZA, Mode.COMPLEMENTARY):
        if config.st_mode in (StanzaMode.EDT, None):
            package = "edt"
        else:
            package = "ewt"
        st_pipe = stanza.Pipeline(
            lang="et", processors="tokenize,pos,lemma", tokenize_pretokenized=True, verbose=False, package=package
        )

    with open(config.path) as file:
        data = file.read()

    parsed = conllu.parse(data)

    true_lemmas: List[List[str]] = []
    vb_lemmas: List[List[Tuple[str]]] = []
    st_lemmas: List[List[str]] = []

    for sentence in tqdm(parsed):
        forms = [el["form"] for el in sentence]
        lemmas = [preprocess_lemma(el["lemma"], config) for el in sentence]
        true_lemmas.append(lemmas)
        if vb_pipe:
            analyzed = vb_pipe(forms)
            vb_lemmas.append([el.lemma_candidates for el in analyzed])
        if st_pipe:
            analyzed = st_pipe([forms])
            st_lemmas.append([preprocess_lemma(el.words[0].lemma, config) for el in analyzed.sentences[0].tokens])

    if st_lemmas and vb_lemmas:
        for true_sent, vb_sent, st_sent in zip(true_lemmas, vb_lemmas, st_lemmas):
            for true_lemma, vb_lemma, st_lemma in zip(true_sent, vb_sent, st_sent):
                total += 1
                if true_lemma == st_lemma or true_lemma in vb_lemma:
                    correct += 1
    elif st_lemmas and not vb_lemmas:
        for true_sent, st_sent in zip(true_lemmas, st_lemmas):
            for true_lemma, st_lemma in zip(true_sent, st_sent):
                total += 1
                if true_lemma == st_lemma:
                    correct += 1
    elif vb_lemmas and not st_lemmas:
        for true_sent, vb_sent in zip(true_lemmas, vb_lemmas):
            for true_lemma, vb_lemma in zip(true_sent, vb_sent):
                total += 1
                if true_lemma in vb_lemma:
                    correct += 1
    return correct / total

