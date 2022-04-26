from dataclasses import dataclass
from typing import List, Union

import torch
from estnltk import Text
from estnltk.taggers import VabamorfTagger
from tqdm.auto import tqdm

import lexenlem.models.common.seq2seq_constant as constant
from lexenlem.models.common.seq2seq_model import Seq2SeqModelCombined
from lexenlem.models.common.utils import prune_decoded_seqs, unsort
from lexenlem.models.common.vabamorf2conll import neural_model_tags
from lexenlem.models.lemma import edit
from lexenlem.models.lemma.data import DataLoaderCombined
from lexenlem.models.lemma.vocab import MultiVocab, Vocab
from lexenlem.models.common.data import get_long_tensor, sort_all

tagger = VabamorfTagger(compound=True, disambiguate=False, guess=False)


@dataclass
class BaseAnalysis:
    token: str
    lemma: str
    part_of_speech: str


@dataclass
class VabamorfAnalysis(BaseAnalysis):
    feats: str


@dataclass
class VabamorfAnalysisConll(BaseAnalysis):
    feats: str
    xpos: bool = True


@dataclass
class AdHocInput:
    src_input: List[int]  # contains form itself + concatenated pos and features
    lemma_input: List[int]
    edit_type: int


@dataclass
class AdHocModelInput:
    src: torch.tensor
    src_mask: torch.tensor
    lem: torch.tensor
    lem_mask: torch.tensor
    orig_idx: List[int]

    def cuda(self):
        self.src.cuda()
        self.src_mask.cuda()
        self.lem.cuda()
        self.lem_mask.cuda()

    def cpu(self):
        self.src.cpu()
        self.src_mask.cpu()
        self.lem.cpu()
        self.lem_mask.cpu()


def get_vabamorf_analysis(token: str) -> List[VabamorfAnalysis]:
    text = Text(token)
    text.tag_layer(tagger.input_layers)
    tagger.tag(text)
    lemmas = text["morph_analysis"][0].root
    forms = text["morph_analysis"][0].form
    parts_of_speech = text["morph_analysis"][0].partofspeech
    return [
        VabamorfAnalysis(token=token, lemma=lemma, part_of_speech=part_of_speech, feats=form)
        for lemma, part_of_speech, form in zip(lemmas, parts_of_speech, forms)
    ]


def remove_pos_from_feats(feats: str) -> str:
    if "POS=" not in feats:
        return feats
    feats = feats.split("|")
    return "|".join([feat for feat in feats if "POS=" not in feat])


def convert_vb_to_conll(vb_analysis: VabamorfAnalysis) -> List[VabamorfAnalysisConll]:
    # conversion from vb to conll may be ambiguous
    candidates: List[str] = neural_model_tags(vb_analysis.token, vb_analysis.part_of_speech, vb_analysis.feats)
    return [
        VabamorfAnalysisConll(
            token=vb_analysis.token,
            lemma=vb_analysis.lemma,
            part_of_speech=vb_analysis.part_of_speech,
            feats=remove_pos_from_feats(feats_candidate),
        )
        for feats_candidate in candidates
    ]


def tokenize(raw_text: str) -> List[str]:
    text = Text(raw_text)
    text.tag_layer("tokens")
    tokenized = []
    for token in text.tokens:
        tokenized.append(token.text)
    return tokenized


def prepare_batch(
        preprocessed_input: List[Union[VabamorfAnalysis, VabamorfAnalysisConll]],
        eos_after: bool,
        use_pos: bool,
        use_feats: bool,
        skip_lemma: bool,
        vocab: Vocab,
) -> List[AdHocInput]:
    batch = []

    for element in tqdm(preprocessed_input, desc="Preparing raw batch...", disable=True):

        if isinstance(element, VabamorfAnalysis):
            raise NotImplementedError("Vabamorf output format not supported at the moment.")

        edit_type: int = edit.EDIT_TO_ID[edit.get_edit_type(word=element.token, lemma=element.lemma)]

        surface_form: List[str] = list(element.token)
        if eos_after:
            surface_form = [constant.SOS] + surface_form
        else:
            surface_form = [constant.SOS] + surface_form + [constant.EOS]

        part_of_speech: List[str] = [f"POS={element.part_of_speech}"]

        feats: List[str] = element.feats.split("|")

        src_input = []
        src_input += surface_form
        if use_pos:
            src_input += part_of_speech
        if use_feats:
            src_input += feats
        if eos_after:
            src_input += [constant.EOS]

        if skip_lemma:
            lemma_input = [constant.SOS, constant.EOS]
        else:
            # TODO: account for a list of lemmas
            lemma_input = [constant.SOS, *list(element.lemma), constant.EOS]

        input_element = AdHocInput(
            src_input=vocab.map(src_input), lemma_input=vocab.map(lemma_input), edit_type=edit_type
        )

        batch.append(input_element)
    return batch


def process_batch(raw_batch: List[AdHocInput]) -> AdHocModelInput:
    raw_batch = [[el.src_input, el.lemma_input] for el in raw_batch]

    batch_size = len(raw_batch)
    batch = list(zip(*raw_batch))
    assert len(batch) == 2

    lens = [len(x) for x in batch[0]]
    batch, orig_idx = sort_all(batch, lens)

    src = batch[0]
    src = get_long_tensor(src, batch_size)
    src_mask = torch.eq(src, constant.PAD_ID)

    lem = batch[1]
    lem = get_long_tensor(lem, batch_size)
    lem_mask = torch.eq(lem, constant.PAD_ID)

    return AdHocModelInput(src, src_mask, lem, lem_mask, orig_idx)


def save_vocab(dataloader: DataLoaderCombined, filename: str):
    params = {"vocab": dataloader.vocab.state_dict()}
    torch.save(params, filename)


def load_vocab(filename: str) -> MultiVocab:
    return MultiVocab.load_state_dict(torch.load(filename)["vocab"])


class VabamorfAdHocProcessor:

    def __init__(
            self,
            path: str,
            use_feats: bool = True,
            convert_to_conllu: bool = True,
            skip_lemma: bool = False,
            use_cuda: bool = False,
            output_compound_separator: bool = False,
            use_upos: bool = True,
            use_dict: bool = True,
    ):
        self.config = None
        self.word_dict = None
        self.composite_dict = None
        self.vocab = None
        self.lexicon = None
        self.output_compound_separator = output_compound_separator
        self.use_feats = use_feats
        self.skip_lemma = skip_lemma
        self.use_upos = use_upos
        self.use_dict = use_dict

        self.convert_to_conllu = convert_to_conllu

        self.use_cuda = use_cuda

        if self.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._init_model(path)

        self.use_pos = self.config["use_pos"]

        self.eos_after = self.config["eos_after"]

        self.analyzer = VabamorfAnalyzer(
            output_compound_separator=self.output_compound_separator,
            convert_to_conllu=self.convert_to_conllu,
            use_upos=True,
        )

    def _init_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.word_dict, self.composite_dict = checkpoint['dicts']
        self.vocab: Vocab = MultiVocab.load_state_dict(checkpoint['vocab'])["combined"]
        self.lexicon = checkpoint['lexicon']
        if self.config['dict_only']:
            raise NotImplementedError
        self.model = Seq2SeqModelCombined(self.config, self.vocab, use_cuda=self.use_cuda)
        self.model.eval()
        self.model.load_state_dict(checkpoint['model'])

    def preprocess_text(self, raw_text: str) -> List[Union[VabamorfAnalysis, VabamorfAnalysisConll]]:
        tokenized = tokenize(raw_text)
        analyzed = []
        for token in tokenized:
            token_analysis = self.analyzer.analyze(token)
            analyzed.append(token_analysis)
        return analyzed

    def lemmatize_dict(
            self, preprocessed: List[Union[VabamorfAnalysis, VabamorfAnalysisConll]]
    ) -> List[Union[str, None]]:
        if self.use_pos:
            return [self.composite_dict.get((el.token, el.part_of_speech), None) for el in preprocessed]
        else:
            return [self.word_dict.get((el.token, el.part_of_speech), None) for el in preprocessed]

    def lemmatize(self, input_str: str):
        preprocessed: List[Union[VabamorfAnalysis, VabamorfAnalysisConll]] = self.preprocess_text(input_str)

        raw_batch = prepare_batch(
            preprocessed,
            use_feats=self.use_feats,
            use_pos=self.use_pos,
            skip_lemma=self.skip_lemma,
            eos_after=self.eos_after,
            vocab=self.vocab
        )

        pt_batch = process_batch(raw_batch)

        if self.use_cuda:
            pt_batch.cuda()

        output_seqs, _, _ = self.model.predict_greedy(
            src=pt_batch.src, src_mask=pt_batch.src_mask, lem=pt_batch.lem, lem_mask=pt_batch.lem_mask, log_attn=False
        )

        output_seqs = [self.vocab.unmap(ids) for ids in output_seqs]

        output_seqs = prune_decoded_seqs(output_seqs)
        output_seqs = ["".join(seq) for seq in output_seqs]
        output_seqs = unsort(output_seqs, pt_batch.orig_idx)

        if self.use_dict and self.use_pos:
            tmp = []
            d_lemmatized = self.lemmatize_dict(preprocessed)
            for d_lemma, hypothesis in zip(d_lemmatized, output_seqs):
                if not d_lemma:
                    tmp.append(hypothesis)
                else:
                    tmp.append(d_lemma)
            output_seqs = tmp

        return output_seqs
