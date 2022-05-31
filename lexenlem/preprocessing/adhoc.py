from typing import List

import torch

from lexenlem.models.common import seq2seq_constant as constant
from lexenlem.models.common.data import sort_all, get_long_tensor
from lexenlem.models.common.seq2seq_model import Seq2SeqModelCombined
from lexenlem.models.common.utils import prune_decoded_seqs, unsort
from lexenlem.models.lemma.vocab import Vocab, MultiVocab
from lexenlem.preprocessing.vabamorf_pipeline import VbPipeline, VbTokenAnalysis, AdHocInput, AdHocModelInput


class AdHocLemmatizer:

    def __init__(
            self,
            path: str,
            *,
            use_feats: bool = True,
            skip_lemma: bool = False,
            use_cuda: bool = False,
            allow_compound_separator: bool = False,
            allow_derivation_sign: bool = False,
            use_stanza: bool = False,
            restore_verb_ending: bool = True,
            use_context: bool = True,
            guess_unknown_words: bool = True,
            use_proper_name_analysis: bool = True,
    ):
        self.use_context = use_context
        self.use_proper_name_analysis = use_proper_name_analysis
        self.allow_compound_separator = allow_compound_separator
        self.guess_unknown_words = guess_unknown_words
        self.restore_verb_ending = restore_verb_ending
        self.allow_derivation_sign = allow_derivation_sign
        self.use_feats = use_feats
        self.skip_lemma = skip_lemma
        self.use_stanza = use_stanza

        if self.use_stanza:
            from lexenlem.preprocessing.stanza_pipeline import StanzaPretokenizedAnalyzer
            self.stanza = StanzaPretokenizedAnalyzer()
        else:
            self.stanza = None

        self.use_cuda = use_cuda

        if self.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._init_model(path)
        self.use_pos = self.config["use_pos"]
        self.eos_after = self.config["eos_after"]

        self.analyzer = VbPipeline(
            use_context=self.use_context,
            use_proper_name_analysis=self.use_proper_name_analysis,
            output_compound_separator=self.allow_compound_separator,
            guess_unknown_words=self.guess_unknown_words,
            output_phonetic_info=False,  # setting to True breaks the pipeline
            restore_verb_ending=self.restore_verb_ending,
            ignore_derivation_symbol=not self.allow_derivation_sign,

        )

    def __call__(self, raw_text: str) -> List[str]:
        return self.lemmatize(raw_text)

    def _init_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint["config"]
        self.word_dict, self.composite_dict = checkpoint["dicts"]
        self.vocab: Vocab = MultiVocab.load_state_dict(checkpoint["vocab"])["combined"]
        self.lexicon = checkpoint.get("lexicon", None)
        if self.config["dict_only"]:
            raise NotImplementedError
        self.model = Seq2SeqModelCombined(self.config, self.vocab, use_cuda=self.use_cuda)
        self.model.eval()
        self.model.load_state_dict(checkpoint['model'])

    def analyze_text(self, raw_text: str) -> List[VbTokenAnalysis]:
        analyzed: List[VbTokenAnalysis] = self.analyzer(raw_text)
        if self.use_stanza:
            analyzed = self.stanza(analyzed)
        return analyzed

    def lemmatize(self, input_str: str) -> List[str]:
        preprocessed: List[VbTokenAnalysis] = self.analyze_text(input_str)

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

        if not self.allow_derivation_sign:
            output_seqs = [el.replace("=", "") for el in output_seqs]

        if not self.allow_compound_separator:
            output_seqs = [el.replace("_", "") for el in output_seqs]

        return output_seqs


def prepare_batch(
        preprocessed_input: List[VbTokenAnalysis],
        eos_after: bool,
        use_pos: bool,
        use_feats: bool,
        skip_lemma: bool,
        vocab: Vocab,
) -> List[AdHocInput]:
    batch = []

    for element in preprocessed_input:

        surface_form: List[str] = list(element.token)
        if eos_after:
            surface_form = [constant.SOS] + surface_form
        else:
            surface_form = [constant.SOS] + surface_form + [constant.EOS]

        part_of_speech: List[str] = [element.part_of_speech]

        feats: List[str] = [element.features]

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
            lemma_input = [constant.SOS, *list(element.processed_lemma_candidates), constant.EOS]

        input_element = AdHocInput(src_input=vocab.map(src_input), lemma_input=vocab.map(lemma_input))

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

    return AdHocModelInput(src=src, src_mask=src_mask, lem=lem, lem_mask=lem_mask, orig_idx=orig_idx)
