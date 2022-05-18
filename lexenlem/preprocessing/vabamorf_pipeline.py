import dataclasses
from dataclasses import dataclass
from typing import List, Union, Tuple, Optional

import torch
from estnltk import Text
from estnltk.taggers import VabamorfTagger, WhiteSpaceTokensTagger, PretokenizedTextCompoundTokensTagger
import stanza
from loguru import logger
from tqdm.auto import tqdm

from lexenlem.models.common import seq2seq_constant as constant
from lexenlem.models.common.data import sort_all, get_long_tensor
from lexenlem.models.common.seq2seq_model import Seq2SeqModelCombined
from lexenlem.models.common.utils import prune_decoded_seqs, unsort
from lexenlem.models.lemma.vocab import Vocab, MultiVocab

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


@dataclass
class VbTokenAnalysis:
    index: int  # 1-based numbering to avoid confusion with conll
    token: str
    disambiguated_lemma: str
    lemma_candidates: Tuple[str]
    candidate_parts_of_speech: Tuple[str]
    part_of_speech: str  # disambiguated only
    features: str  # disambiguated only
    # only available when reading conll file
    true_lemma: Optional[str] = None
    sent_id: Optional[str] = None
    # store model's prediction
    predicted_lemma: Optional[str] = None
    # indicates whether conll data is used for feats and pos
    is_conll_data: bool = False

    @property
    def processed_lemma_candidates(self) -> str:
        return "".join(sorted(list(set(self.lemma_candidates))))


@dataclass(frozen=True)
class AdHocInput:
    src_input: List[int]  # contains form itself + concatenated pos and features
    lemma_input: List[int]
    target_in: Optional[List[int]] = None
    target_out: Optional[List[int]] = None


@dataclass
class AdHocModelInput:
    src: torch.tensor
    src_mask: torch.tensor
    lem: torch.tensor
    lem_mask: torch.tensor
    orig_idx: List[int]
    tgt_in: Optional[torch.tensor] = None
    tgt_out: Optional[torch.tensor] = None

    def cuda(self):
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()
        self.lem = self.lem.cuda()
        self.lem_mask = self.lem_mask.cuda()
        if self.tgt_in is not None:
            self.tgt_in = self.tgt_in.cuda()
        if self.tgt_out is not None:
            self.tgt_out = self.tgt_out.cuda()

    def cpu(self):
        self.src = self.src.cpu()
        self.src_mask = self.src_mask.cpu()
        self.lem = self.lem.cpu()
        self.lem_mask = self.lem_mask.cpu()
        if self.tgt_in is not None:
            self.tgt_in = self.tgt_in.cpu()
        if self.tgt_out is not None:
            self.tgt_out = self.tgt_out.cpu()


class VbPipeline:

    def __init__(
            self,
            use_context: bool = True,
            use_proper_name_analysis: bool = True,
            output_compound_separator: bool = False,
            guess_unknown_words: bool = True,
            output_phonetic_info: bool = False,
            restore_verb_ending: bool = True,
            ignore_derivation_symbol: bool = True,
    ) -> None:
        self.use_context = use_context
        self.use_proper_name_analysis = use_proper_name_analysis
        self.output_compound_separator = output_compound_separator  # doesn't do anything atm
        self.guess_unknown_words = guess_unknown_words
        self.output_phonetic_info = output_phonetic_info
        self.restore_verb_ending = restore_verb_ending
        self.ignore_derivation_symbol = ignore_derivation_symbol
        # ambiguous analyzer
        self._amb_morph_tagger = VabamorfTagger(
            compound=self.output_compound_separator,
            disambiguate=False,
            guess=self.guess_unknown_words,
            slang_lex=False,
            phonetic=self.output_phonetic_info,
            use_postanalysis=True,
            use_reorderer=True,
            propername=self.use_proper_name_analysis,
            predisambiguate=False,
            postdisambiguate=False,

        )
        # disambiguated analyzer
        self._disamb_morph_tagger = VabamorfTagger(
            compound=self.output_compound_separator,
            disambiguate=True,
            guess=self.guess_unknown_words,
            slang_lex=False,
            phonetic=self.output_phonetic_info,
            use_postanalysis=True,
            use_reorderer=True,
            propername=self.use_proper_name_analysis,
            predisambiguate=self.use_context,
            postdisambiguate=self.use_context,
        )
        # workaround taggers to be able with to work with pretokenized input as it's not accepted by `Text` class
        self._whitespace_tagger = WhiteSpaceTokensTagger()
        self._compound_token_tagger = PretokenizedTextCompoundTokensTagger()

    def __call__(
            self, data: Union[str, List[str]]
    ) -> List[VbTokenAnalysis]:
        return self.analyze(data)

    def analyze(self, input_text: Union[List[str], str]) -> List[VbTokenAnalysis]:
        if isinstance(input_text, str):
            ambiguous_analysis = self._ambiguous_str_analysis(input_text)
            disambiguated_analysis = self._disambiguated_str_analysis(input_text)
            # number of tokens must be the same in both analyses
            if len(disambiguated_analysis["morph_analysis"]) != len(ambiguous_analysis["morph_analysis"]):
                raise ValueError("Number of tokens mismatch")
        elif isinstance(input_text, list) and all((isinstance(el, str) and " " not in el for el in input_text)):
            ambiguous_analysis = self._ambiguous_pretokenized_analysis(input_text)
            disambiguated_analysis = self._disambiguated_pretokenized_analysis(input_text)
            # number of tokens must be the same in both analyses and match the number of original tokens
            if not (
                    len(disambiguated_analysis["morph_analysis"]) == len(ambiguous_analysis["morph_analysis"])
                    and len(disambiguated_analysis["morph_analysis"]) == len(input_text)
            ):
                raise ValueError("Number of tokens mismatch")
        else:
            raise ValueError(
                "Input text should may be raw `str` or pretokenized `List[str]` with no whitespaces in each token"
            )

        result = []
        tokens = disambiguated_analysis["morph_analysis"].text
        if self.output_compound_separator:
            disambiguated_lemmas = disambiguated_analysis["morph_analysis"].root
            lemma_candidate_list = ambiguous_analysis["morph_analysis"].root
        else:
            disambiguated_lemmas = disambiguated_analysis["morph_analysis"].lemma
            lemma_candidate_list = ambiguous_analysis["morph_analysis"].lemma
        features_list = disambiguated_analysis["morph_analysis"].form
        pos_list = disambiguated_analysis["morph_analysis"].partofspeech
        ambiguous_pos_list = ambiguous_analysis["morph_analysis"].partofspeech

        for index, (
                token, disambiguated_lemma, lemma_candidates, features, part_of_speech, part_of_speech_candidates
        ) in enumerate(
                zip(tokens, disambiguated_lemmas, lemma_candidate_list, features_list, pos_list, ambiguous_pos_list)
        ):
            disambiguated_lemma: str = disambiguated_lemma[0]
            if self.ignore_derivation_symbol:
                disambiguated_lemma = disambiguated_lemma.replace("=", "")
                lemma_candidates = [lemma_candidate.replace("=", "") for lemma_candidate in lemma_candidates]
            part_of_speech: str = part_of_speech[0]

            # add `ma` back to verbs
            if self.restore_verb_ending and self.output_compound_separator:
                lemma_candidates = [
                    f"{lemma_candidate}ma" if pos == "V" else lemma_candidate
                    for lemma_candidate, pos in zip(lemma_candidates, part_of_speech_candidates)
                ]
                if part_of_speech == "V":
                    disambiguated_lemma = f"{disambiguated_lemma}ma"

            lemma_candidates: List[str] = sorted(lemma_candidates)
            result.append(
                VbTokenAnalysis(
                    index=index + 1,  # 1-based
                    token=token,
                    disambiguated_lemma=disambiguated_lemma,
                    lemma_candidates=tuple(lemma_candidates),
                    features=features[0],
                    part_of_speech=part_of_speech[0],
                    candidate_parts_of_speech=tuple(part_of_speech_candidates),
                )
            )
        return result

    def _ambiguous_str_analysis(self, raw_text: str) -> Text:
        text = Text(raw_text)
        text.tag_layer(self._amb_morph_tagger.input_layers)
        self._amb_morph_tagger.tag(text)
        return text

    def _disambiguated_str_analysis(self, raw_text: str) -> Text:
        text = Text(raw_text)
        text.tag_layer(self._disamb_morph_tagger.input_layers)
        self._disamb_morph_tagger.tag(text)
        return text

    def _ambiguous_pretokenized_analysis(self, tokens: List[str]) -> Text:
        joined_text = " ".join(tokens)
        text = Text(joined_text)
        self._whitespace_tagger.tag(text)
        self._compound_token_tagger(text)
        text.tag_layer(self._amb_morph_tagger.input_layers)
        self._amb_morph_tagger.tag(text)
        return text

    def _disambiguated_pretokenized_analysis(self, tokens: List[str]) -> Text:
        joined_text = " ".join(tokens)
        text = Text(joined_text)
        self._whitespace_tagger.tag(text)
        self._compound_token_tagger(text)
        text.tag_layer(self._amb_morph_tagger.input_layers)
        self._disamb_morph_tagger.tag(text)
        return text


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

    return AdHocModelInput(src, src_mask, lem, lem_mask, orig_idx)


class VabamorfAdHocProcessor:

    def __init__(
            self,
            path: str,
            use_feats: bool = True,
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

        self.use_cuda = use_cuda

        if self.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._init_model(path)

        self.use_pos = self.config["use_pos"]

        self.eos_after = self.config["eos_after"]

        self.analyzer = VbPipeline()

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

    def preprocess_text(self, raw_text: str) -> List[VbTokenAnalysis]:
        return self.analyzer(raw_text)[0]

    def lemmatize_dict(
            self, preprocessed: List[VbTokenAnalysis]
    ) -> List[Union[str, None]]:
        if self.use_pos:
            return [self.composite_dict.get((el.token, el.part_of_speech), None) for el in preprocessed]
        else:
            return [self.word_dict.get((el.token, el.part_of_speech), None) for el in preprocessed]

    def lemmatize(self, input_str: str):
        preprocessed: List[VbTokenAnalysis] = self.preprocess_text(input_str)

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

        # if self.use_dict and self.use_pos:
        #     tmp = []
        #     d_lemmatized = self.lemmatize_dict(preprocessed)
        #     for d_lemma, hypothesis in zip(d_lemmatized, output_seqs):
        #         if not d_lemma:
        #             tmp.append(hypothesis)
        #         else:
        #             tmp.append(d_lemma)
        #     output_seqs = tmp

        return output_seqs


class StanzaPretokenizedAnalyzer:

    def __init__(self):

        self.nlp = stanza.Pipeline(
            lang="et", processors="tokenize,pos", tokenize_pretokenized=True, pos_batch_size=256
        )

    def _analyze(self, tokens: Union[List[List[str]], List[str]]) -> stanza.Document:
        logger.info("Stanza processing...")
        processed = self.nlp(tokens)
        logger.info("Finished Stanza processing...")
        return processed

    def __call__(self, token_analyses: List[List[VbTokenAnalysis]]) -> List[List[VbTokenAnalysis]]:

        if not token_analyses:
            raise RuntimeError("Passed an empty list")

        if len(token_analyses) == 1:
            tokens_to_analyze: List[str] = [el.token for el in token_analyses[0]]
        else:
            tokens_to_analyze: List[List[str]] = []
            for sentence in token_analyses:
                tokens_to_analyze.append([el.token for el in sentence])

        processed = self._analyze(tokens_to_analyze)

        if len(token_analyses) != len(processed.sentences):
            raise RuntimeError("Lengths mismatch")

        reanalyzed_tokens: List[List[VbTokenAnalysis]] = []

        for vb_sentence, stanza_sentence in zip(token_analyses, processed.sentences):
            cur_sentence: List[VbTokenAnalysis] = []
            for vb_token, stanza_token in zip(vb_sentence, stanza_sentence):
                cur_sentence.append(
                    dataclasses.replace(
                        vb_token,
                        part_of_speech=stanza_token.words[0].upos,
                        features=stanza_token.words[0].feats
                    )
                )
            reanalyzed_tokens.append(cur_sentence)

        if len(token_analyses) != len(reanalyzed_tokens):
            raise RuntimeError("Lengths mismatch")

        return reanalyzed_tokens
