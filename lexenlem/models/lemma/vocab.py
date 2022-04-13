from collections import Counter

from lexenlem.models.common.vocab import BaseVocab, BaseMultiVocab, CompositeVocab
from lexenlem.models.common.seq2seq_constant import VOCAB_PREFIX


class Vocab(BaseVocab):
    def build_vocab(self):
        counter = Counter(self.data)
        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w: i for i, w in enumerate(self._id2unit)}


class MultiVocab(BaseMultiVocab):
    @classmethod
    def load_state_dict(cls, state_dict):
        new = cls()
        for k, v in state_dict.items():
            if k == 'feats':
                new[k] = FeatureVocab.load_state_dict(v)
            else:
                new[k] = Vocab.load_state_dict(v)
        return new


class FeatureVocab(CompositeVocab):
    def __init__(self, data=None, lang="", idx=0, sep="|", keyed=True):
        super().__init__(data, lang, idx=idx, sep=sep, keyed=keyed)
