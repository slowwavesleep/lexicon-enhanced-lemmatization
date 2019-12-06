import random
from collections import Counter, defaultdict
import lexenlem.models.common.seq2seq_constant as constant


class Lexicon:
    def __init__(self, dropout=0.1):
        self.pos_lexicon = defaultdict(str)
        self.word_lexicon = defaultdict(str)
        self.dropout = dropout

    def init_lexicon(self, data):
        ctr = Counter()
        ctr.update([(d[0], d[1], d[2]) for d in data])
        for entry, _ in ctr.most_common():
            if random.random() > self.dropout:
                word, pos, lemma = entry
                self.pos_lexicon[(word, pos)] += lemma
                self.word_lexicon[word] += lemma

    def lemmatize(self, word, pos):
        if (word, pos) in self.pos_lexicon:
            return self.pos_lexicon[(word, pos)]
        if word in self.word_lexicon:
            return self.word_lexicon[word]
        else:
            return constant.UNK