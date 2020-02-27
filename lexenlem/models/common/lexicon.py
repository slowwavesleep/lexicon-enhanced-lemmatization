import random
from collections import Counter, defaultdict
import lexenlem.models.common.seq2seq_constant as constant


class Lexicon:
    def __init__(self, unimorph=False):
        #self.pos_lexicon = defaultdict(str)
        self.word_lexicon = defaultdict(str)
        self.unimorph_lexicon = defaultdict(str)
        self.unimorph = unimorph

    def init_unimorph(self):
        ctr = Counter()
        with open(self.unimorph, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    line = line.split('\t')
                    lemma, word = line[0], line[1].split()[-1]
                    ctr[(lemma, word)] += 1
        
        for entry, _ in ctr.most_common():
            lemma, word = entry
            self.unimorph_lexicon[word] += lemma
        print(self.unimorph_lexicon)

    def init_lexicon(self, data):
        ctr = Counter()
        ctr.update([(d[0], d[1], d[2]) for d in data])
        for entry, _ in ctr.most_common():
            word, pos, lemma = entry
            #self.pos_lexicon[(word, pos)] += lemma
            self.word_lexicon[word] += lemma
        if self.unimorph:
            print("[Loading the Unimorph lexicon...]")
            self.init_unimorph()

    def lemmatize(self, word, pos):
        #if (word, pos) in self.pos_lexicon:
        #    return list(self.pos_lexicon[(word, pos)])
        if self.unimorph and word in self.unimorph_lexicon:
            return list(self.unimorph_lexicon[word])
        if word in self.word_lexicon:
            return list(self.word_lexicon[word])
        return []