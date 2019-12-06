## Table of Contents  
[Model Architecture](#methodology)  
[Results](#results)  
[Error Analysys](#errors)  
[ToDo](#todo)  
[References](#references)  


<a name="methodology"></a>
## Model Architecture

The model is a Seq2Seq decoder with the following general structure:

![Overall](img/StanfordNLP_Lemmatizer-Overall.jpg)

#### Encoder

![Encoder](img/StanfordNLP_Lemmatizer-Encoder.jpg)

##### Transform

![Transform](img/StanfordNLP_Lemmatizer-Transform.jpg)

#### Decoder

![Decoder](img/StanfordNLP_Lemmatizer-Decoder.jpg)

##### Attention

**@** - batch matrix-matrix product (https://pytorch.org/docs/stable/torch.html?highlight=torch%20bmm#torch.bmm)

![Attention](img/StanfordNLP_Lemmatizer-Attention.jpg)

### Modified Architecture

To introduce the inputs from the rule-based lemmatizer, we added the second encoder and combined the outputs of two encoders before passing them into the decoder. The decoder sctructure is also modified to icorporate new inputs.

![Overall](img/StanfordNLP_Lemmatizer-Overall_Modified.jpg)

#### Decoder

![Decoder](img/StanfordNLP_Lemmatizer-Decoder_Modified.jpg)

#### Performance

The modified model gives **97.75 score** on **test set with gold morphology** and **96.67 score** on **test set with predicted morphology**.


<a name="results"></a>
## Results

### Estonian

All the models are trained with [Estonian UD v2.4 treebank](https://github.com/UniversalDependencies/UD_Estonian-EDT).

All the scores are measured with the [official CoNLL 2018 evaluation script](http://universaldependencies.org/conll18/evaluation.html).

#### TurkuNLP

[Model repository](https://github.com/jmnybl/universal-lemmatizer/tree/9bc90f81965f2c577e58d0319ba9066641d0e605)

The OPENNMT component in the model was replaced to the newest vestion from [the official repository](https://github.com/OpenNMT/OpenNMT-py/tree/master) to fix the compatibility issues.

|model     |gold score|pred score|
|----------|----------|----------|
|Original  |97.30     |96.02     |
|+vabamorf |97.43     |96.53     |

#### StanfordNLP
[Model repository](https://github.com/stanfordnlp/stanfordnlp)

The numbers in the table show the order of the features in the input for encoder. All the models are trained without the edit classifier.

|src|pos|feats|vabamorf|gold score|pred score|
|---|---|-----|--------|----------|----------|
| 1 | - |  -  |    -   |  95.41   |  95.41   |
| 1 | 2 |  -  |    -   |  95.42   |    -     |
| 2 | 1 |  -  |    -   |  96.56   |  96.02   |
| 3 | 2 |  1  |    -   |  93.13   |  92.48   |
| 2 | 1 |  3  |    -   |  96.71   |  96.19   |
| 4 | 3 |  2  |    1   |  96.32   |  96.04   |

The model currently constructs the embeddings and vocabularies for input, pos and feats separately and later concatenates them together for input. If we concatenate the raw input and build one dictionary and embedding for it, the **gold morph test score** of the model rises up to **97.50**, however, the **predicted morph test score** is **96.17**.  

Tagger dev scores:

|UPOS |XPOS |UFeats|AllTags|
|-----|-----|------|-------|
|97.18|98.37|95.58 |94.15  |

Scores for the lemmatizers on dev and test set:

|model                  |dev       |test      |
|-----------------------|----------|----------|
|default                |96.91     |96.17     |
|lexicon                |79.89     |81.81     |
|lexicon (dropout 0.1)  |93.67     |93.53     |
|vabamorf (default)     |97.37     |96.64     |
|vabamorf (no guesser)  |97.43     |96.78     |

To test if the difference between the `vabamorf (default)` and `vabamorf (no guesser)` models are significant, several tests were conducted. First, we tried to check if there is a statistically significant differents between the means of the scores using the paired Wilcoxon test. Ten additional models were trained with different seeds and the p-value was 0.063 that means that the difference between the scores is not statistically significant. However, in order for the Wilcoxon test to be credible, we need at least 20 pairs of samples. This is very computationally expensive and not reasonable to do every time to perform the analysis. Paired Student's t-test cannot be performed neither, since the independence requirement is not met.

To solve this problem, the paired bootstrap resampling test was used. This test was used in the official [CoNLL-2018 shared task](https://universaldependencies.org/conll18/proceedings/pdf/K18-2001.pdf) to rank the systems. Also, another papers, for example [(Philipp Koehn, 2004)](https://www.aclweb.org/anthology/W04-3250.pdf) use this method to compare machine translation systems. There exists an official evaluation script in the [UDAPI](https://github.com/udapi/udapi-python/blob/master/udapi/block/eval/conll18.py) library which was [revised and reimplemented for our library](https://github.com/501Good/lexicon-enhanced-lemmatization/blob/master/lexenlem/utils/paired_bootstrap.py). Our script was compared with the official script and ensured to produce similar results.

In short, the sentenced for each system gold and predicted are resampled with replacement 1000 times, and the score of each resample is taken. Then, it is counted how many times one system perfomed better or worse than another. Later, the scores for each system are sorted by the middle score. Finally, the p-value is calculated as `number of wins / number of resamples` and 95% confidence interval is taken to calculate the score deviation from the average. 

Results are the following:

```
 1.      Vabamorf no Disambiguator 96.85 ± 0.17 (96.68 .. 97.02) p=0.027
------------------------------------------------------------------------
 2.               Vabamorf Default 96.59 ± 0.19 (96.41 .. 96.78) p=0.001
------------------------------------------------------------------------
 3.                        Default 95.99 ± 0.20 (95.78 .. 96.18) p=0.001
------------------------------------------------------------------------
 4.                    Lexicon 010 94.66 ± 0.23 (94.43 .. 94.89)
```

The line between the systems shows that there is a statistically significant difference between the two systems. We can also see that the `vabamorf (default)` and `vabamorf (no guesser)` models perform significantly different. 

#### Vabamorf

Vabamorf and lexicon scores on dev set:

|lexicon   |vabamorf |
|----------|---------|
|83.77     |86.14    |

If we take all predictions from Vabamorf, they contain the correct lemma in about 87% of cases. 

### Russian

Models for Russian were trained on [UD SynTagRus v2.4](https://github.com/UniversalDependencies/UD_Russian-SynTagRus/tree/master). The corpus was split in two: the first one is original containing over 1m tokens, the second one was reduced to the size of Estonian UD corpus. For each corpus three models were trained: without any additional information, adding lemmas from the lexicon that was built of the training data, adding lemmas from [Pymorphy2](https://github.com/kmike/pymorphy2), a rule-based morphological analyser for Russian and Ukranian. In total, _six_ models were trained.

Scores for the lexicon and Pymorphy2 on the dev set:

|corpus    |lexicon   |pymorphy2|
|----------|----------|---------|
|full      |92.28     |89.94    |
|small     |88.17     |89.55    |

Scores for the lemmatizers on dev and test set:

|model                        |dev       |test      |
|-----------------------------|----------|----------|
|full                         |99.10     |97.60     |
|full_lexicon                 |90.94     |90.12     |
|full_lexicon (dropout 0.1)   |98.14     |97.03     |
|full_pymorphy                |99.43     |97.15     |
|small                        |97.81     |96.63     |
|small_lexicon                |86.39     |85.98     |
|small_lexicon (dropout 0.1)  |97.10     |96.12     |
|small_pymorphy               |99.20     |97.55     |

The decrease in performace with Pymorphy2 can be caused by the poor performance of the rule-based lemmatizer. Currently, the model gets all the outputs from Pymorphy2, which contain the correct lemma only in about 92% of cases. Another reason may be that Pymorphy generates on average more predictions for each word than Vabamorf.

<a name="errors"></a>
## Error analysis

### Estonian 

If we compare the system without double attention and vabamorf input (lemma1) with the system with double attention and vabamorf input (lemma2), we can see improvements.

First, the number of erroneous lemmas, i.e. where strings were not equal, are the following:

| lemma1 | lemma2 |
| ------ | ------ |
|  1429  |  1185  |

Lemma2 system **corrected 611 errors**, i.e. where lemma1 was different from gold but lemma2 was correct.

Lemma 2 system **introduced 367 new errors**, i.e. where lemma1 was the same as gold but lemma2 was different.

The result of `errors / total_number_of_lemmas` coincide with the official evaluation script. That means that the official evaluation script simply compares two strings char-by-char.

This gives **96.80 dev score** for lemma1 and **97.34 score** for lemma2.

If we ignore the special symbols `_+=` in lemmas, we have **98.79 dev score** for lemma1 and **99.13 dev score** for lemma2.

Also, this way, lemma2 system **corrected 292 errors** and **introduced 141 new errors**.

### Lemma Description

Kui on tegemist liitmoodustisega, siis:

- Tüvi on eristatud eelnevast komponendist '\_' märgiga;
- Lõpp on eristatud eelnevast komponendist '+' märgiga; nn. null-lõpp ongi '+0'
- Sufiks on eristatud eelnevast komponendist '=' märgiga.

### Lexicon

The issue with the default lexicon is that it contains all the words from the training set, since it was constructed from the training set. Thus, the model doesn't know how to perform when the lemma is not found in the lexicon. We tried to tackle this problem with discarding each word from the lexicon with the probability of 0.1, so that the model trains with unknowns as well. This help the issue, but the model with the lexicon still the model with the outer lemmatizers as well as the default models. Probably, more experiments needed with different word dropout rates.

<a name="todo"></a>
## ToDo
- [x] Check if the order of inputs has any effect on the performance
- [x] Check if replacing rule-based system with the lexicon has any effect on the performance
- [x] Perform error anaysis
- [x] Check if the official evaluation script takes into account the underscore in lemmas
- [x] Test for other languages
- [x] Train the model without disambiguation and guesser
- [x] Analyze the performance of Vabamorf
- [ ] Check if two inputs are aligned
- [x] Introduce dropout to the lexicon
- [ ] Test if results of the default vabamorf and vabamorf without guesser are significantly different

<a name="references"></a>
## References
1. [Shin, Jaehun, and Jong-hyeok Lee. "Multi-encoder Transformer Network for Automatic Post-Editing." Proceedings of the Third Conference on Machine Translation: Shared Task Papers. 2018.](https://www.aclweb.org/anthology/W18-6470.pdf)
2. [Junczys-Dowmunt, Marcin, and Roman Grundkiewicz. "MS-UEdin Submission to the WMT2018 APE Shared Task: Dual-Source Transformer for Automatic Post-Editing." arXiv preprint arXiv:1809.00188 (2018).](https://arxiv.org/pdf/1809.00188.pdf)
