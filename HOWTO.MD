To reproduce the results from the paper:

1. Requirements

Python >= 3.6

Pytorch

Numpy


2. Data
a) Close the UD v2.5 treebank repository for your language from https://universaldependencies.org/ and place it into data/treebanks folder
b) For Unimoph model, download the data for your language from https://unimorph.github.io/ and place it into extern_data/unimorph/FULL_LANGUAGE_NAME/ISO639-2_CODE.txt. For example, for English the path should look like extern_data/unimorph/English/eng.txt
c) For Apertium model, download and install the following Python Package: https://github.com/apertium/apertium-python

3. Training 
(All examples are given for UD_English-EWT treebank; to train on other languages replace the name of the treebank with the corresponding one)
a) To train the Default model, run: 
	bash scripts/train_lemma.sh UD_English-EWT --cuda True --no_edit --no_dict
b) To train the Lexicon model, run: 
	bash scripts/train_lemma.sh UD_English-EWT --cuda True --lemmatizer lexicon --lexicon_dropout 0.8 --no_edit --no_dict
c) To train the Unimorph model, run:
	bash scripts/train_lemma.sh UD_English-EWT --cuda True --lemmatizer lexicon --lexicon_dropout 0.8 --no_edit --no_dict --unimorph_dir extern_data/unimorph/English/eng.txt
d) To train the Apertium model, run:
	bash scripts/train_lemma.sh UD_English-EWT --cuda True --lemmatizer apertium --no_edit --no_dict
   or use a pregenerated Apertium candidates to save time:
	bash scripts/train_lemma.sh UD_English-EWT --cuda True --lemmatizer apertium_pretrained_eng --no_edit --no_dict
Please, refer to lexenlem/lemmarizer_cmb.py for additional arguments.

4. Testing
***
NB! In the paper, all the tests were performed on the test files with POS and FEATS predicted with the latest Stanza model.
***
a) To test your model, run:
	bash scripts/test_lemma.sh UD_English-EWT --batch_size BATCH_SIZE --cuda True --model_dir PATH_TO_THE_MODEL_FOLDER --no_edit --no_dict --output_file OUTPUT_FILE_NAME
   for example:
	bash scripts/test_lemma.sh UD_English-EWT --batch_size 5000 --cuda True --model_dir saved_models/en_apertium_default --no_edit --no_dict --output_file ./data/lemma/en_ewt.apertium.test.pred.conllu