#!/bin/bash
#
# Prepare data for training and evaluating lemmatizers. Run as:
#   ./prep_lemma_data.sh TREEBANK
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT).
# This script assumes UDBASE and LEMMA_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

treebank=$1; shift
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

# process treebanks that need cross validation
if [ -d "$UDBASE/${treebank}_XV" ]; then
    src_treebank="${treebank}_XV"
    src_short="${short}_xv"
else
    src_treebank=$treebank
    src_short=$short
fi

echo "src_treebank: $src_treebank"

train_conllu=$UDBASE/$src_treebank/${src_short}-ud-train.conllu
dev_conllu=$UDBASE/$src_treebank/${src_short}-ud-dev.conllu # gold dev
dev_gold_conllu=$UDBASE/$src_treebank/${src_short}-ud-dev.conllu
test_conllu=$UDBASE/$src_treebank/${src_short}-ud-test.conllu # gold dev
test_gold_conllu=$UDBASE/$src_treebank/${src_short}-ud-test.conllu

train_in_file=$LEMMA_DATA_DIR/${short}.train.in.conllu
dev_in_file=$LEMMA_DATA_DIR/${short}.dev.in.conllu
dev_gold_file=$LEMMA_DATA_DIR/${short}.dev.gold.conllu
test_in_file=$LEMMA_DATA_DIR/${short}.test.in.conllu
test_gold_file=$LEMMA_DATA_DIR/${short}.test.gold.conllu

# copy conllu file if exists; otherwise create empty files
if [ -e $train_conllu ]; then
    echo 'copying training data...'
    cp $train_conllu $train_in_file
else
    touch $train_in_file
fi

if [ -e $dev_conllu ]; then
    echo 'copying dev data...'
    cp $dev_conllu $dev_in_file
else
    touch $dev_in_file
fi

if [ -e $dev_gold_conllu ]; then
    cp $dev_gold_conllu $dev_gold_file
else
    touch $dev_gold_file
fi

if [ -e $test_conllu ]; then
    echo 'copying test data...'
    cp $test_conllu $test_in_file
else
    touch $test_in_file
fi

if [ -e $test_gold_conllu ]; then
    cp $test_gold_conllu $test_gold_file
else
    touch $test_gold_file
fi

