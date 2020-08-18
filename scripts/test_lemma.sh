#!/bin/bash
#
# Train and evaluate lemmatizer. Run as:
#   ./run_lemma.sh TREEBANK OTHER_ARGS
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT) and OTHER_ARGS are additional training arguments (see lemmatizer code) or empty.
# This script assumes UDBASE and LEMMA_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

treebank=$1; shift
args=$@
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

test_eval_file=${LEMMA_DATA_DIR}/${short}.test.in.conllu
test_output_file=${LEMMA_DATA_DIR}/${short}.test.pred.conllu
test_gold_file=${LEMMA_DATA_DIR}/${short}.test.gold.conllu

echo "Running lemmatizer with $args..."
python -m lexenlem.models.lemmatizer_cmb --data_dir $LEMMA_DATA_DIR --eval_file $test_eval_file \
        --output_file $test_output_file --gold_file $test_gold_file --lang $short --mode predict $args