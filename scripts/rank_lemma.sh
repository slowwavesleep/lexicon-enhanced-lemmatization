#!/bin/bash
#
# Rank different lemmatizers. Run as:
#   ./rank_lemma.sh TREEBANK OTHER_ARGS
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT) and OTHER_ARGS are additional training arguments (see paired_bootstrap.py code) or empty.
# This script assumes UDBASE, LEMMA_DATA_DIR, and RESULTS_DIR are correctly set in config.sh.

source scripts/config.sh

treebank=$1; shift
args=$@
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

gold_file=${LEMMA_DATA_DIR}/${short}.test.gold.conllu
results_dir=${RESULTS_DIR}/${treebank}

echo "Running systems ranking with $args..."
python -m lexenlem.utils.paired_bootstrap --gold_file $gold_file --systems $results_dir $args