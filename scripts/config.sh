#!/bin/bash
#
# Set environment variables for the training and testing of stanfordnlp modules.

# Set UDBASE to the location of CoNLL18 folder 
# For details, see http://universaldependencies.org/conll18/data.html
export UDBASE=./data/treebanks

# Set directories to store processed training/evaluation files
export DATA_ROOT=./data
export LEMMA_DATA_DIR=$DATA_ROOT/lemma
export POS_DATA_DIR=$DATA_ROOT/pos

# Set directories to store external word vector data
export WORDVEC_DIR=./extern_data/word2vec

# Set a directory for the results
export RESULTS_DIR=./results
