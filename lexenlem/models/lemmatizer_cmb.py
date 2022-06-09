"""
Entry point for training and evaluating a lemmatizer.

This lemmatizer combines a neural sequence-to-sequence architecture with an `edit` classifier 
and two dictionaries to produce robust lemmas from word forms.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

import sys
import time
from datetime import datetime
import argparse
import numpy as np
import random
import torch
import importlib

from lexenlem.models.lemma.data import DataLoaderCombined
from lexenlem.models.lemma.trainer import TrainerCombined
from lexenlem.models.lemma import scorer, edit
from lexenlem.models.common import utils
from lexenlem.models.common.lexicon import ExtendedLexicon


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/lemma', help='Directory for all lemma data.')
    parser.add_argument('--unimorph_dir', type=str, default='', help='Directory of unimorph file')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_argument('--gold_file', type=str, default=None, help='Output CoNLL-U file.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')

    parser.add_argument('--no_dict', dest='ensemble_dict', action='store_false',
                        help='Do not ensemble dictionary with seq2seq. By default use ensemble.')
    parser.add_argument('--dict_only', action='store_true', help='Only train a dictionary-based lemmatizer.')

    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--emb_dim', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--emb_dropout', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--max_dec_len', type=int, default=50)
    parser.add_argument('--beam_size', type=int, default=1)

    parser.add_argument('--attn_type', default='soft', choices=['soft', 'mlp', 'linear', 'deep'], help='Attention type')
    parser.add_argument('--no_edit', dest='edit', action='store_false',
                        help='Do not use edit classifier in lemmatization. By default use an edit classifier.')
    parser.add_argument('--no_morph', dest='morph', action='store_false',
                        help='Do not use morphological tags as inputs. By default use pos and morphological tags.')
    parser.add_argument('--no_pos', dest='pos', action='store_false',
                        help='Do not use pos tags as inputs. By default use pos and morphological tags.')
    parser.add_argument('--lemmatizer', type=str, default=None, help='Name of the outer lemmatizer function')
    parser.add_argument('--no_pos_lexicon', dest='use_pos', action='store_false',
                        help='Do not use word-pos dictionary in the lexicon')
    parser.add_argument('--no_word_lexicon', dest='use_word', action='store_false',
                        help='Do not use word dictionary in the lexicon')
    parser.add_argument('--lexicon_dropout', type=float, default=0.8,
                        help='Probability to drop the word from the lexicon')
    parser.add_argument('--eos_after', action='store_true',
                        help='Put <EOS> symbol after all the inputs. Otherwise put it after the end of token')
    parser.add_argument('--num_edit', type=int, default=len(edit.EDIT_TO_ID))
    parser.add_argument('--alpha', type=float, default=1.0)

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--decay_epoch', type=int, default=30, help="Decay the lr starting from this epoch.")
    parser.add_argument('--num_epoch', type=int, default=60)
    parser.add_argument('--early_stop', type=int, default=10,
                        help="Stop training if dev score doesn't improve after the specified number of epochs.")
    parser.add_argument('--min_epochs', type=int, default=10,
                        help="Minimum number of epochs to train before early stopping gets applied.")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--model_dir', type=str, default='saved_models/lemma', help='Root dir for saving models.')
    parser.add_argument('--log_attn', action='store_true', help='Log attention output.')

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)

    args = vars(args)
    print("Running lemmatizer in {} mode".format(args['mode']))

    # manually correct for training epochs
    if args['lang'] in ['cs_pdt', 'ru_syntagrus', 'de_hdt']:
        args['num_epoch'] = 30

    if args['mode'] == 'train':
        train(args)
    else:
        evaluate(args)


def train(args):
    # load data
    print("[Loading data with batch size {}...]".format(args['batch_size']))
    if args['lemmatizer'] is None:
        lemmatizer = None
    elif args['lemmatizer'] == 'lexicon':
        print("[Using the lexicon...]")
        lemmatizer = 'lexicon'
    else:
        print(f"[Loading the {args['lemmatizer']} lemmatizer...]")
        lemmatizer = importlib.import_module('lexenlem.lemmatizers.' + args['lemmatizer'])
    train_batch = DataLoaderCombined(
        args['train_file'], args['batch_size'], args, lemmatizer=lemmatizer, evaluation=False
    )
    vocab = train_batch.vocab
    if args['lemmatizer'] == 'lexicon':
        lemmatizer = train_batch.lemmatizer
    args['vocab_size'] = vocab['combined'].size
    dev_batch = DataLoaderCombined(
        args['eval_file'], args['batch_size'], args, lemmatizer=lemmatizer, vocab=vocab, evaluation=True
    )

    utils.ensure_dir(args['model_dir'])
    model_file = '{}/{}_lemmatizer.pt'.format(args['model_dir'], args['lang'])

    # pred and gold path
    system_pred_file = args['output_file']
    gold_file = args['gold_file']

    utils.print_config(args)

    # skip training if the language does not have training or dev data
    if len(train_batch) == 0 or len(dev_batch) == 0:
        print("[Skip training because no data available...]")
        sys.exit(0)

    # start training
    # train a dictionary-based lemmatizer
    trainer = TrainerCombined(args=args, vocab=vocab, use_cuda=args['cuda'])
    if args['lemmatizer'] == 'lexicon':
        trainer.lexicon = train_batch.lemmatizer
    print("[Training dictionary-based lemmatizer...]")
    trainer.train_dict(train_batch.conll.get(['word', 'upos', 'lemma']))
    print("Evaluating on dev set...")
    dev_preds = trainer.predict_dict(dev_batch.conll.get(['word', 'upos']))
    dev_batch.conll.write_conll_with_lemmas(dev_preds, system_pred_file)
    _, _, dev_f = scorer.score(system_pred_file, gold_file)
    print("Dev F1 = {:.2f}".format(dev_f * 100))

    if args.get('dict_only', False):
        # save dictionaries
        trainer.save(model_file)
    else:
        # train a seq2seq model
        print("[Training seq2seq-based lemmatizer...]")
        global_step = 0
        max_steps = len(train_batch) * args['num_epoch']
        max_dev_steps = len(dev_batch)
        dev_score_history = []
        devs_without_improvements = 0
        best_dev_preds = []
        current_lr = args['lr']
        global_start_time = time.time()
        format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
        format_str_dev = '{}: step {}/{} (epoch {}/{}) ({:.3f} sec/batch)'

        # start training
        for epoch in range(1, args['num_epoch'] + 1):
            dev_step = 0
            train_loss = 0
            for i, batch in enumerate(train_batch):
                start_time = time.time()
                global_step += 1
                loss = trainer.update(batch, eval=False)  # update step
                train_loss += loss
                if global_step % args['log_step'] == 0:
                    duration = time.time() - start_time
                    print(
                        format_str.format(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            global_step,
                            max_steps,
                            epoch,
                            args['num_epoch'],
                            loss,
                            duration,
                            current_lr
                        )
                    )

            # eval on dev
            print("Evaluating on dev set...")
            dev_preds = []
            dev_edits = []
            for i, batch in enumerate(dev_batch):
                start_time = time.time()
                dev_step += 1
                preds, edits, _ = trainer.predict(batch, args['beam_size'])
                dev_preds += preds
                if edits is not None:
                    dev_edits += edits
                if dev_step % args['log_step'] == 0:
                    duration = time.time() - start_time
                    print(
                        format_str_dev.format(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            dev_step,
                            max_dev_steps,
                            epoch,
                            args['num_epoch'],
                            duration
                        )
                    )
            dev_preds = trainer.postprocess(dev_batch.conll.get(['word']), dev_preds, edits=dev_edits)

            # try ensembling with dict if necessary
            if args.get('ensemble_dict', False):
                print("[Ensembling dict with seq2seq model...]")
                dev_preds = trainer.ensemble(dev_batch.conll.get(['word', 'upos']), dev_preds)
            dev_batch.conll.write_conll_with_lemmas(dev_preds, system_pred_file)
            _, _, dev_score = scorer.score(system_pred_file, gold_file)

            train_loss = train_loss / train_batch.num_examples * args['batch_size']  # avg loss per batch
            print("epoch {}: train_loss = {:.6f}, dev_score = {:.4f}".format(epoch, train_loss, dev_score))

            # save best model
            if epoch == 1 or dev_score > max(dev_score_history):
                trainer.save(model_file)
                print("new best model saved.")
                best_dev_preds = dev_preds

            # early stopping
            if epoch > args['min_epochs']:
                if dev_score <= max(dev_score_history):
                    devs_without_improvements += 1
                    print("{} epochs since last dev score improvement.".format(devs_without_improvements))
                else:
                    devs_without_improvements = 0
            if devs_without_improvements > args['early_stop']:
                print("No dev score improvements for {} epocs. Stopping training...".format(args['early_stop']))
                break

            # lr schedule
            if epoch > args['decay_epoch'] and dev_score <= dev_score_history[-1] and \
                    args['optim'] in ['sgd', 'adagrad']:
                current_lr *= args['lr_decay']
                trainer.update_lr(current_lr)

            dev_score_history += [dev_score]
            print("")

        print("Training ended with {} epochs.".format(epoch))

        best_f, best_epoch = max(dev_score_history) * 100, np.argmax(dev_score_history) + 1
        print("Best dev F1 = {:.2f}, at epoch = {}".format(best_f, best_epoch))


def evaluate(args):
    # file paths
    system_pred_file = args['output_file']
    gold_file = args['gold_file']
    model_file = '{}/{}_lemmatizer.pt'.format(args['model_dir'], args['lang'])

    # load model
    use_cuda = args['cuda'] and not args['cpu']
    trainer = TrainerCombined(model_file=model_file, use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab

    for k in args:
        if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand']:
            loaded_args[k] = args[k]

    print("[Loading the outer lemmatizer...]")
    if args['lemmatizer'] != loaded_args['lemmatizer'] and loaded_args['lemmatizer'] is None:
        loaded_args['lemmatizer'] = args['lemmatizer']
    if loaded_args['lemmatizer'] == 'lexicon' and args['lemmatizer'] not in ['lexicon', None]:
        loaded_args['lemmatizer'] = 'lexicon_extended'
    if loaded_args['lemmatizer'] is None:
        lemmatizer = None
    elif loaded_args['lemmatizer'] == 'lexicon':
        print("[Using the lexicon...]")
        lemmatizer = trainer.lexicon
    elif loaded_args['lemmatizer'] == 'lexicon_extended':
        print(f"[Loading the Lexicon extended with the {args['lemmatizer']} lemmatizer...]")
        lemmatizer = ExtendedLexicon(trainer.lexicon,
                                     importlib.import_module('lexenlem.lemmatizers.' + args['lemmatizer']))
    else:
        print(f"[Loading the {loaded_args['lemmatizer']} lemmatizer...]")
        lemmatizer = importlib.import_module('lexenlem.lemmatizers.' + loaded_args['lemmatizer'])

    # laod data
    print("Loading data with batch size {}...".format(args['batch_size']))
    batch = DataLoaderCombined(args['eval_file'], args['batch_size'], loaded_args, lemmatizer=lemmatizer, vocab=vocab,
                               evaluation=True)

    # skip eval if dev data does not exist
    if len(batch) == 0:
        print("Skip evaluation because no dev data is available...")
        print("Lemma score:")
        print("{} ".format(args['lang']))
        sys.exit(0)

    dict_preds = trainer.predict_dict(batch.conll.get(['word', 'upos']))

    if loaded_args.get('dict_only', False):
        preds = dict_preds
    else:
        print("Running the seq2seq model...")
        preds = []
        edits = []
        log_attns = {}
        for i, b in enumerate(batch):
            ps, es, attns = trainer.predict(b, args['beam_size'], log_attn=args['log_attn'])
            if attns:
                for k, _ in attns.items():
                    if k in log_attns:
                        if k == 'attns':
                            if log_attns[k].shape[0] > attns[k].shape[0]:
                                attns[k] = np.vstack((attns[k], np.zeros(
                                    (log_attns[k].shape[0] - attns[k].shape[0], attns[k].shape[1], attns[k].shape[2]))))
                            else:
                                log_attns[k] = np.vstack((log_attns[k], np.zeros((
                                                                                 attns[k].shape[0] - log_attns[k].shape[
                                                                                     0], log_attns[k].shape[1],
                                                                                 log_attns[k].shape[2]))))
                            log_attns[k] = np.concatenate([log_attns[k], attns[k]], axis=2)
                        elif k == 'all_hyp':
                            log_attns[k] = np.concatenate([log_attns[k], attns[k]], axis=0)
                        else:
                            if log_attns[k].shape[1] > attns[k].shape[1]:
                                attns[k] = np.hstack((attns[k], np.zeros(
                                    (attns[k].shape[0], log_attns[k].shape[1] - attns[k].shape[1]))))
                            else:
                                log_attns[k] = np.hstack((log_attns[k], np.zeros(
                                    (log_attns[k].shape[0], attns[k].shape[1] - log_attns[k].shape[1]))))
                            log_attns[k] = np.concatenate([log_attns[k], attns[k]], axis=0)
                    else:
                        log_attns[k] = attns[k]
            preds += ps
            if es is not None:
                edits += es
        if args['log_attn']:
            lem_name = loaded_args['lemmatizer'] if loaded_args['lemmatizer'] is not None else 'nolexicon'
            fname = ''.join([args['lang'], '_', lem_name])
            print(f'[Logging attention to {fname}.npz...]')
            np.savez(fname, **log_attns)
        preds = trainer.postprocess(batch.conll.get(['word']), preds, edits=edits)

        if loaded_args.get('ensemble_dict', False):
            print("[Ensembling dict with seq2seq lemmatizer...]")
            preds = trainer.ensemble(batch.conll.get(['word', 'upos']), preds)

    # write to file and score
    batch.conll.write_conll_with_lemmas(preds, system_pred_file)
    if gold_file is not None:
        _, _, score = scorer.score(system_pred_file, gold_file)

        print("Lemma score:")
        print("{} {:.2f}".format(args['lang'], score * 100))


if __name__ == '__main__':
    main()
