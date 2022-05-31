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
from tqdm.auto import tqdm
from loguru import logger

from lexenlem.models.lemma.data import DataLoaderVbConfig, DataLoaderVb
from lexenlem.models.lemma.trainer import TrainerVb
from lexenlem.models.lemma import scorer, edit
from lexenlem.models.common import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/lemma", help="Directory for all lemma data.")
    parser.add_argument("--unimorph_dir", type=str, default="", help="Directory of unimorph file")
    parser.add_argument("--train_file", type=str, default=None, help="Input file for data loader.")
    parser.add_argument("--eval_file", type=str, default=None, help="Input file for data loader.")
    parser.add_argument("--gold_file", type=str, default=None, help="Output CoNLL-U file.")
    parser.add_argument("--output_file", type=str, default=None, help="Output CoNLL-U file.")

    parser.add_argument("--mode", default="train", choices=["train", "predict"])
    parser.add_argument("--lang", default="et", type=str, help="Language")

    parser.add_argument(
        "--no_dict",
        dest="ensemble_dict",
        action="store_false",
        help="Do not ensemble dictionary with seq2seq. By default use ensemble.",
    )
    parser.add_argument("--dict_only", action="store_true", help="Only train a dictionary-based lemmatizer.")

    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--emb_dim", type=int, default=50)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--emb_dropout", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max_dec_len", type=int, default=50)
    parser.add_argument("--beam_size", type=int, default=1)

    parser.add_argument("--attn_type", default="soft", choices=["soft", "mlp", "linear", "deep"], help="Attention type")
    parser.add_argument(
        "--no_edit",
        dest="edit",
        action="store_false",
        help="Do not use edit classifier in lemmatization. By default use an edit classifier.",
    )
    parser.add_argument(
        '--no_morph',
        dest='morph',
        action='store_false',
        help='Do not use morphological tags as inputs. By default use pos and morphological tags.',
    )
    parser.add_argument(
        "--no_pos",
        dest="pos",
        action="store_false",
        help="Do not use pos tags as inputs. By default use pos and morphological tags.",
    )
    parser.add_argument("--lemmatizer", type=str, default=None, help="Name of the outer lemmatizer function")
    parser.add_argument(
        "--no_pos_lexicon",
        dest="use_pos",
        action="store_false",
        help="Do not use word-pos dictionary in the lexicon",
    )
    parser.add_argument(
        "--no_word_lexicon",
        dest="use_word",
        action="store_false",
        help="Do not use word dictionary in the lexicon"
    )
    parser.add_argument(
        "--lexicon_dropout",
        type=float,
        default=0.8,
        help="Probability to drop the word from the lexicon",
    )
    parser.add_argument(
        "--eos_after",
        action="store_true",
        help="Put <EOS> symbol after all the inputs. Otherwise put it after the end of token",
    )
    parser.add_argument("--num_edit", type=int, default=len(edit.EDIT_TO_ID))
    parser.add_argument("--alpha", type=float, default=1.0)

    parser.add_argument("--sample_train", type=float, default=1.0, help="Subsample training data.")
    parser.add_argument("--optim", type=str, default="adam", help="sgd, adagrad, adam or adamax.")
    parser.add_argument("--lr", type=float, default=1e-3, help='Learning rate')
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--decay_epoch", type=int, default=30, help="Decay the lr starting from this epoch.")
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument(
        "--early_stop",
        type=int,
        default=10,
        help="Stop training if dev score doesn't improve after the specified number of epochs.",
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=10,
        help="Minimum number of epochs to train before early stopping gets applied."
    )
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help="Gradient clipping.")
    parser.add_argument("--log_step", type=int, default=50, help="Print log every k steps.")
    parser.add_argument("--model_dir", type=str, default="./saved_models/lemma", help="Root dir for saving models.")
    parser.add_argument("--log_attn", action="store_true", help="Log attention output.")

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--cpu", action="store_true", help="Ignore CUDA.")
    parser.add_argument("--identity_baseline", action="store_true")
    parser.add_argument("--vabamorf_baseline", action="store_true")
    parser.add_argument("--use_conll_features", action="store_true")
    parser.add_argument("--generate_stanza_features", action="store_true")
    parser.add_argument("--no_vb_context", action="store_true")
    parser.add_argument("--no_proper", action="store_true")
    parser.add_argument("--output_compound_separator", action="store_true")
    parser.add_argument("--no_guess_unknown_words", action="store_true")
    parser.add_argument("--output_phonetic_info", action="store_true")
    parser.add_argument("--no_ignore_derivation_symbol", action="store_true")
    args = parser.parse_args()
    return args


def main():
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)

    args = vars(args)
    if args.get("identity_baseline", False) and args.get("vabamorf_baseline", False):
        raise RuntimeError("vb OR identity")
    logger.info("Running lemmatizer in {} mode".format(args['mode']))

    if args["mode"] == "train":
        train(args)
    else:
        evaluate(args)


def train(args):
    logger.info("[Loading data with batch size {}...]".format(args['batch_size']))
    config = DataLoaderVbConfig(
        morph=args.get("morph", True),
        pos=args.get("pos", True),
        sample_train=args["sample_train"],
        lang="et",
        eos_after=args.get("eos_after", False),
        split_feats=False,
        use_context=not args.get("no_vb_context", False),
        use_proper_name_analysis=not args.get("no_proper", False),
        output_compound_separator=args.get("output_compound_separator", False),
        guess_unknown_words=not args.get("no_guess_unknown_words", False),
        output_phonetic_info=args.get("output_phonetic_info", False),
        ignore_derivation_symbol=not args.get("no_ignore_derivation_symbol", False),
    )
    train_loader = DataLoaderVb(
        input_src=args["train_file"],
        batch_size=args["batch_size"],
        config=config,
        evaluation=False,
        use_conll_features=args.get("use_conll_features", False),
        generate_stanza_features=args.get("generate_stanza_features", False),
    )
    vocab = train_loader.vocab
    args["vocab_size"] = vocab["combined"].size
    dev_loader = DataLoaderVb(
        input_src=args["eval_file"],
        batch_size=args["batch_size"],
        config=config,
        vocab=vocab,
        evaluation=True,
        use_conll_features=args.get("use_conll_features", False),
        generate_stanza_features=args.get("generate_stanza_features", False),
    )
    utils.ensure_dir(args["model_dir"])
    model_file = "{}/{}_lemmatizer.pt".format(args["model_dir"], args["lang"])

    # pred and gold path
    system_pred_file = args["output_file"]
    gold_file = args["gold_file"]

    # utils.print_config(args)

    logger.info("Running with the following configs:")
    for key, value in args.items():
        logger.info(f"{key}: {str(value)}")

    # skip training if the language does not have training or dev data
    if len(train_loader) == 0 or len(dev_loader) == 0:
        logger.info("[Skip training because no data available...]")
        sys.exit(0)

    # start training
    # train a dictionary-based lemmatizer
    trainer = TrainerVb(args=args, vocab=vocab, use_cuda=args['cuda'])

    # if args['lemmatizer'] == 'lexicon':
    #     trainer.lexicon = train_batch.lemmatizer
    # print("[Training dictionary-based lemmatizer...]")
    # trainer.train_dict(train_batch.conll.get(['word', 'upos', 'lemma']))
    # print("Evaluating on dev set...")
    # dev_preds = trainer.predict_dict(dev_batch.conll.get(['word', 'upos']))
    # dev_batch.conll.write_conll_with_lemmas(dev_preds, system_pred_file)
    # _, _, dev_f = scorer.score(system_pred_file, gold_file)
    # print("Dev F1 = {:.2f}".format(dev_f * 100))

    # train a seq2seq model
    logger.info("[Training seq2seq-based lemmatizer...]")
    global_step = 0
    max_steps = len(train_loader) * args["num_epoch"]
    max_dev_steps = len(dev_loader)
    dev_score_history = []
    devs_without_improvements = 0
    best_dev_preds = []
    current_lr = args["lr"]
    global_start_time = time.time()
    format_str = "{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}"
    format_str_dev = "{}: step {}/{} (epoch {}/{}) ({:.3f} sec/batch)"

    # start training
    for epoch in range(1, args["num_epoch"] + 1):
        dev_step = 0
        train_loss = 0
        try:
            for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                start_time = time.time()
                global_step += 1
                loss = trainer.update(batch, evaluate=False)  # update step
                train_loss += loss
                if global_step % args["log_step"] == 0:
                    duration = time.time() - start_time
                    logger.info(
                        format_str.format(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            global_step,
                            max_steps,
                            epoch,
                            args["num_epoch"],
                            loss,
                            duration,
                            current_lr
                        )
                    )
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt detected. Exiting training.")
            sys.exit(0)

        # eval on dev
        logger.info("Evaluating on dev set...")
        dev_preds = []  # predictions for every token
        for i, batch in tqdm(enumerate(dev_loader), total=len(dev_loader)):
            start_time = time.time()
            dev_step += 1
            preds, _ = trainer.predict(batch, 1)
            dev_preds += preds
            if dev_step % args["log_step"] == 0:
                duration = time.time() - start_time
                logger.info(
                    format_str_dev.format(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        dev_step,
                        max_dev_steps,
                        epoch,
                        args["num_epoch"],
                        duration
                    )
                )
        dev_preds = trainer.postprocess(dev_loader.original_tokens, dev_preds)  # list of original forms

        # try ensembling with dict if necessary
        # if args.get('ensemble_dict', False):
        #     print("[Ensembling dict with seq2seq model...]")
        #     dev_preds = trainer.ensemble(dev_batch.conll.get(['word', 'upos']), dev_preds)

        dev_loader.write_to_conll(dev_preds, system_pred_file)
        _, _, dev_score = scorer.score(system_pred_file, gold_file)

        train_loss = train_loss / train_loader.num_examples * args['batch_size']  # avg loss per batch
        logger.info("epoch {}: train_loss = {:.6f}, dev_score = {:.4f}".format(epoch, train_loss, dev_score))

        # save best model
        if epoch == 1 or dev_score > max(dev_score_history):
            trainer.save(model_file)
            logger.info("new best model saved.")
            best_dev_preds = dev_preds

        # early stopping
        if epoch > args['min_epochs']:
            if dev_score <= max(dev_score_history):
                devs_without_improvements += 1
                logger.info("{} epochs since last dev score improvement.".format(devs_without_improvements))
            else:
                devs_without_improvements = 0
        if devs_without_improvements > args['early_stop']:
            logger.info("No dev score improvements for {} epocs. Stopping training...".format(args['early_stop']))
            break

        # lr schedule
        if epoch > args["decay_epoch"] and dev_score <= dev_score_history[-1] and args["optim"] in ["sgd", "adagrad"]:
            current_lr *= args["lr_decay"]
            trainer.update_lr(current_lr)

        dev_score_history += [dev_score]
        # print("")

    logger.info("Training ended with {} epochs.".format(epoch))

    best_f, best_epoch = max(dev_score_history) * 100, np.argmax(dev_score_history) + 1
    logger.info("Best dev F1 = {:.2f}, at epoch = {}".format(best_f, best_epoch))


def evaluate(args):
    # file paths
    system_pred_file = args["output_file"]
    gold_file = args["gold_file"]
    model_file = "{}/{}_lemmatizer.pt".format(args["model_dir"], args["lang"])

    # load model
    use_cuda = args["cuda"] and not args["cpu"]
    trainer = TrainerVb(model_file=model_file, use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab

    for k in args:
        if k.endswith("_dir") or k.endswith("_file") or k in ["shorthand"]:
            loaded_args[k] = args[k]

    logger.info("Loading data with batch size {}...".format(args["batch_size"]))
    config = DataLoaderVbConfig(
        morph=trainer.args.get("morph", True),
        pos=trainer.args.get("pos", True),
        lang="et",
        eos_after=trainer.args.get("eos_after", False),
        split_feats=False,
        use_context=not trainer.args.get("no_vb_context", False),
        use_proper_name_analysis=not trainer.args.get("no_proper", False),
        output_compound_separator=trainer.args.get("output_compound_separator", False),
        guess_unknown_words=not trainer.args.get("no_guess_unknown_words", False),
        output_phonetic_info=trainer.args.get("output_phonetic_info", False),
        ignore_derivation_symbol=not trainer.args.get("no_ignore_derivation_symbol", False),
    )

    dataloader = DataLoaderVb(
        input_src=args["eval_file"],
        batch_size=args["batch_size"],
        config=config,
        vocab=vocab,
        evaluation=True,
        use_conll_features=trainer.args.get("use_conll_features", False),
        generate_stanza_features=args.get("generate_stanza_features", False),
    )

    # skip eval if dev data does not exist
    if len(dataloader) == 0:
        logger.info("Skip evaluation because no dev data is available...")
        logger.info("Lemma score:")
        logger.info("{} ".format(args['lang']))
        sys.exit(0)

    preds = []
    if args["identity_baseline"]:
        preds.extend(dataloader.original_tokens)
    elif args["vabamorf_baseline"]:
        preds.extend([element.disambiguated_lemma for element in dataloader.flat_analysis])
    else:
        for i, batch in tqdm(
                enumerate(dataloader), desc="Running the seq2seq model in `predict` mode...", total=len(dataloader)
        ):
            batch_preds, _ = trainer.predict(batch, args["beam_size"], log_attn=args["log_attn"])
            preds += batch_preds
        preds = trainer.postprocess(dataloader.original_tokens, preds)

    # write to file and score
    dataloader.write_to_conll(preds, system_pred_file)
    if gold_file is not None:
        _, _, score = scorer.score(system_pred_file, gold_file)

        logger.info("Lemma score:")
        logger.info("{} {:.2f}".format(args["lang"], score * 100))


if __name__ == '__main__':
    main()
