import sys
import time
import os

import numpy as np
import torch
from typing import List
from docopt import docopt

from utils import read_corpus, zip_data, write_sents
from vocab import Vocab, VocabEntry
import configuration
from nmt import routine
import paths

from transformer.model import TransformerModel

from configuration import GeneralConfig as general_config
from configuration import TransformerConfig as transformer_config
from configuration import DecodeConfig as decode_config
from configuration import TrainConfig as train_config
from configuration import SupervisionConfig as supervision_config


def train(load_from=None, save_to=None):
    print_file = sys.stderr
    if general_config.printout:
        print_file = sys.stdout

    dev_data_src = read_corpus(paths.dev_source, source='src')
    dev_data_tgt = read_corpus(paths.dev_target, source='tgt')
    dev_data = zip_data(dev_data_src, dev_data_tgt)

    train_data_src = read_corpus(paths.train_source, source='src')
    train_data_tgt = read_corpus(paths.train_target, source='tgt')
    if supervision_config.pos_supervision is not None:
        train_data_src_pos = read_corpus(paths.get_data_path("train", "src", pos=True), source="src", subwords_switch=False)
        train_data_tgt_pos = read_corpus(paths.get_data_path("train", "tgt", pos=True), source="tgt", subwords_switch=False)
        train_data = zip_data(train_data_src, train_data_tgt, train_data_src_pos, train_data_tgt_pos)
        problems = 0
        for i, (src, tgt, src_pos, tgt_pos) in enumerate(train_data):
            if len(src) != len(src_pos):
                if ("&" not in "".join(src) and "-" not in "".join(src) and "cannot" not in "".join(src) and "gonna" not in "".join(src) and "gotta" not in "".join(src)):
                    print(len(src), len(src_pos))
                    print(src)
                    print(src_pos)
                    problems += 1
        print(problems)
    else:
        train_data = zip_data(train_data_src, train_data_tgt)

    train_batch_size = train_config.batch_size
    valid_niter = general_config.valid_niter
    log_every = general_config.log_every
    if save_to is not None:
        model_save_path = save_to
    else:
        model_save_path = paths.model

    max_epoch = train_config.max_epoch

    if general_config.sanity:
        log_every = 1
        train_data = train_data[:150]
        dev_data = dev_data[:150]
        max_epoch = 2
    pretraining = general_config.pretraining
    pretraining_encoder = general_config.pretraining_encoder
    loaded = False
    if load_from is not None:
        try:
            print("Loading from", load_from)
            model = TransformerModel.load(load_from)
            loaded = True
            pretraining = False
            pretraining_encoder = False
        except FileNotFoundError:
            pass
    if not loaded:
        print("No loading file provided or found : training from scratch")
        print("Loading Transformer Model")
        model = TransformerModel(embedding_rank=transformer_config.embedding_rank,
                                 inner_rank=transformer_config.inner_rank,
                                 ffward_rank=transformer_config.ffward_rank,
                                 pos_supervision=supervision_config.pos_supervision)

    if general_config.cuda:
        model.to_gpu()
    else:
        print("No cuda support")

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    lr = train_config.lr
    max_patience = train_config.patience
    max_num_trial = train_config.max_num_trial
    lr_decay = train_config.lr_decay

    model = routine.train_model(model, train_data, dev_data, model_save_path,
                                train_batch_size, valid_niter, log_every, max_epoch, lr, max_patience, max_num_trial,
                                lr_decay, pos_supervision=supervision_config.pos_supervision is not None)
    model.to_cpu()


def decode(load_from=None):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """

    if general_config.test:
        data_src = read_corpus(paths.test_source, source='src')
        data_tgt = read_corpus(paths.test_target, source='tgt')
        data_tgt_path = paths.test_target
    else:
        data_src = read_corpus(paths.dev_source, source='src')
        data_tgt = read_corpus(paths.dev_target, source='tgt')
        data_tgt_path = paths.dev_target

    print(f"load model from {paths.model}", file=sys.stderr)
    if load_from is not None:
        model_load_path = load_from
    else:
        model_load_path = paths.model

    model = TransformerModel.load(model_load_path)
    if general_config.cuda:
        model.to_gpu()
    model.eval()
    max_step = decode_config.max_decoding_time_step
    if general_config.sanity:
        max_step = 2

    hypotheses = routine.batch_beam_search(model, data_src, max_step, batch_size=decode_config.batch_size,
                                           replace=decode_config.replace)

    lines = []
    for src_sent, hyps in zip(data_src, hypotheses):
        top_hyp = hyps[0]
        lines.append(top_hyp.value)
    write_sents(lines, paths.decode_output)

    bleu_command = "perl scripts/multi-bleu.perl " + data_tgt_path + " < " + paths.decode_output
    os.system(bleu_command)
