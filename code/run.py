# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    run.py train [options]
    run.py decode [options]

Options:
    -h --help                               show this screen.
"""

import warnings
warnings.filterwarnings("ignore")

from docopt import docopt
import numpy as np
import torch
import configuration
from nmt import nmt
from transformer import transformer
from positionless import positionless
from vocab import Vocab, VocabEntry
import paths

gconfig = configuration.GeneralConfig()
cconfig = configuration.CompressionConfig()


def lstm_script(args):
    raise NotImplementedError
    # if args['train']:
    #     load_from = paths.model if gconfig.load else None
    #     nmt.train(load_from)
    #     load_from = paths.model
    #     nmt.decode(load_from)
    # elif args['decode']:
    #     load_from = paths.model
    #     nmt.decode(load_from)
    # else:
    #     raise RuntimeError(f'invalid command')


def transformer_script(args):
    if gconfig.mode == "normal":
        if args['train']:
            load_from = paths.model if gconfig.load else None
            transformer.train(load_from)
            load_from = paths.model
            transformer.decode(load_from)
        elif args['decode']:
            load_from = paths.model
            transformer.decode(load_from)
        else:
            raise RuntimeError(f'invalid command')
    elif gconfig.mode == "parts_of_speech":
        if args['train']:
            load_from = paths.model if gconfig.load else None
            transformer.train(load_from)
            load_from = paths.model
            transformer.decode(load_from)
        elif args['decode']:
            load_from = paths.model
            transformer.decode(load_from)
        else:
            raise RuntimeError(f'invalid command')
    elif gconfig.mode == "positionless":
        if args['train']:
            load_from = paths.model if gconfig.load else None
            positionless.train(load_from)
            load_from = paths.model
            positionless.decode(load_from)
        elif args['decode']:
            load_from = paths.model
            positionless.decode(load_from)
        else:
            raise RuntimeError(f'invalid command')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(gconfig.seed)
    np.random.seed(seed * 13 // 7)
    torch.manual_seed(seed * 13 // 7)
    if gconfig.model == "transformer":
        transformer_script(args)
    if gconfig.model == "lstm":
        lstm_script(args)


if __name__ == '__main__':
    main()
