# coding=utf-8

"""
creates the bag-of-words count vectors

Usage:
    bag_of_words.py binary
    bag_of_words.py tfidf
    bag_of_words.py count

Options:
    -h --help                               show this screen.
"""

from itertools import chain
from collections import Counter
import pickle
import json
from docopt import docopt

import configuration
from utils import read_corpus, write_sents
import paths
from vocab import Vocab, VocabEntry

if __name__ == '__main__':
    args = docopt(__doc__)

    vconfig = configuration.VocabConfig()
    assert not vconfig.subwords_source
    assert not vconfig.subwords_target
    vocab = pickle.load(open(paths.get_vocab_path(pos=False), 'rb'))
    vocab_length = len(vocab.tgt)

    if args["binary"]:

        for chunk in ["train", "test", "valid"]:

            tg = paths.get_data_path(chunk, "tgt", pos=False)
            print('read in target sentences: %s' % tg)

            # both in src mode since we'll add the start/stop tokens later
            tgt_sents = read_corpus(tg, source='tgt')
            tf_idf_corpus = []

            for sentence in tgt_sents:
                sentence_tf_idf = {vocab.tgt[k]: 1 for k in sentence}
                tf_idf_corpus.append(sentence_tf_idf)

            tg_out = paths.get_tf_idf_vector_path(chunk)
            json.dump(tf_idf_corpus, open(tg_out, 'w'), indent=2)

    if args["tfidf"]:

        for chunk in ["train", "test", "valid"]:

            tg = paths.get_data_path(chunk, "tgt", pos=False)
            print('read in target sentences: %s' % tg)

            # both in src mode since we'll add the start/stop tokens later
            tgt_sents = read_corpus(tg, source='tgt')
            corpus_counts = Counter(chain(*tgt_sents))
            total_count = sum(corpus_counts.values())
            tf_idf_corpus = []

            for sentence in tgt_sents:
                sentence_tf_idf = {vocab.tgt[k]: (v / len(sentence) / corpus_counts[k]) for k, v in
                                   Counter(sentence).items()}
                tf_idf_corpus.append(sentence_tf_idf)

            tg_out = paths.get_tf_idf_vector_path(chunk)
            json.dump(tf_idf_corpus, open(tg_out, 'w'), indent=2)

    if args["count"]:

        for chunk in ["train", "test", "valid"]:

            tg = paths.get_data_path(chunk, "tgt", pos=False)
            print('read in target sentences: %s' % tg)

            # both in src mode since we'll add the start/stop tokens later
            tgt_sents = read_corpus(tg, source='tgt')
            tf_idf_corpus = []

            for sentence in tgt_sents:
                sentence_tf_idf = {vocab.tgt[k]: (v / len(sentence)) for k, v in
                                   Counter(sentence).items()}
                tf_idf_corpus.append(sentence_tf_idf)

            tg_out = paths.get_tf_idf_vector_path(chunk)
            json.dump(tf_idf_corpus, open(tg_out, 'w'), indent=2)
