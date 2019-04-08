import paths
import utils

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


DECODE_FILE_PATH = "results/decode.en.test.txt"


if __name__ == '__main__':

    test_target = paths.get_data_path("test", "tgt", pos=True)

    tgt_sents = utils.read_corpus(test_target, source='src')
    pred_sents = utils.read_corpus(DECODE_FILE_PATH, source="src")

    for tgt_sent, pred_sent in zip(tgt_sents, pred_sents):

        print(pred_sent)
        print(tgt_sent)