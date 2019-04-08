import spacy
import en_core_web_sm
import de_core_news_sm

import configuration
from utils import read_corpus, write_sents
import paths


def preprocess_sentence(sentence):
    sentence = " ".join(sentence).replace("&apos;", "'")
    return sentence


if __name__ == '__main__':

    vconfig = configuration.VocabConfig()
    assert not vconfig.subwords_source
    assert not vconfig.subwords_target

    for chunk in ["train", "test", "valid"]:

        sc = paths.get_data_path(chunk, "src", pos=False)
        tg = paths.get_data_path(chunk, "tgt", pos=False)
        print('read in source sentences: %s' % sc)
        print('read in target sentences: %s' % tg)

        # both in src mode since we'll add the start/stop tokens later
        src_sents = read_corpus(sc, source='src')
        tgt_sents = read_corpus(tg, source='src')

        src_tagged = []
        tgt_tagged = []

        nlp_en = en_core_web_sm.load()
        nlp_de = de_core_news_sm.load()

        for sentence in src_sents:
            doc = nlp_en(preprocess_sentence(sentence))
            tagged_sentence = [token.tag_ for token in doc]
            src_tagged.append(tagged_sentence)

        print("parsed English")

        for sentence in tgt_sents:
            doc = nlp_de(preprocess_sentence(sentence))
            tagged_sentence = [token.tag_ for token in doc]
            tgt_tagged.append(tagged_sentence)

        print("parsed German")

        out_sc = paths.get_data_path(chunk, "src", pos=True)
        out_tg = paths.get_data_path(chunk, "tgt", pos=True)

        write_sents(src_tagged, out_sc)
        write_sents(tgt_tagged, out_tg)
