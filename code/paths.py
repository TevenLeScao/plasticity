import configuration

gconfig = configuration.GeneralConfig()
vconfig = configuration.VocabConfig()
cconfig = configuration.CompressionConfig()
mconfig = configuration.LSTMConfig()
results_folder = "results/"

parts_of_speech_mode = gconfig.mode == "parts_of_speech"

model = results_folder + "model"
if mconfig.factorization:
    factorized_suffix = ".factorized.{}.{}{}".format(mconfig.embed_size, mconfig.rank_encoder, mconfig.rank_decoder)
    model = model + factorized_suffix
model_compressed_prefix = model + ".compressed"
pruning_where_suffix = cconfig.pruning_where if isinstance(
    cconfig.pruning_where, str) else "_".join(list(cconfig.pruning_where))
model_prune_suffix = ".prune" + \
    str(int(100*cconfig.pruning))+cconfig.pruning_scheme+"_" + pruning_where_suffix
model_postfactorize_suffix = "postfact0.5"

model_pruned_suffix = model_prune_suffix + ".pruned"
model_pruned_retrained_suffix = model_prune_suffix + ".retrained"
model_pruned = model_compressed_prefix + model_pruned_suffix
model_pruned_retrained = model_compressed_prefix + model_pruned_retrained_suffix

model_postfactorized_suffix = model_postfactorize_suffix + ".postfactorized"
model_postfactorized_retrained_suffix = model_postfactorize_suffix + ".retrained"
model_postfactorized = model_compressed_prefix + model_postfactorized_suffix
model_postfactorized_retrained = model_compressed_prefix + model_postfactorized_retrained_suffix

model_quantized_suffix = ".quantized."+cconfig.small_dtype
model_quantized = model_compressed_prefix + model_quantized_suffix

model_mixed = model_quantized + ".mixed_precision"

vocab_folder = "data/vocab/"

decode_output_suffix = ".test.txt" if gconfig.test else ".valid.txt"
decode_output = results_folder+"decode.en"+decode_output_suffix

data_aligned_folder = "data/bilingual/"
data_subwords_folder = "data/subwords/"

train_loss_history_file = results_folder + "train_loss_history.json"
dev_loss_history_file = results_folder + "dev_ppl_history.json"


def get_vocab_path(pos=False):
    vocab = vocab_folder+"vocab" + (".subsrc" if vconfig.subwords_source else ".words") + \
    (".subtgt" if vconfig.subwords_target else ".words") + (".pos" if pos else "") + ".bin"
    return vocab


def get_data_path(chunk, mode, pos=False, token=False):
    prefix = data_aligned_folder
    if mode == "tgt":
        language_suffix = ".de"
    else:
        assert mode == "src"
        language_suffix = ".en"
    if pos:
        pos_suffix = ".pos"
    elif token:
        pos_suffix = ".token"
    else:
        pos_suffix = ""
    return prefix + chunk + ".de-en" + language_suffix + pos_suffix


train_source = get_data_path("train", "src", pos=parts_of_speech_mode)
train_target = get_data_path("train", "tgt", pos=parts_of_speech_mode)
dev_source = get_data_path("valid", "src", pos=parts_of_speech_mode)
dev_target = get_data_path("valid", "tgt", pos=parts_of_speech_mode)
test_source = get_data_path("test", "src", pos=parts_of_speech_mode)
test_target = get_data_path("test", "tgt", pos=parts_of_speech_mode)
newstest_target = "newstest2014-deen-ref.de.sgm"
newstest_source = "newstest2014-deen-src.en.sgm"


def get_tf_idf_vector_path(chunk):
    prefix = data_aligned_folder
    return prefix + "tf_idf." + chunk + ".de-en.json"

