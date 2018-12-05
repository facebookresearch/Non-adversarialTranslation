# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import numpy as np
import utils
import params


src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.src_lang, params.n_eval_ex, False)
tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.tgt_lang, params.n_eval_ex, False)

T = np.load("%s/%s_%s_TY.npy" % (params.cp_dir, params.tgt_lang, params.src_lang))
TranslatedX = src_embeddings.dot(T)
cross_dict = utils.load_dictionary('data/%s-%s.5000-6500.txt' % (params.src_lang, params.tgt_lang), src_word2id, tgt_word2id)
utils.get_word_translation_accuracy(params.src_lang, src_word2id, TranslatedX,
                                    params.tgt_lang, tgt_word2id, tgt_embeddings,
                                    params.method, cross_dict)
