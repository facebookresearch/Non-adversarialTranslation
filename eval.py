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

print("%s_%s" % (params.src_lang, params.tgt_lang))
T = np.load("%s/%s_%s_T.npy" % (params.cp_dir, params.src_lang, params.tgt_lang))
TranslatedX = src_embeddings.dot(np.transpose(T))
cross_dict = utils.load_dictionary('data/%s-%s.5000-6500.txt' % (params.src_lang, params.tgt_lang), src_word2id, tgt_word2id)
utils.get_word_translation_accuracy(params.src_lang, src_word2id, TranslatedX,
                                    params.tgt_lang, tgt_word2id, tgt_embeddings,
                                    params.method, cross_dict)

print("%s_%s" % (params.tgt_lang, params.src_lang))
T = np.load("%s/%s_%s_T.npy" % (params.cp_dir, params.tgt_lang, params.src_lang))
TranslatedY = tgt_embeddings.dot(np.transpose(T))
cross_dict = utils.load_dictionary('data/%s-%s.5000-6500.txt' % (params.tgt_lang, params.src_lang), tgt_word2id, src_word2id)
utils.get_word_translation_accuracy(params.tgt_lang, tgt_word2id, TranslatedY,
                                    params.src_lang, src_word2id, src_embeddings,
                                    params.method, cross_dict)
