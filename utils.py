# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import io
import numpy as np
import faiss


def read_txt_embeddings(emb_path, max_vocab, full_vocab=False):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
            else:
                word, vect = line.rstrip().split(' ', 1)
                if not full_vocab:
                    word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                if word in word2id:
                    if full_vocab:
                        print("Word '%s' found twice in embedding file" &(word))
                else:
                    if not vect.shape == (300,):
                        print("Invalid dimension (%i) for word '%s' in line %i."
                                       % (vect.shape[0], word, i))
                        continue
                    assert vect.shape == (300,), i
                    word2id[word] = len(word2id)
                    vectors.append(vect[None])
            if max_vocab > 0 and len(word2id) >= max_vocab:
                break

    assert len(word2id) == len(vectors)
    print("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.concatenate(vectors, 0)

    assert embeddings.shape == (len(id2word), 300)
    return id2word, word2id, embeddings


def load_dictionary(path, word2id1, word2id2):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
    assert os.path.isfile(path)

    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            word1, word2 = line.rstrip().split()
            if word1 in word2id1 and word2 in word2id2:
                pairs.append(np.array([word2id1[word1], word2id2[word2]]))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    print("Found %i pairs of words in the dictionary (%i unique). "
          "%i other pairs contained at least one unknown word "
          "(%i in lang1, %i in lang2)"
          % (len(pairs), len(set([x for x, _ in pairs])),
             not_found, not_found1, not_found2))

    pairs = np.array(pairs)
    return pairs


def get_nn_avg_dist(emb, query, knn):
    # cpu mode
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    distances, _ = index.search(query, knn)
    return distances.mean(1)


def get_word_translation_accuracy(lang1, word2id1, emb1, lang2, word2id2, emb2, method, dico):
    """
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the precision@k.
    """

    assert np.max(dico[:, 0]) < len(emb1)
    assert np.max(dico[:, 1]) < len(emb2)

    # normalize word embeddings
    emb1 = emb1 / np.linalg.norm(emb1, ord=2, axis=1, keepdims=True)
    emb2 = emb2 / np.linalg.norm(emb2, ord=2, axis=1, keepdims=True)
    emb1 = emb1.astype('float32')
    emb2 = emb2.astype('float32')

    # nearest neighbors
    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query.dot(emb2.T)

    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
        # queries / scores
        query = emb1[dico[:, 0]]
        scores = 2 * query.dot(emb2.T)
        scores -= average_dist1[dico[:, 0]][:, None]
        scores -= average_dist2[None, :]

    else:
        raise Exception('Unknown method: "%s"' % method)

    results = []
    top_matches = np.argsort(-scores, 1)[:, :10]
    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == dico[:, 1][:, None]).sum(1)
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0]):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        print("%i source words - %s - Precision at k = %i: %f" %
              (len(matching), method, k, precision_at_k))
        results.append(('precision_at_%i' % k, precision_at_k))

    return results
