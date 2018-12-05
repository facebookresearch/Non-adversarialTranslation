# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


src_lang = "en"
tgt_lang = "es"
icp_init_epochs = 100
icp_train_epochs = 50
icp_ft_epochs = 50
n_pca = 50
n_icp_runs = 25
n_init_ex = 5000
n_ft_ex = 7500
n_eval_ex = 200000
n_processes = 1
method = 'csls_knn_10' # nn|csls_knn_10
cp_dir = "output"
