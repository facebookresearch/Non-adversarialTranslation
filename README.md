# Non-Adversarial Unsupervised Word Translation

This repo contains PyTorch code replicating the main ideas presented in:

- **Non-Adversarial Unsupervised Word Translation**<br/>
*Yedid Hoshen and Lior Wolf, EMNLP 2018*<br/>
[https://arxiv.org/abs/1801.06126](https://arxiv.org/abs/1801.06126)

## Getting Started
1) Start a new conda repo using Python 3. Install fbpca, faiss, numba:
```
conda install numpy numba scipy tqdm faiss-cpu -c pytorch
pip install fbpca
```

2) Download FastText data and test dictionary for two languages (e.g. en and es):
```
sh download.sh en es
```

3) Create lightweight data files
```
python get_data.py
```

4) Solve the word translation problem with our method (MBC-ICP)
```
python run_icp.py
```

5) Evaluate translation accuracy:
```
python eval.py
```
Where comparison method 'nn' is faster, and 'csls_knn_10' is more accurate. It can be selected in the configuration file params.py

All hyper-parameter choices may be modified in the params.py file. The optimal number of processes should be set in accordance with the specific system configuration. 

Reducing the number of icp_runs (params.n_icp_runs) will reduce run-time. In the case that no good initialization is found, increase the number of icp runs (params.n_icp_runs).

## Credit
Credit to Adam Polyak for significantly speeding up the code.

## License
This project is CC-BY-NC-licensed.

## Changelog
10-dec-2018: Fixed bug in eval.py - results are now in-line with paper on non-european languages
