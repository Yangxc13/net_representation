
##Net Representation Learning

#####Yang Xiaocheng 2017210860

This is the second homework in class *Advanced Machine Learning* of Professor Jie Tang. We are asked to re-implement four network embedding algorithms, and evaluate them on two Dataset, **Cora** and **Tencent Weibo**.

The four algorithms in this project are **DeepWalk**, **Node2vec**, **LINE** and **Graphgan**. The former ones are based on gensim and the latter ones are based on pytorch.

| Algorithm | Cora(Accuracy) | Tencent Weibo(AUC) |
|:---------:|:--------------:|:------------------:|
| DeepWalk  | 0.8148         | 0.5 |
| GraphGAN  | 0.7926         | - |
| Node2vec  | 0.7593         | - |
| LINE      | 0.7556(1st) 0.7000(2nd) | 0.8488(2nd) |

###DeepWalk

Usage:

`python deepwalk.py -i data/cora/cora.cites -o cora_deepwalk_cbow_hs_walk10_length40_window5_seed0_thread4.embeddings -r 10 -t 40 -d 64 -w 5 --seed 0 --workers 4`

Just a simplified version of the [code](https://github.com/phanein/deepwalk) from the authors.

With num\_walks=10, walk\_len=40, feature\_dim=64, window\_size=5; for word2vec, iter=5, sg=1, hs=1; without parameter fine-tuning; for logistic regression, lr=3.36e-2.

I find that using sg(Skip-gram) and hs(Hierarchical Softmax) works better than using cbow and ns(Negative Samlpling).

| Accuracy(Cora) | Cbow | Skip-gram |
|:--------:|:----:|:---------:|
| Hierarchical Softmax | 0.6704 | 0.8148 |
| Negative Samlpling | 0.7667 | 0.7815 |

In my experimrnt, in general, deepwalk outperforms the other three, even with no fine-tuning of deepwalk. The difference between (deepwalk & node2vec) and (LINE & graphgan) is that the former two sample a fixed size of paths first, and then do loss-minimization on those paths; while the latter two keep sampling, and do global expectation loss-minimization on any possible sampling paths. Also, the latter two use negative sampling, and graphgan outperforms deepwalk with ns, however negative sampling is not necessary as the node size on Cora is small. By the way, I only run node2vec once, and I think with fine-tuning node2vec maybe work better that deepwalk.

Reference:

* Perozzi B, Al-Rfou R, Skiena S. Deepwalk: Online learning of social representations[C]//Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2014: 701-710.
* [https://github.com/phanein/deepwalk](https://github.com/phanein/deepwalk)

###LINE

Usage:

`python line.py --learning_rate 0.03 --workers 10 --cuda [--second]`

If `--second` is added, use second proximity, else use first proximity.

Second proximity is about 2x slower than first proximity, but converges much faster.(Especially the loss from negative sampling edges)

When running on Cora, I multiplied the loss on negative sampling edges with (1/1000), and its score was not very satisfactory. When later I ran it on tencent, I removed this coefficient and it converged faster and scored much better. However, it become worse when running on Cora without this coefficient. It needs further experiments on Cora that which coefficient is the best.

Add replay-buffer and warm-up for faster convergence.

The [anthors' code(on C)](https://github.com/tangjianpku/LINE) is faster and outperforms mine.

Add logstic regression to evaluate the embedding after a fixed size of epochs.

Best accuracy on Cora got on batch=30000. Logstic regression lr=3.36e-2.

TODO: Need further exploration why the second proximity works worse than the first proximity.

Reference:

* Tang J, Qu M, Wang M, et al. Line: Large-scale information network embedding[C]//Proceedings of the 24th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2015: 1067-1077.
* [https://github.com/snowkylin/line](https://github.com/snowkylin/line)

###GraphGAN

Usage:

`python graphgan.py -i data/cora/cora.cites -d d.out -g g.out`

I think the structure of the [authors' code](https://github.com/hwwang55/GraphGAN) is a mess and I rewrite it on pytorch.

Add replay-buffer convergence, without paths reuse just using online sampling it is too hard to converge.

In the authors' code, the generator's loss funcntion does not use *Graph Softmax* as which is said in the paper. I implement it myself, but it is too slow and needs further debug. See `Generator.forward2()` and `GraphGan.generator_data_loader2`. Temporarilly I use a word2vec-similar generator loss function as in the authors' code.

Best accuracy on Cora got on epoch=7. Logstic regression lr=0.144.

It is said in the paper that gan is difficult to train and it is better to use a pretraining embedding from deepwalk or line. However, I randomly initialize the embedding and it works well.

Reference:

* Wang H, Wang J, Wang J, et al. GraphGAN: Graph Representation Learning with Generative Adversarial Nets[J]. arXiv preprint arXiv:1711.08267, 2017.
* [https://github.com/hwwang55/GraphGAN](https://github.com/hwwang55/GraphGAN)

###Node2vec

Usage:

`python node2vec.py -i data/cora/cora.cites -o cora_node2vec_r10_t40_d64_w5_thread4.embeddings --num_walks 10 --walk_len 40 --dim 64 --window_size 5`

A simplified version, only run only, without deep exploration.

With num_walks=10, walk_len=40, feature_dim=64, window_size=5; for word2vec, iter=5, sg=1, hs=1; without parameter fine-tuning; for logistic regression, lr=7.85e-3.

I think with fine-tuning node2vec maybe work better that deepwalk.

Reference:

* Tang J, Qu M, Wang M, et al. Line: Large-scale information network embedding[C]//Proceedings of the 24th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2015: 1067-1077.
* [https://github.com/aditya-grover/node2vec](https://github.com/aditya-grover/node2vec)

###Tencent

We tried two algorithms, Deepwalk and LINE(2nd)

`python deepwalk.py -i data/tencent/train_edges.npy --npz_format -o tencent_deepwalk_sg_hs_walk10_length40_window5_seed0_thread4.embeddings -r 10 -t 40 -d 64 -w 5 --seed 0 --workers 4`

`python line.py --learning_rate 0.03 --workers 10 --cuda --input data/tencent/train_edges.npy`

With the Deepwalk method, the AUC score is around 0.5. With LINE(2nd), the AUC score gets 0.8488 after 20000 iterations.

###Others

logistic_regression.py provides independent embedding evaluation and hyperparameter fine-tuning, it also provides functions that can be called inside the net representaion learning algorithms to online evaluation the embedding, but the latter may contain bug as the accuracy tend to be very low. See `class EvalClass` and `train()`.

Usage:
`python logistic_regression.py --cuda -i line_result/out_40000.embeddings -o line_lr_0.03_workers_10_2nd_epoch40000`

Embedding results in my experiments are attached for possible comparing.