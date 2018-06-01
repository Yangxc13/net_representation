import random
import collections
import numpy as np
from six import iterkeys
from argparse import ArgumentParser
from gensim.models import Word2Vec
from multiprocessing import cpu_count


class Graph(collections.defaultdict):

    def __init__(self, edgelist, undirected=True):
        super(Graph, self).__init__(list)
        for i in range(edgelist.shape[0]):
            self[edgelist[i,0]].append(edgelist[i,1])
            if undirected:
                self[edgelist[i,1]].append(edgelist[i,0])
        # sort neighbours and remove self loops
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))
            if k in self[k]: 
                self[k].remove(k)

    def walk(self, gamma=10, t=40, rand=random.Random(0)):
        # 源码中默认为等概率选择，不过认为可以改变
        walks = []
        nodes = list(self.keys())
        
        for _ in range(gamma):
            rand.shuffle(nodes)
            for node in nodes:
                path = [node]
                while len(path) < t and len(self[path[-1]]):
                    path.append(rand.choice(self[path[-1]]))
                walks.append(path)
        return walks


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', nargs='?', required=True,
                      help='Input graph file, with edgelist format.')
    parser.add_argument('-o', '--output', required=True,
                      help='Output representation file')
    parser.add_argument('-r', '--gamma', default=10, type=int,
                      help='Number of random walks to start at each node(epochs num)')
    parser.add_argument('-t', default=40, type=int,
                      help='Length of the random walk started at each node')
    parser.add_argument('-d', default=64, type=int,
                      help='Number of latent dimensions to learn for each node.')
    parser.add_argument('-w', default=5, type=int,
                      help='Window size of skipgram model.')
    parser.add_argument('-p', '--workers', type=int, default=1, # cpu_count(),
                      help='Number of parallel processes.')
    parser.add_argument('--seed', default=0, type=int,
                      help='Seed for random walk generator.')
    parser.add_argument('--npz_format', action='store_true',
                      help='Input file is of npz format if True. Of txt format if False(default).')
    args = parser.parse_args()

    if args.npz_format:
        net = np.load(args.input).astype(str)
    else:
        net = np.loadtxt(args.input, dtype=str)
    G = Graph(net, undirected=True)
    walks = G.walk(args.gamma, args.t, random.Random(args.seed))
    model = Word2Vec(walks, size=args.d, window=args.w, min_count=0, seed=args.seed, sg=1, hs=0, workers=args.workers)
    model.wv.save_word2vec_format(args.output)