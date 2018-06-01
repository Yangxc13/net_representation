# coding: utf-8
import random
import numpy as np

import networkx as nx
from argparse import ArgumentParser
from multiprocessing import cpu_count

from gensim.models import Word2Vec


class AliasSampling:
	# Reference: https://en.wikipedia.org/wiki/Alias_method
	def __init__(self, prob):
		self.n = len(prob)
		self.U = np.array(prob) * self.n
		self.K = [i for i in range(len(prob))]
		overfull, underfull = [], []
		for i, U_i in enumerate(self.U):
			if U_i > 1:
				overfull.append(i)
			elif U_i < 1:
				underfull.append(i)
		while len(overfull) and len(underfull):
			i, j = overfull.pop(), underfull.pop()
			self.K[j] = i
			self.U[i] = self.U[i] - (1 - self.U[j])
			if self.U[i] > 1:
				overfull.append(i)
			elif self.U[i] < 1:
				underfull.append(i)

	def sampling(self, n=1):
		x = np.random.rand(n)
		i = np.floor(self.n * x)
		y = self.n * x - i
		i = i.astype(np.int32)
		res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
		if n == 1:
			return res[0]
		else:
			return res

class Graph:
	'''
		Default undirected graph and no weights
	'''
	def __init__(self, edgelistfile, p, q):
		self.g = nx.read_edgelist(edgelistfile)
		self.p = p
		self.q = q
		self.init_prob()

	def get_alias_edge(self, src, dst):
		raw_probs = []
		for nxt in sorted(self.g.neighbors(dst)):
			if nxt == src:
				raw_probs.append(1./self.p)
			elif self.g.has_edge(nxt, src): # 如果是有向图，需要注意这里的位置
				raw_probs.append(1)
			else:
				raw_probs.append(1./self.q)
		probs = np.array(raw_probs) / np.sum(raw_probs)
		return AliasSampling(prob=probs)

	def init_prob(self):
		# 在源代码中有另一种alias_sampling的方法，这里我们直接统一用Line中的方法了
		# 因为边没有权重，忽略alias_nodes，只考虑把p,q引入alias_edges即可
		self.alias_edges = {}

		for src, dst in self.g.edges():
			self.alias_edges[(src, dst)] = self.get_alias_edge(src, dst)
			self.alias_edges[(dst, src)] = self.get_alias_edge(dst, src)

	def node2vec_walk(self, walk_len, root):
		walk = [root]

		while len(walk) < walk_len:
			work = walk[-1]
			nxt = sorted(self.g.neighbors(work))
			if len(nxt) == 0:
				break
			elif len(walk) == 1:
				walk.append( np.random.choice(nxt, size=1)[0] )
			else:
				prev = walk[-2]
				nxt_index = self.alias_edges[(prev, work)].sampling()
				walk.append( nxt[nxt_index] )

		return walk

	def walk(self, num_walks, walk_len):
		walks = []
		nodes = list(self.g.nodes())

		for _ in range(num_walks):
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_len, node))

		return walks


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-i', '--input', nargs='?', required=True,
						help='Input graph file, with edgelist format')
	parser.add_argument('-o', '--output', required=True,
						help='Output representation file')
	parser.add_argument('--num_walks', default=10, type=int,
						help='Number of random walks to start at each node(epochs num)')
	parser.add_argument('--walk_len', default=80, type=int,
						help='Length of the random walk started at each node')
	parser.add_argument('--dim', default=64, type=int,
						help='Number of latent dimensions to learn for each node.')
	parser.add_argument('--window_size', default=10, type=int,
						help='Window size of skipgram model.')
	parser.add_argument('--workers', type=int, default=cpu_count(),
						help='Number of parallel processes.')
	parser.add_argument('--iter', default=1, type=int,
						help='Number of epochs in SGD')
	parser.add_argument('-p', type=float, default=1,
						help='Return hyperparameter. Default is 1.')
	parser.add_argument('-q', type=float, default=1,
						help='Inout hyperparameter. Default is 1.')
	args = parser.parse_args()

	G = Graph(args.input, args.p, args.q)
	walks = G.walk(args.num_walks, args.walk_len)
	model = Word2Vec(walks, size=args.dim, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)