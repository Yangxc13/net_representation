# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import datetime
import numpy as np
import networkx as nx
from argparse import ArgumentParser

import queue
import threading
from time import sleep
from multiprocessing import cpu_count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.autograd import Variable

from line_utils import *


class Line(nn.Module):
	"""refer: https://github.com/bamtercelboo/pytorch_word2vec/blob/master/model.py
	"""
	def __init__(self, N, dim):
		super(Line, self).__init__()
		self.N = N
		self.dim = dim
		self.vertex_emb = nn.Embedding(N, dim)
		self.context_emb = nn.Embedding(N, dim)
		self.init_emb()
		self.gamma = 2
		self.eps = 1e-7

	def init_emb(self):
		init_range = 0.5 / self.dim
		self.vertex_emb.weight.data.uniform_(-init_range, init_range)
		self.context_emb.weight.data.uniform_(-init_range, init_range)

	def forward(self, u, v, label, proximity): 
		emb_u = self.vertex_emb(u)
		if proximity == 'first':
			emb_v = self.vertex_emb(v)
		else:
			emb_v = self.context_emb(v)

		score = torch.mul(emb_u, emb_v)
		score = torch.sum(score, dim=1)
		score = torch.mul(score, label)
		# Choice 1:
		# loss = torch.sum(-F.logsigmoid(score))
		# Choice 2:
		logscore = -F.logsigmoid(score)
		loss1 = torch.sum(torch.mul(logscore, (label+1)))
		loss2 = torch.sum(torch.mul(logscore, (1-label)))
		# Choice 3:
		# logit = F.sigmoid(score)
		# logit = logit.clamp(self.eps, 1.-self.eps)
		# loss = -torch.log(logit)
		# loss1 = torch.sum(loss * (label+1) * ((1 - logit) ** self.gamma))
		# loss2 = torch.sum(loss * (1-label) * (logit ** self.gamma))

		return loss1, loss2

	def save(self, index2node, file_name, proximity='first'):
		assert(proximity in ['first', 'second'])
		use_cuda = next(self.parameters()).is_cuda
		if use_cuda:
			if True: # proximity == 'first':
				embeddings = self.vertex_emb.weight.data.cpu().numpy()
			else:
				embeddings = np.concatenate((self.vertex_emb.weight.data.cpu().numpy(),
											 self.context_emb.weight.data.cpu().numpy()), axis=1)
		else:
			if True: # proximity == 'first':
				embeddings = self.vertex_emb.weight.data.numpy()
			else:
				embeddings = np.concatenate((self.vertex_emb.weight.data.numpy(),
											 self.context_emb.weight.data.numpy()), axis=1)
		fout = open(file_name, 'w', encoding="UTF-8")
		fout.write('{} {}\n'.format(N, embeddings.shape[1]))
		for i in range(self.N):
			fout.write('{} {}\n'.format(index2node[i], ' '.join(str(n) for n in embeddings[i])))
		fout.close()

		return embeddings
		

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
	def __init__(self, edgelistfile, undirected=True):
		# 1. 注意，这里只支持无向图，对于有向图，至少需要改变以下几个部分
		# nx.read_edgelist(create_using=nx.DiGraph)
		# self.g.degree(node, weight='weight') 目前我不知道有向图这里怎么改
		# edge采样，这里用的最简单的uniform采样
		# 在fetch_batch中，如果是无向图的话，有0.5的概率交换次序
		# 2. 其他
		# self.g.degree(node, weight='weight') 如果edgelistfile中有边的weights参数
		if edgelistfile[-3:] == 'npy':
			self.g = nx.Graph()
			self.g.add_edges_from(np.load(edgelistfile))
		else:
			self.g = nx.read_edgelist(edgelistfile)

		self.node_num = self.g.number_of_nodes()
		self.egde_num = self.g.number_of_edges()
		print('Nodes num {}; edges num {}'.format(self.node_num, self.egde_num))

		self.node2index = {}
		self.index2node = {}
		nodes_raw = self.g.nodes(data=True)
		edges_raw = self.g.edges(data=True)
		for index, (node, _) in enumerate(nodes_raw):
			self.node2index[node] = index
			self.index2node[index] = node
		self.edges = [(self.node2index[u], self.node2index[v]) for u, v, _ in edges_raw]

		self.node_negative_distribution = np.power(
			np.array([self.g.degree(node) for node, _ in nodes_raw], dtype=np.float32), 0.75)
		self.node_negative_distribution /= np.sum(self.node_negative_distribution)
		self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

	def fetch_batch(self, batch_size, K=5):
		# 这里的采样方法不保证每个epoch将所有边迭代一遍
		# K: neg sample ratio
		# uniform sampling: all edges have no weights or have equal weights=1
		batch_edges = np.random.choice(self.egde_num, size=batch_size)

		u_i = []
		u_j = []
		label = []
		for edge_index in batch_edges:
			edge = self.edges[edge_index]
			if self.g.__class__ == nx.Graph: # 如果是无向图
				edge = (edge[1], edge[0]) if np.random.rand() > 0.5 else edge
			u_i.append(edge[0])
			u_j.append(edge[1])
			label.append(1)
			for i in range(K):
				while 1:
					neg_node = self.node_sampling.sampling()
					if not (neg_node == edge[0] or self.g.has_edge(self.index2node[neg_node], self.index2node[edge[0]])):
						break
				u_i.append(edge[0])
				u_j.append(neg_node)
				label.append(-1)

		return np.array(u_i).astype(int), np.array(u_j).astype(int), np.array(label).astype(int)


if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('--second', action='store_true')
	parser.add_argument('--dim', type=int, default=64,
						help='embedding vector dim')
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('-K', type=int, default=128,
						help='negative sample ratio')
	parser.add_argument('--num_batches', type=int, default=300000)
	parser.add_argument('--learning_rate', type=float, default=1.0)
	parser.add_argument('--input', type=str, default='data/cora/cora.cites')
	parser.add_argument('--workers', type=int, default=cpu_count()-1)
	args = parser.parse_args()

	G = Graph(args.input)
	N = G.node_num
	q = queue.Queue()

	model = Line(N, args.dim)
	if args.cuda:
		model = model.cuda()
	optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

	def loop():
		# 可能的实现是，除了q之外，所有函数你用到的全局变量都是复制值
		while True:
			batch_edges = np.random.choice(G.egde_num, size=args.batch_size)

			u_i = []
			u_j = []
			label = []
			for edge_index in batch_edges:
				edge = G.edges[edge_index]
				if G.g.__class__ == nx.Graph: # 如果是无向图
					edge = (edge[1], edge[0]) if np.random.rand() > 0.5 else edge
				u_i.append(edge[0])
				u_j.append(edge[1])
				label.append(1)
				for i in range(args.K):
					while 1:
						neg_node = G.node_sampling.sampling()
						if not (neg_node == edge[0] or G.g.has_edge(G.index2node[neg_node], G.index2node[edge[0]])):
							break
					u_i.append(edge[0])
					u_j.append(neg_node)
					label.append(-1)

			q.put((np.array(u_i).astype(int), np.array(u_j).astype(int), np.array(label).astype(int)))

	threads = []
	for i in range(args.workers):
		threads.append(threading.Thread(target=loop, name='LoopThread_{}'.format(i)))
	for t in threads:
		t.start()

	if args.second:
		proximity = 'second'
	else:
		proximity = 'first'
	eval_dim = args.dim # if proximity == 'first' else args.dim * 2
	eval_class = EvalClass(args.dim, use_cuda=True)

	replay = queue.Queue(maxsize=100)
	sample_count = 0
	# warm-up
	while len(replay.queue) < 50:
		u_i, u_j, label = q.get()
		sample_count += 1
		replay.put((u_i, u_j, label))

	for i in range(args.num_batches):
		if q.empty(): # and (not replay.empty()):
			index = np.random.choice(np.arange(replay.qsize()), size=1)[0]
			u_i, u_j, label = replay.queue[index]
		else:
			u_i, u_j, label = q.get()
			sample_count += 1
			if replay.full():
				replay.get()
			replay.put((u_i, u_j, label))
		# u_i, u_j, label = G.fetch_batch(args.batch_size, args.K)
		optimizer.zero_grad()
		u = Variable(torch.LongTensor(u_i))
		v = Variable(torch.LongTensor(u_j))
		y = Variable(torch.FloatTensor(label))
		if args.cuda:
			u = u.cuda()
			v = v.cuda()
			y = y.cuda()
		loss1, loss2 = model.forward(u, v, y, proximity=proximity)
		loss = loss1 + loss2
		loss.backward()
		optimizer.step()
		# print(-F.logsigmoid(score)[:,None].t().data)

		if i % 100 == 0 or i == (args.num_batches - 1):
			print('{}: Iter {}, loss = {}, loss_1 = {}, loss_2 = {}, lr = {}, sample_count = {}'.format(datetime.datetime.now(), i, loss.data[0], loss1.data[0], loss2.data[0], optimizer.param_groups[0]['lr'], sample_count))

		if i * args.batch_size % 100000 == 0:
			lr = args.learning_rate * (1.0 - 1.0 * i / 100000)
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr

		if i > 0 and (i % 5000 == 0 or i == (args.num_batches - 1)):
			filename = 'line_result/out_{}_{}.embeddings'.format(i,proximity)
			net_features = model.save(G.index2node, filename, proximity=proximity)
			# if args.input[-3:] != 'npy':
			# 	acc = eval_class.train(net_features, 0.1, np.logspace(-4, 0, num=20)[12], np.ceil(i/10000))
			# 	print('\tIter {} acc = {}'.format(i, acc))
