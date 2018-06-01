# coding: utf-8
import random
import numpy as np
import networkx as nx
from argparse import ArgumentParser

import tqdm
import pickle
import datetime

import queue
import threading

from logger import Logger
# import linear_regression 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.autograd import Variable


class LinearRegression(nn.Module):

	def __init__(self, input_dim, output_dim):
		super(LinearRegression, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim)
		self.loss_func = nn.CrossEntropyLoss()

	def reset(self):
		self.linear.weight.data.normal_()
		self.linear.bias.data.zero_()

	def forward(self, x, y):
		out = self.linear(x)
		loss = self.loss_func(out, y)
		return loss

	def pred(self, x_t):
		x = Variable(x_t)
		return self.linear(x)

class EvalClass:

	def __init__(self, feature_dim, batchsize=200, epoch=1000, lr=0.02, test_ratio=0.1, \
				labelfile='data/cora/cora.content', verbose=True, use_cuda=False):
		cora_features = np.loadtxt('data/cora/cora.content', dtype=bytes).astype(str)
		self.node_name = cora_features[:,0].astype(int)
		perm = np.argsort(self.node_name)
		cora_features = cora_features[perm, 1:] # 已经删去了node names

		cora_X = cora_features[:,:-1].astype(bool)
		core_Y = np.zeros(cora_features.shape[0], dtype=int)
		for i, label in enumerate(np.unique(cora_features[:,-1])):
			core_Y[cora_features[:,-1] == label] = i + 1
		assert(np.sum(core_Y == 0) == 0)
		core_Y -= 1
		class_num = len(np.unique(core_Y))
		assert(class_num == np.max(core_Y)+1)

		self.Y = core_Y
		self.class_num = class_num
		self.feature_dim = feature_dim

		self.batchsize = batchsize
		self.epoch = epoch
		self.lr = lr
		self.test_ratio = test_ratio
		self.verbose = verbose
		self.use_cuda = use_cuda

		self.net = LinearRegression(self.feature_dim, self.class_num)
		if use_cuda:
			self.net = self.net.cuda()
		self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)

	def re_arange(self, keep_nodes):
		flag = np.zeros(len(self.node_name))
		for node in keep_nodes:
			flag += self.node_name == int(node) # 我不确定keep_nodes是str还是int，这样保险
		self.Y = self.Y[flag>0]

	def train(self, featurefile, shuffle=False):
		self.net.reset()

		net_features = np.loadtxt(featurefile, skiprows=1)
		perm = np.argsort(net_features[:,0])
		net_features = net_features[perm, 1:]
		assert(net_features.shape[0] == self.Y.shape[0])
		assert(net_features.shape[1] == self.feature_dim)

		core_Y = self.Y.copy()
		if shuffle:
			perm = np.arange(net_features.shape[0])
			np.random.shuffle(perm)
			net_features = net_features[perm]
			core_Y = core_Y[perm]

		X = torch.FloatTensor(net_features)
		Y = torch.LongTensor(core_Y)
		if self.use_cuda:
			X = X.cuda()
			Y = Y.cuda()
		# print(net_features.shape, X.shape)

		# data split
		trainval_X = X[int(X.shape[0]*self.test_ratio):]
		trainval_Y = Y[int(X.shape[0]*self.test_ratio):]
		test_X = X[:int(X.shape[0]*self.test_ratio)]
		test_Y = Y[:int(X.shape[0]*self.test_ratio)]

		for _ in range(self.epoch):
			# perm = np.arange(net_features.shape[0])
			# np.random.shuffle(perm)
			# trainval_X = trainval_X[perm]
			# trainval_Y = trainval_Y[perm]

			batches = int(np.ceil(1.*trainval_X.shape[0] / self.batchsize))

			loss_sum = 0
			for __ in range(batches):
				start = __ * self.batchsize
				end = min((__+1) * self.batchsize, trainval_X.shape[0])

				batch_x = Variable(trainval_X[start:end])
				batch_y = Variable(trainval_Y[start:end])
				loss = self.net.forward(batch_x, batch_y)
				if self.use_cuda:
					loss_sum += loss.data.cpu().numpy()
				else:
					loss_sum += loss.data.numpy()

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			pred = self.net.pred(test_X)
			if self.use_cuda:
				pred = pred.data.cpu().numpy()
				acc = np.sum(np.argmax(pred, axis=1) == test_Y.cpu().numpy()) / test_X.shape[0]
			else:
				pred = pred.data.numpy()
				acc = np.sum(np.argmax(pred, axis=1) == test_Y.numpy()) / test_X.shape[0]
			if self.verbose and _ % 100 == 0:
				print('{}: Epoch {}, loss = {}'.format(datetime.datetime.now(), _, loss_sum/trainval_X.shape[0]))
				print('\tacc = {}'.format(acc))

		return acc


class Discriminator(nn.Module):
	def __init__(self, N, dim, use_bias=True):
		super(Discriminator, self).__init__()
		self.N = N
		self.dim = dim
		self.emb = nn.Embedding(N, dim)
		self.use_bias = use_bias
		if self.use_bias:
			self.bias = nn.Parameter(torch.zeros(N))
		self.init_emb()

	def init_emb(self):
		init_range = 0.5 / self.dim
		self.emb.weight.data.uniform_(-init_range, init_range)

	def load(self, pretrain_array):
		self.emb.weight.data.copy_(torch.from_numpy(pretrain_array))

	def get_rewards2(self, left, right):
		emb_left = self.emb(left)
		emb_right = self.emb(right)

		score = torch.mul(emb_left, emb_right)
		score = torch.sum(score, dim=1)
		if self.use_bias:
			score += torch.gather(self.bias, 0, right)
		reward = torch.log(1 - torch.sigmoid(score))

		if next(self.parameters()).is_cuda:
			return reward.data.cpu().numpy()
		else:
			return reward.data.numpy()

	def get_rewards(self, left, right):
		emb_left = self.emb(left)
		emb_right = self.emb(right)

		score = torch.mul(emb_left, emb_right)
		score = torch.sum(score, dim=1)
		if self.use_bias:
			score += torch.gather(self.bias, 0, right)
		# score = torch.clamp(score, -10, 10)
		# score = torch.log(1 + torch.exp(score))
		# 同样，这里也按照我想的试一下
		score = torch.sigmoid(score)

		if next(self.parameters()).is_cuda:
			return score.data.cpu().numpy()
		else:
			return score.data.numpy()

	def forward(self, left, right, y):
		emb_left = self.emb(left)
		emb_right = self.emb(right)

		score = torch.mul(emb_left, emb_right)
		score = torch.sum(score, dim=1)
		if self.use_bias:
			score += torch.gather(self.bias, 0, right)
		prob = torch.sigmoid(score)
		prob = torch.clamp(prob, 1e-5, 1-1e-5)
		loss = - torch.sum(torch.mul(y, torch.log(prob)) + torch.mul(1-y, torch.log(1-prob)))

		return loss

	def save(self, index2node, file_name):
		if next(self.parameters()).is_cuda:
			embeddings = self.emb.weight.data.cpu().numpy()
		else:
			embeddings = self.emb.weight.data.numpy()
		fout = open(file_name, 'w', encoding="UTF-8")
		fout.write('{} {}\n'.format(self.N, self.dim))
		for i in range(self.N):
			fout.write('{} {}\n'.format(index2node[i], ' '.join(str(n) for n in embeddings[i])))
		fout.close()


class Generator(nn.Module):
	def __init__(self, N, dim, use_bias=True):
		super(Generator, self).__init__()
		self.N = N
		self.dim = dim
		self.emb = nn.Embedding(N, dim)
		self.use_bias = use_bias
		if self.use_bias:
			self.bias = nn.Parameter(torch.zeros(N))
		self.init_emb()

	def init_emb(self):
		init_range = 0.5 / self.dim
		self.emb.weight.data.uniform_(-init_range, init_range)

	def load(self, pretrain_array):
		self.emb.weight.data.copy_(torch.from_numpy(pretrain_array))

	def get_scores(self):
		# 这里是tensor的计算，没有归入图中，无法反传
		scores = torch.matmul(self.emb.weight, torch.t(self.emb.weight))
		if self.use_bias:
			scores += self.bias # 检验过，靠后的维度相加

		if next(self.parameters()).is_cuda:
			return scores.data.cpu().numpy()
		else:
			return scores.data.numpy()
		# 注意区别 self.emb.weight.data.numpy()

	# 该函数按照论文实现
	def forward2(self, traces, adjances, rewards):
		# 这里是variable的计算，归入了计算图，可以反传
		all_scores = torch.matmul(self.emb.weight, torch.t(self.emb.weight))
		if self.use_bias:
			all_scores += self.bias
		all_scores = all_scores.view(-1)
		
		trace_prob = []
		for trace, trace_adjance in zip(traces, adjances):
			left = trace[:-1]
			right = trace[1:]
			top_indices = []
			down_indices = []
			for l,r,node_adjance in zip(left,right,trace_adjance):
				base = l * self.N
				top_indices.append(base+r)
				down_indices.append([a+r for a in node_adjance])

			v_top = Variable(torch.LongTensor(top_indices))
			v_downs = [Variable(torch.LongTensor(d)) for d in down_indices]
			if next(self.parameters()).is_cuda:
				v_top = v_top.cuda()
				v_downs = [d.cuda() for d in v_downs]

			top_scores = torch.gather(all_scores, 0, v_top)
			down_scores = [torch.gather(all_scores, 0, d) for d in v_downs]

			# in pytorch 4.0, the result is changed from size=1 to a 0-dim tensor
			# add 'view(1)' for the following 2 lines
			# TODO: check 我设的这个最小值是否合理
			trace_prob_part = torch.cat([torch.log(torch.sum(torch.exp(d)).clamp(min=1e-6)).view(1) for d in down_scores])
			trace_prob.append((torch.sum(top_scores) - torch.sum(trace_prob_part)).view(1))

		probs = torch.cat(trace_prob)
		loss = torch.sum(torch.mul(probs, rewards))

		return loss

	# reward r from discriminator
	# TODO 注意不要让r的梯度返回。
	# 这个需要检查一下，不过我觉得不是一个optimizer，应该不会出现问题

	# 该函数根据作者源代码实现
	def forward(self, left, right, reward):
		emb_left = self.emb(left)
		emb_right = self.emb(right)

		score = torch.mul(emb_left, emb_right)
		score = torch.sum(score, dim=1)
		if self.use_bias:
			score += torch.gather(self.bias, 0, right)
		prob = torch.sigmoid(score)

		prob = torch.clamp(prob, 1e-5, 1-1e-5)
		# loss = -torch.sum(torch.mul(torch.log(prob), reward))
		# loss = sum log(prob)*(-log D)

		# 我按照论文里实现下loss试一下
		loss = torch.sum(torch.mul(torch.log(prob), torch.log(1-reward)))

		return loss

	def save(self, index2node, file_name):
		if next(self.parameters()).is_cuda:
			embeddings = self.emb.weight.data.cpu().numpy()
		else:
			embeddings = self.emb.weight.data.numpy()
		fout = open(file_name, 'w', encoding="UTF-8")
		fout.write('{} {}\n'.format(self.N, self.dim))
		for i in range(self.N):
			fout.write('{} {}\n'.format(index2node[i], ' '.join(str(n) for n in embeddings[i])))
		fout.close()


class GraphGan(object):
	def __init__(self, feature_dim, edgelistfile, undirected=True, update_ratio=1,
				 use_word2vec=True, window_size=3, small_data=False, lr_d=1e-4, lr_g=1e-3, use_cuda=False,
				 dis_pretrain_file=None, gen_pretrain_file=None):
		# 目前只支持无向图
		self.g = nx.read_edgelist(edgelistfile)
		if small_data:
			keep_nodes = []
			for node in list(self.g.nodes())[:200]:
				keep_nodes.append(node)
				keep_nodes.extend(list(self.g[node].keys()))
			keep_nodes = list(set(keep_nodes))

			for node in list( set(list(self.g.nodes())) - set(keep_nodes) ):
				self.g.remove_node(node)

		self.node_num = self.g.number_of_nodes()
		self.egde_num = self.g.number_of_edges()

		self.node2index = {}
		self.index2node = {}
		nodes_raw = self.g.nodes(data=True)
		edges_raw = self.g.edges(data=True)
		for index, (node, _) in enumerate(nodes_raw):
			self.node2index[node] = index
			self.index2node[index] = node

		self.update_ratio = update_ratio
		self.use_word2vec = use_word2vec
		self.window_size = window_size
		assert(self.window_size > 1)
		self.lr_d = lr_d
		self.lr_g = lr_g

		self.dim = feature_dim
		self.discriminator = Discriminator(self.node_num, feature_dim)
		self.generator = Generator(self.node_num, feature_dim)

		self.eval_class = EvalClass(self.dim)
		if small_data:
			self.eval_class.re_arange(keep_nodes)

		if dis_pretrain_file:
			dis_pretrain_data = np.loadtxt(dis_pretrain_file, skiprows=1)
			assert(dis_pretrain_data.shape[0] == self.node_num and dis_pretrain_data.shape[1] == self.dim + 1)
			dis_pretrain_weight = np.zeros((self.node_num, self.dim))
			dis_pretrain_node_indices = list(dis_pretrain_data[:,0].astype(int))
			for i, dis_pretrain_node_index in enumerate(dis_pretrain_node_indices):
				dis_pretrain_weight[self.node2index[str(dis_pretrain_node_index)],:] = dis_pretrain_data[i,1:]
			assert(np.sum( np.sum(np.abs(dis_pretrain_weight), 1) <= 1e-3 ) == 0)
			self.discriminator.load(dis_pretrain_weight)

			acc_d = self.eval_class.train(dis_pretrain_file)
			print('{}: discriminator pretrain weight loaded. acc = {}'.format(datetime.datetime.now(), acc_d))

		if gen_pretrain_file:
			gen_pretrain_data = np.loadtxt(gen_pretrain_file, skiprows=1)
			assert(gen_pretrain_data.shape[0] == self.node_num and gen_pretrain_data.shape[1] == self.dim + 1)
			gen_pretrain_weight = np.zeros((self.node_num, self.dim))
			gen_pretrain_node_indices = list(gen_pretrain_data[:,0].astype(int))
			for i, gen_pretrain_node_index in enumerate(gen_pretrain_node_indices):
				gen_pretrain_weight[self.node2index[str(gen_pretrain_node_index)],:] = gen_pretrain_data[i,1:]
			assert(np.sum( np.sum(np.abs(gen_pretrain_weight), 1) <= 1e-3 ) == 0)
			self.generator.load(gen_pretrain_weight)

			acc_g = self.eval_class.train(gen_pretrain_file)
			print('{}: generator pretrain weight loaded. acc = {}'.format(datetime.datetime.now(), acc_g))

		self.use_cuda = use_cuda
		if self.use_cuda:
			self.discriminator = self.discriminator.cuda()
			self.generator = self.generator.cuda()
		# 在源代码中，用的是Adam， lr=1e-4(d), 1e-3(g)
		self.optimizer_d = optim.SGD(self.discriminator.parameters(), lr=self.lr_d)
		self.optimizer_g = optim.SGD(self.generator.parameters(), lr=self.lr_g)

		self.root_indices = [i for i in range(self.node_num)]
		print('Make Trees')
		self.bfs_trees = []
		for i in tqdm.tqdm(self.root_indices):
		# for i in self.root_indices:
			self.bfs_trees.append(nx.bfs_tree(self.g, self.index2node[i]))

		self.logger = Logger('./logs')


	def online_generating_method(self, root_index, tree, sample_num, all_scores):
		# 说实话，对g和d的不同还不怎么清楚，待研究
		# assert(typ in ['discriminator', 'generator'])

		# 源代码的里的sample[i]即为这里的traces[i][-1]
		traces = []
		# gen_log_probs = []
		trace_adjance_nodes = []

		root = self.index2node[root_index]
		assert(len(tree[root].keys()))
		# 在cora数据集中，每个结点的度均大于0

		# 如果到达了叶子结点，则叶子结点一定会被返回
		while len(traces) < sample_num:
			work = root
			parent = None
			trace = [work]
			# trace_prob = []
			adjance_nodes = []

			while True:
				candidates = list(tree[work].keys())
				if parent:
					candidates.append(parent)
				candidates_index = [self.node2index[node] for node in candidates]
				raw_prob = all_scores[self.node2index[work], candidates_index]
				# softmax
				prob = np.exp(raw_prob - np.max(raw_prob))
				prob /= np.sum(prob)
				if np.abs(np.sum(prob) - 1) < 0.001:
					print(raw_prob, prob)
				# next_work_id = np.random.choice(np.arange(len(candidates)), size=1, p=prob)[0]
				# next_work = candidates[next_work_id]
				# next_work_prob = prob[next_work_id]
				next_work = np.random.choice(candidates, size=1, p=prob)[0]

				trace.append(next_work)
				# trace_prob.append(next_work_prob)
				adjance_nodes.append(candidates_index)

				if next_work == parent:
					break
				parent, work = work, next_work

			traces.append(trace)
			# path_prob = np.sum(np.log(trace_prob))
			# gen_log_probs.append(path_prob)
			trace_adjance_nodes.append(adjance_nodes)
		return traces, trace_adjance_nodes # , gen_log_probs

	def discriminator_data_loader(self, batch_size, all_scores):
		left = []  # samples_q
		right = [] # samples_rel
		label = []

		print('discriminator_data_loader')
		for index in self.root_indices:
			if np.random.rand() < self.update_ratio:
				pos_nodes = self.g[self.index2node[index]] # 所有邻居结点
				pos = [self.node2index[node] for node in pos_nodes]
				assert(len(pos))
				left.extend(len(pos) * [index])
				right.extend(pos)
				label.extend(len(pos) * [1])

				neg_traces, _ = self.online_generating_method(index, self.bfs_trees[index], len(pos), all_scores)
				neg = [self.node2index[trace[-2]] for trace in neg_traces]
				assert(len(neg) == len(pos))
				left.extend(len(pos) * [index])
				right.extend(neg)
				label.extend(len(pos) * [0])

		# 源代码中，下面的yield会在保持原有的left、right不变的情况下重复config.gen_for_d_iters次，待check
		# 我这里相当于config.gen_for_d_iters = 1
		while len(left) >= batch_size:
			yield left[:batch_size], right[:batch_size], label[:batch_size]
			left = left[batch_size:]
			right = right[batch_size:]
			label = label[batch_size:]
		if len(left):
			yield left, right, label

	def list2long_tensor(self, l):
		v_l = Variable(torch.LongTensor(l))
		if self.use_cuda:
			v_l = v_l.cuda()
		return v_l

	def list2float_tensor(self, l):
		v_l = Variable(torch.FloatTensor(l))
		if self.use_cuda:
			v_l = v_l.cuda()
		return v_l	

	def generator_data_loader2(self, batches, batch_size, n_sample_gen = 20):
		gen_update_iter = 200

		all_scores = self.generator.get_scores()

		roots = []
		samples = []
		traces = []
		adjances = []
		print('generator_data_loader2')
		for _ in tqdm.tqdm(range(batches)): # 这样的话，感觉最后几个点凑不够200个永远不会被训练啊
			count = 0
			loss_sum = 0
			for index in self.root_indices:
				if np.random.rand() < self.update_ratio:
					tree = self.bfs_trees[index]
					node_traces, node_trace_adjances = self.online_generating_method(
						index, tree, n_sample_gen, all_scores)
					assert(len(node_traces) == n_sample_gen)
					assert(len(node_trace_adjances) == n_sample_gen)

					roots.extend(n_sample_gen * [index])
					samples.extend([self.node2index[trace[-2]] for trace in node_traces])
					traces.extend([[self.node2index[node] for node in trace] for trace in node_traces])
					adjances.extend(node_trace_adjances)

				if len(roots) >= gen_update_iter:
					batch_roots = roots[:gen_update_iter]
					batch_samples = samples[:gen_update_iter]
					batch_traces = traces[:gen_update_iter]
					batch_adjances = adjances[:gen_update_iter]

					rewards = self.discriminator.get_rewards2(self.list2long_tensor(batch_roots), \
															  self.list2long_tensor(batch_samples))
					self.optimizer_g.zero_grad()
					loss = self.generator.forward2(batch_traces, batch_adjances, self.list2float_tensor(rewards))
					loss.backward()
					self.optimizer_g.step()
					all_scores = self.generator.get_scores()

					roots = roots[gen_update_iter:]
					samples = samples[gen_update_iter:]
					traces = traces[gen_update_iter:]
					adjances = adjances[gen_update_iter:]

					count += 1
					if self.use_cuda:
						loss_sum += loss.data.cpu().numpy()
					else:
						loss_sum += loss.data.numpy()
			print(_, loss_sum/count)

		if len(roots):
			rewards = self.discriminator.get_rewards2(self.list2long_tensor(roots), \
													  self.list2long_tensor(samples))
			self.optimizer_g.zero_grad()
			loss = self.generator.forward2(traces, adjances, self.list2float_tensor(rewards))
			loss.backward()
			self.optimizer_g.step()


	# 采用一种简单的方法，在每次训练all_scores时保持all_scores不变
	# 可以考虑对每个线程新增一个队列，向线程传送最新的all_scores
	def generator_data_loader(self, batches, gen_times, batch_size, n_sample_gen=20, workers=4):
		all_scores = self.generator.get_scores()
		push_batch_size = 200

		q = queue.Queue()
		gen_times_per_thread = int(np.ceil(1.*gen_times/workers))

		def loop(thread_id):
			thread_nodes = self.root_indices.copy()
			random.shuffle(thread_nodes)
			traces = []
			for _ in range(gen_times_per_thread):
				for index in thread_nodes:
					node_traces, _ = self.online_generating_method(index, self.bfs_trees[index], n_sample_gen, all_scores)
					traces.extend([[self.node2index[node] for node in trace] for trace in node_traces])
					if len(traces) >= push_batch_size:
						q.put(traces)
						traces = []
			q.put(traces)

		threads = []
		for i in range(workers):
			threads.append(threading.Thread(target=loop, args=(i,), name='GenThread_{}'.format(i)))
		for t in threads:
			t.start()

		replay_maxsize = 2000
		replay = queue.Queue(maxsize=replay_maxsize)
		gen_traces_num = 0
		for batch_index in range(batches):
			while len(replay.queue) < 3 * batch_size:
				traces = q.get()
				gen_traces_num += 1
				for trace in traces:
					replay.put(trace)
			if not q.empty():
				traces = q.get()
				gen_traces_num += 1
				drop_times = len(replay.queue) + len(traces) - replay_maxsize
				if drop_times > 0:
					for __ in range(drop_times):
						replay.get()
				# print('{}/{} to put in {}'.format(len(replay.queue), replay_maxsize, len(traces)))
				for ii, trace in enumerate(traces):
					replay.put(trace)
					# if len(replay.queue) > replay_maxsize-10:
					# 	print('{}/{} to put in {}/{}'.format(len(replay.queue), replay_maxsize, ii, len(traces)))

			left = []
			right = []

			sample_indices = np.random.choice(np.arange(len(replay.queue)), size=batch_size)
			sample_traces = [replay.queue[sample_index] for sample_index in sample_indices]
			pairs = list(map(self.generate_window_pairs, traces))  # [[], []] each list contains the pairs along the same path
			for path_pairs in pairs:
				for pair in path_pairs:
					left.append(pair[0])
					right.append(pair[1])

			rewards = self.discriminator.get_rewards(self.list2long_tensor(left), self.list2long_tensor(right))
			self.optimizer_g.zero_grad()
			loss = self.generator.forward(self.list2long_tensor(left), \
										  self.list2long_tensor(right), \
										  self.list2float_tensor(rewards))
			loss.backward()
			self.optimizer_g.step()

			if batch_index % 100 == 0:
				print('{}: generator batch {}, loss = {}, get data batches {}'.format(datetime.datetime.now(), batch_index, loss.data[0], gen_traces_num))
			
			# TODO: 是否向线程传递新的score信息
			# all_scores = self.generator.get_scores()

		for t in threads:
			t.join()
		print('{}: all threads end'.format(datetime.datetime.now()))

	# 这里使用的方法类似word2vec，loss=负对数
	# 具体待研究
	def abondaned_generator_data_loader(self, epoches, batch_size):
		# left = []  # root_nodes_gen
		# right = [] # rel_nodes
		# nodes = []
		# traces = []
		root_indices = 20
		gen_update_iter = 200 / n_sample_gen

		all_scores = self.generator.get_scores()

		print('generator_data_loader')
		for _ in tqdm.tqdm(range(epoches)): # 这样的话，感觉最后几个点凑不够200个永远不会被训练啊
			nodes = []
			traces = []
			for index in self.root_indices:
				if np.random.rand() < self.update_ratio:
					node_traces, _ = self.online_generating_method(index, self.bfs_trees[index], n_sample_gen, all_scores)
					# sample_indices = [self.node2index[trace[-1]] for trace in node_traces]
					assert(len(node_traces) == n_sample_gen)
					nodes.append(index)
					# left.extend(len(sample_indices)*[index])
					# right.extend(sample_indices)
					traces.extend([[self.node2index[node] for node in trace] for trace in node_traces])

				if len(nodes) >= gen_update_iter:
					# generate update pairs along the path, [q_node, rel_node] --> [left, right]
					pairs = list(map(self.generate_window_pairs, traces))  # [[], []] each list contains the pairs along the same path
					left = []
					right = []
					# 我觉得以下循环改为np.array会快很多
					for path_pairs in pairs:
						for pair in path_pairs:
							left.append(pair[0])
							right.append(pair[1])

					rewards = self.discriminator.get_rewards(self.list2long_tensor(left), self.list2long_tensor(right))
					self.optimizer_g.zero_grad()
					loss = self.generator.forward(self.list2long_tensor(left), \
												  self.list2long_tensor(right), \
												  self.list2float_tensor(rewards))
					loss.backward()
					self.optimizer_g.step()
					all_scores = self.generator.get_scores()

					nodes = []
					traces = []

			if len(nodes):
				pairs = list(map(self.generate_window_pairs, traces))  # [[], []] each list contains the pairs along the same path
				left = []
				right = []
				# 我觉得以下循环改为np.array会快很多
				for path_pairs in pairs:
					for pair in path_pairs:
						left.append(pair[0])
						right.append(pair[1])

				rewards = self.discriminator.get_rewards(self.list2long_tensor(left), self.list2long_tensor(right))
				self.optimizer_g.zero_grad()
				loss = self.generator.forward(self.list2long_tensor(left), \
											  self.list2long_tensor(right), \
											  self.list2float_tensor(rewards))
				loss.backward()
				self.optimizer_g.step()

	def train(self, epochs, batch_d, batch_g, batch_size):
		for _ in range(epochs):
			print('{}: total epoch {}'.format(datetime.datetime.now(), _))

			# 在源代码中，一次为discriminator采样的数据会不改任何顺序的使用10次，很奇怪，待改
			# 我这里只会使用1次
			all_scores = self.generator.get_scores()
			for batch_iter in range(batch_d):
				count = 0
				loss_sum = 0
				for left, right, label in self.discriminator_data_loader(batch_size, all_scores):
					self.optimizer_d.zero_grad()
					loss = self.discriminator.forward(self.list2long_tensor(left), \
													  self.list2long_tensor(right), \
													  self.list2float_tensor(label))
					loss.backward()
					self.optimizer_d.step()

					count += 1
					if self.use_cuda:
						loss_sum += loss.data.cpu().numpy()
					else:
						loss_sum += loss.data.numpy()
				print('{}: discriminator batch {}, loss = {}'.format(datetime.datetime.now(), batch_iter, loss_sum/count))

			if self.use_word2vec:
				self.generator_data_loader(batches=batch_g, gen_times=10, batch_size=batch_size, workers=10)
			else:
				self.generator_data_loader2(batch_g, batch_size)

			self.eval_test(_)

	def eval_test(self, epoch_num):
		tic = datetime.datetime.now()
		name = 'epoch_{}_{}'.format(epoch_num, tic).replace(':', '_')
		name_d = 'graphgan_result/{}_d.params'.format(name)
		name_g = 'graphgan_result/{}_g.params'.format(name)
		self.save(name_d, name_g)

		print('----------')
		print('discriminator embeddings checking')
		acc_d = self.eval_class.train(name_d)

		print('----------')
		print('generator embeddings checking')
		acc_g = self.eval_class.train(name_d)

		print('Accuracy: {} {}'.format(acc_d, acc_g))

		if False:
			#============ TensorBoard logging ============#
			# (1) Log the scalar values
			info = {
				'acc_d': acc_d,
				'acc_g': acc_g,
			}

			for tag, value in info.items():
				logger.scalar_summary(tag, value, step+1)

			# (2) Log values and gradients of the parameters (histogram)
			for tag, value in self.generator.named_parameters():
				tag = tag.replace('.', '/')
				logger.histo_summary(tag, to_np(value), step+1)
				logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)


	def generate_window_pairs(self, sample_path):
		"""
		given a sample path list from root to a sampled node, generate all the pairs corresponding to the windows size
		e.g.: [1, 0, 2, 4, 2], window_size = 2 -> [1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]
		:param sample_path:
		:return:
		"""
		sample_path = sample_path[:-1]
		pairs = []

		for i in range(len(sample_path)):
			center_node = sample_path[i]
			for j in range(max(i-self.window_size, 0), min(i+self.window_size+1, len(sample_path))):
				if i == j:
					continue
				node = sample_path[j]
				pairs.append([center_node, node])

		return pairs

	def save(self, d_outfile, g_outfile):
		self.discriminator.save(self.index2node, d_outfile)
		self.generator.save(self.index2node, g_outfile)



if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('-i', '--input', required=True, type=str,
					  help='Input graph file, with edgelist format')
	parser.add_argument('-d', '--d_output', required=True, type=str,
					  help='Output discriminator representation file')
	parser.add_argument('-g', '--g_output', required=True, type=str,
					  help='Output generator representation file')
	parser.add_argument('--d_pretrain', default=None, type=str,
					  help='Pretrain discriminator representation file')
	parser.add_argument('--g_pretrain', default=None, type=str,
					  help='Pretrain generator representation file')
	# 在作者的源代码中，dim=50
	parser.add_argument('-f', default=64, type=int,
					  help='Number of latent dimensions to learn for each node.')
	args = parser.parse_args()

	net = GraphGan(args.f, args.input, lr_d=1e-2, lr_g=1e-2, use_cuda=args.cuda, \
				   dis_pretrain_file=args.d_pretrain, gen_pretrain_file=args.g_pretrain)
	net.train(epochs=20, batch_d=20, batch_g=2000, batch_size=64)
	# net.train(epochs=20, batch_d=30, batch_g=30, batch_size=64)
	net.save(args.d_output, args.g_output)
