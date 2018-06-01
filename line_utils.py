# coding: utf-8
import numpy as np

import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class LinearRegression(nn.Module):

	def __init__(self, input_dim, output_dim):
		super(LinearRegression, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim)
		self.loss_func = nn.CrossEntropyLoss()

	def reset(self, weight_scale):
		self.linear.weight.data.normal_(std=weight_scale)
		self.linear.bias.data.zero_()

	def forward(self, x, y):
		out = self.linear(x)
		loss = self.loss_func(out, y)
		return loss

	def pred(self, x_t):
		x = Variable(x_t)
		return self.linear(x)

class EvalClass:

	def __init__(self, feature_dim, batchsize=200, epoch=3000, test_ratio=0.1, \
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
		self.test_ratio = test_ratio
		self.verbose = verbose
		self.use_cuda = use_cuda

		self.net = LinearRegression(self.feature_dim, self.class_num)
		if use_cuda:
			self.net = self.net.cuda()
		self.optimizer = optim.SGD(self.net.parameters(), lr=0.03)

	def load_features(self, featurefile):
		net_features = np.loadtxt(featurefile, skiprows=1)
		perm = np.argsort(net_features[:,0].astype(int))
		net_features = net_features[perm, 1:]
		if net_features.shape[0] != self.Y.shape[0]:
			print('net_features shape {} dismatch label shape {}'.format(net_features.shape, self.Y.shape))
			assert(0)
		assert(net_features.shape[1] == self.feature_dim)

		return net_features

	def train(self, net_features, weight_scale, learning_rate, duplicate_index, shuffle=False):
		self.net.reset(weight_scale)

		for param_group in self.optimizer.param_groups:
			param_group['lr'] = learning_rate
		print(self.optimizer.param_groups[0]['lr'], learning_rate)

		print('-----------------')
		print('WeightScale {}, LearningRate {}, Times {}'.format(weight_scale, learning_rate, duplicate_index))

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
			if self.verbose and (_+1) % 100 == 0:
				print('{}: Epoch {}, loss = {}'.format(datetime.datetime.now(), _, loss_sum/trainval_X.shape[0]))
				print('\tacc = {}'.format(acc))

		return acc