import numpy as np
import scipy.io as sio
from argparse import ArgumentParser

from multiprocessing import Process, Queue

import datetime

import torch
import torch.nn as nn
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

	def __init__(self, feature_dim, batchsize=200, epoch=1000, test_ratio=0.1, \
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
		assert(net_features.shape[0] == self.Y.shape[0])
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

def train(inputfile, batchsize, epoch, lr, test_ratio, verbose=True, use_cuda=False):
	net_features = np.loadtxt(inputfile, skiprows=1)
	perm = np.argsort(net_features[:,0])
	net_features = net_features[perm, 1:]

	cora_features = np.loadtxt('data/cora/cora.content', dtype=bytes).astype(str)
	perm = np.argsort(cora_features[:,0].astype(int))
	cora_features = cora_features[perm, 1:]

	cora_X = cora_features[:,:-1].astype(bool)
	core_Y = np.zeros(cora_features.shape[0], dtype=int)
	for i, label in enumerate(np.unique(cora_features[:,-1])):
		core_Y[cora_features[:,-1] == label] = i + 1
	assert(np.sum(core_Y == 0) == 0)
	core_Y -= 1
	class_num = len(np.unique(core_Y))
	assert(class_num == np.max(core_Y)+1)

	# shuffle before split
	assert(net_features.shape[0] == core_Y.shape[0])
	perm = np.arange(net_features.shape[0])
	np.random.shuffle(perm)
	net_features = net_features[perm]
	core_Y = core_Y[perm]

	X = torch.FloatTensor(net_features)
	Y = torch.LongTensor(core_Y)
	if use_cuda:
		X = X.cuda()
		Y = Y.cuda()
	print(net_features.shape, X.shape)
	# data split
	trainval_X = X[int(X.shape[0]*test_ratio):]
	trainval_Y = Y[int(X.shape[0]*test_ratio):]
	test_X = X[:int(X.shape[0]*test_ratio)]
	test_Y = Y[:int(X.shape[0]*test_ratio)]

	net = LinearRegression(X.shape[1], class_num)
	if use_cuda:
		net = net.cuda()
	optimizer = optim.SGD(net.parameters(), lr=lr)

	for _ in range(epoch):
		# perm = np.arange(net_features.shape[0])
		# np.random.shuffle(perm)
		# trainval_X = trainval_X[perm]
		# trainval_Y = trainval_Y[perm]

		batches = int(np.ceil(1.*trainval_X.shape[0] / batchsize))

		loss_sum = 0
		for __ in range(batches):
			start = __ * batchsize
			end = min((__+1) * batchsize, trainval_X.shape[0])

			batch_x = Variable(trainval_X[start:end])
			batch_y = Variable(trainval_Y[start:end])
			loss = net.forward(batch_x, batch_y)
			if use_cuda:
				loss_sum += loss.data.cpu().numpy()
			else:
				loss_sum += loss.data.numpy()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		pred = net.pred(test_X)
		if use_cuda:
			pred = pred.data.cpu().numpy()
			acc = np.sum(np.argmax(pred, axis=1) == test_Y.cpu().numpy()) / test_X.shape[0]
		else:
			pred = pred.data.numpy()
			acc = np.sum(np.argmax(pred, axis=1) == test_Y.numpy()) / test_X.shape[0]
		if verbose and _ % 100 == 0:
			print('{}: Epoch {}, loss = {}'.format(datetime.datetime.now(), _, loss_sum/trainval_X.shape[0]))
			print('\tacc = {}'.format(acc))

	return acc


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-i', '--input', required=True, type=str,
					  help='Net features file')
	parser.add_argument('-o', '--output', required=True, type=str,
					  help='Net features file')
	parser.add_argument('-b', '--batchsize', type=int, default=200,
					  help='Batch size')
	parser.add_argument('-e', '--epoch', type=int, default=3000,
					  help='Epochs')
	parser.add_argument('-t', '--test_ratio', type=float, default=0.1,
					  help='Test set ratio')
	# parser.add_argument('-c', '--cross', required=False, type=int,
	# 				  help='Num of cross validation folds')
	parser.add_argument('--cuda', action='store_true')
	args = parser.parse_args()

	# old use, only run once
	# train(args.input, args.batchsize, args.epoch, args.lr, args.test_ratio, use_cuda=args.cuda)

	q = Queue()
	eval_class_demo = EvalClass(64, batchsize=args.batchsize, epoch=args.epoch, test_ratio=args.test_ratio, \
		labelfile='data/cora/cora.content', verbose=True, use_cuda=False)
	net_features = eval_class_demo.load_features(args.input)
	times = 5

	weight_scales = [0.1, 0.03, 0.01]
	learning_rates = np.logspace(-4, 0, num=20)[9:-5] #graphgan in [12:-2]
	results = np.zeros((len(weight_scales), len(learning_rates), times))

	def loop(i, j, q, data):
		net_features = data['net_features']
		weight_scales = data['weight_scales']
		learning_rates = data['learning_rates']
		batchsize = data['batchsize']
		epoch = data['epoch']
		test_ratio = data['test_ratio']
		times = data['times']

		thread_results = np.zeros(times)
		eval_class = EvalClass(64, batchsize=batchsize, epoch=epoch, test_ratio=test_ratio, \
			labelfile='data/cora/cora.content', verbose=False, use_cuda=True)
		for t in range(times):
			acc = eval_class.train(net_features, weight_scales[i], learning_rates[j], t)
			thread_results[t] = acc
		q.put((i, j, thread_results))

	data = {'net_features': net_features, 'weight_scales': weight_scales, 'learning_rates': learning_rates,
			'batchsize': args.batchsize, 'epoch': args.epoch, 'test_ratio': args.test_ratio, 'times': times}

	threads = []
	for i, weight_scale in enumerate(weight_scales):
		for j, learning_rate in enumerate(learning_rates):
			threads.append(Process(target=loop, args=(i,j,q,data)))
	for t in threads:
		t.start()

	print('All process start')

	for t in threads:
		t.join()

	for _ in range(len(weight_scales) * len(learning_rates)):
		i, j, thread_results = q.get()
		results[i,j,:] = thread_results

	sio.savemat('{}_result.mat'.format(args.output), {'result': results})
