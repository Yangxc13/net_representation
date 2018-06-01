import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score

parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True)
args = parser.parse_args()

train_edges = np.load('data/tencent/train_edges.npy')
avaialable_nodes = np.unique(train_edges)

node2index = {}
index2node = {}
for index, node in enumerate(avaialable_nodes):
    node2index[node] = index
    index2node[index] = node

test_edges = np.load('data/tencent/test_edges_filtered.npy')
print('test edges shape {}'.format(test_edges.shape))

test_edges_false = np.load('data/tencent/test_edges_false_filtered.npy')
print('test edges false shape {}'.format(test_edges_false.shape))

net_features = np.loadtxt(args.input, skiprows=1)
perm = [node2index[int(node)] for node in net_features[:,0]]
net_features = net_features[perm, 1:]

net_features = net_features / np.sqrt(np.square(net_features).sum(axis=1,keepdims=True))

def cosine(edges, net_features):
	left = [node2index[int(node)] for node in edges[:,0].flatten()]
	right = [node2index[int(node)] for node in edges[:,1].flatten()]

	left_features = net_features[left,:]
	right_features = net_features[right,:]

	return np.sum(left_features*right_features, axis=1)

result1 = cosine(test_edges, net_features)
result2 = cosine(test_edges_false, net_features)

auc = roc_auc_score([1] * len(result1) + [0] * len(result2), \
                    (np.concatenate((result1, result2))+1)/2)
print('{}: AUC={}'.format(args.input, auc))