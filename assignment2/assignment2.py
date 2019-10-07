import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

path = './aclImdb'

def dataset(path):
	'''
	Load dataset
	'''

	train_pos = []
	train_neg = []
	test_pos = []
	test_neg = []

	train_path_pos = os.path.join(path, 'train', 'pos')
	train_path_neg = os.path.join(path, 'train', 'neg')
	test_path_pos = os.path.join(path, 'test', 'pos')
	test_path_neg = os.path.join(path, 'test', 'neg')

	for fname in sorted(os.listdir(train_path_pos)):
		if fname.endswith('.txt'):
			with open(os.path.join(train_path_pos, fname)) as f:
				train_pos.append(f.readline().strip().lower().split())
	for fname in sorted(os.listdir(train_path_neg)):
		if fname.endswith('.txt'):
			with open(os.path.join(train_path_neg, fname)) as f:
				train_neg.append(f.readline().strip().lower().split())
	for fname in sorted(os.listdir(test_path_pos)):
		if fname.endswith('.txt'):
			with open(os.path.join(test_path_pos, fname)) as f:
				test_pos.append(f.readline().strip().lower().split())
	for fname in sorted(os.listdir(test_path_neg)):
		if fname.endswith('.txt'):
			with open(os.path.join(test_path_neg, fname)) as f:
				test_neg.append(f.readline().strip().lower().split())

	train_pos = np.array(train_pos)
	train_neg = np.array(train_neg)
	test_pos = np.array(test_pos)
	test_neg = np.array(test_neg)

	return train_pos, train_neg, test_pos, test_neg

def preprocess(train_pos, train_neg, test_pos, test_neg):
	'''
	shuffle and preprocess dataset
	'''
	np.random.shuffle(train_pos)
	np.random.shuffle(train_neg)
	np.random.shuffle(test_pos)
	np.random.shuffle(test_neg)

	# Create validation set
	X_val = np.concatenate((train_pos[0:2500], train_neg[0:2500]))
	y_val = [1] * 2500 + [0] * 2500
	y_val = np.array(y_val)

	# Create and shuffle train set
	train_pos = train_pos[2500:]
	train_neg = train_neg[2500:]
	X_train = np.concatenate((train_pos, train_neg))
	y_train = [1] * len(train_pos) + [0] * len(train_neg)
	y_train = np.array(y_train)
	np.random.seed(314)
	np.random.shuffle(X_train)
	np.random.seed(314)
	np.random.shuffle(y_train)

	# Create test set
	X_test = np.concatenate((test_pos, test_neg))
	y_test = [1] * len(test_pos) + [0] * len(test_neg)
	X_test = np.array(X_test)
	y_test = np.array(y_test)

	return X_train, y_train, X_val, y_val, X_test, y_test

def make_vocabulary(X_train):
	'''
	make vocabulary over the most frequent 2000 words
	'''
	vocab = dict()
	for sentence in X_train:
		for w in sentence:
			if w in vocab:
				vocab[w] += 1
			else:
				vocab[w] = 1

	vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
	id2word = [w[0] for w in vocab[0:2000]]
	word2id = dict()
	i = 0
	for w in id2word:
		word2id[w] = i
		i += 1

	return id2word, word2id

def construct_features(X_train, X_val, X_test, id2word, word2id):
	'''
	Consturc feature design matrix from vocabulary
	'''
	# Feature matrix (n_sample x n_feature)
	train_features = np.zeros((X_train.shape[0], len(id2word)))
	val_features = np.zeros((X_val.shape[0], len(id2word)))
	test_features = np.zeros((X_test.shape[0], len(id2word)))
	# Train set features
	i = 0
	for sample in X_train:
		for w in sample:
			if w in id2word:
				train_features[i, word2id[w]] += 1
		i += 1
	# Val set features
	i = 0
	for sample in X_val:
		for w in sample:
			if w in id2word:
				val_features[i, word2id[w]] += 1
		i += 1
	# Test set features
	i = 0
	for sample in X_test:
		for w in sample:
			if w in id2word:
				test_features[i, word2id[w]] += 1
		i += 1

	return train_features, val_features, test_features

def plot(train_acc, val_acc):
	'''
	Plot training logs
	'''
	index = np.arange(train_acc.shape[0])
	plt.plot(index, train_acc, label="Training Accuracy")
	plt.plot(index, val_acc, label="Validation Accuracy")
	plt.legend()
	plt.title("Training/Validation set accuracy dynamics")
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.show()

class VanillaNN:
	"""
	Vanilla neural network
	"""
	def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test):
		# data
		self.n_samples = X_train.shape[0]
		self.n_features = X_train.shape[1]
		self.X_train = X_train
		self.X_val = X_val
		self.X_test = X_test
		self.y_train = y_train.reshape(y_train.shape[0], 1)
		self.y_val = y_val.reshape(y_val.shape[0], 1)
		self.y_test = y_test.reshape(y_test.shape[0], 1)
		# hyperparameters
		self.learning_rate = 0.1 / self.n_samples
		self.learning_rate_decay = 0
		self.batch_size = 20
		self.n_epochs = 300
		self.regularization = None
		self.lamb = 0.1
		self.n_hidden = 200
		# parameters
		self.W_1 = np.random.uniform(-0.5, 0.5, (self.n_features, self.n_hidden))
		self.b_1 = np.random.uniform(-0.5, 0.5, (1, self.n_hidden))
		self.W_2 = np.random.uniform(-0.5, 0.5,(self.n_hidden, 1))
		self.b_2 = np.random.uniform(-0.5, 0.5)
		self.best_acc = -1
		# Training log
		self.train_acc = np.zeros(self.n_epochs)
		self.val_acc = np.zeros(self.n_epochs)

	def train(self):
		'''
		Train model with batch gradient descent
			- Output: traning accuracy vs. epoch
					  validation accuracy vs. epoch
		'''
		for e in range(self.n_epochs):
			# batch gradient descent
			x_batches, y_batches = self.getbatch(self.X_train, self.y_train)
			#print(x_batches.shape, y_batches.shape)
			for b in range(x_batches.shape[0]):
				loss = self.backprop(x_batches[b], y_batches[b], self.regularization)
			
			# training tricks
			self.X_train = self.shuffle_data(e, self.X_train)
			self.y_train = self.shuffle_data(e, self.y_train)
			self.learning_rate = (1-self.learning_rate_decay) * self.learning_rate

			# Training Dynamics
			self.val_acc[e] = self.get_accuracy(self.X_val, self.y_val)
			self.train_acc[e] = self.get_accuracy(self.X_train, self.y_train)
			if self.val_acc[e] > self.best_acc:
				self.best_acc = self.val_acc[e]
			if e % 10 == 0:
				print("Epoch %d ..." % (e+10))
				print("Training loss: ", loss)
				print("Val set accuracy: ", self.val_acc[e])
				print("Train set accuracy: ", self.train_acc[e])
				print("-" * 30)

		return self.train_acc, self.val_acc

	def test(self):
		'''
		Evaluate on test set. Choose the best validaion model
		'''
		print("Val set accuracy: ", self.get_accuracy(self.X_val, self.y_val, True))
		print("test accuracy: ",  self.get_accuracy(self.X_test, self.y_test, True))

	def sigmoid(self, z):
		# Sigmoid function
		return 1 / (1+np.exp(-z))

	def loss(self, x, y, optimal=False):
		'''
		Cross entropy loss function
		- Not used during train/test
		'''
		m = x.shape[0]
		h = self.sigmoid(np.dot(x, self.W_1) + self.b_1)
		y_pred = self.sigmoid(np.dot(h, self.W_2) + self.b_2)

		return -(1/m) * np.sum(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))

	def backprop(self, x, y, regularization=None):
		'''
		Compute loss and perform backpropagation for give batch x and label y
		Input: x feature batch, y batch label, regularization flag for L1/L2 regularization
		Output: gradient vector
		'''
		m = x.shape[0]
		#print(x.shape, self.W_1.shape, self.b_1.shape)
		h = self.sigmoid(np.dot(x, self.W_1) + self.b_1)
		y_pred = self.sigmoid(np.dot(h, self.W_2) + self.b_2)

		delta_loss = y_pred - y
		if regularization == "l1":
			delta_loss = delta_loss + self.lamb * (np.average(np.sign(self.W_1)) + np.average(np.sign(self.W_2)))
		if regularization == "l2":
			delta_loss = delta_loss + self.lamb * np.power((np.average(self.W_1) + np.average(self.W_2)), 2)

		#print(h.shape, y.shape)

		delta_w2 = np.dot(h.T, delta_loss)
		delta_b2 = 1/m * np.sum(y_pred - y)
		delta_h = h * (1-h) * np.dot(delta_loss, self.W_2.T)
		delta_w1 = np.dot(x.T, delta_h)
		delta_b1 = np.sum(delta_h, axis=0)

		self.W_2 = self.W_2 - self.learning_rate*delta_w2
		self.b_2 = self.b_2 - self.learning_rate*delta_b2
		self.W_1 = self.W_1 - self.learning_rate*delta_w1
		self.b_1 = self.b_1 - self.learning_rate*delta_b1

		return self.loss(x, y)
	
	def getbatch(self, x, y):
		'''
		Input: feature ndarray, labels
		Output: feature batches nd-array, corresponding label array
		'''
		x_batches = []
		y_batches = []
		n_sample = x.shape[0]
		for i in range(0, n_sample, self.batch_size):
			x_batches.append(x[i:min(i+self.batch_size, n_sample)])
			y_batches.append(y[i:min(i+self.batch_size, n_sample)])

		return np.array(x_batches), np.array(y_batches)

	def shuffle_data(self, seed, b):
		np.random.seed(seed)
		np.random.shuffle(b)

		return b

	def get_accuracy(self, x, y, optimal=False):
		'''
		Input: x feature batch, y batch label, optimal flag for using best validation weight
		Output: Scaler accuracy
		'''
		m = x.shape[0]
		h = self.sigmoid(np.dot(x, self.W_1) + self.b_1)
		y_pred = self.sigmoid(np.dot(h, self.W_2) + self.b_2)
		y_pred = y_pred >= 0.5
		acc = np.sum(y_pred==y) / x.shape[0]
		return acc

def main():

	# Load and Preprocess data
	try:
		X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(open("data.pkl", "rb"))
		id2word, word2id = pickle.load( open("aux.pkl", "rb"))
		train_features, val_features, test_features = pickle.load(open("features.pkl", "rb"))
	except:
		print("No pickle files found, preprocessing data ...")

		train_pos, train_neg, test_pos, test_neg = dataset(path)

		X_train, y_train, X_val, y_val, X_test, y_test = preprocess(train_pos, train_neg, test_pos, test_neg)

		id2word, word2id = make_vocabulary(X_train)

		train_features, val_features, test_features = construct_features(X_train, X_val, X_test, id2word, word2id)

		pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), open("data.pkl", "wb"))
		pickle.dump((id2word, word2id), open("aux.pkl", "wb"))
		pickle.dump((train_features, val_features, test_features), open("features.pkl", "wb"))

	# Vanilla neural network Model
	model = VanillaNN(train_features, val_features, test_features, y_train, y_val, y_test)
	train_acc, val_acc = model.train()
	model.test()

	# Plot training dynamics
	plot(train_acc, val_acc)

if __name__ == '__main__':
	main()
