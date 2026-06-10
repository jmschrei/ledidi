# toy_models.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0)


class SumModel(torch.nn.Module):
	"""A parameter-free model that sums the one-hot channels per position.

	This model returns, for each of the four channels, the number of positions
	at which that channel is hot. Because it has no parameters and is a simple
	linear function of the input it produces exactly predictable outputs, which
	makes it convenient for reasoning about losses and edits by hand and for
	posing a design objective directly over nucleotide composition.
	"""

	def __init__(self):
		super(SumModel, self).__init__()

	def forward(self, X):
		return X.sum(dim=-1)


class FlattenDense(torch.nn.Module):
	"""A single dense layer over the flattened one-hot sequence.

	This model flattens the one-hot encoding and applies a single linear layer,
	producing `n_outputs` predictions. It is the smallest model with learnable
	parameters and a non-trivial gradient with respect to every input position,
	which is what the Ledidi optimization needs in order to propose edits. The
	definition matches the model of the same name in the tangermeme test suite.


	Parameters
	----------
	seq_len: int, optional
		The length of the sequences this model accepts. Default is 100.

	n_outputs: int, optional
		The number of outputs the dense layer produces. Default is 3.
	"""

	def __init__(self, seq_len=100, n_outputs=3):
		super(FlattenDense, self).__init__()
		self.dense = torch.nn.Linear(seq_len * 4, n_outputs)
		self.seq_len = seq_len

	def forward(self, X, alpha=0, beta=1):
		X = X.reshape(X.shape[0], self.seq_len * 4)
		return self.dense(X) * beta + alpha


class SmallDeepSEA(torch.nn.Module):
	"""A small convolutional model in the style of DeepSEA.

	Two convolution/pool/ReLU blocks followed by two dense layers produce a
	single prediction. The flattened dimension (176) is hard-coded for the two
	max-pool-by-three operations applied to a length-100 sequence, so this model
	requires `seq_len=100`. The definition matches the model of the same name in
	the tangermeme test suite, where it serves as the convolutional oracle for
	the design tests.


	Parameters
	----------
	n_outputs: int, optional
		The number of outputs the final dense layer produces. Default is 1.
	"""

	def __init__(self, n_outputs=1):
		super(SmallDeepSEA, self).__init__()

		self.conv1 = torch.nn.Conv1d(4, 32, (3,), padding='same')
		self.pool1 = torch.nn.MaxPool1d(3)
		self.relu1 = torch.nn.ReLU()

		self.conv2 = torch.nn.Conv1d(32, 16, (3,), padding='same')
		self.pool2 = torch.nn.MaxPool1d(3)
		self.relu2 = torch.nn.ReLU()

		self.linear1 = torch.nn.Linear(176, 20)
		self.relu3 = torch.nn.ReLU()
		self.linear2 = torch.nn.Linear(20, n_outputs)

	def forward(self, X):
		X = self.relu1(self.pool1(self.conv1(X)))
		X = self.relu2(self.pool2(self.conv2(X)))
		X = X.reshape(X.shape[0], -1)
		X = self.relu3(self.linear1(X))
		return self.linear2(X)


class ConvAvgDense(torch.nn.Module):
	"""A convolution followed by global average pooling and a dense layer.

	A single convolution and ReLU are averaged across the length dimension and
	passed through a dense layer, producing a single prediction for sequences of
	any length. The definition matches the model of the same name in the
	tangermeme test suite.


	Parameters
	----------
	n_outputs: int, optional
		The number of outputs the dense layer produces. Default is 1.
	"""

	def __init__(self, n_outputs=1):
		super(ConvAvgDense, self).__init__()

		self.conv = torch.nn.Conv1d(4, 12, (3,))
		self.relu = torch.nn.ReLU()
		self.dense = torch.nn.Linear(12, n_outputs)

	def forward(self, X):
		return self.dense(self.relu(self.conv(X)).mean(dim=-1))


class SumModelKeepdim(torch.nn.Module):
	"""A single-output model whose output keeps the channel dimension.

	This model sums over both the channel and length dimensions and returns a
	tensor of shape `(batch, 1)`. It is used to exercise the `target` slicing
	machinery and the pruning loop where a single-task model is expected.
	"""

	def __init__(self):
		super(SumModelKeepdim, self).__init__()

	def forward(self, X):
		return X.sum(dim=(1, 2), keepdim=True).squeeze(-1)
