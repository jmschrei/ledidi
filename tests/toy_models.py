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
	makes it convenient for reasoning about losses and edits by hand.
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
	which is what the Ledidi optimization needs in order to propose edits.


	Parameters
	----------
	seq_len: int, optional
		The length of the sequences this model accepts. Default is 12.

	n_outputs: int, optional
		The number of outputs the dense layer produces. Default is 3.
	"""

	def __init__(self, seq_len=12, n_outputs=3):
		super(FlattenDense, self).__init__()
		self.seq_len = seq_len
		self.dense = torch.nn.Linear(seq_len * 4, n_outputs)

	def forward(self, X):
		X = X.reshape(X.shape[0], self.seq_len * 4)
		return self.dense(X)


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
