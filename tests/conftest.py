# conftest.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest


def _random_one_hot(shape, random_state):
	"""Construct a deterministic one-hot tensor of the given shape.

	The hot channel at each position is drawn from a seeded generator so that
	every test that requests a sequence gets the same one, which is what lets
	the regression values below stay pinned across runs.


	Parameters
	----------
	shape: tuple
		A `(n, channels, length)` shape for the returned one-hot tensor.

	random_state: int
		The seed for the generator used to choose the hot channels.


	Returns
	-------
	X: torch.Tensor, shape=shape, dtype=torch.float32
		A one-hot encoded tensor.
	"""

	n, channels, length = shape
	generator = torch.Generator().manual_seed(random_state)

	idxs = torch.randint(0, channels, (n, length), generator=generator)
	X = torch.zeros(n, channels, length, dtype=torch.float32)
	X.scatter_(1, idxs.unsqueeze(1), 1.0)
	return X


@pytest.fixture
def X():
	"""A single one-hot encoded sequence of shape (1, 4, 12)."""

	return _random_one_hot((1, 4, 12), random_state=0)
