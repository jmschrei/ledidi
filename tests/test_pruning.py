# test_pruning.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from ledidi.pruning import greedy_pruning

from .toy_models import SumModel

from numpy.testing import assert_array_almost_equal


def _one_hot_from_chars(chars):
	"""Build a (1, 4, len) one-hot tensor from a list of channel indices."""

	X = torch.zeros(1, 4, len(chars))
	for i, c in enumerate(chars):
		X[0, c, i] = 1.0
	return X


@pytest.fixture
def X():
	# Channels at each of 6 positions.
	return _one_hot_from_chars([0, 1, 2, 3, 0, 1])


@pytest.fixture
def X_hat(X):
	# Edit positions 1, 3, and 4 (channels 1->2, 3->0, 0->3).
	X_hat = torch.clone(X)
	X_hat[0, :, 1] = 0.0; X_hat[0, 2, 1] = 1.0
	X_hat[0, :, 3] = 0.0; X_hat[0, 0, 3] = 1.0
	X_hat[0, :, 4] = 0.0; X_hat[0, 3, 4] = 1.0
	return X_hat


###


def test_greedy_pruning_return_shape(X, X_hat):
	X_m = greedy_pruning(SumModel(), X, X_hat, threshold=1)
	assert X_m.shape == X_hat.shape
	# Output is still a valid one-hot encoding.
	assert torch.all(X_m.sum(dim=1) == 1)
	assert set(torch.unique(X_m).tolist()) == {0.0, 1.0}


def test_greedy_pruning_below_threshold_no_pruning(X, X_hat):
	# Every edit changes two channel counts by 1, so reverting any single edit
	# moves the SumModel output by exactly 2 > threshold=1: nothing is pruned.
	X_m = greedy_pruning(SumModel(), X, X_hat, threshold=1)
	assert_array_almost_equal(X_m.numpy(), X_hat.numpy(), 4)


def test_greedy_pruning_above_threshold_full_pruning(X, X_hat):
	# The change from reverting an edit is measured against the original
	# (fully edited) prediction, so the deviation grows cumulatively as edits
	# are reverted: the three edits score 2, 4, and 6. A threshold above 6
	# therefore prunes all of them and recovers the original sequence.
	X_m = greedy_pruning(SumModel(), X, X_hat, threshold=10)
	assert_array_almost_equal(X_m.numpy(), X.numpy(), 4)


def test_greedy_pruning_cumulative_threshold(X, X_hat):
	# Because the deviation is cumulative (2, 4, 6, ...) a threshold of 3 only
	# admits the first revert (score 2); the second would score 4 > 3 and stops
	# the loop, so exactly one of the three edits is pruned.
	X_m = greedy_pruning(SumModel(), X, X_hat, threshold=3)
	n_edits = (X_m != X).sum(dim=1).bool().sum()
	assert n_edits == 2


def test_greedy_pruning_does_not_mutate_input(X, X_hat):
	X_hat_backup = torch.clone(X_hat)
	_ = greedy_pruning(SumModel(), X, X_hat, threshold=3)
	assert_array_almost_equal(X_hat.numpy(), X_hat_backup.numpy(), 4)


def test_greedy_pruning_no_edits(X):
	# When X_hat == X there are no edits to consider and X is returned as-is.
	X_m = greedy_pruning(SumModel(), X, torch.clone(X), threshold=10)
	assert_array_almost_equal(X_m.numpy(), X.numpy(), 4)


def test_greedy_pruning_target_slicing(X, X_hat):
	# With target=0 only channel-0 counts matter. Reverting an edit that does
	# not touch channel 0 leaves the output unchanged (score 0 < 1 -> pruned),
	# while an edit that adds or removes a channel-0 hot moves it by 1 (not
	# pruned). Position 3 (3->0) and position 4 (0->3) touch channel 0 and so
	# survive; position 1 (1->2) does not and is reverted to X.
	X_m = greedy_pruning(SumModel(), X, X_hat, threshold=1, target=0)

	expected = torch.clone(X_hat)
	expected[0, :, 1] = X[0, :, 1]
	assert_array_almost_equal(X_m.numpy(), expected.numpy(), 4)


def test_greedy_pruning_high_threshold_full_with_target(X, X_hat):
	# A large threshold prunes every channel-0-relevant edit too.
	X_m = greedy_pruning(SumModel(), X, X_hat, threshold=10, target=0)
	assert_array_almost_equal(X_m.numpy(), X.numpy(), 4)


###
# greedy_pruning -- input validation


def test_greedy_pruning_invalid_model(X, X_hat):
	with pytest.raises(TypeError):
		greedy_pruning("not a model", X, X_hat)


@pytest.mark.parametrize("threshold", [0, -1.0])
def test_greedy_pruning_invalid_threshold(X, X_hat, threshold):
	with pytest.raises(ValueError):
		greedy_pruning(SumModel(), X, X_hat, threshold=threshold)


def test_greedy_pruning_invalid_target(X, X_hat):
	with pytest.raises(TypeError):
		greedy_pruning(SumModel(), X, X_hat, target=1.5)


def test_greedy_pruning_non_ohe(X):
	X_hat = torch.full((1, 4, 6), 0.25)
	with pytest.raises(ValueError):
		greedy_pruning(SumModel(), X, X_hat)


def test_greedy_pruning_shape_mismatch(X):
	X_hat = torch.zeros(1, 4, 8)
	X_hat[0, 0, :] = 1.0
	with pytest.raises(ValueError):
		greedy_pruning(SumModel(), X, X_hat)
