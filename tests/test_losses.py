# test_losses.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from ledidi.losses import MinGap

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def y_hat():
	return torch.tensor([
		[1.0, 2.0, 3.0, 4.0],
		[0.0, -1.0, 5.0, 2.0]
	])


@pytest.fixture
def in_mask():
	return torch.tensor([True, True, False, False])


###


def test_mingap_init(in_mask):
	loss = MinGap(in_mask)
	assert torch.equal(loss.in_mask, in_mask)


def test_mingap_return_shape(y_hat, in_mask):
	loss = MinGap(in_mask)
	value = loss(y_hat, None)

	assert isinstance(value, torch.Tensor)
	assert value.shape == ()
	assert value.dtype == torch.float32


def test_mingap_value(y_hat, in_mask):
	# Row 0: on-target min(1, 2)=1, off-target max(3, 4)=4 -> gap 3
	# Row 1: on-target min(0, -1)=-1, off-target max(5, 2)=5 -> gap 6
	# mean(3, 6) = 4.5
	loss = MinGap(in_mask)
	value = loss(y_hat, None)
	assert_almost_equal(value.item(), 4.5, 4)


def test_mingap_y_bar_ignored(y_hat, in_mask):
	loss = MinGap(in_mask)

	a = loss(y_hat, None)
	b = loss(y_hat, torch.tensor([100.0, -100.0]))
	c = loss(y_hat, torch.randn(17, 3))

	assert_almost_equal(a.item(), b.item(), 4)
	assert_almost_equal(a.item(), c.item(), 4)


def test_mingap_single_on_target(y_hat):
	# Only the first output is on-target; the other three are off-target.
	# Row 0: on min(1)=1, off max(2, 3, 4)=4 -> 3
	# Row 1: on min(0)=0, off max(-1, 5, 2)=5 -> 5
	# mean(3, 5) = 4
	in_mask = torch.tensor([True, False, False, False])
	loss = MinGap(in_mask)
	assert_almost_equal(loss(y_hat, None).item(), 4.0, 4)


def test_mingap_single_off_target(y_hat):
	in_mask = torch.tensor([True, True, True, False])
	# Row 0: on min(1, 2, 3)=1, off max(4)=4 -> 3
	# Row 1: on min(0, -1, 5)=-1, off max(2)=2 -> 3
	# mean(3, 3) = 3
	loss = MinGap(in_mask)
	assert_almost_equal(loss(y_hat, None).item(), 3.0, 4)


def test_mingap_single_row(in_mask):
	y_hat = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
	loss = MinGap(in_mask)
	assert_almost_equal(loss(y_hat, None).item(), 3.0, 4)


def test_mingap_minimized_when_separated(in_mask):
	# On-targets high, off-targets low -> negative (well-separated) loss.
	y_hat = torch.tensor([[10.0, 9.0, 0.0, 1.0]])
	loss = MinGap(in_mask)
	assert loss(y_hat, None).item() < 0


def test_mingap_gradient(y_hat, in_mask):
	y_hat = y_hat.clone().requires_grad_(True)
	loss = MinGap(in_mask)

	value = loss(y_hat, None)
	value.backward()

	assert y_hat.grad is not None
	assert y_hat.grad.shape == y_hat.shape
	# Each row contributes its argmin on-target (-) and argmax off-target (+).
	expected = torch.tensor([
		[-0.5, 0.0, 0.0, 0.5],
		[0.0, -0.5, 0.5, 0.0]
	])
	assert_array_almost_equal(y_hat.grad.numpy(), expected.numpy(), 4)


###
# MinGap -- input validation


def test_mingap_in_mask_non_bool():
	with pytest.raises(ValueError):
		MinGap(torch.tensor([1, 1, 0, 0]))


def test_mingap_in_mask_all_true():
	with pytest.raises(ValueError):
		MinGap(torch.tensor([True, True, True, True]))


def test_mingap_in_mask_all_false():
	with pytest.raises(ValueError):
		MinGap(torch.tensor([False, False, False, False]))
