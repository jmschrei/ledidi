# test_losses.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from ledidi.losses import MinGap
from ledidi.losses import GapLoss

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


###
# GapLoss


def test_gaploss_init(in_mask):
	loss = GapLoss(in_mask)

	assert torch.equal(loss.in_mask, in_mask)
	assert loss.off_reduction == 'max'
	assert loss.floor is None
	assert loss.ceiling is None
	assert loss.ceiling_weight == 2.0


def test_gaploss_return_shape(y_hat, in_mask):
	loss = GapLoss(in_mask)
	value = loss(y_hat, None)

	assert isinstance(value, torch.Tensor)
	assert value.shape == ()
	assert value.dtype == torch.float32


def test_gaploss_defaults_match_mingap(y_hat, in_mask):
	# With no bounds and a max reduction, GapLoss is exactly MinGap.
	a = GapLoss(in_mask)(y_hat, None)
	b = MinGap(in_mask)(y_hat, None)
	assert_almost_equal(a.item(), b.item(), 4)


def test_gaploss_y_bar_ignored(y_hat, in_mask):
	loss = GapLoss(in_mask, floor=0.0, ceiling=3.0)

	a = loss(y_hat, None)
	b = loss(y_hat, torch.tensor([100.0, -100.0]))

	assert_almost_equal(a.item(), b.item(), 4)


###
# GapLoss -- off-target reduction


def test_gaploss_mean_reduction_value(y_hat, in_mask):
	# Row 0: on min(1, 2)=1, off mean(3, 4)=3.5 -> 2.5
	# Row 1: on min(0, -1)=-1, off mean(5, 2)=3.5 -> 4.5
	# mean(2.5, 4.5) = 3.5
	loss = GapLoss(in_mask, off_reduction='mean')
	assert_almost_equal(loss(y_hat, None).item(), 3.5, 4)


def test_gaploss_mean_differs_from_max(y_hat, in_mask):
	a = GapLoss(in_mask, off_reduction='max')(y_hat, None)
	b = GapLoss(in_mask, off_reduction='mean')(y_hat, None)
	assert b.item() < a.item()


def test_gaploss_mean_sees_all_off_targets(in_mask):
	# Lowering a non-maximal off-target changes the mean but not the max.
	y_a = torch.tensor([[5.0, 5.0, 1.0, 4.0]])
	y_b = torch.tensor([[5.0, 5.0, -3.0, 4.0]])

	max_a = GapLoss(in_mask, off_reduction='max')(y_a, None)
	max_b = GapLoss(in_mask, off_reduction='max')(y_b, None)
	assert_almost_equal(max_a.item(), max_b.item(), 4)

	mean_a = GapLoss(in_mask, off_reduction='mean')(y_a, None)
	mean_b = GapLoss(in_mask, off_reduction='mean')(y_b, None)
	assert mean_b.item() < mean_a.item()


###
# GapLoss -- floor


def test_gaploss_floor_value(y_hat, in_mask):
	# Off-targets clamped up to 3.5 before the max.
	# Row 0: on 1, off max(3.5, 4)=4 -> 3.  Row 1: on -1, off max(5, 3.5)=5 -> 6.
	loss = GapLoss(in_mask, floor=3.5)
	assert_almost_equal(loss(y_hat, None).item(), 4.5, 4)


def test_gaploss_floor_no_credit_below(in_mask):
	# Once every off-target is at or under the floor, pushing them lower
	# changes nothing.
	at_floor = torch.tensor([[5.0, 5.0, -1.0, -1.0]])
	far_below = torch.tensor([[5.0, 5.0, -9.0, -9.0]])
	loss = GapLoss(in_mask, floor=-1.0)

	a = loss(at_floor, None)
	b = loss(far_below, None)
	assert_almost_equal(a.item(), b.item(), 4)


def test_gaploss_floor_no_gradient_below(in_mask):
	y_hat = torch.tensor([[5.0, 5.0, -9.0, -9.0]], requires_grad=True)
	loss = GapLoss(in_mask, floor=-1.0)
	loss(y_hat, None).backward()

	# The clamped off-target entries are flat, so they receive no gradient.
	assert_array_almost_equal(y_hat.grad[:, 2:].numpy(), [[0.0, 0.0]], 4)


def test_gaploss_floor_vector_matches_float(y_hat, in_mask):
	a = GapLoss(in_mask, floor=3.5)(y_hat, None)
	b = GapLoss(in_mask, floor=torch.full((4,), 3.5))(y_hat, None)
	assert_almost_equal(a.item(), b.item(), 4)


def test_gaploss_floor_ignores_on_target_entries(y_hat, in_mask):
	# Only the off-target entries of the floor are read.
	a = GapLoss(in_mask, floor=torch.tensor([0.0, 0.0, 3.5, 3.5]))(y_hat, None)
	b = GapLoss(in_mask, floor=torch.tensor([99.0, 99.0, 3.5, 3.5]))(y_hat, None)
	assert_almost_equal(a.item(), b.item(), 4)


###
# GapLoss -- ceiling


def test_gaploss_ceiling_inactive_below(y_hat, in_mask):
	# Every on-target value here is below the ceiling, so the penalty is zero.
	a = GapLoss(in_mask)(y_hat, None)
	b = GapLoss(in_mask, ceiling=50.0)(y_hat, None)
	assert_almost_equal(a.item(), b.item(), 4)


def test_gaploss_ceiling_value(in_mask):
	# on min(6, 7)=6, ceiling 4 -> over 2, penalty 2 * 2^2 = 8
	# off max(0, 0)=0 -> gap 0 - 6 = -6; total -6 + 8 = 2
	y_hat = torch.tensor([[6.0, 7.0, 0.0, 0.0]])
	loss = GapLoss(in_mask, ceiling=4.0, ceiling_weight=2.0)
	assert_almost_equal(loss(y_hat, None).item(), 2.0, 4)


def test_gaploss_ceiling_weight_scales_penalty(in_mask):
	y_hat = torch.tensor([[6.0, 7.0, 0.0, 0.0]])
	a = GapLoss(in_mask, ceiling=4.0, ceiling_weight=1.0)(y_hat, None)
	b = GapLoss(in_mask, ceiling=4.0, ceiling_weight=2.0)(y_hat, None)
	# Penalties are 1 * 4 = 4 and 2 * 4 = 8 on the same -6 gap.
	assert_almost_equal(a.item(), -2.0, 4)
	assert_almost_equal(b.item(), 2.0, 4)


def test_gaploss_ceiling_uses_minimum_on_target_entry(in_mask):
	# With several on-targets the lowest ceiling entry binds.
	y_hat = torch.tensor([[6.0, 7.0, 0.0, 0.0]])
	a = GapLoss(in_mask, ceiling=torch.tensor([4.0, 9.0, 0.0, 0.0]))(y_hat, None)
	b = GapLoss(in_mask, ceiling=4.0)(y_hat, None)
	assert_almost_equal(a.item(), b.item(), 4)


def test_gaploss_ceiling_stationary_point(in_mask):
	# The on-target gradient -1 + 2*w*(on - ceiling) vanishes at
	# ceiling + 1/(2w), which is where an unconstrained optimizer lands.
	for weight in (1.0, 2.0, 5.0):
		on = torch.tensor([1.0], requires_grad=True)
		optimizer = torch.optim.Adam([on], lr=0.02)
		loss = GapLoss(in_mask, ceiling=4.0, ceiling_weight=weight)

		for i in range(4000):
			y_hat = torch.cat([on, on, torch.zeros(2)]).unsqueeze(0)
			optimizer.zero_grad()
			loss(y_hat, None).backward()
			optimizer.step()

		assert_almost_equal(float(on.detach()), 4.0 + 1 / (2 * weight), 2)


###
# GapLoss -- input validation


def test_gaploss_in_mask_non_bool():
	with pytest.raises(ValueError):
		GapLoss(torch.tensor([1, 1, 0, 0]))


def test_gaploss_in_mask_all_true():
	with pytest.raises(ValueError):
		GapLoss(torch.tensor([True, True, True, True]))


def test_gaploss_in_mask_all_false():
	with pytest.raises(ValueError):
		GapLoss(torch.tensor([False, False, False, False]))


def test_gaploss_bad_off_reduction(in_mask):
	with pytest.raises(ValueError):
		GapLoss(in_mask, off_reduction='sum')


def test_gaploss_floor_wrong_shape(in_mask):
	with pytest.raises(ValueError):
		GapLoss(in_mask, floor=torch.zeros(3))


def test_gaploss_ceiling_wrong_shape(in_mask):
	with pytest.raises(ValueError):
		GapLoss(in_mask, ceiling=torch.zeros(7))


def test_gaploss_ceiling_weight_non_numeric(in_mask):
	with pytest.raises(TypeError):
		GapLoss(in_mask, ceiling=1.0, ceiling_weight="2")


def test_gaploss_ceiling_weight_non_positive(in_mask):
	with pytest.raises(ValueError):
		GapLoss(in_mask, ceiling=1.0, ceiling_weight=0.0)
