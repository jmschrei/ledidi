# test_ledidi.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from ledidi.ledidi import ledidi
from ledidi.ledidi import Ledidi
from ledidi.ledidi import _gumbel_softmax_hard

from .toy_models import SumModel
from .toy_models import FlattenDense
from .toy_models import SmallDeepSEA
from .toy_models import ConvAvgDense

from numpy.testing import assert_raises
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal


def _model():
	torch.manual_seed(0)
	return FlattenDense(seq_len=12, n_outputs=3)


@pytest.fixture
def model():
	return _model()


@pytest.fixture
def y_bar():
	return torch.tensor([[5.0, -5.0, 0.0]])


###
# _gumbel_softmax_hard helper
#
# These guard the claim that the seeded sampling path is identical to
# torch.nn.functional.gumbel_softmax(..., hard=True). They will fail both if
# the helper is edited incorrectly and if a future torch release changes the
# gumbel-softmax implementation out from under it.


def test_gumbel_softmax_hard_is_one_hot():
	logits = torch.randn(4, 4, 8)
	generator = torch.Generator().manual_seed(0)
	y = _gumbel_softmax_hard(logits, 1.0, 1, generator)

	assert torch.all(y.sum(dim=1) == 1)
	assert set(torch.unique(y).tolist()) == {0.0, 1.0}


def test_global_rng_matches_generator_stream():
	# The matched-seed comparisons below rely on the global default RNG and a
	# fresh torch.Generator producing the identical stream for the same seed.
	for seed in range(4):
		torch.manual_seed(seed)
		a = torch.empty(500).exponential_()
		b = torch.empty(500).exponential_(generator=torch.Generator().manual_seed(seed))
		assert torch.equal(a, b)


def test_gumbel_softmax_hard_matches_reference():
	# Seeding the global RNG and the generator identically must yield bit-for-bit
	# identical output. Swept over seeds, temperatures, dtypes, and contiguity --
	# including the expanded, non-contiguous logits that forward() produces.
	mismatches = []
	for seed in range(3):
		for tau in (0.1, 1.0, 10.0):
			for dtype in (torch.float32, torch.float64):
				torch.manual_seed(seed)
				logits = torch.randn(16, 4, 25, dtype=dtype)
				torch.manual_seed(100 + seed)
				ref = torch.nn.functional.gumbel_softmax(logits, tau=tau,
					hard=True, dim=1)
				g = torch.Generator().manual_seed(100 + seed)
				mine = _gumbel_softmax_hard(logits, tau, 1, g)
				if not torch.equal(mine, ref):
					mismatches.append(('contiguous', seed, tau, dtype))

				base = torch.randn(1, 4, 25, dtype=dtype)
				logits = base.expand(16, 4, 25)
				torch.manual_seed(200 + seed)
				ref = torch.nn.functional.gumbel_softmax(logits, tau=tau,
					hard=True, dim=1)
				g = torch.Generator().manual_seed(200 + seed)
				mine = _gumbel_softmax_hard(logits, tau, 1, g)
				if not torch.equal(mine, ref):
					mismatches.append(('expanded', seed, tau, dtype))

	assert mismatches == []


def test_gumbel_softmax_hard_gradient_matches_reference():
	# The optimization depends on the straight-through gradient, so the backward
	# pass must match F.gumbel_softmax bitwise under matched noise.
	weight = torch.arange(1, 5).view(1, 4, 1).float()

	for seed in range(4):
		for tau in (0.1, 1.0, 5.0):
			base = torch.randn(1, 4, 25)

			lf = base.expand(16, 4, 25).clone().requires_grad_(True)
			torch.manual_seed(seed)
			(torch.nn.functional.gumbel_softmax(lf, tau=tau, hard=True, dim=1)
				* weight).sum().backward()

			lm = base.expand(16, 4, 25).clone().requires_grad_(True)
			g = torch.Generator().manual_seed(seed)
			(_gumbel_softmax_hard(lm, tau, 1, g) * weight).sum().backward()

			assert torch.equal(lf.grad, lm.grad)


def test_gumbel_softmax_hard_matches_reference_input_mask_logits():
	# Reproduce the -inf logits input_mask produces: at each masked position one
	# channel (the original character) is finite and the rest are -inf. Output
	# must match F bitwise, contain no NaN, and freeze the masked positions.
	idxs = torch.randint(0, 4, (1, 25), generator=torch.Generator().manual_seed(0))
	X = torch.zeros(1, 4, 25)
	X.scatter_(1, idxs.unsqueeze(1), 1.0)

	mask = torch.zeros(25, dtype=torch.bool)
	mask[::3] = True
	weights = torch.zeros(1, 4, 25)
	weights[:, :, mask] = float("-inf")
	weights[X.bool()] = 0.0
	logits = (torch.log(X + 1e-4) + weights).expand(16, 4, 25)

	torch.manual_seed(11)
	ref = torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=True, dim=1)
	g = torch.Generator().manual_seed(11)
	mine = _gumbel_softmax_hard(logits, 1.0, 1, g)

	assert torch.equal(mine, ref)
	assert not torch.isnan(mine).any()
	assert torch.all(mine[:, :, mask] == X[:, :, mask])


def test_gumbel_softmax_hard_distribution_matches_softmax():
	# Over many independent draws the hard sample selects category k with
	# probability softmax(logits)_k. Both the helper and F.gumbel_softmax must
	# converge to that theoretical distribution.
	torch.manual_seed(0)
	logits = torch.randn(1, 4, 1)
	theory = torch.softmax(logits, dim=1).flatten()

	big = logits.expand(50000, 4, 1)

	g = torch.Generator().manual_seed(0)
	freq_mine = _gumbel_softmax_hard(big, 1.0, 1, g).mean(dim=0).flatten()

	torch.manual_seed(0)
	freq_ref = torch.nn.functional.gumbel_softmax(big, tau=1.0, hard=True,
		dim=1).mean(dim=0).flatten()

	assert_array_almost_equal(freq_mine.numpy(), theory.numpy(), 2)
	assert_array_almost_equal(freq_ref.numpy(), theory.numpy(), 2)


###
# Ledidi.__init__


def test_ledidi_init_freezes_model(model):
	designer = Ledidi(model, shape=(4, 12), verbose=False)

	assert not any(p.requires_grad for p in designer.model.parameters())
	assert not designer.model.training


def test_ledidi_init_default_weights(model):
	designer = Ledidi(model, shape=(4, 12), verbose=False)

	assert isinstance(designer.weights, torch.nn.Parameter)
	assert designer.weights.shape == (1, 4, 12)
	assert designer.weights.requires_grad
	assert_array_almost_equal(designer.weights.detach().numpy(),
		torch.zeros(1, 4, 12).numpy(), 4)


def test_ledidi_init_initial_weights(model):
	weights = torch.randn(1, 4, 12)
	expected = weights.clone()
	designer = Ledidi(model, shape=(4, 12), initial_weights=weights, verbose=False)

	assert designer.weights.requires_grad
	assert_array_almost_equal(designer.weights.detach().numpy(), expected.numpy(), 4)


def test_ledidi_init_target_none(model):
	designer = Ledidi(model, shape=(4, 12), target=None, verbose=False)
	assert designer.target == slice(None)


def test_ledidi_init_target_int(model):
	designer = Ledidi(model, shape=(4, 12), target=2, verbose=False)
	assert designer.target == slice(2, 3)


def test_ledidi_init_random_state_stored(model):
	designer = Ledidi(model, shape=(4, 12), random_state=7, verbose=False)
	assert designer.random_state == 7
	assert designer._generator is None


###
# Ledidi.forward


def test_ledidi_forward_shape(model, X):
	designer = Ledidi(model, shape=(4, 12), batch_size=4, verbose=False)
	X_hat = designer(X)

	assert X_hat.shape == (4, 4, 12)
	assert torch.all(X_hat.sum(dim=1) == 1)
	assert set(torch.unique(X_hat).tolist()) == {0.0, 1.0}


def test_ledidi_forward_batch_size(model, X):
	designer = Ledidi(model, shape=(4, 12), batch_size=9, verbose=False)
	assert designer(X).shape == (9, 4, 12)


def test_ledidi_forward_default_path_runs(model, X):
	# random_state=None draws from the global RNG via F.gumbel_softmax.
	designer = Ledidi(model, shape=(4, 12), batch_size=4, verbose=False)
	torch.manual_seed(0)
	a = designer(X)
	assert a.shape == (4, 4, 12)


def test_ledidi_forward_seeded_reproducible(model, X):
	a = Ledidi(model, shape=(4, 12), batch_size=4, random_state=0, verbose=False)(X)
	b = Ledidi(_model(), shape=(4, 12), batch_size=4, random_state=0, verbose=False)(X)
	assert torch.equal(a, b)


def test_ledidi_forward_seed_advances(X):
	# Successive draws from the same seeded designer differ from one another.
	w = torch.randn(1, 4, 12, generator=torch.Generator().manual_seed(0)) * 12
	designer = Ledidi(_model(), shape=(4, 12), batch_size=4, initial_weights=w,
		random_state=0, verbose=False)
	assert not torch.equal(designer(X), designer(X))


def test_ledidi_forward_does_not_touch_global_rng(model, X):
	designer = Ledidi(model, shape=(4, 12), batch_size=4, random_state=0, verbose=False)

	torch.manual_seed(123)
	before = torch.rand(5)
	_ = designer(X)
	torch.manual_seed(123)
	after = torch.rand(5)
	assert torch.equal(before, after)


def test_ledidi_forward_lazy_generator(model, X):
	designer = Ledidi(model, shape=(4, 12), batch_size=4, random_state=0, verbose=False)
	assert designer._generator is None
	designer(X)
	assert designer._generator is not None


###
# Ledidi.fit_transform


def test_ledidi_fit_transform_shape(model, X, y_bar):
	designer = Ledidi(model, shape=(4, 12), batch_size=4, max_iter=50,
		random_state=0, verbose=False)
	X_hat = designer.fit_transform(X, y_bar)

	assert X_hat.shape == (4, 4, 12)
	assert torch.all(X_hat.sum(dim=1) == 1)
	assert set(torch.unique(X_hat).tolist()) == {0.0, 1.0}


def test_ledidi_fit_transform_reproducible(X, y_bar):
	a = Ledidi(_model(), shape=(4, 12), batch_size=4, max_iter=100,
		random_state=0, verbose=False).fit_transform(X, y_bar)
	b = Ledidi(_model(), shape=(4, 12), batch_size=4, max_iter=100,
		random_state=0, verbose=False).fit_transform(X, y_bar)
	assert torch.equal(a, b)


def test_ledidi_fit_transform_regression(model, X, y_bar):
	designer = Ledidi(model, shape=(4, 12), batch_size=4, max_iter=100,
		random_state=0, verbose=False)
	X_hat = designer.fit_transform(X, y_bar)

	n_edits = (X_hat != X).sum(dim=1).bool().sum(dim=-1)
	assert_array_almost_equal(n_edits.numpy(), [9, 9, 9, 9])

	y_hat = model(X_hat).mean(dim=0)
	assert_array_almost_equal(y_hat.detach().numpy(), [0.5408, -0.7059, 0.327], 4)


def test_ledidi_fit_transform_reduces_output_loss(model, X, y_bar):
	designer = Ledidi(model, shape=(4, 12), batch_size=4, max_iter=100,
		random_state=0, verbose=False)
	X_hat = designer.fit_transform(X, y_bar)

	loss = torch.nn.MSELoss()
	before = loss(model(X), y_bar).item()
	after = loss(model(X_hat).mean(dim=0, keepdim=True), y_bar).item()
	assert after < before


def test_ledidi_fit_transform_history(model, X, y_bar):
	designer = Ledidi(model, shape=(4, 12), batch_size=4, max_iter=50,
		random_state=0, return_history=True, verbose=False)
	X_hat, history = designer.fit_transform(X, y_bar)

	assert X_hat.shape == (4, 4, 12)
	assert history['batch_size'] == 4
	for key in ('edits', 'input_loss', 'output_loss', 'total_loss'):
		assert len(history[key]) == 50


def test_ledidi_fit_transform_no_improvement_fallback(model, X):
	# With a target equal to the current prediction and a huge input penalty,
	# no edit can lower the total loss, so the designer returns the original
	# sequence expanded to the batch -- shape (batch_size, 4, 12), not (1, ...).
	# This is a regression test for a fallback that previously returned (1, ...)
	# and crashed the affinity-catalog path in ledidi().
	y_bar = model(X).detach()
	designer = Ledidi(model, shape=(4, 12), batch_size=4, max_iter=10, l=1e6,
		random_state=0, verbose=False)
	X_hat = designer.fit_transform(X, y_bar)

	assert X_hat.shape == (4, 4, 12)
	for i in range(4):
		assert_array_almost_equal(X_hat[i].numpy(), X[0].numpy(), 4)


def test_ledidi_fit_transform_input_mask(model, X, y_bar):
	# Masked positions can never be edited.
	input_mask = torch.zeros(12, dtype=torch.bool)
	input_mask[:6] = True
	designer = Ledidi(model, shape=(4, 12), batch_size=4, max_iter=50,
		input_mask=input_mask, random_state=0, verbose=False)
	X_hat = designer.fit_transform(X, y_bar)

	edited = (X_hat != X).sum(dim=1).bool()
	assert edited[:, input_mask].sum() == 0


def test_ledidi_fit_transform_updates_weights(model, X, y_bar):
	designer = Ledidi(model, shape=(4, 12), batch_size=4, max_iter=100,
		random_state=0, verbose=False)
	designer.fit_transform(X, y_bar)
	# The weights are replaced with the best weights, which must have moved away
	# from the zero initialization for at least one position.
	assert designer.weights.abs().sum() > 0


###
# ledidi() wrapper -- input validation


@pytest.mark.parametrize("n_repeats", [0, -1, 1.5, "3"])
def test_ledidi_invalid_n_repeats(model, X, y_bar, n_repeats):
	assert_raises(ValueError, ledidi, model, X, y_bar, n_repeats=n_repeats,
		device='cpu', verbose=False)


@pytest.mark.parametrize("n_samples", [0, -1, 2.5, "5"])
def test_ledidi_invalid_n_samples(model, X, y_bar, n_samples):
	assert_raises(ValueError, ledidi, model, X, y_bar, n_samples=n_samples,
		device='cpu', verbose=False)


###
# ledidi() wrapper -- shapes and behavior


def test_ledidi_wrapper_basic(model, X, y_bar):
	X_hat = ledidi(model, X, y_bar, batch_size=4, max_iter=50, random_state=0,
		device='cpu', verbose=False)
	assert X_hat.shape == (4, 4, 12)
	assert torch.all(X_hat.sum(dim=1) == 1)


def test_ledidi_wrapper_reproducible(X, y_bar):
	a = ledidi(_model(), X, y_bar, batch_size=4, max_iter=50, random_state=0,
		device='cpu', verbose=False)
	b = ledidi(_model(), X, y_bar, batch_size=4, max_iter=50, random_state=0,
		device='cpu', verbose=False)
	assert torch.equal(a, b)


def test_ledidi_wrapper_n_repeats(model, X, y_bar):
	X_hat = ledidi(model, X, y_bar, n_repeats=3, batch_size=4, max_iter=20,
		random_state=0, device='cpu', verbose=False)
	assert X_hat.shape == (3, 4, 4, 12)


def test_ledidi_wrapper_catalog(model, X, y_bar):
	X_hat = ledidi(model, X, [y_bar, y_bar * 0], batch_size=4, max_iter=20,
		random_state=0, device='cpu', verbose=False)
	assert X_hat.shape == (2, 4, 4, 12)


def test_ledidi_wrapper_catalog_and_repeats(model, X, y_bar):
	X_hat = ledidi(model, X, [y_bar, y_bar * 0], n_repeats=2, batch_size=4,
		max_iter=20, random_state=0, device='cpu', verbose=False)
	assert X_hat.shape == (2, 2, 4, 4, 12)


def test_ledidi_wrapper_n_samples(model, X, y_bar):
	X_hat = ledidi(model, X, y_bar, n_samples=10, batch_size=4, max_iter=20,
		random_state=0, device='cpu', verbose=False)
	assert X_hat.shape == (10, 4, 12)
	assert torch.all(X_hat.sum(dim=1) == 1)


def test_ledidi_wrapper_return_designer(model, X, y_bar):
	out = ledidi(model, X, y_bar, batch_size=4, max_iter=20, random_state=0,
		device='cpu', verbose=False, return_designer=True)
	assert isinstance(out, list) and len(out) == 2
	assert isinstance(out[1], Ledidi)


def test_ledidi_wrapper_return_history(model, X, y_bar):
	out = ledidi(model, X, y_bar, batch_size=4, max_iter=20, random_state=0,
		device='cpu', verbose=False, return_history=True)
	assert len(out) == 2
	assert sorted(out[1].keys()) == ['batch_size', 'edits', 'input_loss',
		'output_loss', 'total_loss']


def test_ledidi_wrapper_return_designer_and_history(model, X, y_bar):
	out = ledidi(model, X, y_bar, batch_size=4, max_iter=20, random_state=0,
		device='cpu', verbose=False, return_designer=True, return_history=True)
	assert len(out) == 3
	assert isinstance(out[1], Ledidi)
	assert len(out[2]['total_loss']) == 20


def test_ledidi_wrapper_target(model, X):
	# A single-task target with a matching single-value y_bar.
	X_hat = ledidi(model, X, torch.tensor([[5.0]]), target=0, batch_size=4,
		max_iter=20, random_state=0, device='cpu', verbose=False)
	assert X_hat.shape == (4, 4, 12)


###
# Complete design runs on a long (100 bp) sequence, across the toy models from
# the tangermeme test suite and several initializations. These run the full
# optimization to convergence (default max_iter / early stopping) and pin the
# resulting edits as regression values. Because the seeded sampler is fully
# deterministic and the edits are selected by argmax, these gold values are
# stable run-to-run; they will flag any change to the design behavior.


def _long_X(random_state):
	"""A single one-hot encoded sequence of shape (1, 4, 100)."""

	idxs = torch.randint(0, 4, (1, 100),
		generator=torch.Generator().manual_seed(random_state))
	X = torch.zeros(1, 4, 100)
	X.scatter_(1, idxs.unsqueeze(1), 1.0)
	return X


def _edited_positions(X, X_hat, row=0):
	"""The positions at which designed sequence `row` differs from `X`."""

	return torch.where((X != X_hat).sum(dim=1).bool()[row])[0]


# The exact positions edited in the first designed sequence when SumModel is
# pushed toward the composition [40, 10, 40, 10] from the random_state=0 seed.
SUM_MODEL_EDITS_RS0 = [
	2, 4, 5, 6, 7, 13, 21, 22, 27, 28, 30, 34, 37, 38, 43, 44, 45, 47, 49, 52,
	53, 59, 60, 64, 70, 75, 76, 78, 83, 84, 85, 86, 87, 88, 91, 94, 97, 98
]

# The nucleotide each of those positions was edited to (mostly A/G, the
# up-weighted channels, with a few C to satisfy the C=10 part of the target).
SUM_MODEL_EDIT_CHARS_RS0 = "AGGAGGAGGGACAAACAAGGGGGAAGAGAGGAAGGGGG"


def test_design_sum_model_composition_edits():
	# SumModel's output is the per-channel nucleotide count, so a target of
	# [40, 10, 40, 10] is a direct objective over composition: raise A and G,
	# lower C and T. The design should hit that composition exactly and every
	# edit should convert a position to one of the up-weighted nucleotides.
	X = _long_X(0)
	y_bar = torch.tensor([[40.0, 10.0, 40.0, 10.0]])

	X_hat = ledidi(SumModel(), X, y_bar, random_state=0, device='cpu',
		verbose=False)

	assert X_hat.shape == (16, 4, 100)
	assert torch.all(X_hat.sum(dim=1) == 1)

	# The first designed sequence reaches the target composition exactly.
	assert X_hat[0].sum(dim=-1).tolist() == [40, 10, 40, 10]

	# The exact edited positions and the nucleotide each became are pinned as
	# regression values -- this is the literal record of what was edited.
	edited = _edited_positions(X, X_hat).tolist()
	assert edited == SUM_MODEL_EDITS_RS0

	new_chars = "".join("ACGT"[c] for c in X_hat[0, :, edited].argmax(dim=0).tolist())
	assert new_chars == SUM_MODEL_EDIT_CHARS_RS0


def test_design_sum_model_across_initializations():
	# Across several initializations the design is reproducible and reaches (or
	# nearly reaches) the target composition, with a pinned number of edits.
	y_bar = torch.tensor([[40.0, 10.0, 40.0, 10.0]])
	expected = {
		0: {'n_edits': 38, 'comp': [40, 10, 40, 10]},
		1: {'n_edits': 37, 'comp': [40, 10, 40, 10]},
		2: {'n_edits': 26, 'comp': [41, 9, 40, 10]},
	}

	for rs, exp in expected.items():
		X = _long_X(rs)
		X_hat = ledidi(SumModel(), X, y_bar, random_state=rs, device='cpu',
			verbose=False)
		X_hat2 = ledidi(SumModel(), X, y_bar, random_state=rs, device='cpu',
			verbose=False)

		assert torch.equal(X_hat, X_hat2)
		assert len(_edited_positions(X, X_hat)) == exp['n_edits']
		assert X_hat[0].sum(dim=-1).tolist() == exp['comp']


def test_design_flatten_dense():
	# A long-sequence design against the linear FlattenDense oracle, pinning the
	# edit counts and the designed predictions and checking that the design
	# moved the outputs toward the [5, -5, 0] target.
	expected = {
		0: {'n_edits': 56, 'pred': [2.0901, -1.8956, -0.0103]},
		1: {'n_edits': 44, 'pred': [2.1669, -1.8537, 0.0298]},
	}
	y_bar = torch.tensor([[5.0, -5.0, 0.0]])

	for rs, exp in expected.items():
		torch.manual_seed(0)
		model = FlattenDense(seq_len=100, n_outputs=3)
		X = _long_X(rs)

		X_hat = ledidi(model, X, y_bar, random_state=rs, device='cpu',
			verbose=False)
		X_hat2 = ledidi(model, X, y_bar, random_state=rs, device='cpu',
			verbose=False)

		assert torch.equal(X_hat, X_hat2)
		assert len(_edited_positions(X, X_hat)) == exp['n_edits']

		pred = model(X_hat).mean(dim=0)
		assert_array_almost_equal(pred.detach().numpy(), exp['pred'], 4)

		# Output 0 was driven up and output 1 down relative to the original.
		orig = model(X)[0]
		assert pred[0] > orig[0]
		assert pred[1] < orig[1]


def test_design_across_tangermeme_models():
	# A complete design run against each toy model from the tangermeme suite, on
	# a long sequence and a couple of initializations. The convolutional models
	# use a smaller input-loss weight so that edits are actually proposed
	# through their weaker gradients. Each run must be reproducible, return a
	# valid batch of one-hot sequences, and propose at least one edit.
	# `check_repro` re-runs the design and asserts identical output. It is
	# enabled only for the convolutional models, whose determinism through the
	# seeded sampler is not exercised by the linear-model tests above.
	cases = [
		(lambda: SumModel(), torch.tensor([[40.0, 10.0, 40.0, 10.0]]), {}, False),
		(lambda: FlattenDense(seq_len=100, n_outputs=3),
			torch.tensor([[5.0, -5.0, 0.0]]), {}, False),
		(lambda: SmallDeepSEA(n_outputs=1), torch.tensor([[10.0]]),
			{'l': 0.01}, True),
		(lambda: ConvAvgDense(n_outputs=1), torch.tensor([[5.0]]),
			{'l': 0.001}, True),
	]

	for make_model, y_bar, kwargs, check_repro in cases:
		for rs in (0, 1):
			torch.manual_seed(0)
			model = make_model()
			X = _long_X(rs)

			X_hat = ledidi(model, X, y_bar, random_state=rs, device='cpu',
				verbose=False, **kwargs)

			assert X_hat.shape == (16, 4, 100)
			assert torch.all(X_hat.sum(dim=1) == 1)
			assert len(_edited_positions(X, X_hat)) > 0

			if check_repro:
				torch.manual_seed(0)
				model2 = make_model()
				X_hat2 = ledidi(model2, X, y_bar, random_state=rs,
					device='cpu', verbose=False, **kwargs)
				assert torch.equal(X_hat, X_hat2)
