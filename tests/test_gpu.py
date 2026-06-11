# test_gpu.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from ledidi.ledidi import ledidi
from ledidi.ledidi import Ledidi
from ledidi.ledidi import _gumbel_softmax_hard

from ledidi.wrappers import DesignWrapper
from ledidi.pruning import greedy_pruning

from .toy_models import SumModel
from .toy_models import FlattenDense
from .toy_models import SumModelKeepdim


# Every test in this module needs a real CUDA device, so the whole file is
# tagged `gpu` -- which the default `addopts = -m 'not gpu'` in pyproject.toml
# deselects -- and additionally skipped if no device is present. Run them with
# `python -m pytest tests/ -m gpu`. These are the GPU counterparts of the
# CPU tests in the sibling files: they re-prove the seeded sampler is bitwise
# identical to F.gumbel_softmax on the device, that the generator lives on (and
# follows the module to) the GPU without touching the global CUDA RNG, and that
# the forward/fit_transform/wrapper/pruning paths run and stay on the device.
pytestmark = [
	pytest.mark.gpu,
	pytest.mark.skipif(not torch.cuda.is_available(),
		reason="CUDA device not available"),
]


def _model():
	torch.manual_seed(0)
	return FlattenDense(seq_len=12, n_outputs=3).cuda()


def _one_hot(length, random_state=0):
	idxs = torch.randint(0, 4, (1, length),
		generator=torch.Generator().manual_seed(random_state))
	X = torch.zeros(1, 4, length)
	X.scatter_(1, idxs.unsqueeze(1), 1.0)
	return X


@pytest.fixture
def model():
	return _model()


@pytest.fixture
def X():
	return _one_hot(12).cuda()


@pytest.fixture
def y_bar():
	return torch.tensor([[5.0, -5.0, 0.0]]).cuda()


###
# _gumbel_softmax_hard helper on the GPU
#
# The CPU suite proves the helper is bitwise identical to F.gumbel_softmax under
# matched seeds. The CUDA default RNG and an explicit CUDA torch.Generator are a
# different code path, so the equivalence has to be re-proven on the device.


def test_gpu_gumbel_softmax_hard_is_one_hot():
	logits = torch.randn(4, 4, 8, device='cuda')
	generator = torch.Generator(device='cuda').manual_seed(0)
	y = _gumbel_softmax_hard(logits, 1.0, 1, generator)

	assert y.is_cuda
	assert torch.all(y.sum(dim=1) == 1)
	assert set(torch.unique(y).tolist()) == {0.0, 1.0}


def test_gpu_global_rng_matches_generator_stream():
	# The matched-seed comparisons below rely on the CUDA default RNG and a fresh
	# CUDA torch.Generator producing the identical stream for the same seed.
	for seed in range(4):
		torch.cuda.manual_seed(seed)
		a = torch.empty(500, device='cuda').exponential_()
		g = torch.Generator(device='cuda').manual_seed(seed)
		b = torch.empty(500, device='cuda').exponential_(generator=g)
		assert torch.equal(a, b)


def test_gpu_gumbel_softmax_hard_matches_reference():
	# Seeding the CUDA default RNG and the explicit generator identically must
	# yield bit-for-bit identical output, including for the expanded,
	# non-contiguous logits that forward() produces.
	mismatches = []
	for seed in range(3):
		for tau in (0.1, 1.0, 10.0):
			logits = torch.randn(16, 4, 25, device='cuda')
			torch.cuda.manual_seed(100 + seed)
			ref = torch.nn.functional.gumbel_softmax(logits, tau=tau,
				hard=True, dim=1)
			g = torch.Generator(device='cuda').manual_seed(100 + seed)
			mine = _gumbel_softmax_hard(logits, tau, 1, g)
			if not torch.equal(mine, ref):
				mismatches.append(('contiguous', seed, tau))

			base = torch.randn(1, 4, 25, device='cuda')
			logits = base.expand(16, 4, 25)
			torch.cuda.manual_seed(200 + seed)
			ref = torch.nn.functional.gumbel_softmax(logits, tau=tau,
				hard=True, dim=1)
			g = torch.Generator(device='cuda').manual_seed(200 + seed)
			mine = _gumbel_softmax_hard(logits, tau, 1, g)
			if not torch.equal(mine, ref):
				mismatches.append(('expanded', seed, tau))

	assert mismatches == []


def test_gpu_gumbel_softmax_hard_gradient_matches_reference():
	# The straight-through gradient must also match F.gumbel_softmax bitwise on
	# the device under matched noise.
	weight = torch.arange(1, 5).view(1, 4, 1).float().cuda()

	for seed in range(3):
		for tau in (0.1, 1.0, 5.0):
			base = torch.randn(1, 4, 25, device='cuda')

			lf = base.expand(16, 4, 25).clone().requires_grad_(True)
			torch.cuda.manual_seed(seed)
			(torch.nn.functional.gumbel_softmax(lf, tau=tau, hard=True, dim=1)
				* weight).sum().backward()

			lm = base.expand(16, 4, 25).clone().requires_grad_(True)
			g = torch.Generator(device='cuda').manual_seed(seed)
			(_gumbel_softmax_hard(lm, tau, 1, g) * weight).sum().backward()

			assert torch.equal(lf.grad, lm.grad)


def test_gpu_gumbel_softmax_hard_distribution_matches_softmax():
	# Over many independent draws the hard sample selects category k with
	# probability softmax(logits)_k on the device too.
	logits = torch.randn(1, 4, 1, device='cuda')
	theory = torch.softmax(logits, dim=1).flatten()

	big = logits.expand(50000, 4, 1)
	g = torch.Generator(device='cuda').manual_seed(0)
	freq = _gumbel_softmax_hard(big, 1.0, 1, g).mean(dim=0).flatten()

	assert torch.allclose(freq, theory, atol=1e-2)


###
# Ledidi.forward on the GPU


def test_gpu_forward_shape(model, X):
	designer = Ledidi(model, shape=(4, 12), batch_size=4, random_state=0,
		verbose=False).cuda()
	X_hat = designer(X)

	assert X_hat.is_cuda
	assert X_hat.shape == (4, 4, 12)
	assert torch.all(X_hat.sum(dim=1) == 1)
	assert set(torch.unique(X_hat).tolist()) == {0.0, 1.0}


def test_gpu_forward_generator_on_cuda(model, X):
	# The generator is created lazily on the first forward and lives on the same
	# device as the logits.
	designer = Ledidi(model, shape=(4, 12), batch_size=4, random_state=0,
		verbose=False).cuda()

	assert designer._generator is None
	designer(X)
	assert designer._generator is not None
	assert designer._generator.device.type == 'cuda'


def test_gpu_forward_seeded_reproducible(X):
	a = Ledidi(_model(), shape=(4, 12), batch_size=4, random_state=0,
		verbose=False).cuda()(X)
	b = Ledidi(_model(), shape=(4, 12), batch_size=4, random_state=0,
		verbose=False).cuda()(X)
	assert torch.equal(a, b)


def test_gpu_forward_does_not_touch_global_cuda_rng(model, X):
	# Seeding through the private generator must leave the global CUDA RNG
	# untouched, so a separately seeded draw is unperturbed by a design step.
	designer = Ledidi(model, shape=(4, 12), batch_size=4, random_state=0,
		verbose=False).cuda()

	torch.cuda.manual_seed(123)
	before = torch.rand(5, device='cuda')
	designer(X)
	torch.cuda.manual_seed(123)
	after = torch.rand(5, device='cuda')
	assert torch.equal(before, after)


def test_gpu_forward_regenerates_generator_on_device_change():
	# A designer moved between devices must rebuild its generator on the new
	# device (the `self._generator.device != logits.device` branch) rather than
	# reuse a generator bound to the old one.
	X = _one_hot(12)

	designer = Ledidi(FlattenDense(seq_len=12, n_outputs=3), shape=(4, 12),
		batch_size=4, random_state=0, verbose=False)

	cpu_out = designer(X)
	assert designer._generator.device.type == 'cpu'
	assert not cpu_out.is_cuda

	designer = designer.cuda()
	cuda_out = designer(X.cuda())
	assert designer._generator.device.type == 'cuda'
	assert cuda_out.is_cuda


###
# Ledidi.fit_transform on the GPU


def test_gpu_fit_transform_reduces_loss(model, X, y_bar):
	designer = Ledidi(model, shape=(4, 12), batch_size=4, max_iter=100,
		random_state=0, verbose=False).cuda()
	X_hat = designer.fit_transform(X, y_bar)

	assert X_hat.is_cuda
	assert X_hat.shape == (4, 4, 12)
	assert torch.all(X_hat.sum(dim=1) == 1)

	loss = torch.nn.MSELoss()
	before = loss(model(X), y_bar).item()
	after = loss(model(X_hat).mean(dim=0, keepdim=True), y_bar).item()
	assert after < before


###
# ledidi() wrapper with device='cuda'


def test_gpu_ledidi_wrapper_basic(model, X, y_bar):
	X_hat = ledidi(model, X, y_bar, batch_size=4, max_iter=50, random_state=0,
		device='cuda', verbose=False)

	assert X_hat.is_cuda
	assert X_hat.shape == (4, 4, 12)
	assert torch.all(X_hat.sum(dim=1) == 1)


def test_gpu_ledidi_wrapper_moves_cpu_inputs():
	# A CPU model and CPU tensors with device='cuda' are moved to the GPU by the
	# wrapper, and the designed output comes back on the GPU.
	model = FlattenDense(seq_len=12, n_outputs=3)
	X = _one_hot(12)
	y_bar = torch.tensor([[5.0, -5.0, 0.0]])

	X_hat = ledidi(model, X, y_bar, batch_size=4, max_iter=20, random_state=0,
		device='cuda', verbose=False)

	assert X_hat.is_cuda
	assert X_hat.shape == (4, 4, 12)


###
# DesignWrapper and greedy_pruning on the GPU


def test_gpu_designwrapper_runs(X):
	wrapper = DesignWrapper([SumModel(), SumModelKeepdim()]).cuda()
	y = wrapper(X)

	assert y.is_cuda
	assert y.shape == (1, 5)


def test_gpu_design_with_designwrapper(X, y_bar):
	# A complete design run against a multi-model oracle on the GPU.
	wrapper = DesignWrapper([FlattenDense(seq_len=12, n_outputs=3)]).cuda()
	X_hat = ledidi(wrapper, X, y_bar, batch_size=4, max_iter=50, random_state=0,
		device='cuda', verbose=False)

	assert X_hat.is_cuda
	assert torch.all(X_hat.sum(dim=1) == 1)


def test_gpu_greedy_pruning_runs():
	# A large threshold prunes every edit and recovers the original sequence,
	# with all tensors kept on the device.
	chars = [0, 1, 2, 3, 0, 1]
	X = torch.zeros(1, 4, len(chars), device='cuda')
	for i, c in enumerate(chars):
		X[0, c, i] = 1.0

	X_hat = torch.clone(X)
	X_hat[0, :, 1] = 0.0; X_hat[0, 2, 1] = 1.0
	X_hat[0, :, 3] = 0.0; X_hat[0, 0, 3] = 1.0
	X_hat[0, :, 4] = 0.0; X_hat[0, 3, 4] = 1.0

	X_m = greedy_pruning(SumModel().cuda(), X, X_hat, threshold=10)

	assert X_m.is_cuda
	assert torch.equal(X_m, X)
