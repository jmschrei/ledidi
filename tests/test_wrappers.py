# test_wrappers.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from ledidi.wrappers import DesignWrapper

from .toy_models import SumModel
from .toy_models import FlattenDense
from .toy_models import SumModelKeepdim

from numpy.testing import assert_array_almost_equal


torch.manual_seed(0)


@pytest.fixture
def X():
	X = torch.zeros(2, 4, 12)
	idxs = torch.randint(0, 4, (2, 12), generator=torch.Generator().manual_seed(0))
	X.scatter_(1, idxs.unsqueeze(1), 1.0)
	return X


###


def test_designwrapper_is_module():
	wrapper = DesignWrapper([SumModel()])
	assert isinstance(wrapper, torch.nn.Module)


def test_designwrapper_registers_models():
	models = [SumModel(), SumModel()]
	wrapper = DesignWrapper(models)

	assert isinstance(wrapper.models, torch.nn.ModuleList)
	assert len(wrapper.models) == 2


def test_designwrapper_single_model(X):
	model = SumModel()
	wrapper = DesignWrapper([model])

	y = wrapper(X)
	assert y.shape == (2, 4)
	assert_array_almost_equal(y.numpy(), model(X).numpy(), 4)


def test_designwrapper_concatenates_last_dim(X):
	# Two single-output models -> (batch, 2).
	wrapper = DesignWrapper([SumModelKeepdim(), SumModelKeepdim()])
	y = wrapper(X)

	assert y.shape == (2, 2)
	assert_array_almost_equal(y[:, 0].numpy(), y[:, 1].numpy(), 4)


def test_designwrapper_mixed_output_widths(X):
	# One model emits 4 outputs, the other emits 1 -> (batch, 5).
	wrapper = DesignWrapper([SumModel(), SumModelKeepdim()])
	y = wrapper(X)

	assert y.shape == (2, 5)
	assert_array_almost_equal(y[:, :4].numpy(), SumModel()(X).numpy(), 4)
	assert_array_almost_equal(y[:, 4:].numpy(), SumModelKeepdim()(X).numpy(), 4)


def test_designwrapper_matches_manual_concatenation(X):
	models = [FlattenDense(n_outputs=3), FlattenDense(n_outputs=2)]
	wrapper = DesignWrapper(models)

	y = wrapper(X)
	expected = torch.cat([models[0](X), models[1](X)], dim=-1)

	assert y.shape == (2, 5)
	assert_array_almost_equal(y.detach().numpy(), expected.detach().numpy(), 4)


def test_designwrapper_three_models(X):
	wrapper = DesignWrapper([SumModelKeepdim(), SumModelKeepdim(), SumModelKeepdim()])
	y = wrapper(X)
	assert y.shape == (2, 3)


def test_designwrapper_gradient_flows(X):
	X = X.clone().requires_grad_(True)
	wrapper = DesignWrapper([FlattenDense(n_outputs=3), FlattenDense(n_outputs=2)])

	wrapper(X).sum().backward()
	assert X.grad is not None
	assert X.grad.shape == X.shape
