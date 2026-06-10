# test_plot.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ledidi.plot import plot_history
from ledidi.plot import plot_edits


@pytest.fixture(autouse=True)
def close_figures():
	yield
	plt.close("all")


@pytest.fixture
def history():
	# Two iterations of a batch_size=4 run, each with a couple of edits encoded
	# as the (batch_idx, channel_idx, position_idx) triple that torch.where
	# returns inside Ledidi.fit_transform.
	edits = [
		(torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([5, 8])),
		(torch.tensor([0, 2, 3]), torch.tensor([1, 0, 2]), torch.tensor([3, 6, 9]))
	]
	return {'edits': edits, 'batch_size': 4}


@pytest.fixture
def X_orig():
	X = torch.zeros(1, 4, 12)
	X[0, 0, :] = 1.0
	return torch.randn(1, 4, 12) * X


@pytest.fixture
def X_attrs():
	X = torch.zeros(2, 4, 12)
	idxs = torch.randint(0, 4, (2, 12), generator=torch.Generator().manual_seed(0))
	X.scatter_(1, idxs.unsqueeze(1), 1.0)
	return torch.randn(2, 4, 12) * X


###


def test_plot_history_runs(history):
	plt.figure()
	out = plot_history(history)
	assert out is None
	# Every edit position should have produced a scatter point.
	ax = plt.gca()
	assert len(ax.collections) == len(history['edits'])


def test_plot_history_labels(history):
	plt.figure()
	plot_history(history)
	ax = plt.gca()
	assert ax.get_xlabel() == "Position"
	assert ax.get_ylabel() == "Iteration"


def test_plot_edits_returns_axes(X_orig, X_attrs):
	axs = plot_edits(X_orig, X_attrs)
	# One track per edited sequence plus the prepended original.
	assert len(axs) == X_attrs.shape[0] + 1
	assert all(isinstance(ax, plt.Axes) for ax in axs)


def test_plot_edits_accepts_list(X_orig, X_attrs):
	axs = plot_edits(X_orig, [X_attrs[:1], X_attrs[1:]])
	assert len(axs) == 3


def test_plot_edits_color_list(X_orig, X_attrs):
	colors = ['black', 'red', 'blue']
	axs = plot_edits(X_orig, X_attrs, colors=colors)
	assert len(axs) == 3


def test_plot_edits_kwargs(X_orig, X_attrs):
	axs = plot_edits(X_orig, X_attrs, figsize=(4, 6))
	assert len(axs) == 3
