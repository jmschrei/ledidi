# plot.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""Plotting helpers for inspecting a design run.

These utilities visualize the three things you most often want to look at after
running Ledidi: how the losses evolved, when and where edits were proposed, and
how the final edits relate to the motifs they create. :func:`plot_loss` and
:func:`plot_history` both take the history dict returned by ``return_history=True``
and plot, respectively, the input/output loss curves and the position of each
proposed edit over the course of the optimization. :func:`plot_edits` overlays
the proposed edits on a per-base attribution track. They are thin matplotlib
helpers; for attribution computation and sequence-level plotting more broadly,
see `tangermeme <https://github.com/jmschrei/tangermeme>`_.
"""

import numpy
import torch
import matplotlib.pyplot as plt

from tangermeme.plot import plot_logo


def plot_loss(history, ax=None):
	"""Plot the input and output losses across the procedure.

	This function takes in the history dictionary produced by the Ledidi object
	and plots the two losses that drive the optimization: the output loss (the
	distance between the oracle's prediction and the desired output) and the
	input loss (the average number of edits made to the sequence). Looking at
	these curves is the best way to build intuition for what Ledidi is doing: it
	first rapidly acquires edits to drop the output loss, then slowly sheds
	unnecessary edits to drop the input loss.

	The two losses are measured in different units and on different scales, so
	they are drawn against twin y-axes that share the iteration axis. The output
	loss is read off the left axis and the input loss off the right axis, each
	colored to match its curve.


	Parameters
	----------
	history: dict
		The history object produced by Ledidi. To get it, pass in
		`return_history=True` to the initialization of Ledidi.

	ax: matplotlib.axes.Axes or None, optional
		The axis to draw the output loss on. The input loss is drawn on a twin
		of this axis. If None, the current axis (`plt.gca()`) is used. Default
		is None.


	Returns
	-------
	axs: tuple
		The `(output_axis, input_axis)` pair the losses were drawn on, where the
		second is the twin axis created for the input loss.
	"""

	if ax is None:
		ax = plt.gca()

	ax2 = ax.twinx()

	line0, = ax.plot(history['output_loss'], color='0.3', label="Output Loss")
	line1, = ax2.plot(history['input_loss'], color='darkorange',
		label="Input Loss")

	ax.set_xlabel("Iteration", fontsize=12)
	ax.set_ylabel("Output Loss", fontsize=12, color='0.3')
	ax2.set_ylabel("Input Loss", fontsize=12, color='darkorange')

	ax.tick_params(axis='y', labelcolor='0.3')
	ax2.tick_params(axis='y', labelcolor='darkorange')

	ax.spines['top'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax.spines['left'].set_color('0.3')
	ax2.spines['right'].set_color('darkorange')

	ax.legend(handles=[line0, line1], fontsize=10, loc='upper right')
	return ax, ax2


def plot_history(history):
	"""Plot the location of edits across the procedure.

	This function takes in the history dictionary produced by the Ledidi object
	and plot the position of the edits as they were proposed during the editing
	procedure. This plot allows you to see when certain edits were proposed.


	Parameters
	----------
	history: dict
		The history object produced by Ledidi. To get it, pass in
		`return_history=True` to the initialization of Ledidi.
	"""

	ax = plt.gca()
	b = history['batch_size']

	for i, edits in enumerate(history['edits']):
		y, _, x = edits
		x = x.numpy(force=True)
		y = y.numpy(force=True)
		plt.scatter(x, y+i*b, s=0.05, c='0.6')

	ylim = len(history['edits']*b)
	yticks = numpy.array(ax.get_yticks())

	plt.yticks(yticks, yticks // b)
	plt.ylim(ylim, 0)
	plt.xlabel("Position", fontsize=12)
	plt.ylabel("Iteration", fontsize=12)


def plot_edits(X_orig, X_attrs, colors='darkorange', axs=None, **kwargs):
	"""Plot attributions for edited sequences compared to an initial sequence.

	This function will take in attributions for an initial sequence as well as a
	set of edited sequences and display those attribution values as a series of
	stacked logo tracks, with characters colored if they are edited with respect
	to the initial sequence. The original sequence is prepended as the first
	track. All tracks share a common y-axis range so that attribution magnitudes
	are directly comparable, and position ticks are shown only on the bottom
	track. The calculation of attributions is left to the user as there are
	several different methods for calculating attributions and each have their
	own hyperparameters.

	By default the function creates its own figure and one track per sequence.
	Pass `axs` to draw into axes you have already laid out instead -- for
	instance to place the tracks alongside other panels, or to zoom each track
	into a window of interest.


	Parameters
	----------
	X_orig: torch.tensor, shape=(1, 4, len)
		Attributions for the initial unedited sequence. Attributions should only be
		non-zero for the observed character.

	X_attrs: torch.Tensor or list, shape=(n, 4, len)
		Attributions for `n` edited sequences, with values only being non-zero for
		the observed character. If a list is provided, it is concatenated to form
		such a tensor.

	colors: str or list, optional
		What color to use for the edited characters. If a single string is
		provided, that color is used for every track. If a list is provided, it
		is indexed per track and so must have one entry for each track that is
		plotted, i.e. length X_attrs.shape[0] + 1 to account for the original
		sequence being prepended as the first track (its entry is unused since
		that track has no edits). Default is darkorange.

	axs: list of matplotlib.axes.Axes or None, optional
		The axes to draw the tracks into, one per track, i.e. length
		X_attrs.shape[0] + 1 to account for the prepended original sequence. If
		None, a new figure is created with one track per sequence. Default is
		None.

	``**kwargs``: args, optional
		Any additional arguments to pass into plt.figure. Only used when `axs` is
		None, since otherwise the figure already exists.


	Returns
	-------
	axs: list of matplotlib.axes.Axes
		The axes the tracks were drawn into, with the original sequence first.
	"""

	if isinstance(X_attrs, list):
		X_attrs = torch.cat(X_attrs, dim=0)

	X_attrs = torch.cat([X_orig, X_attrs], dim=0)
	X_attrs = X_attrs.numpy(force=True)

	n = len(X_attrs)
	ymin, ymax = X_attrs.min() * 1.1, X_attrs.max() * 1.1
	ref = (X_attrs[0] != 0).argmax(axis=0)

	if axs is None:
		plt.figure(**kwargs)
		axs = [plt.subplot(n, 1, i+1) for i in range(n)]

	for i, (X, ax) in enumerate(zip(X_attrs, axs)):
		c = colors if isinstance(colors, str) else colors[i]

		idx = (X != 0).argmax(axis=0)
		position_colors = ['dimgrey' if j == r else c for r, j in zip(ref, idx)]

		plot_logo(X, ax=ax, color=position_colors, min_height_pct=None)
		ax.spines['left'].set_visible(False)

		ax.set_ylim(ymin, ymax)
		ax.grid(False)
		if i < n - 1:
			ax.set_xticks([])

	axs[-1].set_xlabel("Position", fontsize=12)
	return axs
