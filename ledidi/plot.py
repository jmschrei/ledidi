# plot.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""Plotting helpers for inspecting a design run.

These utilities visualize two things you often want to look at after running
Ledidi: when and where edits were proposed, and how the final edits relate to
the motifs they create. :func:`plot_history` takes the history dict returned by
``return_history=True`` and plots the position of each proposed edit over the
course of the optimization, and :func:`plot_edits` overlays the proposed edits
on a per-base attribution track. They are thin matplotlib helpers; for
attribution computation and sequence-level plotting more broadly, see
`tangermeme <https://github.com/jmschrei/tangermeme>`_.
"""

import numpy
import torch
import pandas
import logomaker
import matplotlib.pyplot as plt


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


def plot_edits(X_orig, X_attrs, colors='darkorange', **kwargs):
	"""Plot attributions for edited sequences compared to an initial sequence.
	
	This function will take in attributions for an initial sequence as well as a
	set of edited sequences and display those attribution values as a series of
	tracks, with characters colored if they are edited with respect to the initial
	sequence. The calculation of attributions is left to the user as there are
	several different methods for calculating attributions and each have their
	own hyperparameters.
	
	
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
	
	``**kwargs``: args, optional
		Any additional arguments to pass into plt.figure.
	"""
	
	if isinstance(X_attrs, list):
		X_attrs = torch.cat(X_attrs, dim=0)
	
	X_attrs = torch.cat([X_orig, X_attrs], dim=0)
	X_attrs = X_attrs.numpy(force=True)
	
	ymin, ymax = X_attrs.min() * 1.1, X_attrs.max() * 1.1
	idxs0 = (X_attrs[0] != 0).argmax(axis=0)
	alpha = numpy.array(['A', 'C', 'G', 'T'])
	
	axs = []
	plt.figure(**kwargs)
	for i, X in enumerate(X_attrs):
		if isinstance(colors, str):
			c = colors
		else:
			c = colors[i]

		ax = plt.subplot(len(X_attrs), 1, i+1)
		axs.append(ax)
		
		idx = (X != 0).argmax(axis=0)
		diff = [alpha[i1] if i0 != i1 else 'N' for i0, i1 in zip(idxs0, idx)]
		diff = ''.join(diff)

		df = pandas.DataFrame(X.T, columns=['A', 'C', 'G', 'T'])
		df.index.name = 'pos'

		logo = logomaker.Logo(df, ax=ax, color_scheme='dimgrey')
		logo.style_spines(visible=False)
		logo.style_glyphs_in_sequence(sequence=diff, color=c)
		ax.set_ylim(ymin, ymax)
		ax.grid(False)
		ax.set_xticks([], [])
	
	return axs