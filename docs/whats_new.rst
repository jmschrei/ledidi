.. currentmodule:: ledidi


===============
Release History
===============


Version 2.1.0
=============

Highlights
----------

	- Added :func:`ledidi.losses.MinGap`, the min-gap loss from Gosai et al. for
	  designing output-specific (e.g., cell type-specific) elements.
	- Added :class:`ledidi.wrappers.DesignWrapper` for combining several models
	  into a single multi-output designer.
	- Expanded plotting utilities in ``ledidi.plot`` with ``plot_edits`` and
	  ``plot_history``.
	- Extended ``ledidi`` to support designing affinity catalogs by passing a
	  list of target values, along with the ``n_repeats`` and ``n_samples``
	  options for drawing multiple designs.


Version 2.0.0
=============

Highlights
----------

	- Rewrote Ledidi in PyTorch. TensorFlow models are no longer supported.
	- Reframed edit design as a continuous optimization over a weight matrix
	  sampled through the Gumbel-softmax straight-through estimator.
	- Added greedy pruning of edits via ``ledidi.pruning.greedy_pruning``.
