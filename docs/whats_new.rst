.. currentmodule:: ledidi


===============
Release History
===============


Version 2.2.0 (unreleased)
==========================

Highlights
----------

	- Hardened inputs across the library: :func:`ledidi.ledidi`,
	  :class:`ledidi.Ledidi`, :func:`ledidi.pruning.greedy_pruning`,
	  :class:`ledidi.wrappers.DesignWrapper`, and :func:`ledidi.losses.MinGap`
	  now validate their arguments and raise informative errors for malformed
	  inputs (non-one-hot sequences, shape and dtype mismatches, non-positive
	  hyperparameters, and so on) rather than failing deep inside the optimizer.
	- Tensor validation reuses ``tangermeme.utils._validate_input``, which is now
	  a dependency.
	- Raised the minimum supported Python to 3.10 and the minimum PyTorch to 2.0.


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
