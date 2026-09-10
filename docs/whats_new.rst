.. currentmodule:: ledidi


===============
Release History
===============


Version 2.2.1 (unreleased)
==========================

Highlights
----------

	- Rewrote every cross-reference in the bundled Claude Code Agent Skill as a
	  plain backticked path (``references/objective.md``) instead of a Markdown
	  link (``[references/objective.md](references/objective.md)``). Nothing that
	  reads a skill renders Markdown, so the link form wrote each path twice for
	  no benefit; 191 links were converted, shrinking the router by 7.2%. If you
	  installed the skill previously, re-run ``ledidi-install-skills --force`` to
	  pick up the corrections.
	- Moved ``tests/test_install_skills.py`` to ``tests/_skills/test_install.py``
	  so the test layout mirrors the package layout, in which ``_skills`` is the
	  only subpackage.


Version 2.2.0
=============

Highlights
----------

	- Hardened inputs across the library: :func:`ledidi.ledidi`,
	  :class:`ledidi.Ledidi`, :func:`ledidi.pruning.greedy_pruning`,
	  :class:`ledidi.wrappers.DesignWrapper`, and :func:`ledidi.losses.MinGap`
	  now validate their arguments and raise informative errors for malformed
	  inputs (non-one-hot sequences, shape and dtype mismatches, non-positive
	  hyperparameters, and so on) rather than failing deep inside the optimizer.
	- Fixed :func:`ledidi.pruning.greedy_pruning` to accept a ``threshold`` of 0
	  (which reverts no edits) again; the input hardening had incorrectly
	  rejected it as non-positive.
	- Tensor validation reuses ``tangermeme.utils._validate_input``, which is now
	  a dependency. The expected tensor formats and the errors raised for
	  malformed inputs are documented on the :doc:`input_output` page, with the
	  most common messages and fixes collected in the :doc:`faq`.
	- Raised the minimum supported Python to 3.10 and the minimum PyTorch to 2.0.
	- Expanded the documentation: a runnable :doc:`getting_started` walkthrough, a
	  :doc:`parameters` reference, and the input/output and FAQ pages above.
	- Added a :doc:`requirements` page documenting the hardware and software
	  Ledidi needs -- measured CPU/GPU runtimes and peak GPU memory as a function
	  of ``batch_size`` and oracle size, and the core and optional dependencies.
	- Split installation out of the landing page into a full :doc:`installation`
	  guide covering virtual environments, the CPU-only PyTorch build and the
	  ``device='cpu'`` argument it requires, conda, installing from source,
	  upgrading, troubleshooting, and a snippet that verifies an installation end
	  to end.
	- Added :func:`ledidi.plot.plot_loss` for drawing the input and output loss
	  curves from a returned history, and gave :func:`ledidi.plot.plot_edits` an
	  ``axs`` argument so its tracks can be drawn into an existing layout.
	- Reimplemented :func:`ledidi.plot.plot_edits` on top of
	  ``tangermeme.plot.plot_logo`` (using its per-position ``color`` argument to
	  highlight the edited bases) rather than ``logomaker``. ``logomaker`` and
	  ``pandas`` are no longer dependencies. This requires ``tangermeme >= 1.3.0``.


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
