.. currentmodule:: ledidi


========================
Input and Output Formats
========================

Ledidi is strict about the tensors it accepts: as of version 2.2.0 the inputs are validated up front and malformed arguments raise an informative ``ValueError`` (or ``TypeError``) immediately, rather than failing deep inside the optimizer. This page documents exactly what is expected.


The sequence tensor
===================

Sequences -- both the template ``X`` you pass in and the designs ``X_hat`` you get back -- are **one-hot encoded** tensors of shape ``(batch, n_channels, length)`` and dtype ``torch.float32``. For DNA that is four channels ordered ``A, C, G, T``; for RNA or protein it is whatever channel order your oracle was trained on.

- The template ``X`` passed to ``ledidi`` has a batch dimension of **1**: shape ``(1, n_channels, length)``.
- Each position is one-hot: the channels at a position sum to 1, with a single 1.0 and the rest 0.0.
- **All-zero columns are allowed** and are treated as unknown / ``N``. These are exactly the positions used for :doc:`in-painting <tutorials/Tutorial_3_-_In-Painting>` -- zeroing a span tells Ledidi it may fill it in freely. Any ``N`` in your template is therefore a candidate for editing and will not be preserved.

If you are starting from a string, one-hot encode it (and decode the result) like this::

   import torch

   def one_hot_encode(seq):
       mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
       X = torch.zeros(1, 4, len(seq))
       for i, char in enumerate(seq):
           X[0, mapping[char], i] = 1.0
       return X

   def decode(X):  # X is a single (4, length) one-hot sequence
       return "".join("ACGT"[c] for c in X.argmax(dim=0))

The sibling library `tangermeme <https://github.com/jmschrei/tangermeme>`_ provides ``one_hot_encode`` and ``characters`` utilities that handle this -- as well as reading FASTA and genomic loci -- for you.


The desired output ``y_bar``
============================

``y_bar`` specifies the value you want the oracle to predict for the edited sequence. It is a tensor of shape ``(1, n_outputs)``: the leading dimension must be **1** (internally Ledidi expands a batch of sequences against it), and ``n_outputs`` must match the number of outputs your model returns.

- For a single-output model, ``y_bar = torch.tensor([[4.5]])``.
- For a multi-task model or a :class:`~ledidi.wrappers.DesignWrapper` that concatenates several models, ``y_bar`` has one entry per output, in the same order the model returns them.
- For an **affinity catalog**, pass a *list* of such tensors. The design is repeated once per element and the results are stacked, adding a leading catalog dimension to ``X_hat``. A list is required even when the targets form a simple range -- this is deliberate, to keep the single-target and catalog cases unambiguous::

     # design against four increasing targets in one call
     y_bar = [torch.tensor([[v]]) for v in (1.0, 2.0, 3.0, 4.0)]
     X_hat = ledidi(model, X, y_bar)   # X_hat.shape == (4, batch_size, n_channels, length)

Some output losses (notably :class:`~ledidi.losses.MinGap`) ignore ``y_bar`` entirely but still require it for the call signature; pass a correctly shaped placeholder such as ``torch.zeros(1, n_outputs)``.


What you get back
=================

``ledidi`` returns ``X_hat``, a batch of independently sampled designed sequences of shape ``(batch_size, n_channels, length)`` (``batch_size`` defaults to 16). All sequences in the batch are drawn from the *same* learned weight matrix, so they are correlated -- if a motif could be placed in several spots, Ledidi commits to one. To get genuinely different designs, run again with a different ``random_state`` or use ``n_repeats``. For an affinity catalog, ``X_hat`` gains a leading dimension of length equal to the catalog.


Masks and priors
================

Two optional arguments shape *where* and *how* edits may be made:

- ``input_mask`` -- a boolean tensor of shape ``(length,)``. It marks positions that may **not** be edited (a hard constraint). It must be a ``torch.bool`` tensor.
- ``initial_weights`` -- a float tensor of shape ``(1, n_channels, length)`` used to seed the optimization. Setting an entry to ``-inf`` forbids that character (the mechanism behind ``input_mask``); finite positive or negative values act as soft priors that nudge, but do not force, the design. See the :doc:`Constraints and Priors tutorial <tutorials/Tutorial_2_-_Constraints_and_Priors>`.


Validation and errors
======================

Tensor validation is delegated to ``tangermeme.utils._validate_input``, which raises a ``ValueError`` -- not a ``TypeError`` -- for a non-tensor, a wrong dtype, a wrong shape, or a non-one-hot sequence. Scalar hyperparameters are checked inline: non-positive ``tau``, ``lr``, ``batch_size``, ``max_iter``, ``early_stopping_iter``, or ``report_iter`` raise ``ValueError``, a negative edit weight ``l`` raises ``ValueError``, and a non-integer ``target`` or a model that is not a ``torch.nn.Module`` raises ``TypeError``. The :doc:`faq` lists the messages you are most likely to see and how to fix them.
