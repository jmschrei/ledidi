.. currentmodule:: ledidi


==========================
FAQ and Troubleshooting
==========================

Running on CPU or GPU
=====================

By default ``ledidi`` moves the model and tensors to the GPU (``device='cuda'``). On a machine without a CUDA GPU you must pass ``device='cpu'`` explicitly, otherwise the call will error with a CUDA-related ``RuntimeError``::

   X_hat = ledidi(model, X, y_bar, device='cpu')

You can also pass any ``torch.device`` or device string, e.g. ``device='cuda:1'``.


How reproducible is a design?
=============================

Pass ``random_state`` to make the Gumbel-softmax sampling reproducible. Unlike calling ``torch.manual_seed``, this draws from a private generator and does **not** mutate the global torch RNG, so it leaves the rest of your script unaffected.

On a GPU, ``random_state`` alone does not guarantee bitwise-identical results across runs. PyTorch performs some CUDA/cuDNN operations approximately for speed, and because Ledidi samples from the model's outputs, even differences at machine precision can change which edits are drawn. The designs will almost always be just as good -- just not identical. For full determinism (e.g., when debugging) set::

   import torch
   torch.use_deterministic_algorithms(True)
   torch.manual_seed(0)

before running. On CUDA this also requires setting the cuBLAS workspace environment variable *before* ``torch`` is imported, or PyTorch will raise an error when a deterministic algorithm is requested::

   import os
   os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

   import torch

Full determinism is noticeably slower, so it is best reserved for cases that truly need it. Note also that an end-to-end design is reproducible on a single machine but can drift by an edit or two across different CPU architectures, due to floating-point non-associativity in the optimizer.


I got a ``ValueError`` (or ``TypeError``) from a Ledidi call
============================================================

As of version 2.2.0 Ledidi validates its inputs up front. The most common messages and their fixes:

- **Non-one-hot sequence** -- ``X`` (or ``X_hat`` in :func:`~ledidi.pruning.greedy_pruning`) is not one-hot encoded. Each position must sum to 1 with a single 1.0 entry; all-zero ``N`` columns are allowed. See :doc:`input_output`.
- **Shape mismatch** -- ``X`` must be ``(1, n_channels, length)``; ``input_mask`` must be ``(length,)``; ``initial_weights`` must be ``(1, n_channels, length)``; in pruning, ``X_hat`` must match ``X``'s shape.
- **``y_bar`` leading dimension** -- ``y_bar`` must have shape ``(1, n_outputs)``. A common mistake is passing ``torch.tensor([4.5])`` (shape ``(1,)``) instead of ``torch.tensor([[4.5]])``.
- **Wrong dtype** -- ``input_mask`` and a :class:`~ledidi.losses.MinGap` mask must be ``torch.bool``.
- **Non-positive hyperparameter** -- ``tau``, ``lr``, ``batch_size``, ``max_iter``, ``early_stopping_iter``, and ``report_iter`` must be positive; ``l`` must be non-negative; ``threshold`` in pruning must be positive.
- **Degenerate ``MinGap`` mask** -- the on-target/off-target mask must contain at least one ``True`` and at least one ``False``; an all-on or all-off mask has no gap to maximize.

Tensor validation is delegated to ``tangermeme.utils._validate_input``, which raises ``ValueError`` (not ``TypeError``) for tensor problems; ``TypeError`` is reserved for a non-``Module`` model or a non-integer ``target``.


Why are all the sequences in ``X_hat`` so similar?
==================================================

Every sequence in the returned batch is sampled from the *same* learned weight matrix, so they are correlated by construction. If a motif could have been inserted at several locations, Ledidi commits to one and all sampled sequences reflect that choice. To get genuinely different sets of edits, run Ledidi again with a different ``random_state``, or pass ``n_repeats`` to do several independent runs in a single call.


Can I sample many designs cheaply?
==================================

Yes. Once the weight matrix is fit, drawing more sequences is just forward sampling and is extremely fast. Pass ``n_samples`` to draw a large number of designs after optimization::

   X_hat = ledidi(model, X, y_bar, n_samples=10000)  # nearly as fast as the default

Equivalently, keep the fitted designer with ``return_designer=True`` and call its ``forward`` repeatedly. See :doc:`Tutorial 8 <tutorials/Tutorial_8_-_The_Ledidi_Object>` for this workflow.
