.. currentmodule:: ledidi


==========================
FAQ and Troubleshooting
==========================

Do I need a GPU?
================

No. Ledidi runs on a CPU, and there is no minimum GPU. The toy oracle in the :doc:`Quickstart <index>` designs in about three seconds on a single CPU thread, and a BPNet-scale design over a 2114 bp sequence takes about nine seconds on a CPU against 0.8 seconds on a GPU. Tutorials 0 and 8 are written to run on a CPU with no downloads.

A GPU is worth having once the oracle is a real genomics model and you are designing many sequences, but it is a speed-up rather than a requirement. See :doc:`requirements` for the measured timings and memory, and :doc:`installation` for how to install the smaller CPU-only PyTorch build.


Running on CPU or GPU
=====================

By default ``ledidi`` moves the model and tensors to the GPU (``device='cuda'``). On a machine without a CUDA GPU you must pass ``device='cpu'`` explicitly::

   X_hat = ledidi(model, X, y_bar, device='cpu')

Forget it and the call fails immediately, with one of two messages depending on which PyTorch build you have:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Your PyTorch build
     - Error you get
   * - CPU-only (version ends in ``+cpu``)
     - ``AssertionError: Torch not compiled with CUDA enabled``
   * - CUDA, but no GPU visible
     - ``RuntimeError: No CUDA GPUs are available``

Both mean the same thing: pass ``device='cpu'``. To write code that runs on either kind of machine, pick the device up front::

   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   X_hat = ledidi(model, X, y_bar, device=device)

You can also pass any ``torch.device`` or device string, e.g. ``device='cuda:1'``.


I ran out of GPU memory
=======================

Peak memory is dominated by the oracle, not by Ledidi: each iteration runs a forward *and* backward pass over ``batch_size`` sequences and retains the activations of all of them. It is therefore roughly linear in ``batch_size`` x design length x oracle size x number of oracles. The learned weight matrix itself is a few tens of kilobytes and is never the problem.

Measure before changing anything::

   torch.cuda.reset_peak_memory_stats()
   X_hat = ledidi(model, X, y_bar, device='cuda')
   print(torch.cuda.max_memory_allocated() / 2**20, "MiB peak")

Then work down this list and stop at the first rung that works:

1. **Lower** ``batch_size``. A linear reduction that requires no code changes and does not bias the design, since both losses are per-sequence means. Try 8, then 4. This is the right answer far more often than anything below it.
2. **Shorten the design field.** The other linear axis -- design the region that matters rather than a large window around it.
3. **Gradient checkpointing on the oracle**, applied in segments rather than to the whole model at once. Wrapping the entire model in a single ``checkpoint(...)`` call saves essentially nothing.

For context on what is normal, a default ``batch_size=16`` design against a BPNet-scale oracle peaks at a few hundred megabytes; see the table in :doc:`requirements`. The bundled agent skill's ``references/memory-and-oom.md`` has the full ladder with measured numbers.


Something went wrong installing Ledidi
======================================

The :doc:`installation` page has a troubleshooting section covering the common cases: ``torch.cuda.is_available()`` returning ``False`` on a machine with a GPU, an install that pulled gigabytes of unwanted ``nvidia-*`` packages, a Python older than the required 3.10, and missing ``pip`` inside a ``uv`` environment. It also has a snippet that verifies an installation end to end, with the expected output for both a CUDA and a CPU-only machine.


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
- **Non-positive hyperparameter** -- ``tau``, ``lr``, ``batch_size``, ``max_iter``, ``early_stopping_iter``, and ``report_iter`` must be positive; ``l`` and ``threshold`` in pruning must be non-negative (a ``threshold`` of 0 is valid and reverts no edits).
- **Negative or out-of-range ``target``** -- ``target`` is an index of a single output, so ``target=-1`` does not mean "the last output" and raises a ``ValueError``; an index past the end of the model's output raises on the first forward pass. Pass a non-negative index, or wrap the model and use ``target=None``.
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
