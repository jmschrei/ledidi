.. currentmodule:: ledidi


===============
Getting Started
===============

This page walks through the :doc:`Quickstart <index>` example one step at a time. It uses a tiny parameter-free oracle so there is nothing to download and the whole thing runs on a CPU in seconds. Once it makes sense, the :doc:`tutorials <index>` show the same workflow with real genomics models.

The same example is also available as a runnable notebook with live outputs and a loss plot in :doc:`Tutorial 0 <tutorials/Tutorial_0_-_Getting_Started>`.

The three ingredients
=====================

Every Ledidi design needs exactly three things:

1. **An oracle model** -- any differentiable ``torch`` model that maps a one-hot sequence to a prediction. Here we use a toy oracle that scores how well a sequence matches the AP-1 motif ``TGACTCA``::

      import torch
      from ledidi import ledidi

      motif = "TGACTCA"
      weights = torch.zeros(1, 4, len(motif))
      for i, char in enumerate(motif):
          weights[0, "ACGT".index(char), i] = 1.0

      class MotifScore(torch.nn.Module):
          def forward(self, X):
              return torch.nn.functional.conv1d(X, weights).amax(dim=-1)

   A perfect match anywhere in the sequence scores 7 (the motif length); a random sequence scores much lower.

2. **A template sequence** -- the sequence to edit, one-hot encoded as a ``(1, 4, length)`` float tensor with channels ordered ``A, C, G, T``. We start from a random 50 bp sequence::

      torch.manual_seed(0)
      idxs = torch.randint(0, 4, (1, 50))
      X = torch.zeros(1, 4, 50).scatter_(1, idxs.unsqueeze(1), 1.0)

3. **A desired output** -- the value you want the oracle to predict for the edited sequence, as a ``(1, n_outputs)`` tensor. We ask for a perfect motif match::

      y_bar = torch.tensor([[7.0]])


Running Ledidi
==============

With those three things in hand, the call itself is one line::

   X_hat = ledidi(MotifScore(), X, y_bar, device="cpu", random_state=0, verbose=False)

``ledidi`` returns a batch of independently sampled designed sequences with shape ``(batch_size, 4, length)`` (``batch_size`` defaults to 16). We pass ``device="cpu"`` because the default is ``"cuda"`` -- on a machine without a GPU you must set this explicitly or the call will error. ``random_state=0`` makes the run reproducible without disturbing the global torch RNG, and ``verbose=False`` silences the per-iteration logging.


Inspecting the edits
====================

Decode a designed sequence back to a string to confirm the motif was inserted::

   designed = "".join("ACGT"[c] for c in X_hat[0].argmax(dim=0))
   print("TGACTCA" in designed)  # True

To see exactly which positions changed, compare the template to a designed sequence::

   seq = X_hat[0]  # one designed sequence, shape (4, length)
   positions = torch.where((X[0] != seq).any(dim=0))[0]
   for p in positions:
       before = "ACGT"[X[0, :, p].argmax()]
       after  = "ACGT"[seq[:, p].argmax()]
       print(f"position {p.item()}: {before} -> {after}")

For this example that prints just a couple of edits -- the ones that complete the ``TGACTCA`` motif at the cheapest location in the random sequence, rather than overwriting a whole stretch of it. Each of the ``batch_size`` sequences in ``X_hat`` is sampled independently, so they may carry slightly different edits.


Where to go next
================

- :doc:`input_output` -- the exact tensor shapes, dtypes, and conventions Ledidi expects, and the errors it raises when they are wrong.
- :doc:`parameters` -- the handful of knobs worth tuning, starting with ``l`` (the edit-vs-output trade-off).
- :doc:`faq` -- reproducibility, running on CPU vs GPU, and common error messages.
- The :doc:`tutorials <index>` -- worked examples with real BPNet, Enformer, and Malinois oracles, covering constraints, in-painting, multiple models, affinity catalogs, and custom losses.
