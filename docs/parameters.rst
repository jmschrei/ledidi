.. currentmodule:: ledidi


==============
Key Parameters
==============

Most designs only require tuning a couple of knobs. The parameters below are passed straight through :func:`ledidi` to the underlying :class:`Ledidi` optimizer. For the full set, see the :doc:`API reference <api/ledidi>`.

.. list-table::
   :header-rows: 1
   :widths: 18 12 70

   * - Parameter
     - Default
     - What it does
   * - ``l``
     - ``0.1``
     - Weight on the edit (input) loss. **The main knob to tune** -- lower values prioritize hitting the target output, higher values prioritize making fewer edits.
   * - ``target``
     - ``None``
     - For a multi-task model, the index of the output to design against. ``None`` uses the whole output.
   * - ``output_loss``
     - ``MSELoss()``
     - The loss comparing the model's prediction to ``y_bar``. Swap in any callable ``f(y_hat, y_bar)``.
   * - ``tau``
     - ``1``
     - Gumbel-softmax temperature; higher is sharper (closer to a hard argmax).
   * - ``batch_size``
     - ``16``
     - Sequences sampled and averaged per iteration.
   * - ``max_iter``
     - ``1000``
     - Maximum optimization iterations.
   * - ``early_stopping_iter``
     - ``100``
     - Stop after this many iterations without improvement.
   * - ``input_mask``
     - ``None``
     - Boolean tensor of shape ``(length,)`` marking positions that may **not** be edited.
   * - ``random_state``
     - ``None``
     - Seed for reproducible sampling. Unlike ``torch.manual_seed``, this does not mutate the global RNG, so it leaves the rest of your script unaffected.
   * - ``device``
     - ``'cuda'``
     - Device to run on. Pass ``'cpu'`` if you have no GPU, or the call will error.
   * - ``verbose``
     - ``True``
     - Print the input, output, and total loss as the design proceeds.


Tuning advice
=============

In practice ``l`` is the parameter you will reach for most often. It sets the exchange rate between the two terms of the objective: the **output loss** (how close the prediction is to ``y_bar``) and the **input loss** (how many edits were made). If a design hits the target but uses too many edits, raise ``l``; if it makes few edits but never reaches the target, lower it. When the output loss is a complicated mixture of terms -- many models, or a custom loss with very small magnitudes -- you will often need to drop ``l`` substantially so the two terms are on a comparable scale.


Wrapper-only parameters
=======================

A few parameters live on :func:`ledidi` itself rather than being forwarded to :class:`Ledidi`:

- ``n_samples`` -- after optimization, draw this many designed sequences from the fitted weight matrix. Sampling is extremely fast once the weights are learned, so this is nearly free.
- ``n_repeats`` -- run the whole design procedure this many times and stack the results, giving genuinely independent sets of edits in a single call.
- ``return_history`` -- also return a per-iteration history dict recording the input, output, and total losses and where edits were proposed. Visualize the map of edits over the run with :func:`~ledidi.plot.plot_history`.
- ``return_designer`` -- also return the fitted :class:`Ledidi` object so you can keep sampling from it.
