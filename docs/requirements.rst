.. currentmodule:: ledidi


============
Requirements
============

Ledidi is a thin optimization loop wrapped around whatever oracle model you hand it, so its own requirements are modest: a recent Python, PyTorch, and three further packages. In practice the hardware you need is set almost entirely by the oracle, not by Ledidi -- if your model can run a forward and backward pass on a batch of sequences, Ledidi can design against it.

Everything below installs with a single ``pip install ledidi``; see :doc:`installation` for the step-by-step guide, including the CPU-only PyTorch build and a snippet that confirms the whole stack works end to end.


Hardware
========

Ledidi runs on CPU and on a CUDA GPU. Neither is a hard requirement, and there is no minimum GPU: the toy examples in the :doc:`Quickstart <index>` and :doc:`getting_started` run on a laptop CPU in seconds. A GPU becomes worthwhile once the oracle is a real genomics model, and it becomes necessary once you want large batches or long sequences.

By default ``ledidi`` moves the model and tensors to the GPU (``device='cuda'``). **On a machine without a CUDA GPU you must pass** ``device='cpu'`` **explicitly**, otherwise the call fails immediately -- with ``AssertionError: Torch not compiled with CUDA enabled`` on a CPU-only PyTorch build, or ``RuntimeError: No CUDA GPUs are available`` on a CUDA build that cannot find a device. See :doc:`installation` for both cases.

Processor
---------

The numbers below were measured on a Linux workstation with PyTorch 2.12 (CUDA 13.2) and an NVIDIA H200; CPU timings are on that machine's CPU, GPU timings on one H200.

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Design run
     - CPU
     - GPU
     - Peak GPU memory
   * - Toy AP-1 oracle, 50 bp, ``batch_size=16``
     - 3.1 s (1 thread)
     - --
     - --
   * - BPNet GATA2 (0.11 M params), 2114 bp, ``batch_size=16``
     - 30 s (1 thread), 8 s (8 threads)
     - 0.8 s
     - 232 MB

Both are complete default runs (``max_iter=1000``), which stop early after 180 iterations for the BPNet example. CPU timings depend strongly on how many threads PyTorch is allowed: the BPNet run takes about 30 s single-threaded, 11 s on four threads, and 8 s on eight. The takeaway is that a small oracle is perfectly comfortable on a CPU -- the tutorials that use one are written to run there -- while a GPU buys roughly an order of magnitude, and much more for larger models and batches.

GPU memory
----------

Ledidi keeps one weight matrix of shape ``(4, length)`` and, at each iteration, backpropagates through ``batch_size`` sequences at once. Its own footprint is negligible; memory is dominated by the oracle's activations, so it scales roughly linearly in ``batch_size`` and in the size of the model.

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Oracle (2114 bp input)
     - ``batch_size``
     - Peak GPU memory
   * - BPNet GATA2 (0.11 M params)
     - 1
     - 75 MB
   * - BPNet GATA2 (0.11 M params)
     - 16 (default)
     - 232 MB
   * - BPNet GATA2 (0.11 M params)
     - 32
     - 408 MB
   * - BPNet GATA2 (0.11 M params)
     - 64
     - 734 MB
   * - BPNet GATA2 (0.11 M params)
     - 128
     - 1402 MB
   * - BPNet ATAC (6.4 M params)
     - 16 (default)
     - 1415 MB

A few hundred megabytes covers a BPNet-scale design, so any modern GPU is enough. Large oracles (Enformer, Borzoi, and other long-context models) are the case that actually strains a card, and there the constraint is the same one you already face when running the model itself. If a design runs out of memory, lower ``batch_size`` first -- it trades wall-clock time and gradient smoothness for memory, and does not change what Ledidi is optimizing. The bundled agent skill's ``references/memory-and-oom.md`` walks through the full recovery ladder.

System memory and disk
----------------------

Peak resident memory was under 1 GB for both runs above, so system RAM is rarely the binding constraint; a CPU-only design needs about as much RAM as running the oracle on the same batch.

The install itself is small -- the ``ledidi`` wheel is under 100 KB -- but a CUDA-enabled PyTorch is not: ``torch`` plus its bundled NVIDIA libraries take roughly 4.3 GB of disk. The CPU-only PyTorch build is far smaller if space is tight -- a complete CPU-only environment measures 1.3-1.5 GB against roughly 4.3 GB for the CUDA one; see :doc:`installation`. Budget separately for the oracle checkpoints and any genome files you use; the BPNet models used in the tutorials are about 31 MB in total, while an ``hg38`` FASTA is around 3 GB.


Software
========

Ledidi requires **Python >= 3.10** and **PyTorch >= 2.0**. The Python floor comes from ``tangermeme``, which requires 3.10 or later. Ledidi is tested on Python 3.10, 3.11, 3.12, and 3.13 on every commit.

Ledidi is pure Python and has no compiled extensions of its own, so it should run anywhere PyTorch does. Continuous integration exercises it on Linux only, so Linux is the best-tested platform.

Core dependencies
-----------------

These four are installed automatically by ``pip install ledidi``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Package
     - Version
     - Why Ledidi needs it
   * - ``torch``
     - ``>= 2.0``
     - The optimization loop, the Gumbel-softmax sampler, and the oracle model itself.
   * - ``tangermeme``
     - ``>= 1.3.0``
     - Input validation (``tangermeme.utils._validate_input``) and the sequence logo drawn by :func:`ledidi.plot.plot_edits`. Also supplies the sequence and model utilities used throughout the tutorials.
   * - ``numpy``
     - any
     - Array handling in the plotting and pruning utilities.
   * - ``matplotlib``
     - any
     - :func:`ledidi.plot.plot_loss` and :func:`ledidi.plot.plot_edits`.

``tangermeme`` brings a scientific stack of its own (``pandas``, ``scipy``, ``scikit-learn``, ``numba``, ``pyfaidx``, ``pybigtools``, ``memelite``, ``tqdm``), so a complete environment is larger than these four suggest -- a clean CPU-only install came to 36 packages. Those extras are what make the genomics I/O and motif utilities used in the tutorials available without further installation.

Optional dependencies
---------------------

None of these are needed to design sequences; install them only for the task at hand.

*Development and testing* -- ``pip install "ledidi[dev]"`` adds ``pytest`` and ``pytest-cov``. Run the suite from the repository root with ``python -m pytest tests/``. It is CPU-only and deterministic; the GPU tests are marked and deselected by default, and can be run on a CUDA machine with ``python -m pytest tests/ -m gpu``.

*Building the documentation* -- ``docs/requirements.txt`` lists ``sphinx-rtd-theme``, ``nbsphinx``, ``pandoc``, and ``jinja2``, alongside ``torch`` and ``matplotlib``.

*Running the tutorials* -- the tutorials use real oracle models, each of which brings its own dependency. Install only the ones for the tutorial you are reading:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Used for
   * - ``bpnetlite``
     - The BPNet oracles in most tutorials. Checkpoints are at `Zenodo record 14604495 <https://zenodo.org/records/14604495>`_.
   * - ``enformer-pytorch``
     - The Enformer oracle used to validate designs in Tutorial 7.
   * - ``boda``
     - The Malinois oracle used for MPRA activity design.
   * - ``pyfaidx``
     - Reading genome FASTA files when designing into real loci.
   * - ``memelite``
     - Scanning designed sequences against a MEME motif database.
   * - ``seaborn``, ``tqdm``
     - Plotting and progress bars in a few of the notebooks.

Tutorials 0 and 8 use a parameter-free toy oracle and need none of the above -- they run anywhere Ledidi itself does.

*Coding agents* -- Ledidi bundles a `Claude Code Agent Skill <https://docs.claude.com/en/docs/claude-code/skills>`_ that is installed with the package and copied into place by the ``ledidi-install-skills`` console script. It requires nothing beyond Ledidi itself. See the :doc:`landing page <index>` for what the skill covers, and :doc:`installation` for keeping the installed copy up to date.


Where to go next
================

- :doc:`installation` -- step-by-step install recipes, CPU-only included, and how to confirm they worked.
- :doc:`getting_started` -- a step-by-step walkthrough of your first design.
- :doc:`input_output` -- the exact tensor shapes and dtypes Ledidi expects.
- :doc:`parameters` -- the knobs worth tuning, starting with ``l`` and ``batch_size``.
- :doc:`faq` -- CPU vs GPU, reproducibility, and common error messages.
