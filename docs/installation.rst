.. currentmodule:: ledidi


============
Installation
============

Ledidi is a small pure-Python package -- the wheel is under 100 KB -- and installing it is a one-liner. Almost everything on this page is really about PyTorch, which is the one dependency big enough to be worth choosing deliberately: whether you want the CUDA build or the much smaller CPU-only build, and how to confirm afterwards that the one you got actually works.

If you are in a hurry::

   pip install ledidi

then jump to :ref:`verifying-your-installation` and run the snippet there. If it prints ``ledidi is installed and working.`` you are done and can move on to :doc:`getting_started`.

For the hardware and software Ledidi needs -- runtimes, GPU memory, and the full dependency list -- see :doc:`requirements`.


Before you start
================

Ledidi requires **Python >= 3.10**. Check what you have::

   python --version

If that reports 3.9 or older, install a newer Python before continuing; ``pip install ledidi`` will otherwise refuse with a message about the required Python version. The floor comes from ``tangermeme``, which is a hard dependency.

We strongly recommend installing into a virtual environment rather than a system or base Python, so that a PyTorch build chosen for Ledidi cannot disturb another project. Every recipe below creates one.


Choosing a PyTorch build
========================

This is the only real decision. PyTorch ships as two very different distributions, and ``pip install ledidi`` will pull whichever one your platform defaults to -- on Linux, that is the CUDA build, even on a machine with no GPU.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Build
     - Disk
     - Choose it when
   * - CUDA
     - ~4.3 GB
     - You have an NVIDIA GPU and want to design against anything larger than a toy oracle.
   * - CPU-only
     - ~1.3-1.5 GB
     - You have no GPU, or you are on a laptop, in CI, or in a container where 3 GB of CUDA libraries is dead weight.

Those figures are measured: complete CPU-only environments built with the commands below came to 1.3 GB on Python 3.10 and 1.5 GB on Python 3.13, of which ``torch`` itself is 695 MB. The CUDA build is ``torch`` at 1.2 GB plus 3.1 GB of bundled NVIDIA libraries.

Ledidi itself is identical either way. The only behavioral difference is the ``device`` argument, covered in :ref:`cpu-only-usage` below.


Installing with a GPU (CUDA)
============================

On Linux with an NVIDIA GPU, the default wheels already include CUDA, so nothing special is needed::

   python -m venv ledidi-env
   source ledidi-env/bin/activate
   pip install ledidi

Confirm that PyTorch can see the GPU before going further::

   python -c "import torch; print(torch.cuda.is_available())"

This must print ``True``. If it prints ``False`` on a machine with a GPU, PyTorch was installed without CUDA support or the driver is too old for the CUDA version in the wheel -- pick the matching command from the `PyTorch install matrix <https://pytorch.org/get-started/locally/>`_ and reinstall ``torch``.


Installing without a GPU (CPU-only)
===================================

To get the small build, install ``torch`` from PyTorch's CPU index *first*, then install Ledidi. Because ``torch`` is already present and satisfies Ledidi's ``>= 2.0`` requirement, the second command will not pull the CUDA build over the top of it::

   python -m venv ledidi-env
   source ledidi-env/bin/activate
   pip install --index-url https://download.pytorch.org/whl/cpu torch
   pip install ledidi

Order matters. Running ``pip install ledidi`` first would download several gigabytes of CUDA wheels that you would then have to replace.

The same thing with `uv <https://docs.astral.sh/uv/>`_, which is considerably faster::

   uv venv ledidi-env --python 3.10
   uv pip install --python ledidi-env/bin/python --index-url https://download.pytorch.org/whl/cpu torch
   uv pip install --python ledidi-env/bin/python ledidi

Recent versions of uv can also select the backend directly, without the explicit index::

   uv pip install --torch-backend=cpu torch

Verify that you got the build you asked for::

   python -c "import torch; print(torch.__version__)"

A CPU-only build reports a version ending in ``+cpu``, for example ``2.13.0+cpu``. A CUDA build ends in something like ``+cu132`` instead.


.. _cpu-only-usage:

Using Ledidi on a CPU-only machine
----------------------------------

There is one thing to remember, and it is the single most common stumbling block for new users: :func:`ledidi` **defaults to** ``device='cuda'``. On a CPU-only machine you must pass ``device='cpu'`` explicitly::

   X_hat = ledidi(model, X, y_bar, device="cpu")

Forget it and the call fails immediately, with one of two messages depending on which PyTorch build you have:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Your PyTorch build
     - Error you get
   * - CPU-only (``+cpu``)
     - ``AssertionError: Torch not compiled with CUDA enabled``
   * - CUDA, but no GPU visible
     - ``RuntimeError: No CUDA GPUs are available``

Both mean the same thing in practice: pass ``device='cpu'``. A convenient idiom that works on either kind of machine is to pick the device up front and pass it through::

   device = "cuda" if torch.cuda.is_available() else "cpu"
   X_hat = ledidi(model, X, y_bar, device=device)

Everything in Ledidi works on a CPU, including the tutorials -- Tutorials 0 and 8 are written to run there with no downloads. Designs against small oracles take seconds; see the :doc:`requirements` page for measured timings.


Installing with conda
=====================

Ledidi is not on conda-forge, so install it with pip inside a conda environment. Let conda provide Python and, if you want it, PyTorch::

   conda create -n ledidi python=3.11
   conda activate ledidi
   pip install ledidi

Mixing conda-installed and pip-installed PyTorch in the same environment is a well-known source of trouble; pick one. If conda already provides ``torch``, pip will leave it alone, since Ledidi only requires ``>= 2.0``.


Installing from source
======================

For development, or to get an unreleased fix. Ledidi is packaged with a ``pyproject.toml`` and built with hatchling::

   git clone https://github.com/jmschrei/ledidi
   cd ledidi
   uv pip install -e ".[dev]"

The ``dev`` extra adds ``pytest`` and ``pytest-cov``. The equivalent with pip is ``pip install -e ".[dev]"``.

Confirm the checkout is healthy by running the test suite from the repository root::

   python -m pytest tests/

It is CPU-only and deterministic, and takes about ten seconds. The GPU tests are marked and deselected by default; on a CUDA machine you can run them with ``python -m pytest tests/ -m gpu``, and they skip themselves if no CUDA device is present.

To build the distributions::

   uv build

This writes a wheel and an sdist to ``dist/``.


What gets installed
===================

Ledidi declares four dependencies -- ``torch``, ``tangermeme``, ``numpy``, and ``matplotlib`` -- but ``tangermeme`` brings a scientific stack of its own, so a complete environment is larger than that list suggests. A clean CPU-only install on Python 3.10 produced 36 packages, including ``pandas``, ``scipy``, ``scikit-learn``, ``numba``, ``pyfaidx``, ``pybigtools``, ``memelite``, and ``tqdm``. Those are what make the genomics I/O and motif utilities used in the tutorials available without any further installation.

Installing Ledidi also puts the ``ledidi-install-skills`` console script on your ``PATH``. It copies the bundled Claude Code Agent Skill into ``~/.claude/skills/ledidi``; see the :doc:`landing page <index>` for what the skill covers. It is entirely optional and installs nothing else.


.. _verifying-your-installation:

Verifying your installation
===========================

Run this after installing, whichever route you took. It prints the versions of everything Ledidi depends on, reports whether a GPU was found, and then runs a complete design against a parameter-free oracle -- so it exercises the whole stack, optimizer, sampler, and gradient flow, without downloading a model checkpoint. It takes a few seconds and works identically on a CPU-only machine::

   import platform
   import torch
   import numpy
   import matplotlib
   import tangermeme
   import ledidi

   print("python     ", platform.python_version())
   print("torch      ", torch.__version__)
   print("numpy      ", numpy.__version__)
   print("matplotlib ", matplotlib.__version__)
   print("tangermeme ", tangermeme.__version__)
   print("ledidi     ", ledidi.__version__)

   device = "cuda" if torch.cuda.is_available() else "cpu"
   print("device     ", device)

   # A parameter-free oracle that scores how well a sequence matches the AP-1
   # motif TGACTCA, and a random 50 bp sequence to edit.
   weights = torch.zeros(1, 4, 7)
   for i, char in enumerate("TGACTCA"):
       weights[0, "ACGT".index(char), i] = 1.0

   class MotifScore(torch.nn.Module):
       def forward(self, X):
           return torch.nn.functional.conv1d(X, weights.to(X.device)).amax(dim=-1)

   torch.manual_seed(0)
   idxs = torch.randint(0, 4, (1, 50))
   X = torch.zeros(1, 4, 50).scatter_(1, idxs.unsqueeze(1), 1.0)

   X_hat = ledidi.ledidi(MotifScore(), X, torch.tensor([[7.0]]), device=device,
       random_state=0, verbose=False)

   designed = "".join("ACGT"[c] for c in X_hat[0].argmax(dim=0).cpu())
   assert "TGACTCA" in designed, "design failed -- see the FAQ page"
   print("\nledidi is installed and working.")

The version numbers will differ from yours, but the last line is what matters. On a CUDA machine::

   python      3.13.5
   torch       2.12.0+cu132
   numpy       2.4.6
   matplotlib  3.10.0
   tangermeme  1.4.1
   ledidi      2.2.0
   device      cuda

   ledidi is installed and working.

and from a clean CPU-only environment built with the commands above, where the whole script takes a few seconds::

   python      3.10.20
   torch       2.13.0+cpu
   numpy       2.2.6
   matplotlib  3.10.9
   tangermeme  1.4.1
   ledidi      2.2.0
   device      cpu

   ledidi is installed and working.

Note that ``device`` correctly reports ``cpu`` in the second case, and the design still succeeds -- a CPU-only installation is fully functional, not a degraded one.


Troubleshooting
===============

``torch.cuda.is_available()`` prints ``False`` on a machine with a GPU
   PyTorch was installed without CUDA support. Check ``torch.__version__``: a
   version ending in ``+cpu`` is the CPU-only build. Reinstall ``torch`` using
   the command from the `PyTorch install matrix
   <https://pytorch.org/get-started/locally/>`_ for your CUDA version. If the
   version string does contain ``+cu`` and CUDA is still unavailable, the NVIDIA
   driver is likely older than the wheel requires; ``nvidia-smi`` will show the
   driver version.

``AssertionError: Torch not compiled with CUDA enabled``
   You have the CPU-only build and something asked for the GPU -- almost always
   :func:`ledidi`'s ``device='cuda'`` default. Pass ``device='cpu'``.

``RuntimeError: No CUDA GPUs are available``
   Same fix, but this is the CUDA build failing to find a device: either the
   machine has no GPU, or ``CUDA_VISIBLE_DEVICES`` is set to an empty or invalid
   value.

The install pulled gigabytes of ``nvidia-*`` packages I do not want
   That is the CUDA build of PyTorch. Follow the CPU-only recipe above, being
   careful to install ``torch`` from the CPU index *before* installing Ledidi.

``ERROR: Package 'ledidi' requires a different Python``
   Your interpreter is older than 3.10. Create an environment on a newer Python;
   ``uv venv --python 3.10`` will fetch one for you if none is installed.

``pip: command not found`` inside a ``uv venv``
   Environments created by ``uv venv`` do not include ``pip`` by default. Use
   ``uv pip install`` instead, or create the environment with
   ``uv venv --seed``.

An import fails after upgrading
   Upgrade Ledidi and its dependencies together with ``pip install --upgrade
   ledidi``. Note that :func:`ledidi.plot.plot_edits` requires
   ``tangermeme >= 1.3.0``; an older ``tangermeme`` in the environment will
   surface as an import or attribute error from the plotting module.


Upgrading and uninstalling
==========================

Upgrade in place with::

   pip install --upgrade ledidi

If you use the bundled agent skill, refresh the installed copy afterwards with ``ledidi-install-skills --force``, since upgrading the package does not update the copy in ``~/.claude/skills``.

To remove Ledidi::

   pip uninstall ledidi

This leaves the dependencies behind. The simplest way to reclaim the disk is to delete the whole virtual environment, which is the main reason to install into one.


Where to go next
================

- :doc:`getting_started` -- a step-by-step walkthrough of your first design.
- :doc:`requirements` -- hardware and software requirements in detail, with measured runtimes and memory.
- :doc:`input_output` -- the exact tensor shapes and dtypes Ledidi expects.
- :doc:`faq` -- CPU vs GPU, reproducibility, and common error messages.
