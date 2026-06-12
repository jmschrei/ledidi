.. ledidi documentation master file, created by
   sphinx-quickstart on Tue Feb 20 13:46:59 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


ledidi
==========



Ledidi turns any pre-trained machine learning model(s) into an editor of biological sequences to make them exhibit desired characteristics. Ledidi is fast and programmable, meaning that precise characteristics can be induced by simultaneously using several models for design. For example, Ledidi can use one model that predicts chromatin accessibility and a second model that predicts the binding of CTCF to design sites that are accessible due to high binding of CTCF, or are accessible and explicitly do not have CTCF binding to them. Ledidi can design cell type-specific regions by using models that each predict activity in a different cell line. Essentially, you can use Ledidi to design anything that you can find one or more machine learning models to make predictions for.

Ledidi works by phrasing the design process as a continuous optimization problem and then solving this problem using off-the-shelf techniques. Specifically, Ledidi calculates a loss that is comprised of an output loss, measuring how closely the edited sequence matches the desired characteristics as predicted by the machine learning models, and an input loss, measuring the number of edits made thus far. By minimizing this loss, Ledidi designs sequences that closely match the desired output using as few edits as possible. A technical challenge is that the gradient of this loss cannot be directly applied to a categorical sequence, such as DNA. Ledidi circumvents this challenge by learning a continuous weight matrix from which edits to an initial sequence are sampled, and updates this weight matrix using the straight-through estimator at each step. This trick allows the full information in the gradient to be used while also only passing categorical sequences into the machine learning oracle models, which have likely only been trained on categorical sequences.

Here, we have included documentation for how we used Ledidi in the paper alongside additional examples showing Ledidi in practice. We will be updating this repository with cool use-cases we find.



Installation
============

Ledidi is on PyPI and can be installed with pip::

   pip install ledidi

If you already have PyTorch installed this takes less than a minute; otherwise most of the time is spent installing PyTorch. Installing Ledidi automatically pulls in `tangermeme <https://github.com/jmschrei/tangermeme>`_, which Ledidi uses for input validation and which provides the sequence and model utilities used throughout the tutorials. Ledidi requires **Python >= 3.10** and **PyTorch >= 2.0**.

To install from source for development, Ledidi is packaged with a ``pyproject.toml`` and is built with `uv <https://docs.astral.sh/uv/>`_::

   git clone https://github.com/jmschrei/ledidi
   cd ledidi
   uv pip install -e ".[dev]"

The test suite runs on CPU with ``python -m pytest tests/``.



Quickstart
==========

Here is a complete, runnable example that uses a tiny parameter-free oracle, so there is nothing to download. The oracle scores how well a sequence matches the AP-1 motif ``TGACTCA``, and Ledidi designs the edits that maximize that score::

   import torch
   from ledidi import ledidi

   # A toy oracle: it slides the AP-1 motif TGACTCA across the sequence and
   # returns the best match (a perfect match scores 7). Any differentiable torch
   # model that maps a one-hot sequence to a prediction can be used in its place.
   motif = "TGACTCA"
   weights = torch.zeros(1, 4, len(motif))
   for i, char in enumerate(motif):
       weights[0, "ACGT".index(char), i] = 1.0

   class MotifScore(torch.nn.Module):
       def forward(self, X):
           return torch.nn.functional.conv1d(X, weights).amax(dim=-1)

   # A random one-hot starting sequence of shape (1, 4, length).
   torch.manual_seed(0)
   idxs = torch.randint(0, 4, (1, 50))
   X = torch.zeros(1, 4, 50).scatter_(1, idxs.unsqueeze(1), 1.0)

   # Ask Ledidi for edits that make the oracle output a perfect match (7).
   y_bar = torch.tensor([[7.0]])
   X_hat = ledidi(MotifScore(), X, y_bar, device="cpu", random_state=0, verbose=False)

   # X_hat has shape (batch_size, 4, length); decode the first designed sequence.
   designed = "".join("ACGT"[c] for c in X_hat[0].argmax(dim=0))
   print("TGACTCA" in designed)  # True -- the motif was inserted in a few edits

Ledidi finds the cheapest place to introduce the motif and edits only the positions it needs to, rather than overwriting a whole stretch of sequence. See :doc:`getting_started` for a step-by-step walkthrough of this example, :doc:`input_output` for the exact tensor formats Ledidi expects, and :doc:`parameters` for the handful of knobs worth tuning.



.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   self
   getting_started.rst
   input_output.rst
   parameters.rst
   faq.rst
   whats_new.rst


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorials

   tutorials/Tutorial_0_-_Getting_Started.ipynb
   tutorials/Tutorial_1_-_Design_of_Protein_Binding_Sites.ipynb
   tutorials/Tutorial_2_-_Constraints_and_Priors.ipynb
   tutorials/Tutorial_3_-_In-Painting.ipynb
   tutorials/Tutorial_4_-_Multiple_Models.ipynb
   tutorials/Tutorial_5_-_Affinity_Catalogs.ipynb
   tutorials/Tutorial_6_-_Custom_Loss_Functions.ipynb
   tutorials/Tutorial_7_-_Validating_Your_Designs.ipynb
   tutorials/Tutorial_8_-_The_Ledidi_Object.ipynb



.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API

   api/ledidi.rst
   api/pruning.rst
   api/plot.rst
   api/losses.rst
   api/wrappers.rst
