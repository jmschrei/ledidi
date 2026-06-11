# Ledidi

[![PyPI version](https://img.shields.io/pypi/v/ledidi.svg)](https://pypi.org/project/ledidi/)
[![Python versions](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://pypi.org/project/ledidi/)
[![Tests](https://github.com/jmschrei/ledidi/actions/workflows/test.yml/badge.svg)](https://github.com/jmschrei/ledidi/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/ledidi/badge/?version=latest)](https://ledidi.readthedocs.io/en/latest/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jmschrei/ledidi/blob/master/LICENSE)
[![PyPI Downloads](https://img.shields.io/pypi/dm/ledidi.svg)](https://pepy.tech/projects/Ledidi)

<img src="https://github.com/user-attachments/assets/500e5f18-f9af-4cc9-b76c-23296d60640d" width="40%" height="40%">

[[preprint](https://www.biorxiv.org/content/10.1101/2025.04.22.650035v1)][[docs](https://ledidi.readthedocs.io/en/latest/)]

Ledidi is an approach for designing edits to biological sequences that induce desired properties. It does this by inverting the normal way that one thinks about machine learning models: normally during training, the data is held constant and model weights are updated; here, Ledidi holds the model constant and updates the *data*. When using Ledidi, the data are the sequences that are being edited and the "updates" to these sequences are the edits. Using this simple paradigm, Ledidi *turns any trained machine learning model into a biological sequence editor*, regardless of the original purpose of the model. 

A challenge with designing edits for categorical sequences (e.g., DNA sequences with four possible nucleotides) is that the inputs are usually one-hot encoded but gradients are continuous. Applying the gradients directly would yield oddities like 1.2 of an A and -0.3 of a G. These values yield two problems: (1) such values are not biologically meaningful as you cannot design a sequence that is "1.2 A and -0.3 G" and (2) because most machine learning models have been trained on one-hot encoded data and are calibrated for that distribution, such sequences would likely confuse the model, push it off the training data manifold, and result in anomalous sequences even if one managed to turn the output back into a valid sequence.

Ledidi resolves this challenge by phrasing edit design as an optimization problem over a continuous weight matrix `W` that gradients *can* be directly applied to. Specifically, at each iteration, Ledidi draws one-hot encoded sequences one position at a time from a Gumbel-softmax distribution defined by `log(X + eps) + W` where `X` is the original one-hot encoded sequence and `eps` is a small value. When the drawn character is different from the character in `X`, likely because `W` has been updated at that position to prefer another character, an "edit" is being made. These edited sequences are then passed through the frozen pre-trained model and the output is compared to the desired output. The total loss is then derived from this output loss as well as an input loss that is the count of proposed edits, limiting the number that Ledidi will propose. A gradient is calculated from this loss and `W` is updated, keeping `X` unchanged.

![image](https://github.com/user-attachments/assets/41ffe180-8171-4f28-a88e-41cc1c79985a)

Applied out-of-the-box across dozens of sequence-based ML models, Ledidi can take a sequence that does not exhibit some form of activity (e.g., TF binding, transcription, etc) and design edits that turn it into one exhibiting strong activity! Despite major differences in these models, this design does not require any hyperparameter tuning to get good results, and only requires balancing the input and output loss in a simple way to get the best results.

<img width="1367" alt="image" src="https://github.com/user-attachments/assets/39edd8fc-be58-4fb3-8006-18401fc81336" />

Although our examples right now are largely nucleotide sequence-based, one can also apply Ledidi out-of-the-box to RNA or protein models (or really, to any model with a sequence of categorical inputs such as small molecules). The only limitation is what will fit in your GPU or what you are willing to wait for on your CPU!

### Installation
`pip install ledidi`

If you already have PyTorch installed, this should take less than a minute. Otherwise, it may take several minutes, of which most of the time is spent installing PyTorch.

To install from source for development, Ledidi is packaged with a `pyproject.toml` and is built with [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/jmschrei/ledidi
cd ledidi
uv pip install -e .
```

### Quickstart

Here is a complete, runnable example that uses a tiny parameter-free oracle, so there is nothing to download. The oracle scores how well a sequence matches the AP-1 motif `TGACTCA`, and Ledidi designs the edits that maximize that score.

```python
import torch
from ledidi import ledidi

# A toy oracle: it slides the AP-1 motif TGACTCA across the sequence and returns
# the best match (a perfect match scores 7). Any differentiable torch model that
# takes a one-hot sequence and returns a prediction can be used in its place.
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
print("TGACTCA" in designed)  # True -- Ledidi inserted the motif in just a couple of edits
```

Ledidi finds the cheapest place to introduce the motif and edits only the positions it needs to, rather than overwriting a whole stretch of sequence.

### Input and output format

Ledidi works with one-hot encoded sequences as `torch.float32` tensors of shape `(1, n_channels, length)` — for DNA that is `(1, 4, length)`, with channels ordered `A, C, G, T`. The returned `X_hat` has shape `(batch_size, n_channels, length)`: a batch of independently sampled designed sequences (`batch_size` defaults to 16).

If you are starting from a string, you can one-hot encode it and decode the result directly:

```python
import torch

def one_hot_encode(seq):
	mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
	X = torch.zeros(1, 4, len(seq))
	for i, char in enumerate(seq):
		X[0, mapping[char], i] = 1.0
	return X

def decode(X):  # X is a single (4, length) one-hot sequence
	return "".join("ACGT"[c] for c in X.argmax(dim=0))

X = one_hot_encode("ACGTACGTACGT")
# ... run Ledidi to get X_hat ...
# decode(X_hat[0]) turns the first designed sequence back into a string
```

The sibling library [tangermeme](https://github.com/jmschrei/tangermeme) provides `one_hot_encode` and `characters` utilities that handle this (and reading FASTA/loci) for you.

### Running on CPU or GPU

By default `ledidi(...)` moves the model and tensors to the GPU (`device='cuda'`). On a machine without a CUDA GPU you must pass `device='cpu'` explicitly, otherwise the call will error:

```python
X_hat = ledidi(model, X, y_bar, device='cpu')
```

### Usage

Please see the [documentation site](https://ledidi.readthedocs.io/en/latest/) for more complete tutorials on how to use Ledidi. You can find some example BPNet models -- including the one used in the tutorials -- at https://zenodo.org/records/14604495.

At a high level, Ledidi requires a sequence-based machine learning model, an initial sequence, and a desired output from the model. If the model is multi-task, this desired output does not need to cover all of the tasks and can even just use a single one (see this tutorial on [PyTorch wrappers](https://tangermeme.readthedocs.io/en/latest/vignettes/Wrappers_are_Productivity_Hacks.html) for practical advice on how to manipulate models to make them easier to work with Ledidi). These three items are then passed into `ledidi` for design! Note that there is also a `Ledidi` object which is an internal wrapper. No need to use that yet.

```python
import torch
from bpnetlite.bpnet import ControlWrapper
from bpnetlite.bpnet import CountWrapper
from ledidi import ledidi

X = .. your sequence ..

model = torch.load("GATA2.torch", weights_only=False)  # Load a BPNet model predicting GATA2 binding
model = CountWrapper(ControlWrapper(model))  # Use only the count predictions

y_bar = torch.zeros(1, dtype=torch.float32) + 4.5  # Set a desired output of a high value
X_hat = ledidi(model, X, y_bar)  # Run Ledidi!
```

The `ledidi` method will return a batch of sequences whose size can be controlled with the `batch_size` parameter. These sequences are independently generated by sampling from the underlying Gumbel-softmax distributions learned by Ledidi. Importantly, each batch of edits will be correlated in the sense that they all come from the single set of Ledidi weights. So, if a motif could have been substituted in at multiple locations Ledidi will end up only choosing one and these edits will all be drawn from that choice. <b>If you need many different sets of edits you will need to run Ledidi many times with different random seeds</b>. However, an upside is that once the Ledidi weight matrix is learned, edited sequences can be sampled extremely quickly using the forward function.

```
X_hat = ledidi(model, X, y_bar, n_samples=10000)  # Runs almost as fast as the default!
```

If we take a look at the edits being proposed in this sequence, we can see that frequently Ledidi can make designs with fewer edits than the length of the motif that should be inserted. Essentially, Ledidi automatically finds locations on the sequence that are close to the motifs that need to be inserted and precisely makes edits there rather than forcing an entire motif in somewhere in the sequence. Here we can see attributions from our GATA prediction model before edits are made (low because GATA does not bind to this example region, see the tutorial), after edits are made with the edited positions shown in orange, and after a greedy pruning algorithm is used to trim the edit space down in magenta.

<img src="https://github.com/user-attachments/assets/6ea64ed7-bfad-479b-a433-116ad7465ba6" width="70%" height="70%">

When `verbose` is set to `True` you will get logs showing the input and output losses, and the total loss from combining them with the mixing weight. The logs may look something like this:

![image](https://github.com/user-attachments/assets/35040f80-a997-46f2-9855-a4db1a8337b6)

This means that by iteration 100, an average of 35.75 edits were proposed per sequence, and that the best iteration at convergence only had 28.19 edits per sequence on average. The output loss went from 16 to 0.18, which is a pretty major drop in the loss. If you pass in `return_history=True` you can get these losses over the course of the entire process. Here is an example from the tutorial.

<img src="https://github.com/user-attachments/assets/e9801dbc-1821-476a-9b1b-1962c0552f29" width="60%" height="60%">

You can also look at where edits were proposed across the entire process. This gives you a sense for how long edits persisted across the process. In this example, tons of edits are proposed initially and then many of them are undone because the input loss tries to minimize this number.

<img src="https://github.com/user-attachments/assets/f115605e-cc26-4ba8-b3ed-ff0b336f06b4" width="60%" height="60%">

### Key parameters

Most designs only require tuning a couple of knobs. These are passed straight through `ledidi(...)`:

| Parameter | Default | What it does |
|---|---|---|
| `l` | `0.1` | Weight on the edit (input) loss. **The main knob to tune** — lower values prioritize hitting the target output, higher values prioritize making fewer edits. |
| `target` | `None` | For a multi-task model, the index of the output to design against. `None` uses the whole output. |
| `output_loss` | `MSELoss()` | The loss comparing the model's prediction to `y_bar`. Swap in any callable `f(y_hat, y_bar)`. |
| `tau` | `1` | Gumbel-softmax temperature; higher is sharper (closer to a hard argmax). |
| `batch_size` | `16` | Sequences sampled and averaged per iteration. |
| `max_iter` | `1000` | Maximum optimization iterations. |
| `early_stopping_iter` | `100` | Stop after this many iterations without improvement. |
| `input_mask` | `None` | Boolean mask, length `length`, of positions that may **not** be edited. |
| `random_state` | `None` | Seed for reproducible sampling. Unlike calling `torch.manual_seed`, this does not mutate the global RNG, so it leaves the rest of your script unaffected. |
| `device` | `'cuda'` | Device to run on (pass `'cpu'` if you have no GPU). |
| `verbose` | `True` | Print the input, output, and total loss as the design proceeds. |

A few parameters live on `ledidi()` itself rather than being passed through: `n_samples` (draw this many designed sequences after optimization), `n_repeats` (run the whole procedure multiple times), and `return_history` / `return_designer` (also return the loss history / the fitted designer).

### Multiple models, custom losses, and pruning

The top-level package exports `ledidi` and `Ledidi`; the other tools live in their own submodules.

**Balance several oracles** by combining them with `DesignWrapper`, which concatenates their predictions. Pass a vector `y_bar` with one target per model — for example, to raise one model's output while holding another's fixed:

```python
from ledidi.wrappers import DesignWrapper

model = DesignWrapper([model_a, model_b])  # predictions concatenated along the last axis
X_hat = ledidi(model, X, y_bar, device="cpu")  # y_bar has one entry per model output
```

**Design cell-type-specific elements** with the `MinGap` loss, which maximizes a set of on-target outputs while minimizing the off-target ones. It ignores `y_bar` (still required by the signature, so pass a placeholder):

```python
import torch
from ledidi.losses import MinGap

in_mask = torch.tensor([True, False, False])  # output 0 on-target; 1 and 2 off-target
X_hat = ledidi(model, X, torch.zeros(1, 3), output_loss=MinGap(in_mask), device="cpu")
```

**Trim unnecessary edits** after design with `greedy_pruning`, which removes edits one at a time as long as each removal changes the model output by less than `threshold`:

```python
from ledidi.pruning import greedy_pruning

X_pruned = greedy_pruning(model, X, X_hat[:1], threshold=1)  # prunes a single sequence
```

There are also plotting helpers in `ledidi.plot` (`plot_edits`, `plot_history`).

### Inspecting the edits

To see exactly which positions changed and how, compare the original sequence to a designed one:

```python
seq = X_hat[0]  # one designed sequence, shape (4, length)
positions = torch.where((X[0] != seq).any(dim=0))[0]
for p in positions:
	before = "ACGT"[X[0, :, p].argmax()]
	after  = "ACGT"[seq[:, p].argmax()]
	print(f"position {p.item()}: {before} -> {after}")
```

For the Quickstart example this prints just two edits, the ones that complete the `TGACTCA` motif. Each of the `batch_size` sequences in `X_hat` is sampled independently, so they may carry slightly different edits.

### Tutorials

Worked examples as Jupyter notebooks (also rendered on the [documentation site](https://ledidi.readthedocs.io/en/latest/)). Several use BPNet oracle models available at https://zenodo.org/records/14604495.

1. [Design of Protein Binding Sites](https://github.com/jmschrei/ledidi/blob/master/docs/tutorials/Tutorial_1_-_Design_of_Protein_Binding_Sites.ipynb)
2. [Constraints and Priors](https://github.com/jmschrei/ledidi/blob/master/docs/tutorials/Tutorial_2_-_Constraints_and_Priors.ipynb)
3. [In-Painting](https://github.com/jmschrei/ledidi/blob/master/docs/tutorials/Tutorial_3_-_In-Painting.ipynb)
4. [Multiple Models](https://github.com/jmschrei/ledidi/blob/master/docs/tutorials/Tutorial_4_-_Multiple_Models.ipynb)
5. [Affinity Catalogs](https://github.com/jmschrei/ledidi/blob/master/docs/tutorials/Tutorial_5_-_Affinity_Catalogs.ipynb)
6. [Custom Loss Functions](https://github.com/jmschrei/ledidi/blob/master/docs/tutorials/Tutorial_6_-_Custom_Loss_Functions.ipynb)

### Roadmap

Ledidi is research software under active development. The broad direction, roughly in order of priority:

1. **Hardening and usability.** Expand the test suite and coverage, improve robustness and error messages, and grow the tutorials so that going from a trained model to a finished design is straightforward and hard to get wrong.

2. **Validation, trust, and benchmarking.** Gradient-based design can exploit an oracle and produce sequences that score well but are not biologically meaningful. We want first-class evaluation built in: round-tripping designs through independent held-out models, attribution-based checks that the intended motifs were actually created, realism checks against natural sequence, and a reproducible benchmark across many oracles.

3. **Interoperability and end-to-end pipelines.** Tight integration with sibling tools — such as [tangermeme](https://github.com/jmschrei/tangermeme) for attribution and in silico mutagenesis, and efficient oracle models — so that one can go from raw data all the way to finished designs using command-line tools alone.

4. **Multi-objective design under biological constraints.** Designing for several properties at once while respecting hard constraints such as GC content, restriction sites, codon usage, and the absence of off-target effects.

5. **Insertions and deletions.** Extending Ledidi beyond substitutions so that designs can change the length of a sequence, not just its content.

6. **Minimal constructs.** Merging the idea of design with that of in silico marginalization to build the smallest possible construct that still exhibits a desired property, generalizing the current greedy pruning of edits.

7. **Uncertainty and ensembling.** Confidence estimates for designs through oracle ensembles and calibration, so that both people and automated systems know when a design can be trusted.

8. **Agentic interfaces.** Programmatic, agent-facing entry points built on top of the command-line pipeline, so that an agent can carry out the full design loop end to end without a human manually performing any of the steps.

Two threads cut across all of these: **reproducibility and provenance** — capturing the full recipe behind every design so it can be audited and regenerated — and **modality breadth**, with first-class support for RNA, protein, and other categorical sequences beyond DNA.

### Citation

If you use Ledidi in your work, please cite the preprint:

> Schreiber, J., Lorbeer, F. K., Heinzl, M., Lu, Y., Stark, A., & Noble, W. S. (2025). Programmatic design and editing of cis-regulatory elements. *bioRxiv*. https://doi.org/10.1101/2025.04.22.650035

```bibtex
@article{schreiber2025ledidi,
  title={Programmatic design and editing of cis-regulatory elements},
  author={Schreiber, Jacob and Lorbeer, Franziska Katharina and Heinzl, Monika and Lu, Yang and Stark, Alexander and Noble, William Stafford},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.04.22.650035},
  url={https://www.biorxiv.org/content/10.1101/2025.04.22.650035v1}
}
```

### License

Ledidi is released under the [Apache 2.0 License](https://github.com/jmschrei/ledidi/blob/master/LICENSE).
