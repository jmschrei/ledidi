# Designing with several independent models

Real designs usually care about more than one property: raise MYC binding *without*
losing transcription, open chromatin *because of* a specific factor, keep three
cell types apart. The relevant models are separate objects, so they have to be
combined into one oracle before ledidi can use them.

## DesignWrapper

```python
from ledidi.wrappers import DesignWrapper

oracle = DesignWrapper([chrombpnet, bpnet_max, bpnet_e2f6])
y_bar = torch.tensor([[5.0, 4.0, 3.0]])           # one entry per concatenated output
X_bar = ledidi(oracle, X, y_bar, device='cuda')
```

`DesignWrapper` runs `X` through every model and concatenates the results along
the **last** axis. It behaves exactly like one multi-task model afterward.

- Each model may contribute any number of outputs. Two models returning
  `(batch, 1)` and `(batch, 3)` give a wrapper output of `(batch, 4)`.
- All models must agree on every dimension except the last.
- **You own the index bookkeeping.** Nothing labels the columns, so the mapping
  from `y_bar` entry to model is positional and entirely yours to track. Write the
  list of models and `y_bar` next to each other, and keep the order stable when you
  add a model — inserting one shifts every index after it.
- Constructor validation: a single `Module` instead of a list raises `TypeError`
  ("models must be a list or tuple"), an empty list raises `ValueError`, and a
  non-`Module` element raises `TypeError` naming its index.

## When to hand-roll instead

`DesignWrapper` only concatenates. Write your own `Module` when you need anything
else — per-model input preprocessing, per-model output normalization, weighting
one oracle above another, or reducing several models to a single scalar objective:

```python
class Balanced(torch.nn.Module):
	def __init__(self, acc, tf):
		super().__init__()
		self.acc, self.tf = acc, tf

	def forward(self, X):
		a = self.acc(X)
		t = (self.tf(X) - 2.1) / 0.7      # z-score to the accessibility model's scale
		return torch.cat([a, t], dim=-1)
```

Trimming or tiling inputs per model is common enough to have its own file →
`references/receptive-field.md`.

## Comparable dynamic ranges

Predictions from independently trained models live on different scales, because
read depth, data quality, and assay all affect the range. A model predicting 10–20
and one predicting 0–4 cannot be balanced by a shared `l`, and with `MinGap`
(whose whole objective is the *gap* between outputs) mismatched ranges make the
design impossible rather than merely awkward: an off-target model that never
predicts below 10 can never be pushed under an on-target model that never exceeds 4.

Before designing, predict on a few hundred real sequences with each model, compare
the ranges, and z-score inside your wrapper if they differ by more than a few fold.

## Cost

Every added oracle costs a forward *and* backward pass per iteration, so
wall-clock per iteration grows roughly linearly with the number of models — and
with their size, not their count: one 512-filter ChromBPNet typically dominates
four 64-filter BPNets. GPU memory grows the same way, which is the usual reason a
multi-model design fails outright → `references/memory-and-oom.md`.

## Validating multi-model designs

A design balanced across several oracles should be checked against a model that
was *not* in the wrapped oracle, not merely re-scored with the same ensemble — an
ensemble can be exploited jointly. See `references/validating-designs.md`.

## Related references

`references/oracle-contract.md` for the contract each wrapped model must
meet, `references/multi-task-models.md` for selecting outputs and the
hold-at-baseline idiom (which applies unchanged to concatenated models),
`references/custom-losses.md` for `MinGap` across models,
`references/memory-and-oom.md` when several oracles will not fit.
