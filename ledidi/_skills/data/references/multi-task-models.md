# Designing against some tasks of a multi-task model

You have one model with many outputs — Enformer's 5,313 tracks, Malinois's three
cell lines, a BPNet with counts and profiles — and you want to design against a
subset of them. There are two routes, and one of them is strongly preferred.

## Use a wrapper (preferred)

Write a `torch.nn.Module` that returns exactly the outputs you care about, in the
order you want them, and hand *that* to `ledidi` with `target=None`. This is the
recommended route unless the user has explicitly said not to add a wrapper.

```python
import torch

class Tasks(torch.nn.Module):
	"""Expose DNase, GATA2, and MAX (in that order) from a multi-task model."""

	def __init__(self, model, idxs):
		super().__init__()
		self.model = model
		self.idxs = idxs

	def forward(self, X):
		return self.model(X)[:, self.idxs]        # shape (batch, len(idxs))

model_bar = Tasks(model, [4766, 4834, 5111])
y_bar = torch.tensor([[6.0, 4.0, 0.0]])           # one entry per exposed output
X_bar = ledidi(model_bar, X, y_bar, device='cuda')
```

Why a wrapper rather than the built-in slicing:

- It is the **only** way to select more than one output (see below).
- The exposed output order is yours, so `y_bar` reads in the order you wrote it
  and stays readable months later.
- It composes with everything else — you can normalize, reweight, or sum tracks in
  the same wrapper, and the same object works for `greedy_pruning` and for
  `tangermeme.predict` when you validate.
- It makes `target=None` correct, which removes the index arithmetic that causes
  the silent failures below.

## `target`: one output, non-negative, `int` only

The built-in selector is `target`. It slices as `[:, target:target+1]`, so:

- `target=None` → all outputs.
- `target=int` → **exactly one** output. There is no way to pass several.
- Anything else — `tuple`, `list`, `slice`, a boolean `torch.Tensor` mask, a 0-d
  tensor, `numpy.int64`, `float` — raises `TypeError`. A `numpy` integer from
  `numpy.argmax` or a dataframe lookup must be cast with `int()` first.
- `bool` is accepted, because `isinstance(True, int)` is true in Python:
  `target=True` selects output 1 and `target=False` selects output 0. Never write
  this on purpose.

### Footgun: a negative or out-of-range `target` silently designs nothing

`target=-1` becomes `slice(-1, 0)` and `target=7` on a 3-output model becomes
`slice(7, 8)`. Both select an **empty** tensor of shape `(batch_size, 0)`. MSE
against an empty tensor is `nan`, `nan < best_total_loss` is never true, so no
iteration ever improves, the run early-stops, and `fit_transform` returns the
**unedited template**. The only signal is a torch broadcasting `UserWarning`,
which `verbose=False` hides.

Measured on a 3-output motif oracle asked for a value of 6:

| `target` | selected shape | initial loss | mean edits/seq | final output |
|---|---|---|---|---|
| `2` | `(1, 1)` | 4.0 | 2 | 6.00 |
| `-1` | `(1, 0)` | `nan` | 0 | 4.00 (unchanged) |
| `7` | `(1, 0)` | `nan` | 0 | 4.00 (unchanged) |

`-1` for "the last output" is a standard PyTorch habit and it does not work here.
Count your model's outputs and pass a non-negative index, or wrap and use
`target=None`. **If a design returns zero edits, check `target` first.**

The same slicing code is in `greedy_pruning`, where an empty selection makes every
candidate score 0.0 and therefore prunes **every** edit — see [pruning.md](pruning.md).

## Masking inside the loss (the fallback)

If a wrapper is genuinely off the table, a custom `output_loss` can do the
selection, because it receives the full output when `target=None`:

```python
in_idxs = torch.tensor([4766, 4834, 5111])

def masked_mse(y_hat, y_bar):
	return torch.nn.functional.mse_loss(y_hat[:, in_idxs], y_bar)

X_bar = ledidi(model, X, y_bar, output_loss=masked_mse, device='cuda')
```

Here `y_bar` must match the shape the loss compares against, `(1, 3)`, not the
model's full output width. This works, but the selection is now buried in the loss
where `greedy_pruning` and your validation code cannot see it — you will have to
repeat it everywhere. Prefer the wrapper.

## Holding outputs at baseline instead of dropping them

Excluding an output means the design is free to wreck it. Usually what you want is
to hold it *where it is*: predict on the template first and overwrite only the
entries you intend to change.

```python
y_bar = model_bar(X).detach().clone()   # current predictions for every exposed task
y_bar[0, 0] = 6.0                       # raise DNase
y_bar[0, 2] = 0.0                       # suppress MAX; GATA2 held at baseline
```

This is the multi-task version of a mask: rather than naming the motifs that must
survive, you tell the oracle which of its readouts must not move. It also allows
ledidi to rearrange sequence freely as long as those readouts stay put. Note that
`y_bar` must have a leading dimension of exactly 1, which `model_bar(X)` on a
single template already satisfies.

## Related references

[oracle-contract.md](oracle-contract.md) for the wrapper basics,
[multiple-models.md](multiple-models.md) when the tasks live in different models,
[custom-losses.md](custom-losses.md) for `MinGap` (which requires `target=None`)
and other objectives, [io-and-validation.md](io-and-validation.md) for `y_bar`
shape rules, [pruning.md](pruning.md) for keeping `target` consistent afterward.
