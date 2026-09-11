# Custom output losses

The output loss *is* the design objective. Swapping it is where most of the
expressive power of ledidi lives, and it is a two-line change.

## The contract

```python
def my_loss(y_hat, y_bar):
	...
	return scalar_tensor      # shape (), differentiable
```

- Called as `output_loss(y_hat, y_bar)` — predictions first, target second. This is
  the opposite order from most PyTorch losses' documentation and from
  `tangermeme.design`'s `loss(y, y_hat)`. Getting it backwards is silent for
  symmetric losses like MSE and wrong for everything else.
- `y_hat` is the output **after `target` slicing**, and `y_bar` must match that width
  → `references/io-and-validation.md`.
- **Must return a scalar** — see the footgun below.
- Must be differentiable with respect to `y_hat`.
- Any callable works: a function, a `torch.nn.Module`, or a plain class with
  `__call__`.

### Footgun: the loss must reduce to a scalar yourself

`total_loss.backward()` requires a scalar, so any loss that reduces per-example —
`reduction='none'`, or `torch.nn.functional.kl_div` with the wrong `reduction` — must
be reduced by you, usually with `.mean()`. This is the most common mistake when
writing a custom loss here, and it is also the opposite of the contract in
`tangermeme.design`, which *requires* `reduction='none'` so it can rank candidates.

## `MinGap` — output-specific design with no target values

`ledidi.losses.MinGap` implements the min-gap loss of Gosai et al.: maximize the gap
between the **minimum on-target** output and the **maximum off-target** output. Use it
for cell type-specific or assay-specific elements, where "specific" matters more than
any particular value.

```python
import torch
from ledidi.losses import MinGap

in_mask = torch.tensor([False, True, False])     # output 1 on-target, 0 and 2 off
X_bar = ledidi(model, X, torch.zeros(1, 3), output_loss=MinGap(in_mask), device='cuda')
```

- `in_mask` is `torch.bool`, one entry per model output. It must contain **at least
  one `True` and one `False`** — an all-on or all-off mask raises `ValueError`, since
  there is no gap to maximize.
- It **ignores `y_bar` entirely** but the signature still requires it; pass a
  correctly shaped placeholder such as `torch.zeros(1, n_outputs)`.
- Taking the *minimum* over on-target outputs means all of them must come up; taking
  the *maximum* over off-target ones means all of them must go down. That is the
  point, and also the weakness: one on-target output the model predicts poorly, or one
  off-target output correlated with the on-target ones, can stall the whole design.
- There are no target values, so nothing forces the on-target predictions to be
  *high* — a large gap between two negative values satisfies it. If you need absolute
  levels too, combine it with an MSE term.
- `MinGap` is a plain class implementing `__call__`, not a `torch.nn.Module`. Do not
  expect `.to(device)`, parameters, or a `forward` method.

### `MinGap` requires `target=None`

The two cannot be combined. With `target=int` the loss sees a single column, so:

- a length-1 `in_mask` fails `MinGap`'s own constructor check
  (`ValueError: in_mask must contain at least one on-target (True) and one off-target`), and
- a full-width `in_mask` against the 1-wide sliced output raises
  `IndexError: The shape of the mask [3] at index 0 does not match ...`.

Leave `target=None` and expose exactly the outputs you want to contrast with a wrapper
→ `references/multi-task-models.md`.

Across several models, `MinGap` additionally requires **comparable dynamic ranges**,
or the gap it maximizes is unreachable → `references/multiple-models.md`.

## Rewarding a direction instead of matching a value

MSE needs a specific number for every output, and often you do not have one: you know
you want a particular accessibility, but not what the corresponding TF-binding
prediction should be. Reward the direction instead, and only specify targets for what
you actually want to control.

```python
def acc_and_max(y_hat, y_bar):
	# y_hat: (batch, 2) = [accessibility, MAX binding]; y_bar: (batch, 1) = accessibility
	mse = torch.nn.functional.mse_loss(y_hat[:, 0:1], y_bar)
	return mse - y_hat[:, 1].mean()      # subtract, because ledidi MINIMIZES
```

Subtracting is how you maximize. The scale of the reward term relative to the MSE term
is now yours to balance, and it interacts with `l` — see the scaling note below.

## One-sided and ballpark losses

**One-sided** — "at least this strong, and better is fine":

```python
def at_least(y_hat, y_bar):
	return torch.clamp(y_bar - y_hat, min=0).pow(2).mean()   # no penalty above target
```

A softer variant rewards exceeding the target less strongly than it penalizes falling
short, rather than not at all.

**Ballpark** — "anywhere within a radius of the target is perfect":

```python
def ballpark(y_hat, y_bar, radius=1.0):
	return torch.clamp((y_hat - y_bar).abs() - radius, min=0).pow(2).mean()
```

Expect a ballpark design to land near the *cheap* edge of the acceptable band: the
input loss breaks the tie, so a target of 7 with radius 1 tends to settle near 6.
That is correct behavior, not a bug.

## Structured outputs: profiles

BPNet-style base-pair-resolution profiles are probability distributions, and MSE
handles them badly — it smears density outside the region you asked for. KL divergence
is sharper:

```python
def kl(y_hat, y_bar):
	log_p = torch.nn.functional.log_softmax(y_hat, dim=-1)
	return torch.nn.functional.kl_div(log_p, y_bar, reduction='batchmean')
```

Two things to keep in mind with profiles: the target must be a *feasible*
distribution (a value of 2 at one position is impossible if the profile sums to 1),
and stranded profiles are offset from the binding event in opposite directions
because BPNet is trained on fragment starts, so a symmetric target is not what you
want.

## Scaling and `l`

Custom losses are frequently orders of magnitude smaller than MSE on counts — KL on
a 1 kbp profile, or a reward term of a few units. At `l=0.1` the input loss then
dominates and the design barely moves. Set `l=0` first to confirm the objective is
optimizable at all, then raise it until the edit count is acceptable →
`references/objective.md`.

## Related references

`references/objective.md` for how the output loss combines with the input loss,
`references/multi-task-models.md` for exposing the outputs your loss needs
and for the hold-at-baseline idiom, `references/io-and-validation.md` for
`y_bar` shapes including placeholders.
