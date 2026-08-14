# Sampling many designs and reusing a fit

Fitting the weight matrix is the expensive part; drawing sequences from a fitted
matrix is nearly free. There are three ways to exploit that, and they differ in what
you keep.

## Why the batch is correlated

Every sequence in a returned batch is sampled from the **same** learned weight
matrix. If a motif could have gone in three places, the optimization committed to one
and all `batch_size` sequences reflect that choice; they differ only in the
Gumbel-softmax noise at individual positions. So a batch is variations on one design,
not several independent solutions.

- want more sequences from **one** design → `n_samples` or the designer's `forward`.
- want **different** designs → `n_repeats`, or re-run with a different `random_state`
  → [catalogs-and-repeats.md](catalogs-and-repeats.md).

A different seed makes different designs *possible*, not certain. When one solution
is clearly cheapest, every seed finds it: on a toy oracle asked to build a single
motif, `random_state=0` and `random_state=7` both edited exactly positions 26 and 28.
Seeds diversify the search, they do not force diversity — if you need genuinely
distinct solutions, vary the objective (an affinity catalog) or constrain the obvious
one away with a mask.

## Option 1: `n_samples`

```python
X_bar = ledidi(model, X, y_bar, n_samples=10000, device='cuda')
```

Draws that many sequences after optimization, at almost no extra time cost. Internally
it calls the designer `n_samples // batch_size + 1` times and truncates, so the
returned count is exact.

**This is not free in memory.** The draw is not wrapped in `torch.no_grad()`, so every
call retains an autograd graph and the returned designs carry it. Use option 2 for
large draws — the measured cost is in [memory-and-oom.md](memory-and-oom.md).

## Option 2: keep the fitted designer

```python
X = X.cuda()                 # ledidi() moves the MODEL in place, not your template
X_bar, designer = ledidi(model, X, y_bar, return_designer=True, device='cuda')

with torch.no_grad():
	more = torch.cat([designer(X).cpu() for _ in range(100)])   # 100 * batch_size
```

The `X.cuda()` is not optional: `ledidi()` moves the model and returns designs on the
device, but leaves your template where it was, so `designer(X)` with a CPU `X` raises
`RuntimeError: Expected all tensors to be on the same device`.

`designer` is the fitted `Ledidi` object. Calling it returns `batch_size` freshly
sampled sequences each time. This is the memory-safe way to draw a lot, and it lets
you interleave sampling with other work, save the designer, or inspect
`designer.weights` to see what was learned (useful for checking whether a prior
survived → [initial-weights.md](initial-weights.md)).

With `n_repeats` or a catalog, `return_designer` gives a list (or list of lists)
following the same collapsing rules as the designs
→ [catalogs-and-repeats.md](catalogs-and-repeats.md).

## Option 3: build `Ledidi` yourself

```python
from ledidi import Ledidi

designer = Ledidi(model, shape=X.shape[-2:], l=0.5, verbose=False,
	random_state=0).to('cuda')
X_bar = designer.fit_transform(X.to('cuda'), y_bar.to('cuda'))
```

You now own device placement, and `shape` is `(n_channels, length)` — two positive
integers, **no batch dimension** (a wrong shape raises `ValueError`). Every keyword
`ledidi()` forwards is available here directly. Use this when you need to subclass, to
write a custom optimization loop (see the accumulation recipes in
[memory-and-oom.md](memory-and-oom.md)), or to hold the object across a long session.

### Footgun: refitting resumes from the previous best weights

`fit_transform` ends by assigning the best iterate's weights back to `self.weights`.
Calling it again therefore starts from those weights, **not** from zeros. Verified: a
fresh designer has all-zero weights; after one `fit_transform` the maximum absolute
weight is ~7.7, and a second call continues from there.

That is occasionally what you want (continue a design, or retarget a warm start), but
it makes "run it twice and compare" misleading. For an independent second run,
construct a new `Ledidi`, or use `n_repeats`.

## The history object

`return_history=True` adds a dict per run:

```python
X_bar, history = ledidi(model, X, y_bar, return_history=True, device='cuda')
history.keys()     # 'edits', 'input_loss', 'output_loss', 'total_loss', 'batch_size'
```

- `input_loss` / `output_loss` / `total_loss` — one float per iteration. Plot with
  [`plot_loss`](plotting.md).
- `edits` — one `torch.where(...)` tuple per iteration, recording every position that
  differed from the template at that step. Plot with [`plot_history`](plotting.md).
- `batch_size` — carried along so the plotting helpers can convert row indices to
  iterations.
- These tensors live on the **design device**, so a CUDA run accumulates them in GPU
  memory → [memory-and-oom.md](memory-and-oom.md).

Note that history is recorded for *every* iteration, including ones worse than the
best, while the returned design comes from the best iterate only — the final point on
the loss curve is not the design you have.

### Footgun: `forward` does not check one-hotness

`Ledidi.forward` validates only the shape of its input, not that it is one-hot
(`fit_transform` checks both). Sampling with a non-one-hot tensor therefore succeeds
and returns meaningless sequences. Pass the same template you fit with.

## Related references

[catalogs-and-repeats.md](catalogs-and-repeats.md) for independent designs and return
shapes, [memory-and-oom.md](memory-and-oom.md) for the sampling memory cost and custom
loops, [objective.md](objective.md) for what "best iterate" means,
[plotting.md](plotting.md) for consuming the history.
