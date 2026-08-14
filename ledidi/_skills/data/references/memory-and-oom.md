# CUDA out-of-memory during design

**Do not apply anything in this file preemptively.** The default configuration —
`batch_size=16`, one BPNet-scale oracle, a ~2 kbp field — costs a few hundred MiB
and needs none of it. Reach for this file only after a real
`torch.cuda.OutOfMemoryError`, and measure rather than guess:

```python
torch.cuda.reset_peak_memory_stats()
X_bar = ledidi(model, X, y_bar, device='cuda')
print(torch.cuda.max_memory_allocated() / 2**20, "MiB peak")
```

## Where the memory goes

Each iteration runs a forward *and* backward pass through every oracle on a batch
of `batch_size` sequences, and the activations of all of them must be retained
between the two. Peak memory is therefore roughly linear in **`batch_size` × design
length × oracle size × number of oracles**. The learned weight matrix itself is
`(1, n_channels, length)` — a few tens of KiB, never the problem. The
`best_sequence`/`best_weights` clones are one batch and one weight matrix, also
negligible.

## The ladder — stop at the first rung that works

### 1. Lower `batch_size`

Linear reduction, no code changes, and it does not bias the design: both losses are
per-sequence means, so the gradient keeps the same scale. What you lose is
gradient-estimate quality per step, which typically costs some extra iterations.
Try 8, then 4. This is the right answer far more often than anything below it.

### 2. Shorten the design field

The other linear axis. Design the 2 kbp that matters rather than the 20 kbp around
it, or reduce the number of chunks / overlap in a tiled wrapper
([receptive-field.md](receptive-field.md)).

### 3. Gradient checkpointing on the oracle

Trades roughly one extra forward pass for a large cut in retained activations.
Safe here because oracles are frozen and in `.eval()` mode, so recomputation is
deterministic. Works through the normal `ledidi()` call, inside a wrapper.

**It must be segmented to help.** Wrapping the whole model in a single
`checkpoint(...)` call saves essentially nothing, because backward still
materializes every intermediate activation while recomputing that one block. The
saving comes from splitting the model into segments so only one segment's
activations are live at a time. Measured on a 16-layer, 128-filter conv stack at
`length=4096`, `batch_size=32`:

| approach | peak |
|---|---|
| no checkpointing | 1220.6 MiB |
| whole model in one `checkpoint(...)` | 1218.4 MiB — **no benefit** |
| `checkpoint_sequential(net, 4, ...)` | **642.4 MiB** |

```python
from torch.utils.checkpoint import checkpoint_sequential

class Checkpointed(torch.nn.Module):
	"""Segment a Sequential oracle so only one segment's activations are live."""

	def __init__(self, model, n_segments=4):
		super().__init__()
		self.model = model
		self.n_segments = n_segments

	def forward(self, X):
		X = checkpoint_sequential(self.model.net, self.n_segments, X,
			use_reentrant=False)
		return X.mean(dim=-1)[:, :1]        # your model's own head
```

`use_reentrant=False` is required for this to behave correctly when only the input
requires grad. If the oracle is not a `Sequential`, call `checkpoint` on individual
sub-blocks instead — the principle is one live segment, not one live model.

### 4. Gradient accumulation over micro-batches

Split one `batch_size=16` step into four `batch_size=4` forward/backward passes,
accumulating gradients on the weight matrix before stepping. This is
*mathematically equivalent* to the large batch for losses that are a per-example
mean — the default `MSELoss()` and `MinGap` both qualify, since `MinGap`'s min/max
run across outputs and only its final `mean` touches the batch.

Measured on a fixed batch of 16 split into 4 micro-batches of 4, comparing the
accumulated gradient on the weight matrix against the full-batch gradient:

| output loss | relative gradient difference |
|---|---|
| `MSELoss()` (mean) | 5.2e-08 — equivalent |
| `MinGap` | 5.3e-08 — equivalent |
| `mean + y.std(dim=0)` | 3.4e-01 — **not** equivalent |
| `mean + y.max(dim=0)` | 7.0e-01 — **not** equivalent |

So the rule holds in both directions: per-example losses decompose exactly, and any
loss coupling examples (a batch-level max, std, or a diversity penalty) does not.
Note the counterexample only shows up once the batch actually varies — at iteration
0 the weight matrix is zeros, every sampled sequence is identical, and a coupling
term contributes no gradient, so testing this at initialization will wrongly suggest
equivalence.

Also mind the normalization: dividing each micro-batch loss by the number of
micro-batches is correct for a *mean*-reduced loss but wrong for a `sum`-reduced one
(measured 75% off), and unequal micro-batch sizes need proportional weighting.

This requires writing your own loop, because `Ledidi.fit_transform` owns the
optimizer step. See [designer-object.md](designer-object.md) for the object you
subclass, and expect to reimplement the best-iterate tracking, early stopping, and
history yourself.

### 5. One oracle at a time (multi-model designs)

When the objective **decomposes over oracles** — the default MSE over concatenated
outputs does — you can score one model per backward pass and let the gradients
accumulate on the weight matrix, so only one model's activations are alive at a
time.

**Mind the normalization, exactly as in rung 4.** `MSELoss()` is a *mean* over the
concatenated outputs, not a sum, so each per-model term needs a `1/n_models` factor
(or use `reduction='sum'` consistently on both sides). Measured with three
one-output oracles against the joint gradient:

| per-model term | max &#124;difference&#124; | ratio to joint |
|---|---|---|
| unscaled sum | 1.9e-04 | **3.00** — 3× too large |
| divided by `n_models` | 3.1e-10 | 1.0000 |

Getting this wrong does not error; it silently rescales the output loss against
`l * input_loss` and changes the design.

This is **invalid for `MinGap` across models**, whose min/max couple the outputs and
cannot be decomposed per model — there is no per-model term to accumulate. Also
requires a custom loop.

### 6. CPU↔GPU shuttling (last resort)

Only if the oracle *parameters* themselves do not fit. One constraint governs the
whole approach: backward needs each model's weights on the same device as the
activations its forward saved, so the move must bracket **forward and backward
together** and therefore cannot live in an `nn.Module.forward` — it needs a custom
loop. Cost is a PCIe round trip per model per iteration, which at `max_iter=1000`
usually dominates runtime.

### 7. Shrink the objective

Drop an oracle, swap a large replicate for a small one, or reduce the chunk count.
Less of the objective, but honest and fast.

## Memory that is not the oracle

These are surprising because they are not part of the optimization at all.

- **The default return path retains an autograd graph.** `fit_transform` returns a
  clone of the best iterate, so a design that improved on its template comes back
  with `requires_grad=True`. `.detach()` designs you intend to hold in bulk. The
  `n_samples` path is drawn under `torch.no_grad()` and comes back detached —
  measured at `length=2114`, `n_samples=5000` peaks at **486 MiB** (the designed
  tensor alone is 161 MiB); before that draw was wrapped it cost 648 MiB and
  returned a graph.
- **Large draws are still cheaper via the designer**, because you can move each
  chunk off the GPU as you go rather than concatenating 5,000 designs in device
  memory:

  ```python
  X = X.cuda()                      # ledidi() moves the MODEL, not your template
  X_bar, designer = ledidi(model, X, y_bar, return_designer=True, device='cuda')
  with torch.no_grad():
      draws = torch.cat([designer(X).cpu() for _ in range(n // designer.batch_size + 1)])[:n]
  ```

  Each chunk lands on the CPU, so device memory never holds more than one batch.
- **`return_history=True` accumulates on the design device.** It stores
  `torch.where(X_hat != X_)` index tensors per iteration, on the GPU — confirmed
  `cuda:0` in a CUDA run. Measured on a realistic design (2114 bp, `batch_size=16`,
  ~25 edits/sequence): **21.3 MiB per 1000 iterations**, and it scales with
  `batch_size × edits × max_iter` — at `batch_size=50` that is ~65 MiB. Usually
  affordable, but it is GPU memory nobody expects to be holding.
- **Catalogs and repeats stack on-device before returning.** An affinity catalog of
  20 targets × 16 sequences × 4 × 2114 float32 is 10.3 MiB resident (measured peak
  21.3 MiB during the stack), and `n_repeats` multiplies it again
  → [catalogs-and-repeats.md](catalogs-and-repeats.md). Rarely the problem on its
  own; it matters when it lands on top of an already-tight design.
- **Fragmentation across runs.** Many sequential designs in one process can fail
  with plenty of free memory. `torch.cuda.empty_cache()` between catalog or repeat
  steps usually clears it; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set
  before `torch` is imported is the heavier lever.

## Reproducibility interaction

Micro-batching changes the number and shape of Gumbel draws, so a split run is not
bitwise identical to the unsplit one → [reproducibility.md](reproducibility.md).

## Validation-time OOM is a different problem

Running out of memory while *checking* a design — attributing 50 designs, scanning
motifs, scoring against several models — has its own knobs, and tangermeme's skill
owns them: `references/deep_lift_shap.md` (lower `batch_size`/`n_shuffles`, and note
that its `batch_size` counts example×reference *pairs*, not examples) and
`references/comparing-models.md` (pass `device=` per call so only one model occupies
the GPU at a time). Do not re-derive those here.

## Related references

[objective.md](objective.md) for what `batch_size` means to the gradient,
[multiple-models.md](multiple-models.md) and
[receptive-field.md](receptive-field.md) for the two ways oracle cost multiplies,
[designer-object.md](designer-object.md) for writing a custom loop.
