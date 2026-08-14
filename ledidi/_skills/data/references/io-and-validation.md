# Shapes, dtypes, and errors

ledidi validates its inputs up front and raises immediately rather than failing deep
inside the optimizer. Tensor checks are delegated to
`tangermeme.utils._validate_input`, which raises **`ValueError`** — not
`TypeError` — for a non-tensor, wrong dtype, wrong shape, or non-one-hot sequence.
`TypeError` is reserved for a non-`Module` model and a non-integer `target`.

## The template `X`

Shape `(1, n_channels, length)`, dtype `torch.float32`, one-hot along the channel
axis. DNA is 4 channels ordered `A, C, G, T`; RNA or protein is whatever order the
oracle was trained on.

- The **leading dimension must be exactly 1**. ledidi expands the single template
  to `batch_size` internally. To design for several templates, loop.
- Each position must sum to 1 with a single `1.0`, **or** be all zeros.
- **All-zero columns are legal** and mean unknown / `N`. They are also free to edit
  and cost nothing in the input loss, which is the entire mechanism behind
  in-painting → [inpainting.md](inpainting.md). Any `N` in a real template is
  therefore a candidate for editing and will *not* be preserved.

### Footgun: `one_hot_encode` returns `int8`

`tangermeme.utils.one_hot_encode` defaults to `dtype=torch.int8` for memory
reasons, and tangermeme's own entry points upcast each batch to the model's dtype
for you — so int8 sequences are idiomatic *there*. **ledidi does not upcast.**
Passing int8 fails with a message that never mentions your one-hot encoding, and
whose wording depends on which op hits it first:

```
RuntimeError: expected scalar type Char but found Float            # conv1d
RuntimeError: Input type (signed char) and bias type (float) ...   # a real BPNet
RuntimeError: mat1 and mat2 must have the same dtype ...           # a Linear head
```

Always cast and add the batch dimension:

```python
from tangermeme.utils import one_hot_encode

X = one_hot_encode("ACGTACGT...").unsqueeze(0).float()    # (1, 4, length), float32
```

tangermeme's skill owns the wider I/O story — reading FASTA, extracting loci from a
genome, `characters` for decoding — see its `references/io-loci.md`.

## The desired output `y_bar`

Shape `(1, n_outputs)`. The leading dimension must be 1; ledidi expands it against
the batch.

- Single-output model: `torch.tensor([[4.5]])`.
- Multi-task model or `DesignWrapper`: one entry per output, in the order the model
  returns them → [multi-task-models.md](multi-task-models.md),
  [multiple-models.md](multiple-models.md).
- **`n_outputs` must match the output width *after* `target` slicing**, not the
  model's full width. With `target=2` on a 3-output model the loss receives
  `(batch_size, 1)`, so `y_bar` is `(1, 1)`.
- A **list** of such tensors means an affinity catalog, not a batch →
  [catalogs-and-repeats.md](catalogs-and-repeats.md).
- Some losses ignore `y_bar` but still require it — `MinGap` needs a correctly
  shaped placeholder such as `torch.zeros(1, n_outputs)` →
  [custom-losses.md](custom-losses.md).

### Footgun: a missing bracket does not raise, it broadcasts

`torch.tensor([4.5])` has shape `(1,)`, whose *leading* dimension is 1 — so it passes
validation and silently broadcasts against the model output. Only a leading dimension
≠ 1 raises. Measured:

| `y_bar` | outcome |
|---|---|
| `[[4.5]]` — `(1, 1)` | correct |
| `[4.5]` — `(1,)` | **runs, no error** (a torch broadcasting `UserWarning` only) |
| `[1., 2., 3.]` — `(3,)` | `ValueError: y_bar must have a leading dimension of size 1` |

With a multi-output model the `(1,)` form fails later and elsewhere, as a
`RuntimeError` about mismatched tensor sizes rather than a `ValueError` from ledidi.
Write the inner brackets and check `y_bar.shape` before a long run.

## What you get back

`X_bar`, shape `(batch_size, n_channels, length)`, `float32`, one-hot — a batch of
independently sampled designs from one learned weight matrix, so they are
correlated by construction. Extra leading dimensions appear for catalogs and
repeats → [catalogs-and-repeats.md](catalogs-and-repeats.md).

**It carries an autograd graph.** Both the default return and the `n_samples` draw
come back with `requires_grad=True`. Call `.detach()` before storing many of them,
and see [memory-and-oom.md](memory-and-oom.md) for the memory this costs.

## Masks and priors

- `input_mask` — `torch.bool`, shape `(length,)`. `True` marks positions that may
  **not** be edited → [masks.md](masks.md).
- `initial_weights` — `float`, shape `(1, n_channels, length)`. Seeds the
  optimization; `-inf` forbids a character, finite values are soft priors →
  [initial-weights.md](initial-weights.md).

## Error reference

| Message / condition | Type | Fix |
|---|---|---|
| `X` not one-hot | `ValueError` | each column sums to 1 with one `1.0`; all-zero is allowed |
| `X` wrong shape | `ValueError` | must be `(1, n_channels, length)` |
| a dtype error naming `Char`/`signed char` | `RuntimeError` | `.float()` your `int8` one-hot |
| `y_bar must have a leading dimension of size 1` | `ValueError` | leading dim must be 1 — e.g. `(3,)` or `(2, 1)` raises; `(1,)` does **not** |
| `input_mask` wrong shape or dtype | `ValueError` | `(length,)` and `torch.bool` |
| `initial_weights` wrong shape | `ValueError` | `(1, n_channels, length)` |
| `X_hat` shape mismatch in pruning | `ValueError` | must equal `X`'s shape |
| `model must be a torch.nn.Module` | `TypeError` | wrap it |
| `target must be an integer or None` | `TypeError` | `int()` a `numpy` integer; no masks, lists, or `bool` |
| `target must be non-negative` | `ValueError` | `-1` is not "the last output" |
| `target=N selects no outputs from a model that returns M` | `ValueError` | count the outputs after wrapping |
| `shape must be a tuple of two positive integers` | `ValueError` | `Ledidi(shape=X.shape[-2:])` |
| `tau`/`lr`/`eps` non-positive | `ValueError` | must be > 0 |
| `batch_size`/`max_iter`/`early_stopping_iter`/`report_iter` non-positive | `ValueError` | positive integers |
| `n_repeats`/`n_samples` non-positive | `ValueError` | positive integers (`n_samples` may be `None`) |
| `l` negative | `ValueError` | `l >= 0`; `0` disables the input loss |
| `threshold` negative in pruning | `ValueError` | `>= 0`; `0` is a valid no-op |
| `in_mask` not `torch.bool`, or all-`True`/all-`False` | `ValueError` | `MinGap` needs both groups non-empty |
| unexpected keyword argument | `TypeError` | `Ledidi.__init__` takes no `**kwargs`; ledidi has no `output_mask`/`args`/`func` |

## Silent failures validation does *not* catch

These raise nothing and return plausible-looking results:

- a `y_bar` of shape `(1,)` → broadcasts silently (above)
- an inverted `input_mask` → protects the region you meant to edit
  ([masks.md](masks.md))
- a `target` that does not match how you sliced at design time, in
  `greedy_pruning` → prunes against the wrong objective ([pruning.md](pruning.md))
- `initial_weights` mutated in place, so a reused tensor carries the last run's
  learned values ([initial-weights.md](initial-weights.md))
- a refit `Ledidi` object resuming from its previous best weights
  ([designer-object.md](designer-object.md))
- a design whose oracle is invariant to its input → a plausible loss and no edits
  ([oracle-contract.md](oracle-contract.md))

## Related references

[objective.md](objective.md), [first-design.md](first-design.md) for a worked
minimal example, [reproducibility.md](reproducibility.md) for `random_state` and
`device`.
