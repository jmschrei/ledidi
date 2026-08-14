# Affinity catalogs and repeats

Two ways to get more than one design out of one call, with return shapes that are
easy to get wrong.

## Affinity catalogs: a list of `y_bar`

An affinity catalog runs the whole design once per target strength, from the same
template. Pass a **list** of `y_bar` tensors:

```python
y_bars = [torch.tensor([[v]]) for v in (1.0, 2.0, 3.0, 4.0)]
X_bar = ledidi(model, X, y_bars, device='cuda')
print(X_bar.shape)          # (4, batch_size, n_channels, length)
```

- **The list is required even when the targets are a plain range.** There is no
  `n_steps` argument and no shortcut for `torch.linspace`; passing a tensor with an
  extra dimension is ambiguous with a multi-output `y_bar`, so ledidi insists on the
  explicit form.
- Each element must independently satisfy the `y_bar` rules — leading dimension 1,
  width matching the sliced output ([io-and-validation.md](io-and-validation.md)).
- Catalogs compose with everything else: masks, in-painting, multi-output targets. For
  a multi-output catalog, each list element is a full `(1, n_outputs)` tensor.

Catalogs are worth running even when you only care about one target value. They show
which edits persist across strengths (signal) versus which appear at one value only
(noise or an artifact of that target), they make the output loss interpretable by
giving you the whole easy-to-hard curve, and they reveal when a chosen target forces
the design into oddities that a nearby value would avoid.

Expect motifs to appear and disappear across steps rather than accumulating
monotonically: ledidi is matching a *precise* value with as few edits as possible, so
it will reuse and re-tune the same near-motif sites rather than adding sites one at a
time.

## Repeats: `n_repeats`

`n_repeats` runs the entire procedure several times from a zero weight matrix,
producing genuinely independent sets of edits:

```python
X_bar = ledidi(model, X, y_bar, n_repeats=3, device='cuda')
print(X_bar.shape)          # (3, batch_size, n_channels, length)
```

This is the fix for "all my designs look the same" — the `batch_size` sequences of a
single run are drawn from one weight matrix and are correlated by construction
([designer-object.md](designer-object.md)). `n_repeats` gives independent solutions;
`n_samples` does not.

## Return shapes

Dimensions are added at the front and **collapse when they equal 1**:

| call | `X_bar.shape` |
|---|---|
| single `y_bar`, `n_repeats=1` | `(batch_size, n_channels, length)` |
| single `y_bar`, `n_repeats=R` | `(R, batch_size, n_channels, length)` |
| catalog of `K`, `n_repeats=1` | `(K, batch_size, n_channels, length)` |
| catalog of `K`, `n_repeats=R` | `(K, R, batch_size, n_channels, length)` |

So a leading dimension of size `K` is ambiguous on its own — it may be a catalog or
repeats. Do not index by position on faith; check what you passed. `return_designer`
and `return_history` follow the same collapsing rules, giving a bare object, a list,
or a list of lists.

With `n_samples`, `batch_size` above is replaced by `n_samples`.

## Seeding

Each designer gets its own seed offset, so catalog steps and repeats are independent
of one another while the whole call stays reproducible
→ [reproducibility.md](reproducibility.md).

## Cost

A catalog of `K` targets with `R` repeats is `K * R` full design runs — the time
multiplies, there is no sharing between them, and this is almost always the cost that
matters. All results are also stacked **on-device** before returning, which is
modest by comparison → [memory-and-oom.md](memory-and-oom.md).

## Related references

[designer-object.md](designer-object.md) for `n_samples` and why batch members
correlate, [io-and-validation.md](io-and-validation.md) for per-element `y_bar` rules,
[validating-designs.md](validating-designs.md) for reading a catalog with attributions.
