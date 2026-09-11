# Trimming edits with greedy_pruning

ledidi already penalizes edit count during optimization, but a finished design
usually still carries edits that contribute almost nothing. `greedy_pruning` removes
them after the fact, giving the smallest set of edits that still achieves the goal.

```python
from ledidi.pruning import greedy_pruning

X_bar_p = greedy_pruning(model, X, X_bar[:1], threshold=1)
```

## How it works

Each round, it tries reverting every remaining edit one at a time, measures
`|y_hat_full - y_hat_reverted|` summed over the selected outputs, and permanently
reverts the single edit with the **smallest** effect — provided that effect is below
`threshold`. When the cheapest available revert would exceed the threshold, it stops
and returns what is left. It runs under `@torch.no_grad()`.

## Arguments

- **`X`** — the original template, `(1, n_channels, length)`, one-hot (all-zero
  columns allowed).
- **`X_hat`** — the designed sequence, same shape as `X`. **One sequence at a time**;
  loop over a batch:

  ```python
  X_bar_p = torch.cat([greedy_pruning(model, X, X_bar[i:i+1], threshold=1)
      for i in range(len(X_bar))])
  ```

- **`threshold`** (default 1) — maximum tolerated change in output from reverting one
  edit. Must be **non-negative**; `0` is valid and is an exact no-op (nothing can be
  below 0), returning the input unchanged. Larger values prune more aggressively and
  drift further from the design's prediction.
- **`target`** (default `None`) — which output to measure the change on. Same rules as
  in design: an `int` selects exactly one output, `None` uses all of them.
- **`verbose`** (default `False`, unlike `ledidi`) — prints each pruned index and its
  delta.

Note the threshold is on the change from reverting a **single** edit, not on
cumulative drift. Many small reverts can move the prediction well past `threshold` in
total, so check the final prediction rather than assuming a bound.

## `target` must match how you sliced during design

If you designed against a wrapped model with `target=None`, prune with the same
wrapper and `target=None`. If you designed with `target=3` on the raw model, prune
with `target=3`. **A mismatch prunes against the wrong objective and is not
reported** — it is a silently different question, not an error.

Negative and out-of-range values are rejected here for the same reason as in design
(`ValueError`) → `references/multi-task-models.md`. Older releases
accepted them, and the consequence here was worse than in design: an empty selection
made `torch.abs(y_hat - y_mod).sum()` equal `0.0` for every candidate, below any
threshold, so **every edit was pruned** and the template came back. Measured on a
3-output oracle with 2 planted edits at `threshold=0.5`, `target=-1` kept 0 of 2
edits and dropped the output from 5.0 to 4.0.

## Cost

Round `k` evaluates every remaining edit, so pruning `n` edits costs on the order of
`n²/2` forward passes — for 40 edits that is ~800 predictions. This is usually the
slowest step after design itself. It is also `batch_size=1` work, so the GPU is
underused; pruning a batch of 50 designs sequentially is often longer than the design
was.

## Using it well

- **Sweep the threshold** to expose the trade-off between edit count and output rather
  than trusting one value:

  ```python
  for t in (0.1, 0.25, 0.5, 1.0, 2.0):
      Xp = greedy_pruning(model, X, X_bar[:1], threshold=t)
      n = int((X[0] != Xp[0]).any(dim=0).sum())
      print(t, n, model(Xp).item())
  ```

- **Check what survived.** Pruning tends to keep exactly the high-attribution edits and
  discard low-attribution ones, which is a useful cross-check on both the pruning and
  the design → `references/validating-designs.md`.
- It is independent of how the edits were produced — any pair of (template, edited
  sequence) works, not just ledidi output.

## Related references

`references/multi-task-models.md` for `target` semantics and the empty-slice
failure, `references/validating-designs.md` for confirming pruned designs
still hold up, `references/pipeline.md` for where pruning sits in the workflow.
