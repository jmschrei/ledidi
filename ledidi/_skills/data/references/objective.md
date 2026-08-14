# The ledidi objective

Every design ledidi performs minimizes one scalar:

```
total_loss = output_loss(y_hat, y_bar) + l * input_loss
```

- **output loss** — how far the oracle's prediction on the edited sequence is from
  the desired output. Default `torch.nn.MSELoss()`.
- **input loss** — how many edits were made. Default `torch.nn.L1Loss(reduction='sum')`,
  divided by `2 * batch_size`, which makes it the **mean number of edited
  positions per sequence** (each edit changes two entries in a one-hot column,
  hence the 2). The number printed as `input_loss` in the logs is directly
  readable as "edits per sequence".
- **`l`** — the exchange rate between them, applied to the input loss.

Everything else in the library is a modification of one of these two terms.

## `l` is the knob you tune

Default `0.1`. It must be non-negative; `l=0` disables the input loss entirely
and lets ledidi edit as much as it likes.

- design hits the target but uses too many edits → **raise `l`**
- design makes few edits but never reaches the target → **lower `l`**

The two terms must be on comparable scales for `l` to mean anything. When the
output loss is small in magnitude — base-pair-resolution profiles, probability
distributions, a custom loss built from several terms, or many oracles averaged
together — the input loss dominates at `l=0.1` and the design will barely move.
Drop `l` by one to three orders of magnitude in those settings, or set `l=0`
first to confirm the output loss can be optimized at all, then raise it until the
edit count is acceptable. See [custom-losses.md](custom-losses.md).

## The output loss sees the *sliced* output

`fit_transform` computes `y_hat = model(X_hat)[:, target]` and only then calls
`output_loss(y_hat, y_bar)`, so `y_bar` must match the width **after** slicing
→ [io-and-validation.md](io-and-validation.md).

`target` is an `int` selecting exactly one output, or `None` for all of them. It
is not a mask and not a list, and a negative or out-of-range value fails
silently — see [multi-task-models.md](multi-task-models.md) before using it.

## The other two knobs: `eps` and `tau`

Both are best left alone, but they are worth understanding because they explain
why designs behave the way they do.

### `eps` (default `1e-4`) sets the price of an edit

`forward` builds its logits as `log(X + eps) + W`. That means the template's own
character starts at `log(1 + eps) ≈ 0` while every other character starts at
`log(eps)`, so `eps` fixes the gap the learned weight must overcome before an edit
can be drawn at all:

| `eps` | logit gap |
|---|---|
| `1e-2` | 4.62 |
| `1e-4` (default) | 9.21 |
| `1e-6` | 13.82 |

Measured at the default, sweeping a fixed weight at one non-template character:

| `W` | fraction of draws that take the edit |
|---|---|
| 5.0 | 2.0% |
| 9.0 | 45.2% |
| 9.5 | 57.7% |
| 15.0 | 99.9% |

The crossover sits right at the gap. So a *smaller* `eps` makes edits more
expensive to propose, and weight values in the fitted matrix should be read
against this scale — a weight of 2 is a nudge, a weight of 10 is a decision.

### `tau` (default `1`) changes the gradient, not the sample

`tau` divides the perturbed logits before the softmax. Two things follow, and the
repo's prose docs get both backwards, so trust this section over them:

1. **The returned sample is hard one-hot at every `tau`.** `forward` returns a
   scatter of `argmax(y_soft)`, and dividing by a positive scalar cannot reorder
   an argmax — the drawn sequence is bitwise identical across `tau` for a fixed
   noise draw. `tau` cannot make the output "closer to the argmax"; it is already
   the argmax.
2. **What `tau` moves is the straight-through gradient**, through `y_soft`. The
   size of that gradient is `(1/tau) × softmax-Jacobian`, and those two factors
   pull against each other, so the relationship is **not monotonic**. Measured
   through `Ledidi.forward` on a real design:

| `tau` | mean &#124;dL/dW&#124; | softmax `p_max` |
|---|---|---|
| 0.1 | 5.7e-24 | 1.0000 |
| 1.0 (default) | 7.1e-05 | 0.9997 |
| 2.0 | 1.2e-03 | 0.9709 |
| 10.0 | **1.5e-03** | 0.4557 |
| 100.0 | 1.3e-04 | 0.2677 |

Small `tau` saturates the softmax and the gradient vanishes; large `tau` flattens
it and the `1/tau` factor takes over. The peak is around `tau ≈ 10` for the
default `eps`, because what actually controls saturation is **`gap / tau`** —
at the default `tau=1` the 9.21 gap leaves the softmax 99.97% saturated.

Practically: leave `tau` at 1 and tune `l` and `lr` instead. If a design will not
move at all and you have already ruled out the usual causes, raising `tau` is a
more principled lever than raising `lr`, because it attacks the saturation
directly. Note that raising `tau` does **not** make the design noisier — the
sampling is unchanged.

## Reading the verbose logs

`verbose=True` (the default) prints one line before the loop, one every
`report_iter` iterations (default 100), and one at the end:

```
iter=I	input_loss=0.0	output_loss= 4.0	total_loss= 4.0	time=0.0
iter=100	input_loss= 2.0	output_loss= 0.0	total_loss= 0.2	time=0.08828
iter=F	input_loss= 2.0	output_loss= 0.0	total_loss= 0.2	time=0.09576
```

(captured from a toy oracle; on a real oracle the losses are larger and the run
takes hundreds of iterations — e.g. an `input_loss` of 35.75 at iteration 100
settling to 28.19 by convergence)

- `iter=I` is the **initial** state: the unedited template, so `input_loss` is 0
  and `output_loss` is the distance from your template's prediction to `y_bar`.
  If this number is already near zero your design task is vacuous.
- `iter=F` is the **best** iterate, not the last one. `input_loss` there is the
  mean edits per sequence in the design you are getting back.
- `time` is seconds since the previous report line, except on `iter=F` where it
  is the total elapsed time.

A healthy run drops `output_loss` fast while `input_loss` climbs, then slowly
sheds edits — `input_loss` falling late in the run is the input term doing its
job. Plot it with [`plot_loss`](plotting.md).

## What you get back is the best iterate

`fit_transform` tracks the lowest `total_loss` seen and returns a clone of the
sequences from that iteration, and it also resets `self.weights` to the weights
from that iteration. Consequences:

- The design is **not** the final state of the optimizer, so a run that diverges
  late still returns its best result.
- `best_sequence` is initialized to the *unedited* template. If no iteration ever
  improves on the initial total loss, you get your template back with zero edits.
  A `nan` output loss produces exactly this, because `nan < best_total_loss` is
  never true — that is the mechanism behind the silent failures in
  [multi-task-models.md](multi-task-models.md).

## Stopping

- `max_iter` (default 1000) is the hard cap.
- `early_stopping_iter` (default 100) stops after that many **consecutive**
  iterations with no improvement in total loss. The counter resets on every
  improvement, so this is not a patience budget for the whole run.
- `lr` (default 1.0, `AdamW` on the weight matrix) is unusually large by
  neural-network standards because it optimizes logits, not model weights. Lower
  it if the loss oscillates; raise `max_iter` rather than `lr` if it is merely slow.

## Related references

[io-and-validation.md](io-and-validation.md) for the exact shapes,
[custom-losses.md](custom-losses.md) for replacing the output loss,
[masks.md](masks.md) and [initial-weights.md](initial-weights.md) for constraining
*where* the input loss is allowed to be paid, [designer-object.md](designer-object.md)
for `return_history` and reusing a fit.
