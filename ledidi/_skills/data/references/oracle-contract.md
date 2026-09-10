# The oracle contract

ledidi backpropagates from the oracle's output all the way to a weight matrix
over the input sequence. Your model must therefore satisfy a short contract, and
almost every real-world design problem is a matter of *wrapping* a model until it
does.

## What ledidi requires

1. **A `torch.nn.Module`.** Anything else raises `TypeError`. A bare function or
   a lambda will not do — wrap it in a `Module`.
2. **`model(X)` takes one argument**: a one-hot tensor
   `(batch_size, n_channels, length)`. Multi-input models (control tracks,
   cell-state vectors) must have the extra inputs bound inside a wrapper, because
   ledidi never passes anything but `X`.
3. **A single tensor out, sliceable as `[:, target]`**, with the batch dimension
   first. A tuple/list/dict return must be reduced to one tensor in a wrapper.
4. **Differentiable end to end.** The gradient must reach the input. Any
   `.detach()`, `torch.no_grad()`, `.item()`, `numpy` round-trip, or
   non-differentiable op (argmax, thresholding, sampling) inside the forward pass
   silently severs it and the design will not move.
5. **Nothing else.** ledidi calls `model.eval()` and sets `requires_grad = False`
   on every parameter for you — you do not need to freeze the model yourself, and
   you should not rely on it being left in training mode afterward.

## The minimal wrapper

```python
import torch

class Wrapper(torch.nn.Module):
	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, X):
		return self.model(X)[:, 5:6]      # one output, shape (batch, 1)
```

A wrapper is the right place for *anything* that adapts the model: selecting
outputs, reordering them, summing over a length axis, normalizing predictions,
trimming or tiling the input, supplying a control track. Keep it out of the loss
— the loss should express the design goal, not repair the model.

For the general art of wrapping genomics models, tangermeme's skill owns this
topic (`references/model-wrapping.md`) and its wrappers vignette goes further
than anything here:
https://tangermeme.readthedocs.io/en/latest/vignettes/Wrappers_are_Productivity_Hacks.html

## Which situation are you in

The contract is easy to state and easy to violate. Three common situations each
have a dedicated file:

- **One multi-task model, you want a subset of its outputs** →
  `references/multi-task-models.md`. `target` selects exactly one
  output; anything else is a wrapper.
- **Several separate models balanced in one design** →
  `references/multiple-models.md`. `DesignWrapper` or hand-rolled.
- **The model's input window does not match the sequence you want to design** →
  `references/receptive-field.md`. Tiling, centering, and what the model
  cannot see.

Oracle size and count are also what drive GPU memory; if you hit a CUDA OOM, go
to `references/memory-and-oom.md` rather than guessing at `batch_size`.

## Sanity checks before designing

Run these once, before any design. They catch most contract violations in
seconds.

```python
X_test = X.clone().requires_grad_(True)
y_hat = model(X_test)
print(y_hat.shape, y_hat.requires_grad)

if y_hat.requires_grad:
	y_hat.sum().backward()
	print("gradient reaching X:", float(X_test.grad.abs().sum()))
```

Note the `requires_grad_(True)` on the *input*. Calling `model(X)` on a plain
template and checking `y_hat.requires_grad` tells you nothing — it reports whether
the model's own parameters need gradients, not whether the graph reaches your
sequence.

- **Shape** — count the outputs explicitly; `target` indexing and `y_bar` width
  both depend on it.
- **Gradient flow** — `requires_grad` must be `True` and the gradient reaching `X`
  non-zero. A `detach()` in the forward pass gives `False` here; a severed-but-
  graphed model gives zeros. Either way, no design is possible.
- **Baseline value** — is the prediction far from `y_bar` in the direction you
  intend? Designing toward a value the template already predicts is a no-op, and
  designing a property the model cannot represent will simply fail.
- **Does an edit change the output at all?** Flip a few positions and re-predict.
  A model that is invariant to your template's content (a length- or
  composition-only summary, a mis-wired wrapper) will report a plausible loss and
  design nothing.

## Related references

`references/objective.md` for what the sliced output feeds into,
`references/custom-losses.md` for objectives that a wrapper cannot express,
`references/validating-designs.md` for confirming the oracle was not
merely exploited.
