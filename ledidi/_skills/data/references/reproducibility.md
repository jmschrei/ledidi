# Reproducibility and devices

## `device` defaults to `'cuda'`

Not "CUDA if available". On a machine without a CUDA GPU every call raises a
CUDA-related `RuntimeError` until you pass `device='cpu'` explicitly:

```python
X_bar = ledidi(model, X, y_bar, device='cpu')
```

Any device string or `torch.device` works, including `'cuda:1'`. The `ledidi()`
wrapper moves the model, template, and target for you; if you build `Ledidi` yourself
you own placement — `.to(device)` the designer and both tensors
([designer-object.md](designer-object.md)).

## `random_state` does not touch the global RNG

```python
X_bar = ledidi(model, X, y_bar, random_state=0, device='cuda')
```

When `random_state` is an integer, sampling is drawn from a private
`torch.Generator` created inside the designer, via a reimplementation of
`F.gumbel_softmax(hard=True)` that accepts a generator. The reference PyTorch
function draws from the global RNG, which cannot be seeded without perturbing every
other random draw in your script — so ledidi deliberately avoids it. Calling
`torch.manual_seed` is therefore *not* equivalent and is not needed.

Measured: after `torch.manual_seed(0)`, `torch.randn(3)` returns
`[1.5410, -0.2934, -2.1788]` whether or not a `random_state=0` design ran in
between — bit-identical. Run the same test with `random_state=None` and the draw
becomes `[0.9782, 0.3518, -0.4055]`: the default path *does* consume the global
RNG.

- `random_state=None` (the default) uses the global RNG exactly as before.
- The private generator **advances** across calls, so successive iterations and
  successive `forward` draws differ from one another while the run as a whole stays
  reproducible.
- The generator follows the module across devices: it is rebuilt if the designer moves,
  so `.to('cuda')` after construction is safe.
- With a catalog or `n_repeats`, each designer is seeded with `random_state + i * n_repeats + j`
  so the entries stay independent while the whole call is reproducible
  → [catalogs-and-repeats.md](catalogs-and-repeats.md).

## What `random_state` does *not* buy you

**On a GPU it does not guarantee bitwise-identical designs across runs.** Some
CUDA/cuDNN kernels are non-deterministic for speed, and because ledidi samples from
the model's own outputs, a difference at machine precision can change which character
is drawn and cascade into a different edit. The designs will be equally good, just not
identical. For full determinism:

```python
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'    # must precede `import torch`

import torch
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
```

`CUBLAS_WORKSPACE_CONFIG` must be set before torch is imported, or requesting
deterministic algorithms raises. This is noticeably slower; reserve it for debugging.

**Across machines it does not hold either.** A design is reproducible on one machine
but can drift by an edit or two across CPU architectures, because floating-point
addition is not associative and the optimizer's reduction order differs. Do not pin
exact edit positions or counts in a test; assert portable invariants instead — the
target was reached within a tolerance, the prediction moved in the right direction, the
edit count is in a sane range, and a repeated run on the same machine matches.

**Micro-batching changes the draws.** Splitting a batch for memory reasons changes how
many Gumbel samples are taken and in what shape, so an accumulated run is reproducible
but not bitwise equal to the unsplit one. Do not compare a memory-tuned run against a
stored gold design → [memory-and-oom.md](memory-and-oom.md).

## Reproducing a design later

Record everything that feeds the result: `random_state`, `l`, `target`, `batch_size`,
`max_iter`, `early_stopping_iter`, the exact `y_bar`, the template, any `input_mask` or
`initial_weights` (remembering that `initial_weights` is **modified in place** by the
run, so save a copy before, not after → [initial-weights.md](initial-weights.md)), and
the oracle checkpoint plus its wrapper. The wrapper is part of the recipe: the same
model wrapped two ways is two different oracles.

## Related references

[first-design.md](first-design.md) for the minimal seeded example,
[catalogs-and-repeats.md](catalogs-and-repeats.md) for seed offsetting,
[memory-and-oom.md](memory-and-oom.md) for the accumulation caveat,
[designer-object.md](designer-object.md) for manual device placement.
