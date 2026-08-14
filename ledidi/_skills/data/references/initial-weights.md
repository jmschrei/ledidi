# Per-character constraints and priors

`initial_weights` is the general mechanism behind every constraint in ledidi.
`input_mask` ([masks.md](masks.md)) is a thin convenience layer over it. Reach for
`initial_weights` directly when you need control over *which characters* may appear,
not just which positions may change.

Shape `(1, n_channels, length)`, float. These are logits added to
`log(X + eps)` before the Gumbel-softmax, so:

- `-inf` → that character can never be drawn at that position (**hard** constraint).
- positive → more likely to be drawn (**soft** prior).
- negative but finite → less likely (**soft** prior).
- `0` everywhere → the default, unconstrained design.

## Footgun: your tensor is modified in place

`Ledidi.__init__` sets `requires_grad = True` on the tensor you pass and wraps it in
a `torch.nn.Parameter` that **shares its storage**, so the optimizer writes the
learned values into your tensor.

This happens whenever the tensor already lives on the design device. Measured with
`torch.zeros(1, 4, 60)` over a 50-step run:

| `device=` | your tensor | outcome |
|---|---|---|
| `'cpu'` | CPU | **mutated** — max abs 9.40 |
| `'cuda'` | CUDA | **mutated** — max abs 9.33 |
| `'cuda'` | CPU | untouched — max abs 0.0 |

The exception is not a safety net: `.to(device)` replaces the parameter's storage,
so a CPU tensor handed to a CUDA run simply is not the tensor being optimized. Since
you generally do not want to depend on which of these you are in, a weight matrix
reused across runs can silently carry the previous run's learned logits. Always pass
a fresh copy:

```python
X_bar_1 = ledidi(model, X, y_bar, initial_weights=w.clone(), device='cuda')
X_bar_2 = ledidi(model, X, y_bar, initial_weights=w.clone(), device='cuda')
```

## Blocking a character everywhere

Base and prime editors cannot make every substitution, and some designs must avoid
introducing particular characters. Set that channel to `-inf`:

```python
initial_weights = torch.zeros(1, 4, 2114)
initial_weights[:, 1, :] = float("-inf")        # never draw a C (channel order ACGT)
```

Careful: this forbids C *everywhere*, so existing Cs in the template get edited away
too — the design will convert them to something else while pursuing your target.
Usually what you want is "no *new* Cs", which means exempting positions that are
already C:

```python
initial_weights = torch.zeros(1, 4, 2114)
initial_weights[:, 1, :] = float("-inf")
initial_weights[:, 1, :][X[:, 1, :].bool()] = 0.0    # existing Cs may stay
```

The same pattern blocks a single substitution at a single position — for example
preventing the `G` of a `GATAA` from becoming a `T`, which would turn a GATA site
into a TATA box. In high-stakes designs this matters: knocking out one factor's motif
can accidentally knock *in* another's, and only a per-character block prevents it.

## Forcing a character in

Set every channel except the one you want to `-inf`, and that position is pinned to
the character you chose:

```python
initial_weights = torch.zeros(1, 4, 2114)
motif = "GATAA"
for i, char in enumerate(motif):
	initial_weights[0, :, 1000+i] = float("-inf")
	initial_weights[0, "ACGT".index(char), 1000+i] = 0.0
```

This is genuinely different from editing the template before or after a run:

- Edit **before** and ledidi may remove it again — it is just sequence, and the input
  loss counts nothing for it.
- Edit **after** and the rest of the design was optimized without it, so the
  combination may overshoot, undershoot, or be internally inconsistent.
- Force it **during** and every other edit is chosen in the presence of the motif,
  while the motif itself cannot be undone.

Note the input loss still counts these forced positions as edits if they differ from
the template, so they consume part of your edit budget.

## Soft priors

Finite values nudge without forcing. This is the tool for "prefer to work in this
region", "prefer this kind of change", or seeding a guess about where a motif should
go.

```python
initial_weights = torch.zeros(1, 4, 2114)
initial_weights[0, 1, 1000:1010] = 5.0     # a soft preference for Cs in this window
```

**Priors can be washed out and do not come back.** They are the *initial* value of a
tensor the optimizer is free to move; if each step decreases a prior it will
eventually vanish, and the run then behaves as though it was never set. A prior is
therefore a starting hypothesis, not a constraint — if the design must contain
something, use `-inf` on the alternatives instead, and check afterward whether the
prior survived:

```python
print(designer.weights[0, 1, 1000:1010])   # did the prior hold, or get optimized away?
```

(`designer` here is the fitted `Ledidi` object from `return_designer=True` →
[designer-object.md](designer-object.md). Note these are the **best-iterate** weights,
not the initial ones, which is exactly what makes the comparison meaningful.)

## Interaction with `input_mask`

**Passing `input_mask` at all erases every prior you placed on a template
character, at every position — not just inside the mask.** It also breaks the
forced-edit idiom above. Priors on non-template characters outside the mask do
survive. This is measured and explained in [masks.md](masks.md); the short version
is: use `input_mask` or `initial_weights` in a given design, not both.

## Related references

[masks.md](masks.md) for whole-position constraints, [inpainting.md](inpainting.md)
for freeing a span instead of constraining it, [io-and-validation.md](io-and-validation.md)
for the shape requirement, [objective.md](objective.md) for how `eps` sets the scale
these logits compete against.
