# Forbidding edits at specific positions

`input_mask` is the hard constraint on *where* edits may be made. Use it when
certain positions must not change — a gene body, a TATA box, an initiation site, a
binding site whose role the oracle does not model.

```python
input_mask = torch.zeros(2114, dtype=torch.bool)
input_mask[1057:] = True                   # protect everything after the TSS

X_bar = ledidi(model, X, y_bar, input_mask=input_mask, device='cuda')
```

## Semantics

- Shape `(length,)`, dtype **`torch.bool`** — a `ValueError` otherwise. One entry per
  position, no channel axis.

### Footgun: `True` means the position may NOT be edited

This polarity is the opposite of the "mask of positions I care about" convention
used elsewhere. Getting it backwards protects exactly the region you meant to edit,
and the design still runs and reports a plausible loss. Verify it
([below](#verifying-a-mask-did-what-you-meant)) rather than trusting it.
- Positions need not be contiguous. Any pattern of `True` works.
- It is a **hard** constraint, not a preference: masked positions receive `-inf`
  weights, so no amount of optimization pressure can overcome them. Verified in the
  tutorials by expanding a mask outward and observing that edits never appear inside
  it.

## What it does under the hood

At the start of `fit_transform`, ledidi writes `-inf` into the weight matrix at every
masked position, and then writes `0` back at the character the template already has:

```python
self.weights[:, :, self.input_mask] = float("-inf")
self.weights[X.type(torch.bool)] = 0
```

So a masked position can only ever draw its original character.

`input_mask` is a convenience layer over `initial_weights`. Anything finer —
forbidding *particular characters* rather than whole positions, or forcing a
character in — is done by building the weight matrix yourself →
[initial-weights.md](initial-weights.md).

### Footgun: passing `input_mask` erases priors at *every* position

Look carefully at the second line above. The `-inf` write is confined to the mask,
but `self.weights[X.type(torch.bool)] = 0` is **not masked** — it resets the
template character's channel to `0` across the entire sequence.

So if you pass `input_mask` and `initial_weights` together, any prior you placed on
a template character is silently discarded, everywhere, not just inside the mask.
Measured with a uniform prior of `2.0` and positions 0–9 masked on a 60 bp
template: **60 of 60** template-channel entries came back `0.0`, and position 30 —
well outside the mask — reads `[2.0, 0.0, 2.0, 2.0]`, with the `0.0` sitting on the
template's own base.

Two practical consequences:

- Priors on **non-template** characters survive outside the mask; priors on the
  template character do not survive anywhere. If your prior was "keep preferring
  what is already here", it is gone.
- The forced-edit idiom from [initial-weights.md](initial-weights.md) — `-inf` on
  every character *except* the one you want, including the template's own — is
  **broken by any `input_mask`**, because the `-inf` you placed on the template
  character is reset to `0` and that character becomes drawable again. Use one
  mechanism or the other in a given design, not both.

## Effects worth anticipating

- Edits concentrate in whatever region remains. The design is still chasing the same
  `y_bar` with less sequence to work with, so a tight mask means more edits packed
  into fewer positions, and an aggressive enough mask makes the target unreachable —
  the output loss simply plateaus.
- Edits tend not to appear at the extreme flanks of the input window even when they
  are permitted, because many oracles are less sensitive there. Do not read that as
  the mask working; check the mask directly.
- Masking is a statement about the *sequence*, not the *oracle*. To constrain
  predictions rather than positions — "keep transcription where it is" — hold outputs
  at baseline instead ([multi-task-models.md](multi-task-models.md)), which lets
  ledidi rearrange sequence freely as long as the readouts do not move.

## Verifying a mask did what you meant

```python
X_bar = X_bar.detach().cpu()                   # designs come back on the design device
edited = (X[0] != X_bar[0]).any(dim=0)
assert not bool(edited[input_mask].any())      # no edits inside the protected span
print(int(edited.sum()), "edits, all outside the mask")
```

The `.cpu()` matters: with the default `device='cuda'`, `ledidi()` returns designs on
the GPU while your `X` stays where you left it, and comparing them raises
`RuntimeError: Expected all tensors to be on the same device`.

Cheap, and it catches an inverted mask immediately.

## Related references

[initial-weights.md](initial-weights.md) for per-character constraints, forcing
edits, and soft priors; [inpainting.md](inpainting.md) for the opposite move
(marking a span as free to fill); [io-and-validation.md](io-and-validation.md) for
the dtype and shape errors; [receptive-field.md](receptive-field.md) for masking off
regions no oracle actually scores.
