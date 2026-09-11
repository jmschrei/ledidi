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
and the design still runs and reports a plausible loss. Verify it — see
"Verifying a mask did what you meant" below — rather than trusting it.
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
`references/initial-weights.md`.

The template-character restore is scoped to the mask, so `initial_weights` you set
elsewhere survive untouched — including a prior sitting on a template character.
Inside the mask, whatever you set is replaced by `-inf`/`0`, because the mask is a
hard constraint and takes precedence.

Older versions of ledidi applied that restore to the *whole* sequence, which
silently discarded template-character priors everywhere and broke the forced-edit
idiom in `references/initial-weights.md` whenever a mask was also passed.
If you are reading code written against an older release, that is why it may have
combined the two mechanisms carefully or not at all.

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
  at baseline instead (`references/multi-task-models.md`), which lets
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

`references/initial-weights.md` for per-character constraints, forcing
edits, and soft priors; `references/inpainting.md` for the opposite move
(marking a span as free to fill); `references/io-and-validation.md` for
the dtype and shape errors; `references/receptive-field.md` for masking off
regions no oracle actually scores.
