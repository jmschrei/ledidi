# In-painting: letting ledidi fill a blank span

Zeroing a span of the template tells ledidi that span is unknown and free to fill.
This is the mechanism behind inserting a motif without saying where it goes,
removing a site, and moving a site — and it is a single line of tensor code, not an
argument.

```python
X_ip = X.clone()
X_ip[:, :, 1000:1050] = 0.0      # blank 50 bp; ledidi may write anything here

X_bar = ledidi(model, X_ip, y_bar, device='cuda')
```

## Why zeroing works

Two independent facts combine:

1. **All-zero columns are valid input.** Validation runs with `allow_N=True`, so a
   column of zeros passes as unknown/`N`. In `forward`, `log(X + eps)` makes all
   `n_channels` equally likely at such a position, so the first draw there is uniform
   rather than anchored to a template character.
2. **Blanked positions are free.** `fit_transform` computes

   ```python
   inpainting_mask = X[0].sum(dim=0) == 1
   input_loss = self.input_loss(X_hat[:, :, inpainting_mask],
       X_[:, :, inpainting_mask]) / (X_hat.shape[0] * 2)
   ```

   so only positions that were one-hot to begin with contribute to the edit count.
   Whatever ledidi writes into a zeroed span costs **nothing** in the input loss.

That second point is what makes in-painting qualitatively different from just
designing: inside the blank, ledidi has no incentive to be conservative, so it builds
whatever the oracle wants at full strength. Outside it, the usual edit penalty still
applies.

Measured on a 60 bp template with positions 20–40 blanked and a motif oracle asked
for a perfect match: all 20 blanked positions were written, **zero** edits were made
outside the blank, the reported `input_loss` was **0.00**, and the oracle reached its
maximum score. Twenty characters were designed and the edit counter never moved.

## Footgun: `N` in a real template is not preserved

If your template came from a genome with `N` runs, or from a one-hot encoder that
zeroes unknown characters, those positions are already in-painting targets. ledidi
will fill them and will not tell you it did. Check before designing:

```python
n_cols = int((X[0].sum(dim=0) == 0).sum())
if n_cols:
	print("warning: {} unknown columns will be filled freely".format(n_cols))
```

If you want them left alone, mask them ([masks.md](masks.md)): a masked all-zero
column comes back all-zero, still `N`, with no `nan` anywhere in the design — so
masking genuinely preserves `N` runs. Cropping the region away is the other clean
option.

## The three canonical uses

**Add a motif to a background without choosing a position.** Blank a window wider
than the motif and set `y_bar` high. ledidi decides where inside the window to build
the site, and because the span is free it will build a strong one.

**Remove a binding site.** Blank the span containing the site and set `y_bar` to a
low value. This is often more reliable than asking for a knockout in place: with the
site blanked, the design cannot simply weaken it by a base or two — it has to fill
the span with something the oracle reads as inactive.

**Move a motif.** Blank both the current location and the destination, and keep
`y_bar` at roughly the template's original prediction. The design is then free to
delete the site from one place and rebuild it in the other while holding the oracle's
readout constant.

## Interaction with the edit budget

Because blanked positions are free, `l` no longer governs the design inside the
blank. If you blank 500 of 2114 positions you have effectively removed the edit
penalty from a quarter of the sequence, and the design will look much less minimal
than usual. Blank the smallest span that gives the design room, and read `input_loss`
in the logs as "edits *outside* the blank".

Comparing an in-painted design's edit count against a non-in-painted one is
meaningless for the same reason — they are counting different position sets.

## Related references

[masks.md](masks.md) and [initial-weights.md](initial-weights.md) for the opposite
operation (constraining rather than freeing), [io-and-validation.md](io-and-validation.md)
for why all-zero columns validate, [objective.md](objective.md) for what the input
loss counts, [validating-designs.md](validating-designs.md) — in-painted spans deserve
extra scrutiny precisely because nothing penalized what went into them.
