# Plotting a design

Three helpers in `ledidi.plot`. They are thin matplotlib wrappers; attribution
computation and general sequence plotting belong to tangermeme. In particular
`plot_edits` **renders each track by calling `tangermeme.plot.plot_logo`**, so every
logo-level question — color forms, annotation overlays, `min_height_pct`, PWM vs
attribution input — is answered by that skill's `references/plot.md`, not here. This
file covers only what is ledidi-specific.

All three need `return_history=True` or attributions you computed yourself — none of
them take raw designed sequences.

## plot_loss — the two loss curves

```python
from ledidi.plot import plot_loss

X_bar, history = ledidi(model, X, y_bar, return_history=True, device='cuda')
ax, ax_input = plot_loss(history)
```

- Draws the **output loss** on the given axis (left, grey) and the **input loss** on a
  twin axis (right, orange), because they are in different units and on different
  scales.
- Returns the `(output_axis, input_axis)` pair; the second is the twin it created. If
  you want to restyle the input-loss axis, use the second return value — reaching for
  `plt.gca()` afterward gets you the twin, not the original.
- `ax=None` uses `plt.gca()`, so it composes into a subplot grid.

Read it as: output loss should drop fast while input loss climbs, then input loss
should fall as unnecessary edits are shed. An input loss that never falls means `l` is
too small to matter; an output loss that never drops means the objective is not
reachable → `references/objective.md`.

## plot_history — where and when edits were proposed

```python
from ledidi.plot import plot_history

plot_history(history)
```

Scatters every proposed edit with position on the x-axis and iteration on the y-axis
(inverted, so the run reads top to bottom). This is the plot that shows edits being
proposed en masse early and progressively abandoned — the input loss doing its work.

- Uses `history['batch_size']` to convert row indices back to iteration numbers, so
  pass the history dict unmodified.
- Takes no `ax` argument; it draws on `plt.gca()`.
- Dense runs produce a lot of points (`batch_size` × edits × iterations). Subsetting
  the history before plotting is reasonable for very long runs.

## plot_edits — edits on attribution tracks

```python
from ledidi.plot import plot_edits

axs = plot_edits(X_attr, [X_bar_attr, X_bar_p_attr],
	colors=['0.5', 'darkorange', 'magenta'], figsize=(10, 6))
```

The most useful and the most error-prone of the three.

- **It takes attributions, not sequences.** `X_orig` is `(1, 4, length)` attributions
  for the template and `X_attrs` is `(n, 4, length)` for the designs (a list is
  concatenated for you). Attributions must be non-zero only at the observed character —
  which is what `deep_lift_shap` gives you. Passing one-hot sequences produces flat,
  meaningless logos rather than an error.
- **The original is prepended as track 0.** So there are `n + 1` tracks, and both
  `colors` (when a list) and `axs` must have `n + 1` entries. The first `colors` entry
  is unused, since track 0 has no edits to highlight — supply a placeholder anyway.
- Characters that differ from the template are drawn in the track's color; unchanged
  ones in grey. So each track shows *its own* edits relative to the original.
- All tracks share one y-range so magnitudes are comparable, and only the bottom track
  keeps position ticks.
- `axs=None` creates its own figure with one track per sequence and forwards `**kwargs`
  to `plt.figure` (`figsize=`). Pass `axs` to place tracks in an existing layout or to
  zoom each track into a window: slice the attributions themselves, e.g.
  `X_attr[:, :, 900:1100]`.
- Returns the list of axes.

A quick way to get the third track right: designs, then pruned designs, in the same
order as your colors — mismatched lengths raise an `IndexError` from the color lookup,
but mismatched *order* silently mislabels which track is which.

## Conventions

`plot_loss` hides the top spines and colors each y-axis to match its curve, and it
adds its own two-entry legend. `plot_edits` hides the left spine and turns off the
grid on every track. `plot_history` does neither — despine it yourself if you are
placing it next to the others.

## Related references

`references/designer-object.md` for what is in the history dict,
`references/validating-designs.md` for computing the attributions
`plot_edits` needs, `references/objective.md` for interpreting the loss curves.
