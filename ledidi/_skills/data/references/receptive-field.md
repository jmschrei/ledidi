# When the model's window does not match the design field

ledidi edits a sequence of fixed length — whatever you pass as `X`. Oracles have
their own input widths, and those two rarely agree. Any model with a dense layer
(Beluga, Basset, Malinois) accepts exactly one length; convolutional models (BPNet,
ChromBPNet) are flexible but were trained at one scale and behave oddly far from
it. Every fix is a wrapper.

## Designing a field longer than the model's window: tiling

A 2 kbp oracle can design a 20 kbp field if you apply it to every chunk and treat
the results as one output tensor.

```python
class Tiled(torch.nn.Module):
	"""Apply a fixed-width model across a longer sequence in chunks."""

	def __init__(self, model, width, stride=None):
		super().__init__()
		self.model = model
		self.width = width
		self.stride = stride or width          # stride < width => overlapping

	def forward(self, X):
		starts = range(0, X.shape[-1] - self.width + 1, self.stride)
		return torch.cat([self.model(X[:, :, s:s+self.width]) for s in starts],
			dim=-1)                            # (batch, n_chunks * outputs_per_chunk)
```

- **Contiguous chunks** (`stride == width`) tile the field exactly once. Cheapest,
  but a motif straddling a chunk boundary is split, and every chunk edge is an
  artificial sequence end where the model sees padding-like context it was never
  trained on. Designs preferentially avoid or exploit those seams.
- **Overlapping chunks** (`stride < width`, e.g. half the width) mean every
  position is scored in at least two contexts, which removes the boundary blind
  spot at proportional extra cost. Prefer this when the design target could land
  anywhere, or when you saw suspicious activity at a seam.
- Cost scales with the number of chunks — `n_chunks` forward *and* backward passes
  per iteration, and activation memory to match → `references/memory-and-oom.md`.

### Ask the user how to aggregate the chunks

The wrapper above returns one output per chunk. Whether that is what you want is a
design decision, not a technical one, and the two options produce genuinely
different experiments. **Present both to the user and let them choose before you
write `y_bar`:**

- **One output per chunk** — `y_bar` has an entry per chunk. This gives positional
  control: drive signal into chunk 7 while holding chunks 1–6 at their baseline
  predictions (see the hold-at-baseline idiom in
  `references/multi-task-models.md`), and afterward you can say *where*
  the signal landed. More bookkeeping.
- **Aggregated to a scalar** — `.sum(dim=-1, keepdim=True)` or `.mean(...)` inside
  the wrapper, so `y_bar` stays a single value and there is nothing to track. The
  design is then free to put the signal anywhere in the field, which may or may not
  be acceptable.

Do not ask when the request already settles it — "put the peak in the middle 2 kb"
implies per-chunk, "raise total accessibility across the locus" implies aggregate.
State the inference and move on. Ask only when the request is silent on whether
position matters.

## Models with different windows: center them

When several oracles have different fixed widths, make the design field as long as
the widest model needs and give each narrower model a centered crop:

```python
class Centered(torch.nn.Module):
	"""Crop a design field down to a model's fixed input width, centered."""

	def __init__(self, model, width):
		super().__init__()
		self.model = model
		self.width = width

	def forward(self, X):
		start = (X.shape[-1] - self.width) // 2
		return self.model(X[:, :, start:start+self.width])

oracle = DesignWrapper([bpnet_2114, Centered(beluga, 2000)])
```

Consequences worth stating explicitly, because they are easy to forget once it runs:

- The narrower model's predictions describe **only its crop**, not the whole design
  field. Trimming 2,114 → 2,000 bp is immaterial; cropping 10 kbp → 2 kbp means
  that model's target value refers to a tenth of what you are designing.
- Edits are still **proposed** everywhere in the field. Nothing stops ledidi from
  editing the flanks a cropped model cannot see — those edits are simply not scored
  by it. Restricting *where* edits may happen is a separate mechanism
  (`references/masks.md`); centering a model does not constrain the design.
- So the flanks are optimized against fewer constraints than the center, and are
  where oracle exploitation is most likely to hide.

## Advanced: check the edges afterward

Because the narrow models never saw the flanks during design, re-apply them there
once the design is finished — slide the cropped model to the left and right ends of
the field and confirm nothing strange was built in the unmonitored regions.

```python
w, L = 2000, X_bar.shape[-1]
for name, s in [("left", 0), ("center", (L - w) // 2), ("right", L - w)]:
	print(name, beluga(X_bar[:1, :, s:s+w]).item())
```

A flank value that is wildly different from the center — a spurious peak, or a
prediction far outside the range the model gives natural sequence — means the design
put something there that no term of your objective was watching. This is a
validation step, not a design step; fold it into the checks in
`references/validating-designs.md`.

## Related references

`references/oracle-contract.md` for the wrapper contract,
`references/multiple-models.md` for combining the wrapped models,
`references/masks.md` to actually forbid edits in regions no model scores,
`references/memory-and-oom.md` because tiling multiplies memory by the
number of chunks.
