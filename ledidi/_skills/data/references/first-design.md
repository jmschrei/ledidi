# A first design

The smallest complete design, with a toy oracle so there is nothing to download and
it runs on a CPU in seconds. Use this to learn the mechanics and to sanity-check an
installation; for a real oracle end to end, see [pipeline.md](pipeline.md).

## The three ingredients

Every design needs exactly three things: an oracle, a template, and a desired
output.

```python
import torch
from ledidi import ledidi

# 1. An oracle. This one slides the AP-1 motif TGACTCA across the sequence and
#    returns the best match, so a perfect match anywhere scores 7.
motif = "TGACTCA"
weights = torch.zeros(1, 4, len(motif))
for i, char in enumerate(motif):
	weights[0, "ACGT".index(char), i] = 1.0

class MotifScore(torch.nn.Module):
	def forward(self, X):
		return torch.nn.functional.conv1d(X, weights).amax(dim=-1)

# 2. A template: one-hot, float32, shape (1, 4, length), channels A, C, G, T.
torch.manual_seed(0)
idxs = torch.randint(0, 4, (1, 50))
X = torch.zeros(1, 4, 50).scatter_(1, idxs.unsqueeze(1), 1.0)

# 3. A desired output, shape (1, n_outputs).
y_bar = torch.tensor([[7.0]])

X_bar = ledidi(MotifScore(), X, y_bar, device='cpu', random_state=0, verbose=False)
```

`X_bar` has shape `(batch_size, 4, 50)` with `batch_size=16` by default.

Three arguments in that call are worth noticing:

- **`device='cpu'` is mandatory here.** The default is `'cuda'` — not "CUDA if
  available" — so on a CPU-only machine every call raises without it.
- **`random_state=0`** makes sampling reproducible without touching the global torch
  RNG → [reproducibility.md](reproducibility.md).
- **`verbose=False`** silences the per-iteration log. Leave it on while learning; the
  log is the fastest way to see whether the design is working
  ([objective.md](objective.md) explains the lines).

## Did it work

```python
designed = "".join("ACGT"[c] for c in X_bar[0].argmax(dim=0))
print(motif in designed)     # True
```

## What changed

Diff the template against a design to list the edits. This idiom is worth
remembering; you will use it constantly.

```python
seq = X_bar[0]                                        # one design, shape (4, 50)
positions = torch.where((X[0] != seq).any(dim=0))[0]
for p in positions:
	before = "ACGT"[X[0, :, p].argmax()]
	after = "ACGT"[seq[:, p].argmax()]
	print("position {}: {} -> {}".format(p.item(), before, after))
```

For this example that prints just a couple of edits — the ones that complete
`TGACTCA` at whichever position in the random sequence was already closest, rather
than overwriting a whole stretch of it.

The `batch_size` designs are all sampled from the *same* learned weight matrix, so
they are variations on one design rather than independent solutions →
[designer-object.md](designer-object.md).

## If a design returns zero edits

In order of likelihood:

1. **The template already predicts `y_bar`.** Check `model(X)` first; the `iter=I`
   log line shows this as an output loss near zero.
2. **`l` is too high** for the scale of your output loss →
   [objective.md](objective.md).
3. **The gradient does not reach the input**, usually a `detach`/`no_grad`/argmax in
   the model, or an oracle that is invariant to its input
   → [oracle-contract.md](oracle-contract.md).
4. **The oracle cannot represent what you asked for**, so the output loss plateaus
   well above zero.

## Related references

[pipeline.md](pipeline.md) for the same workflow with a real oracle plus pruning and
validation, [io-and-validation.md](io-and-validation.md) for shapes and error
messages, [objective.md](objective.md) for `l` and the logs.
