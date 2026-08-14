# A full design, end to end

Start here when the task is "I have a model and a region, design something" and you
need the whole arc: wrap the oracle, confirm the task is feasible, design, prune,
validate, plot. Each step links to the file that owns its details.

The example uses a small GATA2 BPNet oracle (~0.5 MB from Zenodo, needs
`bpnetlite`) so that the shapes and magnitudes are realistic. For a
nothing-to-download version, see [first-design.md](first-design.md).

## 0. Wrap the oracle and check the contract

```python
import torch
import urllib.request
from bpnetlite.bpnet import ControlWrapper
from bpnetlite.bpnet import CountWrapper

urllib.request.urlretrieve(
	"https://zenodo.org/api/records/14604495/files/GATA2.torch/content", "GATA2.torch")
model = torch.load("GATA2.torch", weights_only=False, map_location="cpu")
model = CountWrapper(ControlWrapper(model))     # sequence in, one count out
```

`ControlWrapper` supplies BPNet's all-zero control track and `CountWrapper` selects
the count head — together they turn a multi-input, multi-output BPNet into the
single-tensor oracle ledidi requires. Whatever your model is, get it to that shape
first and verify it → [oracle-contract.md](oracle-contract.md).

Which situation you are in decides the wrapper:
[multi-task-models.md](multi-task-models.md) (a subset of one model's outputs),
[multiple-models.md](multiple-models.md) (several models),
[receptive-field.md](receptive-field.md) (window ≠ design field).

## 1. Load the template and confirm the design is feasible

```python
import pyfaidx
from tangermeme.utils import one_hot_encode
from tangermeme.predict import predict

chrom, mid = "chr1", 246_507_312                  # the SMYD3 promoter
seq = pyfaidx.Fasta("hg38.fa")[chrom][mid-1057:mid+1057].seq.upper()
X = one_hot_encode(seq).unsqueeze(0).float()      # (1, 4, 2114)

print(predict(model, X))     # 0.4569 -> GATA2 is barely predicted to bind here
```

Cast to `float()`: `one_hot_encode` returns `int8` and ledidi needs `float32`
→ [io-and-validation.md](io-and-validation.md). (`tangermeme.io.extract_loci` is the
other route when you have a BED file of loci.)

**Do not skip the baseline prediction.** A template that already predicts your
target makes the design vacuous, and a baseline in the wrong direction usually means
the wrapper is wrong. BPNet counts are log fold enrichment over control, so ~0 means
no binding.

## 2. Choose `y_bar` relative to that baseline

```python
y_bar = model(X).detach() + 4.0      # four log-counts above the current prediction
```

Setting the target relative to the observed baseline keeps it inside the model's
dynamic range. An absolute number picked from intuition is the most common way to
get a design that either does nothing or wildly overshoots into territory the model
never saw in training.

## 3. Design

```python
from ledidi import ledidi

X = X.cuda()          # ledidi() moves the model in place, but NOT your template
X_bar, history = ledidi(model, X, y_bar.cuda(), device='cuda', batch_size=50,
	random_state=0, return_history=True)
```

Move `X` yourself. `ledidi()` returns designs on the device while leaving your
template where it was, so every later step that combines the two — pruning, the edit
diff, `designer(X)` — otherwise raises `RuntimeError: Expected all tensors to be on
the same device`.

Watch the log: `output_loss` should fall quickly while `input_loss` (mean edits per
sequence) rises, then shed edits late → [objective.md](objective.md). If the design
does not move, `l` is the first knob ([objective.md](objective.md)); if it hits the
target with too many edits, raise `l`.

To constrain *where* or *what* may be edited, add
[`input_mask`](masks.md) or [`initial_weights`](initial-weights.md) here; to let
ledidi fill a blanked span, see [inpainting.md](inpainting.md). For a range of
target strengths in one call, see [catalogs-and-repeats.md](catalogs-and-repeats.md).

## 4. Prune

```python
from ledidi.pruning import greedy_pruning

X_bar_p = torch.cat([greedy_pruning(model, X, X_bar[i:i+1], threshold=1)
	for i in range(len(X_bar))])
```

`greedy_pruning` handles one sequence at a time, hence the loop. It reverts edits
whose removal barely changes the prediction, giving the smallest set that still
works → [pruning.md](pruning.md).

## 5. Validate — the step that is easiest to skip and hardest to do without

A design that satisfies the oracle may still be an artifact of *exploiting* the
oracle. At minimum: attributions should show the motif you expect, motif scanning
should show an increase specifically in the intended motif, and an **independently
trained** model should move in the intended direction.

```python
from tangermeme.deep_lift_shap import deep_lift_shap

X_attr = deep_lift_shap(model, X)                    # (1, 4, 2114)
X_bar_attr = deep_lift_shap(model, X_bar[:1].detach())
X_bar_p_attr = deep_lift_shap(model, X_bar_p[:1])
```

`.detach()` the designs first — they come back carrying an autograd graph
([io-and-validation.md](io-and-validation.md)).

Full protocol, including motif hits and round-tripping through other models →
[validating-designs.md](validating-designs.md). If you used a cropped or tiled
oracle, also check the regions it could not see
([receptive-field.md](receptive-field.md)).

## 6. Plot

```python
from ledidi.plot import plot_loss
from ledidi.plot import plot_edits

plot_loss(history)
plot_edits(X_attr, [X_bar_attr, X_bar_p_attr], colors=['0.5', 'darkorange', 'magenta'])
```

`plot_edits` takes **attributions**, not sequences, and prepends the original as the
first track — so `colors` and `axs` need one entry more than the number of designs →
[plotting.md](plotting.md).

## Easy to skip, and worth checking

- `target` is non-negative and correct, or `None` with a wrapper — a bad one returns
  your template unedited ([multi-task-models.md](multi-task-models.md)).
- The edit count is plausible for the effect size you asked for.
- At least one **independently trained** model agrees
  ([validating-designs.md](validating-designs.md)).
- On a CUDA OOM: [memory-and-oom.md](memory-and-oom.md).
