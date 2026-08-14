# Validating a design

Gradient-based design optimizes an oracle, and an oracle can be *exploited*: ledidi
will happily find sequences that score beautifully and mean nothing biologically. A
design is not finished when the loss is low. It is finished when independent evidence
agrees.

Do not report a design as successful on the strength of the oracle's own prediction.

Most of the machinery here belongs to tangermeme, whose skill owns these tools in
detail — see its `references/deep_lift_shap.md`, `references/annotate.md`, and
`references/comparing-models.md`. What follows is the protocol and the ledidi-specific
parts.

## 1. Attributions: did the edits build what you expect?

```python
from tangermeme.deep_lift_shap import deep_lift_shap
from ledidi.plot import plot_edits

X_attr = deep_lift_shap(model, X)
X_bar_attr = deep_lift_shap(model, X_bar[:1].detach())
X_bar_p_attr = deep_lift_shap(model, X_bar_p[:1])

plot_edits(X_attr, [X_bar_attr, X_bar_p_attr],
	colors=['0.5', 'darkorange', 'magenta'])
```

`.detach()` the designs first — they carry an autograd graph
([io-and-validation.md](io-and-validation.md)). `plot_edits` prepends the original as
track 0, so `colors` needs one more entry than you have designs
([plotting.md](plotting.md)).

What to look for:

- The template should show **no** high-attribution characters where you designed, or
  the task was not what you thought.
- The edits should assemble a recognizable motif for the factor you targeted —
  including on the reverse strand, so a GATA design may read as `TTATC`.
- High-attribution edits should be the ones pruning **kept**; low-attribution ones the
  ones it discarded. The correspondence is not perfect but the trend should be clear,
  and its absence suggests the pruning threshold or the `target` was wrong
  ([pruning.md](pruning.md)).
- Beware designs that build a *weak* version of the site rather than a strong one, or
  several weak sites. That is often correct — ledidi is matching a precise target and
  synergistic sites can reach it more cheaply than one strong site — but it should be
  explained rather than assumed.

## 2. Motif hits: did the right sites appear, and only those?

Scan before and after with FIMO and compare counts. Count decoys too: an increase
confined to your intended motif is evidence; a broad increase across unrelated motifs
suggests the model was pushed somewhere strange.

```python
import numpy
from tangermeme.io import read_meme
from memelite import fimo

motifs = read_meme("JASPAR2024_CORE_non-redundant_pfms_meme.txt")

# JASPAR is large and redundant: subset to your target plus a few decoys
names = numpy.array(list(motifs.keys()))
subset = [[n for n in names if k in n][0] for k in ('GATA', 'CTCF', 'MAX', 'KLF')]
motifs = {n: motifs[n] for n in subset}

hits_before = fimo(motifs, X.cpu(), return_counts=True, threshold=0.001)
hits_after = fimo(motifs, X_bar_p.cpu(), return_counts=True, threshold=0.001) / len(X_bar_p)
```

`return_counts=True` gives one count per motif instead of a DataFrame of hits, which
is what you want here; divide the design counts by the number of designs to compare
per-sequence rates. Measured on the GATA2 design above:

| motif | before | after (pruned) |
|---|---|---|
| GATA1 | 1.0 | **3.0** |
| CTCF | 9.0 | 8.0 |
| MAX | 8.0 | 8.0 |
| KLF5 | 18.0 | 17.0 |

The intended motif tripled while the three decoys held or slipped by one — that is the
shape of a result worth trusting.

Expect a decrease in some unrelated motifs — a design that is not asked to preserve
other activity will convert other sites into the one you asked for. That is worth
noticing and reporting, not hiding: it may be unacceptable for your application, in
which case hold those outputs at baseline
([multi-task-models.md](multi-task-models.md)) or protect them
([masks.md](masks.md)) and design again.

## 3. Independent models: does the effect generalize?

The decisive check. Score the designs with a model **not** used during design —
ideally trained by different people, on different data, with a different architecture.

```python
from tangermeme.predict import predict

y_enf = predict(enformer_gata2, torch.cat([X, X_bar[:1].detach(), X_bar_p[:1]]))
```

- A matched output (another GATA2 model for a GATA2 design) should move in the intended
  direction. Absolute values will not agree across models and need not.
- An indirect readout is also informative: factor binding usually opens chromatin, so
  an accessibility model should shift a little even though nothing asked it to.
- Pruned designs typically score slightly *lower* than unpruned ones on external
  models — evidence that pruning removed edits that were doing a little real work, and
  a reason to report both.
- For a design balanced across several oracles, the independent model must be outside
  that set; re-scoring with the same ensemble proves nothing, since an ensemble can be
  exploited jointly ([multiple-models.md](multiple-models.md)).

Pass `device=` per call so only one model occupies the GPU at a time — tangermeme's
`references/comparing-models.md` covers this pattern.

## 4. ledidi-specific checks

- **Regions no oracle could see.** If you cropped or tiled a model, re-apply it at the
  flanks and confirm nothing odd was built where nothing was watching
  → [receptive-field.md](receptive-field.md).
- **In-painted spans.** Blanked regions carry no edit penalty, so the design had free
  rein there. Scrutinize them first → [inpainting.md](inpainting.md).
- **Affinity catalogs validate themselves, partly.** Predicting across catalog steps
  should trace the requested curve, and motifs that persist across steps are more
  trustworthy than ones appearing at a single target
  → [catalogs-and-repeats.md](catalogs-and-repeats.md). Note this uses the design
  oracle, so it is a consistency check, not independent evidence.
- **Repeats agreeing is weak evidence.** Independent runs converging on the same motif
  is reassuring about the *optimization*; it says nothing about the biology, since every
  run shares the same oracle.

## The reporting standard

State which oracle designed the sequence, which independent model confirmed it, what
the attributions show, and what the motif counts did — including the decreases. If a
check was skipped, say so. A design reported without at least one independently trained
model has not been validated, and should be described that way.

## Related references

[pipeline.md](pipeline.md) for where validation sits in the workflow,
[plotting.md](plotting.md) for `plot_edits`, [pruning.md](pruning.md) for the
attribution/pruning correspondence, [memory-and-oom.md](memory-and-oom.md) if
attributing many designs runs out of memory.
