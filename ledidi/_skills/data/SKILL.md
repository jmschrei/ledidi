---
name: ledidi
description: Use for any task involving the ledidi library — gradient-based design of minimal edits to categorical sequences (DNA/RNA/protein) so that a frozen oracle model predicts a desired output. Triggers on designing or editing a sequence, inserting or knocking out a motif or binding site, hitting a target model output, cell type-specific or output-specific element design, affinity catalogs, in-painting, constraining where edits may be made, pruning edits, or balancing several oracle models in one design. This is a router skill — read the relevant file under references/ for details and footguns before writing ledidi code.
---

# ledidi

`ledidi` inverts the usual training loop: the oracle model is frozen and the
*data* is optimized. It learns a continuous weight matrix, samples one-hot edits
to a template sequence from a Gumbel-softmax, and pushes those edits until the
oracle predicts what you asked for — while an input loss keeps the number of
edits small. Any differentiable PyTorch model that maps a one-hot sequence to a
prediction becomes a sequence editor.

This skill is a **router**. Each topic below has a reference file with exact
signatures and footguns. **Read the relevant reference file before writing
code** — do not rely on memory of the API. Several of ledidi's failure modes are
silent: a plausible-looking argument can return your template completely
unedited, or make pruning revert every edit, with no exception raised.

`tangermeme` is a hard dependency of ledidi (it supplies the input validation),
and it owns everything that happens *around* a design — one-hot encoding, FASTA
and loci I/O, attributions, motif scanning, logo plotting. **Install its skill
too** (`tangermeme-install-skills`) and consult it for those steps rather than
reinventing them here.

**Is ledidi even the right tool?** ledidi is gradient-based and finds small,
targeted edits to an existing template. For *discrete* design — implanting motifs
from a library, screening random candidates, greedy or beam substitution — use
`tangermeme.design` instead (`screen`, `greedy_substitution`, `beam_substitution`,
`greedy_marginalize`); its skill's `references/design.md` covers them. Note that
`tangermeme.design` requires a per-candidate loss (`reduction='none'`) to rank
edits, which is the **opposite** of ledidi's requirement that the output loss
return a scalar — do not carry that habit across.

## Read these first

- **[The objective](references/objective.md)** — every design is
  `output_loss(y_hat, y_bar) + l * input_loss`, where the input loss is the mean
  number of edits per sequence. `l` is the exchange rate between "hit the target"
  and "make few edits", and it is the knob you will actually tune. Also covers
  what the `verbose` log lines mean and why the returned design is the
  best-scoring iterate rather than the last one.

- **[The oracle contract](references/oracle-contract.md)** — what your model must
  satisfy before any of this works: differentiable, `model(X)` sliceable as
  `[:, target]`, frozen and `.eval()`ed for you. Read it before wrapping anything.

## Getting the oracle side right

Most real designs fail here, not in the optimizer. Three distinct situations,
each with its own file:

| Situation | Read |
|---|---|
| one multi-task model, you want **some of its outputs** | [references/multi-task-models.md](references/multi-task-models.md) |
| **several independent models** balanced in one design | [references/multiple-models.md](references/multiple-models.md) |
| the model's **input window ≠ the sequence you want to design** | [references/receptive-field.md](references/receptive-field.md) |

## Task → reference file

| If the task is… | Read |
|---|---|
| **starting from scratch** — a real oracle, end to end: design → prune → validate → plot | [references/pipeline.md](references/pipeline.md) |
| a first design / learning the mechanics with no downloads | [references/first-design.md](references/first-design.md) |
| tensor shapes, dtypes, or a `ValueError`/`TypeError` you do not understand | [references/io-and-validation.md](references/io-and-validation.md) |
| forbidding edits at certain **positions** | [references/masks.md](references/masks.md) |
| forbidding or forcing **specific characters**, or setting soft priors | [references/initial-weights.md](references/initial-weights.md) |
| letting ledidi **fill in** a blanked region (in-painting) | [references/inpainting.md](references/inpainting.md) |
| a non-MSE objective: `MinGap`, rewards, one-sided or ballpark losses, profiles | [references/custom-losses.md](references/custom-losses.md) |
| designing against a **range** of target strengths (affinity catalog), or repeats | [references/catalogs-and-repeats.md](references/catalogs-and-repeats.md) |
| sampling many designs cheaply, or reusing a fitted designer | [references/designer-object.md](references/designer-object.md) |
| **trimming** unnecessary edits after design | [references/pruning.md](references/pruning.md) |
| checking a design is real and not oracle exploitation | [references/validating-designs.md](references/validating-designs.md) |
| plotting losses, edit maps, or edits on attribution tracks | [references/plotting.md](references/plotting.md) |
| a **CUDA out-of-memory** error | [references/memory-and-oom.md](references/memory-and-oom.md) |
| reproducibility, seeding, CPU vs GPU | [references/reproducibility.md](references/reproducibility.md) |

## The rest of the package

- `ledidi.ledidi` — the function you almost always call. Handles device
  placement, repeats, affinity catalogs, and post-fit sampling.
- `ledidi.Ledidi` — the underlying `torch.nn.Module` optimizer
  (`fit_transform`, `forward`). Reach for it only to fit once and sample
  repeatedly → [designer-object.md](references/designer-object.md).
- `ledidi.losses.MinGap` — output-specific design without target values.
- `ledidi.wrappers.DesignWrapper` — concatenate several oracles into one.
- `ledidi.pruning.greedy_pruning` — post-hoc edit trimming.
- `ledidi.plot` — `plot_loss`, `plot_history`, `plot_edits`.

## Conventions

- **Tensor layout** `(batch, n_channels, length)`, `torch.float32`, one-hot along
  the channel axis (DNA: 4 channels ordered `A, C, G, T`). The template `X` you
  pass in has a batch dimension of exactly **1**; the returned designs have a
  batch dimension of `batch_size`.
- **Naming** `X` template, `X_bar` designed sequences, `y_bar` desired output,
  `y_hat` predictions, `X_attr` attributions.
- **`device` defaults to `'cuda'`**, not to "CUDA if available". On a CPU-only
  machine you must pass `device='cpu'` explicitly or the call raises.
- **`ledidi()` moves the model in place, but not your template.** Designs come back
  on the device while your `X` stays where it was, so `.to(device)` your `X`
  yourself — otherwise pruning, edit diffs, and `designer(X)` all raise
  `RuntimeError: Expected all tensors to be on the same device`. This is the most
  common first error.
- **Substitutions only.** ledidi changes characters in place; it never inserts or
  deletes, so the length is fixed for the whole design.
- **ledidi's keywords are not tangermeme's.** `ledidi()` forwards `**kwargs` to
  `Ledidi.__init__`, which takes no `**kwargs`, so a tangermeme habit like
  `output_mask=`, `args=`, or `func=` raises `TypeError` rather than being silently
  ignored. Output selection is `target` →
  [multi-task-models.md](references/multi-task-models.md).
- **Returned designs carry an autograd graph.** Both the default return and the
  `n_samples` draw come back with `requires_grad=True`; `.detach()` them before
  holding many, and see [memory-and-oom.md](references/memory-and-oom.md).
