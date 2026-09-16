# losses.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""Output loss functions for output-specific design.

The default output loss in Ledidi is an ``MSELoss`` that drives every model
output toward an explicit target value. That works well when you know exactly
what each output should be, but some design goals are more naturally phrased as
a contrast between groups of outputs rather than as fixed targets.

This module collects such alternative losses. The first is :class:`MinGap`, the
min-gap loss of Gosai et al. for designing output-specific (e.g., cell
type-specific) elements by maximizing the gap between the weakest on-target
output and the strongest off-target output, with no target values required.
:class:`GapLoss` generalizes it: the off-target outputs can be averaged instead
of maximized, and either side can be bounded, so the optimizer stops pushing
once an output is as low or as high as anything the model meaningfully
produces. ``GapLoss(in_mask)`` is exactly ``MinGap(in_mask)``.

Any callable with the signature ``f(y_hat, y_bar)`` can be passed to ``ledidi``
via ``output_loss``, so these classes are examples as much as they are
batteries. Composing them is just arithmetic — to hold one output near a value
while separating the groups, add a squared term to a gap loss::

	gap = GapLoss(in_mask)
	output_loss = lambda y_hat, y_bar: (gap(y_hat, y_bar)
		+ (y_hat[:, i] - y_bar[:, i]).pow(2).mean())

Note that such a target is not where the design lands: the summed objective is
stationary where the squared term's gradient cancels the gap term's, which is
offset from the target itself.
"""

import torch
from tangermeme.utils import _validate_input


class MinGap():
	"""The MinGap loss function for producing task-specific outputs.

	The MinGap function was proposed by Gosai et al. for designing cell
	type-specific regulatory elements. This function tries to maximize the output
	from one or a set of outputs while minimizing the output from all others. In
	the context of cell type-specific design this means having a high response
	from one cell type and a low response in the others. 

	Perhaps initially counterintuitively, MinGap attempts to maximize response for
	the on-target elements by taking the minimum predicted value across them. By
	maximizing this minimum value, we are trying to get high responses from all
	on-target outputs. Likewise, we take the maximum off-target value to ensure
	that ALL off-target values have low values.

	This has advantages and disadvantages compared to using the average on- and
	off-target values. For instance, if one of the on-target values cannot be
	optimized (perhaps because the underlying model does not make good predictions
	for it), the entire optimization procedure can fail. Likewise, if any of the
	off-target values happens to correlate with the on-target ones the
	optimization can struggle even if all other off-target values are low.


	Parameters
	----------
	in_mask: torch.Tensor, shape=(n,), dtype=bool
		A boolean mask over the `n` outputs from the underlying predictive model.
		True marks an output as on-target, i.e., one whose value should be
		maximized, and False marks an output as off-target, i.e., one whose value
		should be minimized. It must contain at least one on-target (True) and one
		off-target (False) output, since the loss takes a minimum over the former
		and a maximum over the latter.
	"""

	def __init__(self, in_mask):
		_validate_input(in_mask, "in_mask", dtype=torch.bool)

		if bool(in_mask.all()) or not bool(in_mask.any()):
			raise ValueError("in_mask must contain at least one on-target "
				"(True) and one off-target (False) output")

		self.in_mask = in_mask

	def __call__(self, y_hat, y_bar):
		"""Compute the min-gap loss for a batch of predictions.

		Note that `y_bar` is accepted only to match the `(y_hat, y_bar)`
		signature that Ledidi expects of an output loss; the min-gap loss has no
		target values and so `y_bar` is ignored entirely.


		Parameters
		----------
		y_hat: torch.Tensor, shape=(batch_size, n)
			The predicted outputs from the underlying model for a batch of
			edited sequences.

		y_bar: torch.Tensor
			Ignored. Present only for signature compatibility with Ledidi.


		Returns
		-------
		loss: torch.Tensor, shape=()
			The mean over the batch of the gap between the maximum off-target
			value and the minimum on-target value. Minimizing this maximizes the
			separation between the on- and off-target outputs.
		"""

		on_target = y_hat[:, self.in_mask].min(dim=-1).values
		off_target = y_hat[:, ~self.in_mask].max(dim=-1).values
		return torch.mean(off_target - on_target)


class GapLoss():
	"""A generalization of the min-gap loss with optional bounds.

	Like :class:`MinGap`, this loss contrasts a group of on-target outputs
	against a group of off-target ones and needs no target values. It adds
	three knobs that control *when the optimizer should stop pushing*, which
	matters because a plain gap has no lower or upper limit: it will keep
	spending edits driving off-target outputs below anything the model can
	meaningfully represent, and driving on-target outputs far past the largest
	value it has ever produced. Both directions reduce the loss and neither
	necessarily improves the sequence.

	The on-target side is always reduced with a minimum, for the reason given
	in :class:`MinGap`: maximizing the weakest on-target output is what forces
	all of them up rather than letting a single easy output carry the loss.

	The off-target side can be reduced with either a maximum or a mean, and the
	choice is a real trade-off. A maximum is the stricter objective — it
	guarantees that *every* off-target output is separated, because the worst
	one is the only one visible to the loss. A mean is more sensitive to a
	shift that lifts all off-target outputs together, since a maximum only ever
	sees one of them, but it can be satisfied by a sequence that is on-target
	and also high in a single off-target output, provided the remaining ones sit
	low enough to carry the average. Prefer a maximum when any single off-target
	response is unacceptable, and a mean when a diffuse response across all
	off-target outputs is the failure you are guarding against.

	`floor` and `ceiling` bound the two sides. Below its floor an off-target
	output earns no further credit, so once every off-target output is at or
	under its floor the off-target term stops producing gradient and the only
	remaining way to reduce the loss is to raise the on-target outputs. Above
	its ceiling an on-target output is penalized quadratically, which leaves
	activity below the ceiling entirely unpenalized while giving the loss a
	finite optimum: the on-target gradient `-1 + 2 * ceiling_weight * (on -
	ceiling)` vanishes at `ceiling + 1 / (2 * ceiling_weight)`. Choosing both
	from data — a floor from outputs the model produces on inputs known to be
	inactive, a ceiling from a high quantile of outputs on real inputs — keeps
	the design inside the range the model was fit on without naming a target.

	The quadratic ceiling term has a second effect worth knowing. Without it
	the loss is linear in the on-target outputs, so the marginal value of an
	edit barely decreases as the design improves and the edit count tends to be
	all-or-nothing as `l` is varied: either edits pay for themselves everywhere
	or nowhere, with little in between. The ceiling term restores curvature and
	with it a usable trade-off between `l` and the number of edits.

	`GapLoss(in_mask)` with no other arguments is identical to
	`MinGap(in_mask)`.


	Parameters
	----------
	in_mask: torch.Tensor, shape=(n,), dtype=bool
		A boolean mask over the `n` outputs from the underlying predictive
		model. True marks an output as on-target, i.e., one whose value should
		be maximized, and False marks an output as off-target, i.e., one whose
		value should be minimized. It must contain at least one on-target
		(True) and one off-target (False) output, since the loss takes a
		minimum over the former and a maximum or mean over the latter.

	off_reduction: str, optional
		How to reduce the off-target outputs, either 'max' or 'mean'. See the
		trade-off described above. Default is 'max'.

	floor: torch.Tensor, shape=(n,), or float or None, optional
		A lower bound on each output, below which no further credit is given.
		Off-target outputs are clamped to this value before being reduced, so
		an off-target output already below its floor contributes no gradient. A
		float is broadcast to every output. Entries corresponding to on-target
		outputs are ignored. If None, the off-target outputs are unbounded
		below. Default is None.

	ceiling: torch.Tensor, shape=(n,), or float or None, optional
		An upper bound on the on-target outputs, above which they are penalized
		quadratically. Activity at or below this value is not penalized at all.
		With several on-target outputs the binding value is the minimum over
		their entries, matching the minimum taken on the on-target side. A
		float is broadcast to every output. Entries corresponding to off-target
		outputs are ignored. If None, the on-target outputs are unbounded
		above and the loss is linear in them. Default is None.

	ceiling_weight: float, optional
		The weight on the quadratic over-ceiling penalty. The on-target outputs
		settle at `ceiling + 1 / (2 * ceiling_weight)`, so a larger value holds
		them closer to the ceiling. Ignored when `ceiling` is None. Default is
		2.0.
	"""

	def __init__(self, in_mask, off_reduction='max', floor=None, ceiling=None,
		ceiling_weight=2.0):
		_validate_input(in_mask, "in_mask", dtype=torch.bool)

		if bool(in_mask.all()) or not bool(in_mask.any()):
			raise ValueError("in_mask must contain at least one on-target "
				"(True) and one off-target (False) output")

		if off_reduction not in ('max', 'mean'):
			raise ValueError("off_reduction must be 'max' or 'mean', not "
				"`{}`".format(off_reduction))

		if not isinstance(ceiling_weight, (int, float)):
			raise TypeError("ceiling_weight must be a float, not `{}`".format(
				type(ceiling_weight)))

		if ceiling_weight <= 0:
			raise ValueError("ceiling_weight must be positive, not `{}`".format(
				ceiling_weight))

		floor = self._as_vector(floor, "floor", in_mask)
		ceiling = self._as_vector(ceiling, "ceiling", in_mask)

		self.in_mask = in_mask
		self.off_reduction = off_reduction
		self.floor = floor
		self.ceiling = ceiling
		self.ceiling_weight = float(ceiling_weight)

	@staticmethod
	def _as_vector(value, name, in_mask):
		"""Broadcast a float bound to a vector, or validate a given one."""

		if value is None:
			return None

		if isinstance(value, (int, float)):
			return torch.full_like(in_mask, float(value), dtype=torch.float32)

		_validate_input(value, name, shape=in_mask.shape)
		return value

	def __call__(self, y_hat, y_bar):
		"""Compute the bounded gap loss for a batch of predictions.

		Note that `y_bar` is accepted only to match the `(y_hat, y_bar)`
		signature that Ledidi expects of an output loss; this loss has no
		target values and so `y_bar` is ignored entirely.


		Parameters
		----------
		y_hat: torch.Tensor, shape=(batch_size, n)
			The predicted outputs from the underlying model for a batch of
			edited sequences.

		y_bar: torch.Tensor
			Ignored. Present only for signature compatibility with Ledidi.


		Returns
		-------
		loss: torch.Tensor, shape=()
			The mean over the batch of the reduced off-target value minus the
			minimum on-target value, plus the over-ceiling penalty when a
			ceiling is given. Minimizing this maximizes the separation between
			the on- and off-target outputs.
		"""

		on_target = y_hat[:, self.in_mask].min(dim=-1).values

		off_target = y_hat[:, ~self.in_mask]
		if self.floor is not None:
			off_target = torch.clamp(off_target, min=self.floor[~self.in_mask])

		if self.off_reduction == 'max':
			off_target = off_target.max(dim=-1).values
		else:
			off_target = off_target.mean(dim=-1)

		loss = off_target - on_target

		if self.ceiling is not None:
			ceiling = self.ceiling[self.in_mask].min()
			over = torch.clamp(on_target - ceiling, min=0.0) ** 2
			loss = loss + self.ceiling_weight * over

		return torch.mean(loss)
