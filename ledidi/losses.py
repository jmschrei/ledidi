# losses.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

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
