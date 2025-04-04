# losses.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

import torch


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
	in_mask: torch.Tensor, shape=(1, n), dtype=bool
		A boolean tensor over the outputs from the underlying predictive model.
		True means to maximize values 
	"""
	
	def __init__(self, in_mask):
		self.in_mask = in_mask
        
	def __call__(self, y_hat, y_bar):
		on_target = y_hat[:, self.in_mask].min(dim=-1).values
		off_target = y_hat[:, ~self.in_mask].max(dim=-1).values
		return torch.mean(off_target - on_target)
