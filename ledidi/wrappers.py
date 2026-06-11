# wrappers.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

import torch


class DesignWrapper(torch.nn.Module):
	"""A wrapper for using multiple models in design.
	
	This wrapper will accept multiple models and concatenate their predictions
	along the last dimension. Each model may emit any number of outputs, but all
	of the models must agree on every dimension except the last one, which is the
	dimension being concatenated over. For instance, if three models are passed in
	that each make predictions of shape (batch_size, 1), the return from this
	wrapper would have shape (batch_size, 3). If instead one model returned
	(batch_size, 3) and a second returned (batch_size, 1), the return would have
	shape (batch_size, 4).
	
	This wrapper is used to design edits when you want to balance the predictions
	from several models, e.g., by increasing predictions from one model without
	changing predictions from a second model. In practice, one would now pass in
	a vector of desired targets instead of a single value and the loss would be
	calculated over each of them.
	
	
	Parameters
	----------
	models: list or tuple
		A non-empty list or tuple of torch.nn.Module objects.
	"""
	
	def __init__(self, models):
		super(DesignWrapper, self).__init__()

		if isinstance(models, torch.nn.Module):
			raise TypeError("models must be a list or tuple of torch.nn.Module "
				"objects, not a single torch.nn.Module")

		if not isinstance(models, (list, tuple)) or len(models) == 0:
			raise ValueError("models must be a non-empty list or tuple of "
				"torch.nn.Module objects")

		for i, model in enumerate(models):
			if not isinstance(model, torch.nn.Module):
				raise TypeError("models[{}] must be a torch.nn.Module, not "
					"`{}`".format(i, type(model)))

		self.models = torch.nn.ModuleList(models)

	def forward(self, X):
		return torch.cat([model(X).clone() for model in self.models], dim=-1)
