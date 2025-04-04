# wrappers.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

import torch


class DesignWrapper(torch.nn.Module):
    """A wrapper for using multiple models in design.

    This wrapper will accept multiple models and turn their predictions into a
    vector. A requirement is that the output from each individual model is a
    tensor whose last dimension is 1, and that all of the models have the same
    other dimensions. For instance, if three models are passed in that each make 
    predictions of shape (batch_size, 1), the return from this wrapper would have
    shape (batch_size, 3).

    This wrapper is used to design edits when you want to balance the predictions
    from several models, e.g., by increasing predictions from one model without
    changing predictions from a second model. In practice, one would now pass in
    a vector of desired targets instead of a single value and the loss would be
    calculated over each of them.


    Parameters
    ----------
    models: list, tuple
        A set of torch.nn.Module objects.
    """

    def __init__(self, models):
        super(DesignWrapper, self).__init__()
        self.models = models
    
    def forward(self, X):
        return torch.cat([model(X) for model in self.models], dim=-1)
