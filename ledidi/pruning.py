# pruning.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

@torch.no_grad()
def greedy_pruning(model, X, X_hat, threshold=1, verbose=False):
	"""A method for pruning edits to remove those that are irrelevant.

	This method will greedily go through all of the proposed edits and evaluate
	the effect of removing them, one at a time. As a greedy method, this will
	iteratively scan over all edits and remove the one with the smallest change
	in model output assuming that change is below the predefined threshold.
	Once the change in output from the edit with the smallest change is above
	the threshold, the procedure will stop and return the remaining edits.

	Note: Only one sequence is pruned at a time.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model used to evaluate the edits.

	X: torch.tensor, shape=(1, d, length)
		A tensor where the second dimension is the number of categories (e.g., 4 
		for DNA) and the third dimension is the length of the sequence, and is
		one-hot encoded.

	X_hat: torch.tensor, shape=(1, d, length)
		A tensor of the same shape as `X` except that it contains the proposed
		edits.

	threshold: float, optional
		A threshold on the maximum change in model output that removing an edit
		can have. Default is 1.

	verbose: bool, optional
		Whether to print out the index and delta at each iteration.

	Returns
	-------
	X_m: torch.tensor, shape=(1, d, length)
		A tensor of the same shape as `X_hat` except with some of the edits
		reverted back to what they were in `X`.
	"""

	diff_idxs_ = torch.where((X != X_hat).sum(axis=1) > 0)[1]
	diff_idxs = set([idx.item() for idx in diff_idxs_])

	y_hat = model(X_hat)
	
	for i in range(len(diff_idxs)):
		best_score, best_idx = float("inf"), -1    

		for idx in diff_idxs:
			X_mod = torch.clone(X_hat)
			X_mod[0, :, idx] = X[0, :, idx]

			y_mod = model(X_mod)
			score = torch.abs(y_hat - y_mod).sum()

			if score < best_score:
				best_score = score
				best_idx = idx 
				
		if best_score < threshold:
			diff_idxs.remove(best_idx)
			X_hat[0, :, best_idx] = X[0, :, best_idx]

			if verbose:
				print("Pruned Index: {}, Score: {:4.4}".format(best_idx, 
					best_score))

		else:
			break
		
	return X_hat
	