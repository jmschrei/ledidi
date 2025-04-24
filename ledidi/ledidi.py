# ledidi.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
#          adapted from code written by Yang Lu

import time
import torch


def ledidi(model, X, y_bar, n_repeats=1, n_samples=None, return_designer=False,
	return_history=False, device='cuda', **kwargs):
	"""Ledidi is a method for editing sequences to exhibit desired properties.
	
	Ledidi is a method for designing compact sets of edits to categorical
	sequences, such as DNA, to make them exhibit desired characteristics as
	predicted by an oracle model. This is done by rephrasing the edit design task
	as a continuous optimization problem that can be solved using off-the-shelf
	optimizers and strategies. In this problem, Ledidi is trying to minimize an
	objective function comprised of an output loss, which measures how far away
	predictions on the edited sequence are from the target predictions, and the
	input loss, which measures the number of edits.

	Because gradients cannot be directly applied to one-hot encoded categorical
	sequences, Ledidi learns an underlying continuous weight matrix from which
	categorical sequences are sampled. The distribution used for sampling is the
	Gumbel-softmax distribution and this is referred to as the straight-through
	estimator. Essentially, the process allows us the flexibility of a simple
	continuous optimization problem and the consistency of still sampling one-hot
	encoded sequences to run through the oracle model.

	The oracle model can bd a single task from a single model, multiple tasks
	from the same model, or even multiple tasks from multiple models. All that
	matters is that the entire thing is differentiable. PyTorch makes the use of
	multiple tasks/models easy through the use of wrappers. See
	
		https://tangermeme.readthedocs.io/en/latest/vignettes/Wrappers_are_Productivity_Hacks.html
	
	for more information on how to construct wrappers that may be helpful.
	
	This function is a wrapper around the Ledidi class, which must inherit from
	torch.nn.Module because a torch.nn.Parameter must be initialized for the 
	weight matrix, and a call to `Ledidi.fit_transform`. Basically, this wrapper 
	turns the two line implementation into a one line one that is consistent with
	other design implementations.
	
	Additionally, one can design an affinity catalog by passing in a list of
	target values in `y_bar` instead of a single value. When a list is provided,
	an additional dimension is added to the front of the returned tensor of
	designed sequences. 
	
	
	If one wants to perform design multiple times they can set `n_repeats` to a
	value above 1. The initial weight matrix will be zero but different samples
	will be drawn from the Gumbel-softmax distribution, potentially leading to
	different outcomes.
	
	Finally, by default one batch of designed sequences is returned. If you would
	like more than one batch of samples returned, you can specify the number of
	samples drawn from Ledidi's learned distributions. 
	
	
	Parameters
	----------
	model: torch.nn.Module
		A model to use as an oracle that will be frozen as a part of the Ledidi
		procedure.
	
	X: torch.Tensor, shape=(1, n_channels, length)
		A tensor containing a single one-hot encoded sequence to propose edits for. 
		This sequence is then expanded out to the desired batch size to generate a 
		batch of edits.

	y_bar: torch.Tensor or list, shape=(1, *)
		The desired output from the model. Any shape for this tensor is permissable
		so long as the `output_loss` function can handle comparing it to the output 
		from the given model. If a list is provided then each item in the list must
		have those properties.
	
	n_repeats: int, optional
		The number of times to run the Ledidi procedure. If 1, do not include this
		as a dimension in the returned blob of sequences. If above 1, include this
		as the first or second dimension, depending on whether an affinity catalog
		is being designed (second if so, first if not). Default is 1.
	
	n_samples: int or None, optional
		The number of samples to draw from Ledidi after the optimization process.
		If None, draw one batch as defined by `batch_size`. Otherwise, draw the
		number of sequences specified.
	
	return_designer: bool, optional
		Whether to return the designers for each design setting. If multiple
		repeats are done, each designer will be returned. Orthogonally, if
		an affinity catalog is being designed, return designers for each step.
		Default is False.
	
	return_history: bool, optional
		Whether to return a history for each run of Ledidi. This history includes
		each loss and other statistics. Default is False.
	
	device: str or torch.device, optional
		The device to move all the tensors and models to as a convenience. Default
		is 'cuda'.
	
	**kwargs
		Any additional arguments to be passed into the Ledidi object.
	
	
	Returns
	-------
	y: torch.Tensor, shape=(*ny, *n_repeats, n_sample, n_channels, length)
		A tensor containing a batch of one-hot encoded sequences which may contain 
		one or more edits compared to the sequence that was passed in. If a list of
		`y_bar` values has been passed in, indicating that one would like to design
		an affinity catalog, that becomes the first dimension. If multiple repeats
		are being done, prepend that as well as either the first dimension, if no
		affinity catalog is being designed, or the second dimension, if the catalog
		is being designed.
	"""
	
	if not isinstance(n_repeats, int) or n_repeats <= 0:
		raise ValueError("n_repeats must be a positive integer, not `{}`".format(
			n_repeats))
	
	if n_samples is not None and (not isinstance(n_samples, int) or n_samples <= 0):
		raise ValueError("n_samples must be a positive integer, not `{}`".format(
			n_samples))
	
	if not isinstance(y_bar, list):
		y_bar = [y_bar]
	
	ny = len(y_bar)
	
	X = X.to(device)
	X_bar = [[] for i in range(ny)]
	designers = [[] for i in range(ny)]
	histories = [[] for i in range(ny)]
	
	for i, y_bar_i in enumerate(y_bar):
		y_bar_i = y_bar_i.to(device)
		
		for j in range(n_repeats):
			designer = Ledidi(model, shape=X.shape[-2:], return_history=return_history, 
				**kwargs).to(device)
			X_bar_ = designer.fit_transform(X, y_bar_i)
			
			if return_history:
				X_bar_, history = X_bar_
				histories[i].append(history)
			
			designers[i].append(designer)
			
			if n_samples is not None:
				n_iter = n_samples // designer.batch_size + 1
				X_bar_ = torch.cat([designer(X) for _ in range(n_iter)], dim=0)[:n_samples]
			
			X_bar[i].append(X_bar_)
		
		X_bar[i] = torch.stack(X_bar[i])
	
	X_bar = torch.stack(X_bar)
	
	if n_repeats == 1:
		X_bar = X_bar[:, 0]
		designers = [d[0] for d in designers]
		histories = None if not return_history else [h[0] for h in histories]
	
	if len(y_bar) == 1:
		X_bar = X_bar[0]
		designers = designers[0]
		histories = None if not return_history else histories[0]
	
	ledidi_output = [X_bar]
	
	if return_designer:
		ledidi_output.append(designers)
	
	if return_history:
		ledidi_output.append(histories)
	
	return ledidi_output[0] if len(ledidi_output) == 1 else ledidi_output


class Ledidi(torch.nn.Module):
    """Ledidi is a method for editing sequences to exhibit desired properties.

    Ledidi is a method for editing categorical sequences, such as those
    comprised of nucleotides or amino acids, to exhibit desired properties in
    a small number of edits. It does so through the use of an oracle model,
    which is a differentiable model that accepts a categorical sequence as
    input and makes relevant predictions. For instance, the model might take
    in one-hot encoded nucleotide sequence and predict the strength of binding 
    for a particular transcription factor. 

    Given a sequence and a desired output, Ledidi uses gradient descent to 
    design edits that bring the predicted output from the model closer to the
    desired output. Because the sequences that predictions are being made for
    must be categorical this involves using the Gumbel-softmax 
    reparameterization trick.


    Parameters
    ----------
    model: torch.nn.Module
        A model to use as an oracle that will be frozen as a part of the
        Ledidi procedure.

    shape: tuple of two integers
        The number of categories and the number of positions, respectively,
        in the sequence to be edited. For nucleotides this might be (4, 1000).

    target: int or None
        When given a multi-task model, the target to slice out and feed into
        output_loss when calculating the gradient. If None, perform no slicing.
        Default is None.

    input_loss: torch.nn.Loss, optional
        A loss to apply to the input space. By default this is the L1 loss
        which corresponds to the number of positions that have been edited.
        This loss is also divided by 2 to account for each edit changing
        two values within that position. Default is torch.nn.L1Loss.

    output_loss: torch.nn.Loss, optional
        A loss to apply to the output space. By default this is the L2 loss
        which corresponds to the mean squared error between the predicted values
        and the desired values.

    tau: float, positive, optional
        The sharpness of the sampled values from the Gumbel distribution used
        to generate the one-hot encodings at each step. Higher values mean
        sharper, i.e., more closely match the argmax of each position.
        Default is 1.

    l: float, positive, optional
        The mixing weight parameter between the input loss and the output loss,
        applied to the input loss. The smaller this value is the more important
        it is that the output loss is minimized. Default is 0.01.

    batch_size: int, optional
        The number of sequences to generate at each step and average loss over. 
        Default is 64.

    max_iter: int, optional
        The maximum number of iterations to continue generating samples.
        Default is 1000.

    report_iter: int optional
        The number of iterations to perform before reporting results of the
        optimization. Default is 100.

    lr: float, optional
        The learning rate of the procedure. Default is 0.1.

    input_mask: torch.Tensor or None, shape=(shape[-1],)
        A mask where 1 indicates what positions cannot be edited. This will 
        set the initial weights mask to -inf at those positions. If None, no 
        positions are masked out. Default is None.

    initial_weights: torch.Tensor or None, shape=(1, shape[0, shape[1])
        Initial weights to use in the weight matrix to specify priors in the
        composition of edits that can be made. Positive values mean more likely
        that certain edits are proposed, negative values mean less likely that
        those edits are proposed.

    eps: float, optional
        The epsilon to add to the one-hot encoding. Because the first step
        of the procedure is to take log(X + eps) the smaller eps is the
        higher a value in the design weight needs to be achieved before
        an edit can be induced. Default is 1e-4.

    random_state: int or None, optional
        Whether to force determinism.

    verbose: bool, optional
        Whether to print the loss during design. Default is True.
    """

    def __init__(self, model, shape, target=None, input_loss=torch.nn.L1Loss(
        reduction='sum'), output_loss=torch.nn.MSELoss(), tau=1, l=0.1, 
        batch_size=16, max_iter=1000, early_stopping_iter=100, report_iter=100, 
        lr=1.0, input_mask=None, initial_weights=None, eps=1e-4, 
        return_history=False, verbose=True):
        super().__init__()
        
        for param in model.parameters():
            param.requires_grad = False
            
        self.model = model.eval()
        self.input_loss = input_loss
        self.output_loss = output_loss
        self.tau = tau
        self.l = l
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.early_stopping_iter = early_stopping_iter
        self.report_iter = report_iter
        self.lr = lr
        self.input_mask = input_mask
        self.eps = eps
        self.return_history = return_history
        self.verbose = verbose

        if target is None:
            self.target = slice(target)
        else:
            self.target = slice(target, target+1)

        if initial_weights is None:
            initial_weights = torch.zeros(1, *shape, dtype=torch.float32,
                requires_grad=True)
        else:
            initial_weights.requires_grad = True
        
        self.weights = torch.nn.Parameter(initial_weights)
        

    def forward(self, X):
        """Generate a set of edits given a sequence.

        This method will take in the one-hot encoded sequence and the current
        learned weight filter and propose edits based on the Gumbel-softmax
        distribution.


        Parameters
        ----------
        X: torch.Tensor, shape=(1, n_channels, length)
            A tensor containing a single one-hot encoded sequence to propose 
            edits for. This sequence is then expanded out to the desired batch 
            size to generate a batch of edits.


        Returns
        -------
        y: torch.Tensor, shape=(batch_size, n_channels, length)
            A tensor containing a batch of one-hot encoded sequences which
            may contain one or more edits compared to the sequence that was
            passed in.
        """

        logits = torch.log(X + self.eps) + self.weights
        logits = logits.expand(self.batch_size, *(-1 for i in range(X.ndim-1)))
        return torch.nn.functional.gumbel_softmax(logits, tau=self.tau, 
            hard=True, dim=1)
        

    def fit_transform(self, X, y_bar):
        """Apply the Ledidi procedure to design edits for a sequence.

        This procedure takes in a single sequence and a desired output from
        the model and designs edits that cause the model to predict the desired
        output. This is done primarily by learning a weight matrix of logits
        that can be added the log'd one-hot encoded sequence. These weights
        are the only weights learned during the procedure.


        Parameters
        ----------
        X: torch.Tensor, shape=(1, n_channels, length)
            A tensor containing a single one-hot encoded sequence to propose 
            edits for. This sequence is then expanded out to the desired batch 
            size to generate a batch of edits.

        y_bar: torch.Tensor, shape=(1, *)
            The desired output from the model. Any shape for this tensor is
            permissable so long as the `output_loss` function can handle
            comparing it to the output from the given model.


        Returns
        -------
        y: torch.Tensor, shape=(batch_size, n_channels, length)
            A tensor containing a batch of one-hot encoded sequences which
            may contain one or more edits compared to the sequence that was
            passed in.
        """

        optimizer = torch.optim.AdamW((self.weights,), lr=self.lr)
        history = {'edits': [], 'input_loss': [], 'output_loss': [], 
            'total_loss': [], 'batch_size': self.batch_size}

        if self.input_mask is not None:
            self.weights.requires_grad = False
            self.weights[:, :, self.input_mask] = float("-inf")
            self.weights[X.type(torch.bool)] = 0
            self.weights.requires_grad = True
        
        inpainting_mask = X[0].sum(dim=0) == 1
        y_hat = self.model(X)[:, self.target]
        
        n_iter_wo_improvement = 0
        output_loss = self.output_loss(y_hat, y_bar).item()

        best_input_loss = 0.0
        best_output_loss = output_loss
        best_total_loss = output_loss
        best_sequence = X
        best_weights = torch.clone(self.weights)
        
        X_ = X.expand(self.batch_size, *X.shape[1:])
        y_bar = y_bar.expand(self.batch_size, *y_bar.shape[1:])
        
        tic = time.time()
        initial_tic = time.time()
        if self.verbose:
            print(("iter=I\tinput_loss=0.0\toutput_loss={:4.4}\t" +
                "total_loss={:4.4}\ttime=0.0").format(output_loss, 
                    best_total_loss))

        for i in range(1, self.max_iter+1):
            X_hat = self(X)
            y_hat = self.model(X_hat)[:, self.target]
            
            input_loss = self.input_loss(X_hat[:, :, inpainting_mask], X_[:, :, inpainting_mask]) / (X_hat.shape[0] * 2)
            output_loss = self.output_loss(y_hat, y_bar)
            total_loss = output_loss + self.l * input_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            input_loss = input_loss.item()
            output_loss = output_loss.item()
            total_loss = total_loss.item()
            
            if self.verbose and i % self.report_iter == 0:
                print(("iter={}\tinput_loss={:4.4}\toutput_loss={:4.4}\t" +
                    "total_loss={:4.4}\ttime={:4.4}").format(i, input_loss, 
                        output_loss, total_loss, time.time() - tic))
            
                tic = time.time()               

            if self.return_history:
                history['edits'].append(torch.where(X_hat != X_))
                history['input_loss'].append(input_loss)
                history['output_loss'].append(output_loss)
                history['total_loss'].append(total_loss)

            if total_loss < best_total_loss:
                best_input_loss = input_loss
                best_output_loss = output_loss
                best_total_loss = total_loss

                best_sequence = torch.clone(X_hat)
                best_weights = torch.clone(self.weights)

                n_iter_wo_improvement = 0
            else:
                n_iter_wo_improvement += 1
                if n_iter_wo_improvement == self.early_stopping_iter:
                    break

        optimizer.zero_grad()
        self.weights = torch.nn.Parameter(best_weights)

        if self.verbose:
            print(("iter=F\tinput_loss={:4.4}\toutput_loss={:4.4}\t" +
                "total_loss={:4.4}\ttime={:4.4}").format(best_input_loss, 
                    best_output_loss, best_total_loss, 
                    time.time() - initial_tic))

        if self.return_history:
            return best_sequence, history
        return best_sequence
