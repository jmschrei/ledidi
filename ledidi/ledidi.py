# ledidi.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
#          adapted from code written by Yang Lu

import time
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


class Ledidi(torch.nn.Module):
    """Ledidi is a method for editing categorical sequences.

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
