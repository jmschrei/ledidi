# ledidi.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
#          adapted from code written by Yang Lu

import time
import torch

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
        applied to the output loss. The larger this value is the more important
        it is that the output loss is minimized. Default is 100.

    batch_size: int, optional
        The number of sequences to generate at each step and average loss over. 
        Default is 32.

    max_iter: int, optional
        The maximum number of iterations to continue generating samples.
        Default is 5000.

    report_iter: int optional
        The number of iterations to perform before reporting results of the
        optimization. Default is 100.

    lr: float, optional
        The learning rate of the procedure. Default is 1e-2.

    input_mask: torch.Tensor or None, shape=(shape[-1],)
        A mask indicating what positions cannot be edited. This will set the
        initial weights mask to -inf at those positions. If None, no positions
        are masked out. Default is None.

    eps: float, optional
        The epsilon to add to the one-hot encoding. Because the first step
        of the procedure is to take log(X + eps) the smaller eps is the
        higher a value in the design weight needs to be achieved before
        an edit can be induced. Default is 1e-4.

    verbose: bool, optional
        Whether to print the loss during design. Default is True.
    """

    def __init__(self, model, shape, target=None, input_loss=torch.nn.L1Loss(
        reduction='sum'), output_loss=torch.nn.MSELoss(), tau=1, l=100, 
        batch_size=32, max_iter=5000, report_iter=100, lr=1e-2, input_mask=None,
        eps=1e-4, verbose=True):
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
        self.report_iter = report_iter
        self.lr = lr
        self.input_mask = input_mask
        self.eps = eps
        self.verbose = verbose

        if target is None:
            self.target = slice(target)
        else:
            self.target = target

        self.weights = torch.nn.Parameter(torch.zeros(1, *shape, 
            dtype=torch.float32, requires_grad=True))
        
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
        """Appply the Ledidi procedure to design edits for a sequence.

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

        if self.input_mask is not None:
            self.weights.requires_grad = False
            self.weights.T[self.input_mask] = float("-inf")
            self.weights[X.type(torch.bool)] = 0
            self.weights.requires_grad = True
        
        y_hat = self.model(X)[:, self.target]

        output_loss = self.output_loss(y_hat, y_bar).item()
        best_total_loss = self.l * output_loss
        best_sequence = X
        
        if self.verbose:
            print(("iter=I\tinput_loss=0\toutput_loss={:4.4}\t" + 
                "total_loss={:4.4}").format(output_loss, best_total_loss))
            tic = time.time()
        
        for i in range(self.max_iter+1):
            X_hat = self(X)
            y_hat = self.model(X_hat)[:, self.target]
            
            input_loss = self.input_loss(X_hat, X) / (X_hat.shape[0] * 2)
            output_loss = self.output_loss(y_hat, y_bar)
            
            total_loss = input_loss + self.l * output_loss
            total_loss_ = total_loss.item()
            
            if self.verbose and i % self.report_iter == 0:
                print(("iter={}\tinput_loss={:4.4}\toutput_loss={:4.4}\t" +
                    "total_loss={:4.4}\ttime={:4.4}").format(i, 
                        input_loss.item(), output_loss.item(), 
                        total_loss_, time.time() - tic))
                tic = time.time()
                      
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if total_loss_ < best_total_loss:
                best_total_loss = total_loss_
                best_sequence = torch.clone(X_hat)

        return best_sequence
