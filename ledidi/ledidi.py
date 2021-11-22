# ledidi.py
# Authors: Yang Lu <ylu465@uw.edu> and Jacob Schreiber <jmschreiber91@gmail.com>

MIN_W = 0.001

import numpy
from scipy.special import logsumexp

import tensorflow as tf
import tensorflow.keras.backend as k

MIN_W = 0.001

class TensorFlowRegressor():
    """A wrapper for a TensorFlow regression model.

    This wrapper holds a TensorFlow model that has regression outputs. The
    methods implemented here are useful for calculating gradients and losses
    given certain masks.
    """

    def __init__(self, model, verbose=False):
        """
        Parameters
        ----------
        model : TensorFlow model
            The regression model that is being wrapped.

        verbose : bool, optional
            Whether to print out logs related to use of this object. 
            Default is False.
        """

        self.verbose = verbose
        self._model = model
        self._input = model.input
        self._output = model.output

        self._input_shape = k.int_shape(self._input)
        self._output_shape = k.int_shape(self._output)
        
        if self.verbose == True:
            print("TensorFlowRegressor model input_shape={}".format(
                self._input_shape))
            print("TensorFlowRegressor model output_shape={}".format(
                self._output_shape))

    def loss_gradient(self, x, y, mask=None):
        """Compute the gradient of the loss function || f(x)-y ||^2 w.r.t. `x`.
        
        Parameters
        ----------
        x : numpy.ndarray
            The input sequence to the model.

        y : numpy.ndarray
            The output to calculate the loss w.r.t.

        mask : numpy.ndarray or None, optional
            A binary mask indicating the outputs to calculate the loss over.
            Default is None.

        Returns
        -------
        grads : numpy.ndarray
            An array of gradients with the same shape as `x`.
        """

        if mask is None:
            mask = numpy.ones(y.shape, dtype='float32')

        mask = tf.convert_to_tensor(mask / mask.sum())

        x_var = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            tape.watch(x_var)
            pred_y = self._model(x_var, training=False)
            loss = k.sum(tf.multiply(mask, k.square(pred_y - y)))

        grads = tape.gradient(loss, [x_var])[0]
        assert grads.shape == x.shape
        return grads


    def loss(self, x, y, mask=None):
        """Compute the loss function || f(x)-y ||^2
        
        Parameters
        ----------
        x : numpy.ndarray
            The input sequence to the model.

        y : numpy.ndarray
            The output to calculate the loss w.r.t.

        mask : numpy.ndarray or None, optional
            A binary mask indicating the outputs to calculate the loss over.
            Default is None.

        Returns
        -------
        loss : float64
            The loss calculated between the provided output and the model 
            predictions.
        """

        if mask is None:
            mask = numpy.ones(y.shape, dtype='float64')

        pred_y = self._model.predict(x.astype('float64'))
        assert pred_y.shape == y.shape

        loss = numpy.sum(mask * numpy.square(pred_y - y)) / mask.sum()
        return loss

class Ledidi(object):
    """The Ledidi sequence designer.

    This object is a wrapper for a differentiable model and the hyperparameters
    of the optimization process. Any model object can be used as long as the
    functions `loss_gradient` and `loss` are implemented (see above).

    Ledidi optimizes the following objective function:

        min_x ||X - X_{0}||_{1} + lambda||f(X) - y_hat||_{2}^{2}

    where the first term is the sequence loss, i.e. the number of edits made to
    the sequence, and the second term is the output loss, i.e. the Euclidean
    distance between the output from the model and the given output.

    This objective is not easy to solve directly because X is discrete but
    Ledidi uses a Gumbel-softmax reparameterization to overcome this difficulty.

    Parameters
    ----------
    model : object
        A neural network or a wrapper for a neural network that has the
        `loss` and `loss_gradient` functions implemented. These models can come
        from any framework.

    tau : float64, positive, optional
        The temperature of the Gumbel-softmax reparameterization. High values
        force the continuous version of the sequence to approach the uniform
        distribution whereas low values force it to match the discrete
        distribution smoothly. Default is 3.

    l : float64, positive, optional
        The weight of the second term in the loss function. Setting l to a
        small value encourages making a small number of edits, whereas setting
        it to a large value encourages matching the given distribution more
        closely. Default is 10.

    max_iter : int, positive
        The maximum number of iterations to perform before ending optimization.
        Default is 100.

    lr : float, positive
        The learning rate, i.e. the step size when making updates. When this is
        small, more precise edits can be found, but optimization takes longer.
        Default is 1e-3.

    mask : numpy.ndarray, optional
        An array with the same shape as the output of the model that indicates
        which losses should be used in optimization, i.e. the outputs that
        the user cares to optimize over.

    early_stopping : int, optional
        The number of iterations with no improvement in the objective function
        to perform before ending optimization, i.e. the patience. Default is
        10.

    min_x : float, optional
        A parameter of the Gumbel-softmax distribution. Default is 0.01.

    max_x : float, optional
        A parameter of the Gumbel-softmax distribution. Default is 0.99.

    random_state : int or None, optional
        The seed to use for random calculations.

    verbose: bool, optional
        Whether to print logs associated with this object. Default is True.
    """ 

    def __init__(self, model, tau=3, l=10, max_iter=100, lr=1e-3, mask=None,
        early_stopping=-1, min_x=0.01, max_x=0.99, random_state=None,
        verbose=True):
        self.model = model
        self.tau = tau
        self.l = l
        self.max_iter = max_iter
        self.lr = lr
        self.early_stopping = early_stopping
        self.min_x = min_x
        self.max_x = max_x
        self.mask = mask
        self.random_state = numpy.random.RandomState(random_state)
        self.verbose = verbose

    def _from_x_to_w(self, x, tau, g, min_x=0.01, max_x=0.99):
        x = numpy.maximum(x, min_x)
        x = numpy.minimum(x, max_x)
        w = numpy.exp(numpy.log(x) * tau - g)
        w = numpy.maximum(w, MIN_W)
        return w

    def _from_w_to_x(self, w, tau, g):
        w = numpy.maximum(w, MIN_W)
        x = numpy.array((numpy.log(w) + g) / tau)[0]
        x = numpy.exp(x.T - logsumexp(x, axis=1)).T
        x = numpy.expand_dims(x, 0)
        return x

    def fit_transform(self, seq, epi_bar):
        seq = numpy.array(seq, ndmin=3)

        missing_indices = numpy.where(numpy.sum(seq[0], axis=1)<=0)[0]
        tau = self.tau

        if self.verbose:
            print('batch_missing_loc_indices={}'.format(missing_indices.shape[0]))

        g = -numpy.log(-numpy.log(self.random_state.uniform(MIN_W, 1, size=seq.shape)))
        curr_w = self._from_x_to_w(seq, tau, g, self.min_x, self.max_x)
        curr_x = self._from_w_to_x(curr_w, tau, g) 
        curr_x[0, missing_indices, :] = 0

        ref_x = seq.copy()

        curr_w_surrogate = curr_w.copy()
        curr_x_surrogate = curr_x.copy()

        best_total_loss = float("inf")
        best_total_loss_discrete = float("inf")
        best_seq = None
        early_stopping_iters = 0

        for i in range(self.max_iter):
            curr_x_discrete = numpy.zeros_like(curr_x, dtype=int)
            curr_x_discrete[0, numpy.arange(seq.shape[1]), numpy.argmax(curr_x[0], axis=1)] = 1
            curr_x_discrete[0, missing_indices, :] = 0

            seq_loss = numpy.sum(numpy.fabs(curr_x - ref_x))
            seq_loss_discrete = numpy.sum(numpy.abs(curr_x_discrete - seq)) / 2

            epi_loss = self.model.loss(curr_x, epi_bar, mask=self.mask)
            epi_loss_discrete = self.model.loss(curr_x_discrete, epi_bar, mask=self.mask)

            total_loss = seq_loss + self.l * epi_loss
            total_loss_discrete = seq_loss_discrete + self.l * epi_loss_discrete

            if self.verbose:
                print('iter={}\tseq_loss={:4.4}\tseq_loss_discrete={:4.4}\tepi_loss={:4.4}\tepi_loss_discrete={:4.4}\ttotal_loss={:4.4}\ttotal_loss_discrete:{:4.4}'.format(
                    i, seq_loss, seq_loss_discrete, epi_loss, epi_loss_discrete, total_loss, total_loss_discrete))

            loss_to_x_grad = self.model.loss_gradient(curr_x_surrogate, epi_bar, mask=self.mask)

            x_to_w_grad = (curr_x_surrogate - curr_x_surrogate*curr_x_surrogate) / tau
            x_to_ref_sgn = numpy.asarray((curr_x_surrogate - ref_x)>=0, dtype=float)
            x_to_ref_sgn[x_to_ref_sgn<=0]=-1
            loss_to_w_grad = (x_to_ref_sgn + self.l * loss_to_x_grad) * x_to_w_grad

            new_w = curr_w_surrogate - self.lr * loss_to_w_grad
            curr_w_surrogate = new_w + (1.0 * i / (i+2)) * (new_w - curr_w)
            curr_w = new_w

            g = -numpy.log(-numpy.log(self.random_state.uniform(MIN_W, 1, size=seq.shape)))
            curr_x = self._from_w_to_x(curr_w, tau, g)
            curr_x_surrogate = self._from_w_to_x(curr_w_surrogate, tau, g)

            curr_x[0, missing_indices, :] = 0
            curr_x_surrogate[0, missing_indices, :] = 0

            if total_loss_discrete < best_total_loss_discrete:
                best_total_loss_discrete = total_loss_discrete
                best_sequence = curr_x_discrete.copy()

            if total_loss < best_total_loss:
                best_total_loss = total_loss
                early_stopping_iters = 0 
            else:
                early_stopping_iters += 1

            if early_stopping_iters == self.early_stopping:
                break

        return best_sequence

