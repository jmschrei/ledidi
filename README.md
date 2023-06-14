# ledidi

> **Note**:
> ledidi has recently been rewritten in PyTorch. Please see the tutorials folder for examples on how to use the current version with your PyTorch models. Unfortunately, ledidi no longer supports TensorFlow models. 

ledidi is an approach for designing edits to biological sequences that induce desired properties. It does this by inverting the normal way that people use machine learning models. Normally, one holds the data constant and uses gradients to update the model weights; ledidi holds the model constant and uses gradients to update the *data*. In this case, the data are sequence that edits are being designed for. Using this simple paradigm, ledidi *turns any machine learning model into a biological sequence editor*, regardless of the original purpose of the model. 

### Installation
`pip install ledidi`

### tl;dr
A challenge with designing edits for categorical sequences is that you cannot smoothly apply gradients to them as one could with continuous weight vectors -- when the sequence has to be one-hot encoded you can't pass in `[0.1, 0.02, 0.3, -0.1]`, for example. A wetlab would be very confused if told to construct that sequence. Further, most machine learning models applied to categorical sequences are calibrated towards expecting exactly a one-hot encoded input and may fall out-of-distribution in weird ways otherwise.

ledidi phrases the design problem as an optimization over a weight matrix. Specifically, given a one-hot encoded sequence `X`, some small `eps` value, ledidi will convert the sequence into logits `log(X + eps) + weights` and then use these logits to generate a new sequence given the Gumbel-softmax distribution. The new one-hot encoded sequence is then passed through the oracle model, gradients are calculated with respect to the desired model output, and the weight matrix is updated in such a way that negative values encourage the sampled one-hot encoded sequences to not take certain values at certain positions and positive values vice-versa.

Take a look at our [preprint](https://www.biorxiv.org/content/10.1101/2020.05.21.109686v1)!


### Example

![](ctcf_perturbation.png)

The input to Ledidi is a biological sequence and a desired output from the model (A, cyan) and the output is an edited sequence and the output from the paired predictive model (A, magenta). A hyperparameter in the optimization process, lambda, controls the number of sequence edits made (B) and the distance between the returned output and the desired output (C). 

In this example Ledidi is knocking out or knocking in CTCF binding. When CTCF is knocked out, Ledidi makes an average of ~5 edits per locus, and these edits occurs primarily at the most conserved positions in the CTCF motif (on both strands, D/E). Ledidi was able to knock out or knock in CTCF binding at almost all loci. 



sequence = numpy.load("CTCF/CTCF-seqs.npz")['arr_0'].astype('float32')[0].reshape(1, 131072, 4)
epi = model.predict(sequence)

desired_epi = epi.copy()
desired_epi[0, 487:537, 687] = 0.0

edited_sequence = mutator.fit_transform(sequence, desired_epi)
found_epi = model.predict(edited_sequence.astype('float32'))[0, :, 687]
```
