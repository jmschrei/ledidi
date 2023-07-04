# ledidi

> **Note**:
> ledidi has recently been rewritten in PyTorch. Please see the tutorials folder for examples on how to use the current version with your PyTorch models. Unfortunately, ledidi no longer supports TensorFlow models. 

ledidi is an approach for designing edits to biological sequences that induce desired properties. It does this by inverting the normal way that people use machine learning models. Normally, one holds the data constant and uses gradients to update the model weights; ledidi holds the model constant and uses gradients to update the *data*. In this case, the data are sequence that edits are being designed for. Using this simple paradigm, ledidi *turns any machine learning model into a biological sequence editor*, regardless of the original purpose of the model. 

A challenge with designing edits for categorical sequences is that the inputs are usually one-hot encoded but gradients are continuous. This challenge poses two problems: (1) ledidi cannot directly apply gradients to edit an input sequence because doing so would produce something that is not one-hot encoded, and (2) most models have been trained on exclusively (or almost exclusively) one-hot encoded sequence and are calibrated for that distribution, and so passing in an edited sequence that is not one-hot encoded would fall off the manifold that the model expects and yield anomalous results.

ledidi resolves this challenge by phrasing edit design as an optimization problem over a weight matrix. Specifically, given a one-hot encoded sequence `X`, some small `eps` value, ledidi converts the one-hot encoded sequence into logits `log(X + eps) + weights` and then samples a one-hot encoded sequence from these logits assuming given the Gumbel-softmax distribution. The new one-hot encoded sequence is passed through the provided predictive model, gradients are calculated with respect to the difference between the actual and desired model output, and the `weights` matrix is updated in such a way that negative values encourage the sampled one-hot encoded sequences to not take certain values at certain positions and positive values vice-versa.

ledidi works on any sequence model in genomics. Although our examples right now are largely nucleotide sequence-based, one can apply ledidi out-of-the-box to RNA or protein models. The only limitation is what will fit in your GPU! So designing nucleotide edits to control gene expression using Enformer, or designing amino acid edits to control protein shape with AlphaFold2, may prove time-consuming.

Take a look at our [preprint](https://www.biorxiv.org/content/10.1101/2020.05.21.109686v1)!

### Installation
`pip install ledidi`

### Example: eliminating JunD binding using Beluga/DeepSEA

Using ledidi to insert a motif is as simple as wrapping your predictive model in the ledidi object and going brr! Using Beluga to design edits that knock out JunD binding given its predictions for JunD in HepG2 requires only a few lines of code and a few minutes (see tutorials folder for more).

```python
from ledidi import Ledidi

y_bar = torch.zeros(1, dtype=torch.float32, device='cuda') - 2

designer = Ledidi(model, X.shape[1:], target=309, tau=0.1, l=1, max_iter=20000, report_iter=1000).cuda()
X_hat = designer.fit_transform(X, y_bar)
```

Although there is stochasticity due to the sampling process, edits will usually reside at positions within the motifs that are predicted to completely knock out binding, causing the predicted logit to go from approx 1.2 to approx -1 (grey bar flanking the edits). 

![image](https://github.com/jmschrei/ledidi/assets/3916816/a81d814f-eea7-4738-b139-9a98744a736d)

A cool part of the procedure employed by ledidi is that after fitting the weight matrix you can quickly sample as many sequences as you want without needing to use the predictive model at all and manually screen those for other desirable properties!

```python
X_hat = torch.cat([designer(X) for i in range(100)])
```

![image](https://github.com/jmschrei/ledidi/assets/3916816/edb46b73-7db9-41d6-8f81-2d8179b7e255)

### Controlling design via the inputs

Sometimes, you have inputs that you do not want to touch. Maybe you're working with a complex locus with important binding sites that cannot be touched. We've got you. You can pass in a binary mask that prevents edits from being proposed at certain positions. Here, we want to minimize JunD binding but only allow edits in the first half of the shown sequence window. 

```python
y_bar = torch.zeros(1, dtype=torch.float32, device='cuda') - 1
input_mask = torch.ones(2000, dtype=torch.bool, device='cuda')
input_mask[950:990] = False

designer = Ledidi(model, X.shape[1:], target=309, tau=0.1, l=1, input_mask=input_mask,
                  max_iter=10000, report_iter=1000).cuda()
X_hat = designer.fit_transform(X, y_bar)
```

![image](https://github.com/jmschrei/ledidi/assets/3916816/6d8c2a16-7f07-4a08-80ae-8c3d9ec4cb52)

Now, we do the same but only allowing edits in the second half of the shown sequence.

```python
y_bar = torch.zeros(1, dtype=torch.float32, device='cuda') - 1
input_mask = torch.ones(2000, dtype=torch.bool, device='cuda')
input_mask[990:1030] = False

designer = Ledidi(model, X.shape[1:], target=309, tau=0.1, l=1, input_mask=input_mask,
                  max_iter=10000, report_iter=1000).cuda()

X_hat = designer.fit_transform(X, y_bar)
```

![image](https://github.com/jmschrei/ledidi/assets/3916816/0e455622-ac34-4b7e-b195-1aa2fd8215e7)

We can see this in a more powerful formulation when using BPNet to design edits that slightly reduce SPI1 signal (from 8.6 to 7), but do not totally eliminate it, from an array of SPI1 binding sites. Maybe the maximum occupancy has been reached. We only allow ledidi to make edits in the second half of the sequence and see that, rather than simply eliminating all binding like a wrecking ball, it precisely knocks out one motif, weakens a second, and leaves the third one entirely alone. 

![image](https://github.com/jmschrei/ledidi/assets/3916816/71bd9ced-8515-4dcc-9352-491604a5b6ff)

But that wasn't the only proposed solution. Sampling edits from ledidi allow us to see a spectrum of sequences with differing numbers of edits and predicted counts.

![image](https://github.com/jmschrei/ledidi/assets/3916816/45c4cab5-eb41-4341-ab50-6c433f85516c)

### Controlled design via the outputs

There are tons of interesting ways to control design using the model outputs. Here, using the same BPNet model trained on SPI1 signal, we can ask it to design edits that cause strong binding in a certain portion of the profile. Let's start off by trying to maximize binding in the slice of 200-400 bp.

```python
class ProfileWrapper(torch.nn.Module):
    def __init__(self, model, start=0, end=-1):
        super().__init__()
        self.model = model
        self.start = start
        self.end = end
    
    def forward(self, X):
        X_ctl = torch.zeros(X.shape[0], 2, X.shape[-1], dtype=X.dtype, device=X.device)
        
        y = self.model(X, X_ctl)[0]
        y = torch.nn.functional.softmax(y.reshape(y.shape[0], -1), dim=-1).reshape(*y.shape)
        y = y[:, :, self.start:self.end].sum(dim=(1, 2))
        return y.unsqueeze(1)

wrapper = ProfileWrapper(model, 200, 400)
y_bar = torch.zeros(1, dtype=torch.float32, device='cuda') + 1

designer = Ledidi(wrapper, X.shape[1:], tau=0.1, l=10, max_iter=10000, report_iter=1000).cuda()
X_hat = designer.fit_transform(X, y_bar)
```

![image](https://github.com/jmschrei/ledidi/assets/3916816/cdf1bc0c-9307-44ae-9fc9-c492a9bc66e2)

Now, let's use the same wrapper to maximize binding in the slice between 600-800 bp.

![image](https://github.com/jmschrei/ledidi/assets/3916816/f2afc4bd-beca-48fa-8d54-96a8bc22ce7b)

If you don't have basepair resolution outputs, don't worry! There's still lots of cool stuff one can try.

- Given a multi-task model, alter predictions for some, but not all tasks, e.g. increase OCT-SOX binding but do not decrease NANOG binding, or maintain SOX binding while eliminating OCT binding to disrupt OCT-SOX binding sites
- Use MULTIPLE MODELS to design edits. Ledidi doesn't have to operate on only one model! If you have multiple BPNet models you can simply run the same sequence through both models and get a gradient.
- Replace or augment the learned weight matrix with a neural network that takes in some manner of co-factors, conditions, or prior knowledge that can guide the design process

### Usage hints

There are only a few parameters to be concerned about when using ledidi.

The first is specifying a target if you have a multitask model. This parameter works the same way as it does in Captum, where you can pass in an integer, a list of targets, or just None if you want everything to be returned. The output from the model, after slicing using the target, is then passed into the `output_loss` which defaults to `torch.nn.MSELoss`. 

The next is `tau`. This controls the "sharpness" of the generation. When `tau` is 1, one-hot encoded sequences are generated from an underlying matrix of logits according to the categorical distribution implied by the logits. As `tau` approaches 0, the one-hot encoded sequences lose diversity and are generated by a distribution that more closely resembles that implied by the argmax of the logits. As `tau` approaches infinity, the generated sequences have more diversity and the distribution more closely resembles the uniform distribution. As a practical matter: as `tau` increases there will be more random edits per sampled sequence. So, if you have a particularly long sequence or notice that there are too many edits being made initially, you may want to start with a lower `tau`. Also, when your model has worse attributions or other underlying issues, you may want to have a lower `tau`.

The last is `l`, which is the mixing parameter between the input loss and the output loss. This has to be set entirely based on your goals and you range of your signal. You should increase it when you want to achieve a desired objective and do not care about the number of edits you make and decrease it when you care much more about only making a small number of edits even when they do not completely achieve the objective.

You may ask: don't `tau` and `l` do the same thing, i.e., control the number of edits per sequence? The answer is a resounding "sort of." `tau` controls the number of edits made per sequence in the <i>generation</i> strp whereas `l` controls the proportion of the <i>loss</i> made up by those edits. Basically, because there is an element of randomness in the generation step, `tau` controls that randomness as sort of a "temperature" parameter whereas `l` controls the learning. As a concrete example: after Ledidi is done learning the underlying design matrix one can continue to generate new sequences and those sequences can be controlled by changing `tau` but `l` will not effect them.
