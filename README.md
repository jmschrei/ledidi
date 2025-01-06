# ledidi

[![PyPI Downloads](https://static.pepy.tech/badge/ledidi)](https://pepy.tech/projects/ledidi)

![image](https://github.com/user-attachments/assets/41ffe180-8171-4f28-a88e-41cc1c79985a)

> **Note**:
> ledidi has recently been rewritten in PyTorch. Please see the tutorials folder for examples on how to use the current version with your PyTorch models. Unfortunately, ledidi no longer supports TensorFlow models. 

ledidi is an approach for designing edits to biological sequences that induce desired properties. It does this by inverting the normal way that one thinks about machine learning models: normally during training, the data is held constant and model weights are updated; here, ledidi holds the model constant and updates the *data*. When using ledidi, the data are the sequences that are being edited and the "updates" to these sequences are the edits. Using this simple paradigm, ledidi *turns any trained machine learning model into a biological sequence editor*, regardless of the original purpose of the model. 

A challenge with designing edits for categorical sequences (e.g., DNA sequences with four possible nucleotides) is that the inputs are usually one-hot encoded but gradients are continuous. Applying the gradients directly would yield oddities like 1.2 of an A and -0.3 of a G. These values yield two problems: (1) such values are not biologically meaningful as you cannot design a sequence that is "1.2 A and -0.3 G" and (2) because most machine learning models have been trained on one-hot encoded data and are calibrated for that distribution, such sequences would likely confuse the model, push it off the training data manifold, and result in anomalous sequences even if one managed to turn the output back into a valid sequence.

ledidi resolves this challenge by phrasing edit design as an optimization problem over a continuous weight matrix `W` that gradients *can* be directly applied to. Specifically, at each iteration, ledidi draws one-hot encoded sequences one position at a time from a Gumbel-softmax distribution defined by `log(X + eps) + W` where `X` is the original one-hot encoded sequence and `eps` is a small value. When the drawn character is different from the character in `X`, likely because `W` has been updated at that position to prefer another character, an "edit" is being made. These edited sequences are then passed through the frozen pre-trained model and the output is compared to the desired output. The total loss is then derived from this output loss as well as an input loss that is the count of proposed edits, limiting the number that ledidi will propose. A gradient is calculated from this loss and `W` is updated, keeping `X` unchanged.

Although our examples right now are largely nucleotide sequence-based, one can also apply ledidi out-of-the-box to RNA or protein models (or really, to any model with a sequence of categorical inputs such as small molecules). The only limitation is what will fit in your GPU or what you are willing to wait for on your CPU!

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
