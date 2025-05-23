# Ledidi

[![PyPI Downloads](https://static.pepy.tech/badge/Ledidi)](https://pepy.tech/projects/Ledidi)

<img src="https://github.com/user-attachments/assets/500e5f18-f9af-4cc9-b76c-23296d60640d" width="40%" height="40%">

[[preprint](https://www.biorxiv.org/content/10.1101/2025.04.22.650035v1)][[docs](https://ledidi.readthedocs.io/en/latest/)]

> **Note**:
> Ledidi has recently been rewritten in PyTorch. Please see the tutorials folder for examples on how to use the current version with your PyTorch models. Unfortunately, Ledidi no longer supports TensorFlow models. 

Ledidi is an approach for designing edits to biological sequences that induce desired properties. It does this by inverting the normal way that one thinks about machine learning models: normally during training, the data is held constant and model weights are updated; here, Ledidi holds the model constant and updates the *data*. When using Ledidi, the data are the sequences that are being edited and the "updates" to these sequences are the edits. Using this simple paradigm, Ledidi *turns any trained machine learning model into a biological sequence editor*, regardless of the original purpose of the model. 

A challenge with designing edits for categorical sequences (e.g., DNA sequences with four possible nucleotides) is that the inputs are usually one-hot encoded but gradients are continuous. Applying the gradients directly would yield oddities like 1.2 of an A and -0.3 of a G. These values yield two problems: (1) such values are not biologically meaningful as you cannot design a sequence that is "1.2 A and -0.3 G" and (2) because most machine learning models have been trained on one-hot encoded data and are calibrated for that distribution, such sequences would likely confuse the model, push it off the training data manifold, and result in anomalous sequences even if one managed to turn the output back into a valid sequence.

Ledidi resolves this challenge by phrasing edit design as an optimization problem over a continuous weight matrix `W` that gradients *can* be directly applied to. Specifically, at each iteration, Ledidi draws one-hot encoded sequences one position at a time from a Gumbel-softmax distribution defined by `log(X + eps) + W` where `X` is the original one-hot encoded sequence and `eps` is a small value. When the drawn character is different from the character in `X`, likely because `W` has been updated at that position to prefer another character, an "edit" is being made. These edited sequences are then passed through the frozen pre-trained model and the output is compared to the desired output. The total loss is then derived from this output loss as well as an input loss that is the count of proposed edits, limiting the number that Ledidi will propose. A gradient is calculated from this loss and `W` is updated, keeping `X` unchanged.

![image](https://github.com/user-attachments/assets/41ffe180-8171-4f28-a88e-41cc1c79985a)

Applied out-of-the-box across dozens of sequence-based ML models, Ledidi can design edits turning a sequence that do not exhibit some form of activity (e.g., TF binding, transcription, etc) and turn it into one exhibiting strong activity! Despite major differences in these models, this design does not require any hyperparameter tuning to get good results, and only requires balancing the input and output loss in a simple way to get the best results.

<img width="1367" alt="image" src="https://github.com/user-attachments/assets/39edd8fc-be58-4fb3-8006-18401fc81336" />

Although our examples right now are largely nucleotide sequence-based, one can also apply Ledidi out-of-the-box to RNA or protein models (or really, to any model with a sequence of categorical inputs such as small molecules). The only limitation is what will fit in your GPU or what you are willing to wait for on your CPU!

### Installation
`pip install ledidi`

### Usage

Please see the [documentation site](https://ledidi.readthedocs.io/en/latest/) for more complete tutorials on how to use Ledidi. You can find some example BPNet models -- including the one used in the tutorials -- at https://zenodo.org/records/14604495.

At a high level, Ledidi requires a a sequence-based machine learning model, an initial sequence, and a desired output from the model. If the model is multi-task, this desired output does not need to cover all of the tasks and can even just use a single one (see this tutorial on [PyTorch wrappers](https://tangermeme.readthedocs.io/en/latest/vignettes/Wrappers_are_Productivity_Hacks.html) for practical advice on how to manipulate models to make them easier to work with Ledidi). These three items are then passed into `ledidi` for design! Note that there is also a `Ledidi` object which is an internal wrapper. No need to use that yet.

```python
import torch
from bpnetlite.bpnet import ControlWrapper
from bpnetlite.bpnet import CountWrapper
from Ledidi import ledidi

X = .. your sequence ..

model = torch.load("GATA2.torch", weights_only=False)  # Load a BPNet model predicting GATA2 binding
model = CountWrapper(ControlWrapper(model))  # Use only the count predictions

y_bar = torch.zeros(1, dtype=torch.float32) + 4.5  # Set a desired output of a high value
X_hat = ledidi(model, X, y_bar)  # Run Ledidi!
```

The `ledidi` method will return a batch of sequences whose size can be controlled with the `batch_size` parameter. These sequences are independently generated by sampling from the underlying Gumbel-softmax distributions learned by Ledidi. Importantly, each batch of edits will be correlated in the sense that they all come from the single set of Ledidi weights. So, if a motif could have been substituted in at multiple locations Ledidi will end up only choosing one and these edits will all be drawn from that choice. <b>If you need many different sets of edits you will need to run Ledidi many times with different random seeds</b>. However, an upside is that once the Ledidi weight matrix is learned, edited sequences can be sampled extremely quickly using the forward function.

```
X_hat = ledidi(model, X, y_bar, n_samples=10000)  # Runs almost as fast as the default!
```

If we take a look at the edits being proposed in this sequence, we can see that frequently Ledidi can make designs with fewer edits than the length of the motif that should be inserted. Essentially, Ledidi automatically finds locations on the sequence that are close to the motifs that need to be inserted and precisely makes edits there rather than forcing an entire motif in somewhere in the sequence. Here we can see attributions from our GATA prediction model before edits are made (low because GATA does not bind to this example region, see the tutorial), after edits are made with the edited positions shown in orange, and after a greedy pruning algorithm is used to trim the edit space down in magenta.

<img src="https://github.com/user-attachments/assets/6ea64ed7-bfad-479b-a433-116ad7465ba6" width="70%" height="70%">

When `verbose` is set to `True` you will get logs showing the input and output losses, and the total loss from combining them with the mixing weight. The logs may look something like this:

![image](https://github.com/user-attachments/assets/35040f80-a997-46f2-9855-a4db1a8337b6)

This means that by iteration 100, an average of 35.75 edits were proposed per sequence, and that the best iteration at convergence only had 28.19 edits per sequence on average. The output loss went from 16 to 0.18, which is a pretty major drop in the loss. If you pass in `return_history=True` you can get these losses over the course of the entire process. Here is an example from the tutorial.

<img src="https://github.com/user-attachments/assets/e9801dbc-1821-476a-9b1b-1962c0552f29" width="60%" height="60%">

You can also look at where edits were proposed across the entire process. This gives you a sense for how long edits persisted across the process. In this example, tons of edits are proposed initially and then many of them are undone because the input loss tries to minimize this number.

<img src="https://github.com/user-attachments/assets/f115605e-cc26-4ba8-b3ed-ff0b336f06b4" width="60%" height="60%">
