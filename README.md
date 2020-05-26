# Ledidi

Ledidi is an approach for designing edits to biological sequences that induce desired properties. A difficulty with this task is that biological sequences are discrete and so direct optimization can be difficult. Ledidi overcomes this challenge through using the Gumbel-softmax reparameterization trick to turn a discrete sequence input into a continuous representation where standard gradient descent methods can be applied easily. Ledidi differs from most current biological sequence design methods in that they generally design entire sequences that satisfy certain properties, whereas Ledidi designs compact sets of edits to given sequences that result in the desired outcome.

Currently, we've paired Ledidi with the [Basenji model](https://github.com/calico/basenji) and designed edits to the human genome that create CTCF binding, knock out CTCF binding, and induce cell-type specific binding of JUND. 

Take a look at our [preprint](https://www.biorxiv.org/content/10.1101/2020.05.21.109686v1)!

### Installation

You can install Ledidi with `pip install ledidi`.

### Example

![](ctcf_perturbation.png)

The input to Ledidi is a biological sequence and a desired output from the model (A, cyan) and the output is an edited sequence and the output from the paired predictive model (A, magenta). A hyperparameter in the optimization process, lambda, controls the number of sequence edits made (B) and the distance between the returned output and the desired output (C). 

In this example Ledidi is knocking out or knocking in CTCF binding. When CTCF is knocked out, Ledidi makes an average of ~5 edits per locus, and these edits occurs primarily at the most conserved positions in the CTCF motif (on both strands, D/E). Ledidi was able to knock out or knock in CTCF binding at almost all loci. 

### Usage

Ledidi can be paired with any differentiable model, but provides a wrapper for regression models that are implemented in TensorFlow. 
