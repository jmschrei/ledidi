# Ledidi

Ledidi is an approach for designing edits to biological sequences that induce desired properties. A difficulty with this task is that biological sequences are discrete and so direct optimization can be difficult. Ledidi overcomes this challenge through using the Gumbel-softmax reparameterization trick to turn a discrete sequence input into a continuous representation where standard gradient descent methods can be applied easily. Ledidi differs from most current biological sequence design methods in that they generally design entire sequences that satisfy certain properties, whereas Ledidi designs compact sets of edits to given sequences that result in the desired outcome.

Currently, we've paired Ledidi with the [Basenji model](https://github.com/calico/basenji) and designed edits to the human genome that create CTCF binding, knock out CTCF binding, and induce cell-type specific binding of JUND. 

Read our paper here (not here yet :().

### Installation

You can install Ledidi with `pip install ledidi`.

### Usage

Ledidi can be paired with any differentiable model, but provides a wrapper for regression models that are implemented in TensorFlow. 
