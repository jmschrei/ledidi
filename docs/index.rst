.. ledidi documentation master file, created by
   sphinx-quickstart on Tue Feb 20 13:46:59 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


    .. image:: logo/pomegranate-logo.png
        :width: 300px


    .. image:: https://readthedocs.org/projects/pomegranate/badge/?version=latest
       :target: http://pomegranate.readthedocs.io/en/latest/?badge=latest


ledidi
==========



Ledidi turns any pre-trained machine learning model(s) into an editor of biological sequences to make them exhibit desired characteristics. Ledidi is fast and programmable, meaning that precise characteristics can be induced by simultaneously using several models for design. For example, Ledidi can use one model that predicts chromatin accessibility and a second model that predicts the binding of CTCF to design sites that are accessible due to high binding of CTCF, or are accessible and explicitly do not have CTCF binding to them. Ledidi can design cell type-specific regions by using models that each predict activity in a different cell line. Essentially, you can use Ledidi to design anything that you can find one or more machine learning models to make predictions for.

Ledidi works by phrasing the design process as a continuous optimization problem and then solving this problem using off-the-shelf techniques. Specifically, Ledidi calculates a loss that in comprised of an output loss, measuring how closely the edited sequence matches the desired characteristics as predicted by the machine learning models, and an input loss, measuring the number of edits made thus far. By minimizing this loss, Ledidi designs sequences that closely match the desired output using as few edits as possible. A technical challenge is that the gradient of this loss cannot be directly applied to a categorical sequence, such as DNA. Ledidi circumvents this challenge by learning a continuous weight matrix from which edits to an initial sequence are sampled, and updates this weight matrix using the straight-through estimator at each step. This trick allows the full information in the gradient to be used while also only passing categorical sequences into the machine learning oracle models, which have likely only been trained on categorical sequences.

Here, we have included documentation for how we used Ledidi in the paper alongside additional examples showing Ledidi in practice. We will be updating this repository with cool use-cases we find.



Installation
============

`pip install ledidi`



.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   self
   whats_new.rst


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorials

   tutorials/Tutorial_1_-_Design_of_Protein_Binding_Sites.ipynb
   tutorials/Tutorial_2_-_Constraints_and_Priors.ipynb
   tutorials/Tutorial_3_-_In-Painting.ipynb
   tutorials/Tutorial_4_-_Multiple_Models.ipynb
   tutorials/Tutorial_5_-_Affinity_Catalogs.ipynb
   tutorials/Tutorial_6_-_Custom_Loss_Functions.ipynb



.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API

   api/ledidi.rst
   api/pruning.rst
   api/plot.rst
   api/losses.rst
   api/wrappers.rst

