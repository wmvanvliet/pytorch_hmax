[![DOI](https://zenodo.org/badge/143711096.svg)](https://zenodo.org/badge/latestdoi/143711096)

PyTorch implementation of HMAX
==============================

PyTorch implementation of the HMAX model that closely follows that of the
MATLAB implementation of The Laboratory for Computational Cognitive
Neuroscience:

http://maxlab.neuro.georgetown.edu/hmax.html

The S and C units of the HMAX model can almost be mapped directly onto
TorchVision's Conv2d and MaxPool2d layers, where channels are used to store the
filters for different orientations. However, HMAX also implements multiple
scales, which doesn't map nicely onto the existing TorchVision functionality.
Therefore, each scale has its own Conv2d layer, which are executed in parallel.

Here is a schematic overview of the network architecture:

    layers consisting of units with increasing scale
    S1 S1 S1 S1 S1 S1 S1 S1 S1 S1 S1 S1 S1 S1 S1 S1
     \ /   \ /   \ /   \ /   \ /   \ /   \ /   \ /
      C1    C1    C1    C1    C1    C1    C1    C1
       \     \     \    |     /     /     /     /
               ALL-TO-ALL CONNECTIVITY
       /     /     /    |     \     \     \     \
      S2    S2    S2    S2    S2    S2    S2    S2
       |     |     |     |     |     |     |     |
      C2    C2    C2    C2    C2    C2    C2    C2


Installation
============

This script depends on the [NumPy, SciPy](https://www.scipy.org), [PyTorch and
TorchVision](https://pytorch.org) packages.


Clone the repository somewhere and run the `example.py` script:

    git clone https://github.com/wmvanvliet/pytorch_hmax
    python example.py


Usage
=====

See the `example.py` script on how to run the model on 10 example images.


Explanation of the output
=========================

The `hmax.get_all_layers` method returns a 4-tuple: `s1`, `c1`, `s2`, `c2`.
Here is a detailed explanation of the dimensions of each of these variables:

`s1`
----
These are the first simple units in the model, that perform a 2D convolution with Gabor filters. There are 4 Gabor filters, oriented at 90, -45, 0 and 45 degrees. Each filter is defined at 16 different scales. The `s1` variable is a list of length 16, containing the output at each scale. Each element is a NumPy array of shape `#images x #rotations x image_height x image_width` that is the result of the convolution operation.

`c1`
----
The output of the `s1` units is processed by the `c1` units, which perform a maxpool operation. This is done in 8 scales (pooling across a different number of pixels). The `c1` variable is alist of lengh 8, containing the output at each `s1` scale. Each element is a NumPy array of shape `#images x #rotations x height x width`.

`s2`
----
The output of the `c1` units is processed by the `s2` units, which perform 2D convolution again (not with Gabor filters this time, but pre-trained filters loaded from the `universal_patch_set.mat` file). This is done in 8 scales, operating on each of the 8 scales of the c1 output. The `s2` variable is a list of lengh 8, containing the output at each scale. Each element is again a list of length 8, matching the 8 scales of the `c1` units. The elements of this list are NumPy arrays of shape `#images x #filters x height x width` containing the convolution output.

`c2`
----
The output of the `s2` units is processed by the `c2` units, which perform a maxpool operation for each `s2` filter. The `c2` variable is a list of length 8, containing the output at each `s2` scale. Each element is a NumPy array of shape `#images x #filters` containing the result of the maxpool operation.
