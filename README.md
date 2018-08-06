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
