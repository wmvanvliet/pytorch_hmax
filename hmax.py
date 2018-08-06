"""
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

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn

def gabor_filter(size, wavelength, orientation):
    """Create a single gabor filter.

    Parameters
    ----------
    size : int
        The size of the filter, measured in pixels. The filter is square, hence
        only a single number (either width or height) needs to be specified.
    wavelength : float
        The wavelength of the grating in the filter, relative to the half the
        size of the filter. For example, a wavelength of 2 will generate a
        Gabor filter with a grating that contains exactly one wave. This
        determines the "tightness" of the filter.
    orientation : float
        The orientation of the grating in the filter, in degrees.

    Returns
    -------
    filt : ndarray, shape (size, size)
        The filter weights.
    """
    lambda_ = size * 2. / wavelength
    sigma = lambda_ * 0.8
    gamma = 0.3  # spatial aspect ratio: 0.23 < gamma < 0.92
    theta = np.deg2rad(orientation + 90)

    # Generate Gabor filter
    x, y = np.mgrid[:size, :size] - (size // 2)
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)
    filt = np.exp(-(rotx**2 + gamma**2 * roty**2) / (2 * sigma ** 2))
    filt *= np.cos(2 * np.pi * rotx / lambda_)
    filt[np.sqrt(x**2 + y**2) > (size / 2)] = 0

    # Normalize the filter
    filt = filt - np.mean(filt)
    filt = filt / np.sqrt(np.sum(filt ** 2))

    return filt


class S1(nn.Module):
    """A layer of S1 units with different orientations but the same scale.

    The S1 units are at the bottom of the network. They are exposed to the raw
    pixel data of the image. Each S1 unit is a Gabor filter, which detects
    edges in a certain orientation. They are implemented as PyTorch Conv2d
    modules, where each channel is loaded with a Gabor filter in a specific
    orientation.

    Parameters
    ----------
    size : int
        The size of the filters, measured in pixels. The filters are square,
        hence only a single number (either width or height) needs to be
        specified.
    wavelength : float (default: 2)
        The wavelength of the grating in the filter, relative to the half the
        size of the filter. For example, a wavelength of 2 will generate a
        Gabor filter with a grating that contains exactly one wave. This
        determines the "tightness" of the filter.
    orientations : list of float
        The orientations of the Gabor filters, in degrees.
    """
    def __init__(self, size=7, wavelength=4., orientations=[90, -45, 0, 45]):
        super().__init__()
        self.num_orientations = len(orientations)

        # Use PyTorch's Conv2d as a base object. Each "channel" will be an
        # orientation.
        self.gabor = nn.Conv2d(1, self.num_orientations, size,
                               padding=size // 2, bias=False)

        # Fill the Conv2d filter weights with Gabor kernels: one for each
        # orientation
        for channel, orientation in enumerate(orientations):
            self.gabor.weight.data[channel, 0] = torch.Tensor(
                gabor_filter(size, wavelength, orientation))

        # A convolution layer filled with ones. This is used to normalize the
        # result in the forward method.
        self.uniform = nn.Conv2d(1, 4, size, padding=size // 2, bias=False)
        nn.init.constant_(self.uniform.weight, 1)

        # Since everything is pre-computed, no gradient is required
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, X):
        """Apply Gabor filters, take absolute value, and normalize."""
        out = torch.abs(self.gabor(X))
        norm = torch.sqrt(self.uniform(X ** 2))
        norm.data[norm == 0] = 1  # To avoid divide by zero
        out /= norm
        return out


class C1(nn.Module):
    """A layer of C1 units with different orientations but the same scale.

    Each C1 unit pools over the S1 units that are assigned to it.

    Parameters
    ----------
    s1_units : list of S1
        The S1 layers assigned to this C1 layer. Typically, each S1 layer has
        filters of a different scale.
    size : int
        Size of the MaxPool2d operation being performed by this C1 layer.
    stride : int
        The stride of the MaxPool2d operation being performed by this C1 layer.
        Defaults to half the size.
    """
    def __init__(self, s1_units, size, stride=None):
        super().__init__()
        self.num_orientations = s1_units[0].num_orientations
        self.s1_units = s1_units
        if stride is None:
            stride = size // 2
        self.local_pool = nn.MaxPool2d(size, stride=stride, padding=size // 2)

    def forward(self, X):
        """Max over scales, followed by a MaxPool2d operation."""
        s1_outputs = torch.cat([s1(X).unsqueeze(0) for s1 in self.s1_units], 0)

        # Pool over all scales
        s1_output, _ = torch.max(s1_outputs, dim=0)

        # Pool over local (c1_space x c1_space) neighbourhood
        return self.local_pool(s1_output)


class S2(nn.Module):
    """A layer of S2 units with different orientations but the same scale.

    The activation of these units is computed by taking the distance between
    the output of the C layer below and a set of predefined patches. This
    distance is computed as:

      d = sqrt( (w - p)^2 )
        = sqrt( w^2 - 2pw + p^2 )
    """
    def __init__(self, c1_units, patches):
        super().__init__()
        self.c1_units = c1_units

        num_patches, num_orientations, size, _ = patches.shape
        assert c1_units[0].num_orientations == num_orientations

        # Main convolution layer
        self.conv = nn.Conv2d(in_channels=num_orientations,
                              out_channels=num_orientations * num_patches,
                              kernel_size=size,
                              padding=size // 2,
                              groups=num_orientations,
                              bias=False)
        self.conv.weight.data = torch.Tensor(
            patches.transpose(1, 0, 2, 3).reshape(1600, 1, size, size))

        # A convolution layer filled with ones. This is used for the distance
        # computation
        self.uniform = nn.Conv2d(1, 1, size, padding=size // 2, bias=False)
        nn.init.constant_(self.uniform.weight, 1)

        # This is also used for the distance computation
        self.patches_sum_sq = nn.Parameter(
            torch.Tensor((patches ** 2).sum(axis=(1, 2, 3))))

        self.num_patches = num_patches
        self.num_orientations = num_orientations
        self.size = size

        # No gradient required for this layer
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, X):
        outputs = []
        for c1 in self.c1_units:
            c1_output = c1(X)
            s2_output = self.conv(c1_output)

            # Unstack the orientations
            s2_output_size = s2_output.shape[3]
            s2_output = s2_output.view(
                -1, self.num_orientations, self.num_patches, s2_output_size,
                s2_output_size)

            # Pool over orientations
            s2_output = s2_output.sum(dim=1)

            # Compute distance
            c1_sq = self.uniform(
                torch.sum(c1_output ** 2, dim=1, keepdim=True))
            dist = c1_sq - 2 * s2_output
            dist += self.patches_sum_sq[None, :, None, None]
            dist[dist < 0] = 0  # Negative values should never occur
            torch.sqrt_(dist)
            outputs.append(dist)
        return outputs


class C2(nn.Module):
    """A layer of C2 units operating on a layer of S2 units."""
    def __init__(self, s2_unit):
        super().__init__()
        self.s2_unit = s2_unit

    def forward(self, X):
        """Take the minimum value of the underlying S2 units."""
        s2_outputs = self.s2_unit(X)
        mins = [s2.min(dim=3)[0] for s2 in s2_outputs]
        mins = [m.min(dim=2)[0] for m in mins]
        mins = torch.cat([m[:, None, :] for m in mins], 1)
        return mins.min(dim=1)[0]


class HMAX(nn.Module):
    """The full HMAX model.

    Parameters
    ----------
    universal_patch_set : str
        Filename of the .mat file containing the universal patch set.
    """
    def __init__(self, universal_patch_set):
        super().__init__()

        # S1 layers, consisting of units with increasing size
        s1_sizes = np.arange(7, 39, 2)
        s1_wavelengths = np.arange(4, 3.15, -0.05)
        self.s1_units = [
            S1(size=size, wavelength=wavelength)
            for size, wavelength in zip(s1_sizes, s1_wavelengths)
        ]
        for i, s1 in enumerate(self.s1_units):
            self.add_module('s1_%d' % i, s1)

        # Each C1 layer pools across two S1 layers
        self.c1_units = [C1(self.s1_units[i:i + 2], size=i + 8)
                         for i in range(0, len(self.s1_units), 2)]
        for i, c1 in enumerate(self.c1_units):
            self.add_module('c1_%d' % i, c1)

        # Read the universal patch set for the S2 layer
        m = loadmat(universal_patch_set)
        patches = [patch.reshape(shape[[2, 1, 0, 3]]).transpose(3, 0, 2, 1)
                   for patch, shape in zip(m['patches'][0], m['patchSizes'].T)]

        # One S2 layer for each patch scale, operating on all C1 layers
        self.s2_units = [S2(self.c1_units, patches=patch)
                         for patch in patches]
        for i, s2 in enumerate(self.s2_units):
            self.add_module('s2_%d' % i, s2)

        # One C2 layer operating on each scale
        self.c2_units = [C2(s2) for s2 in self.s2_units]
        for i, c2 in enumerate(self.c2_units):
            self.add_module('c2_%d' % i, c2)

    def forward(self, X):
        """Run through everything and concatenate the output of the C2s."""
        c2_output = [c2(X)[:, :, None] for c2 in self.c2_units]
        return torch.cat(c2_output, 2)
