"""Everything used in the MPCR lab, in one import:

    !pip install git+https://github.com/williamedwardhahn/mpcr
    from mpcr import *

gives a notebook numpy, torch, torchvision, matplotlib, scipy, scikit-image, scikit-learn, pandas and
seaborn under their usual names, plus the lab's helpers (plot, montage_plot, GPU, GPU_data, one_hot,
softmax, cross_entropy, acc, ...). Version 1 keeps every name version 0 provided, but installs
nothing at import time, works outside Colab, and picks CUDA, Apple MPS or the CPU automatically.
wandb (`wb`), nltk, cv2, flashtorch and barbar are imported the first time you use them.
"""
__version__ = '1.0.0'

# ------------------------------------------------------------------ the standard library
import os
import sys
import time
import copy
import json
import math
import random
import itertools
import subprocess
import urllib.request
from urllib.request import Request, urlopen
from abc import ABCMeta, abstractmethod
import warnings as _warnings

# ------------------------------------------------------------------ the scientific stack
import numpy as np
from numpy import linalg as LA
import pandas as pd
import requests
import scipy
from scipy import stats
from scipy import signal
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
with _warnings.catch_warnings():
    _warnings.simplefilter('ignore')
    import pylab
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.axes_rgb import make_rgb_axes, RGBAxes
import seaborn as sns

import imageio
from imageio import *                                       # imopen, imiter, v2, v3, ... as in version 0
from imageio.v2 import *                                    # imread, imwrite, ... without the v3 warning
try:
    from imageio.v2 import imsave, mimsave
except ImportError:
    from imageio.v2 import imwrite as imsave, mimwrite as mimsave

import skimage
from skimage import io
from skimage import io as skio
from skimage.transform import resize
from skimage.color import rgb2hsv
from skimage.util import view_as_windows, view_as_blocks

from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression as LR

# ------------------------------------------------------------------ PyTorch
import torch
from torch import matmul
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

import torchvision
import torchvision.utils
from torchvision import datasets, models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torchvision.models import *                            # alexnet, resnet18, vgg16, ...
from torch.nn.functional import *                           # conv2d, relu, linear, max_pool2d, ...
from torchvision.transforms.functional import normalize

# version 0 imported `random` the module and then `random` the function; notebooks saw the function
from random import random, randint
from bisect import bisect

try:
    from torchsummary import summary
except ImportError:
    def summary(model, input_size, *args, **kwargs):
        raise ImportError('mpcr: torchsummary is not installed (pip install torchsummary)')

# ------------------------------------------------------------------ libraries loaded on first use
from ._lazy import LazyModule, LazyAttr

wb = LazyModule('wandb')
nltk = LazyModule('nltk')
cv2 = LazyModule('cv2', 'pip install opencv-python-headless')
Backprop = LazyAttr('flashtorch.saliency', 'Backprop')
apply_transforms = LazyAttr('flashtorch.utils', 'apply_transforms')
Bar = LazyAttr('barbar', 'Bar')

try:
    from IPython.display import HTML
except ImportError:
    HTML = LazyAttr('IPython.display', 'HTML')

# Google Drive: the real thing on Colab, a harmless stand-in elsewhere
try:
    from google.colab import drive
except ImportError:
    class _NoDrive:
        """Outside Colab there is no Google Drive to mount; your files are already on disk."""

        def mount(self, path='/content/drive', *args, **kwargs):
            _warnings.warn(f'mpcr: not running on Colab, so drive.mount({path!r}) does nothing', stacklevel=2)

        def flush_and_unmount(self, *args, **kwargs):
            pass

        def __repr__(self):
            return '<google.colab.drive stand-in (not on Colab)>'
    drive = _NoDrive()

# ------------------------------------------------------------------ the lab's helpers (these win over
# torch.nn.functional's softmax, cross_entropy and one_hot, as in version 0)
from .core import *
from .nlp import *
from .drawnet import *
from .bio import *

# `from mpcr import *` exports every public name above, but not the submodules themselves
__all__ = sorted(n for n, v in list(globals().items())
                 if not n.startswith('_') and n not in ('core', 'nlp', 'bio')
                 and not (n == 'drawnet' and not callable(v)))
