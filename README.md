# mpcr

Everything used in the MPCR lab, in one import.

```python
!pip install git+https://github.com/williamedwardhahn/mpcr
from mpcr import *
```

That gives a notebook the usual scientific stack under the usual names, plus the lab's small helpers.
It works on Colab, in Jupyter on your own machine, and in plain Python scripts, and it uses an
NVIDIA GPU, an Apple-silicon GPU or the CPU, whichever you have.

## What you get

| | names |
|---|---|
| arrays and maths | `np`, `LA` (numpy.linalg), `scipy`, `stats`, `signal`, `math`, `random`, `randint` |
| data and plots | `pd`, `plt`, `pylab`, `sns`, `Image` (PIL), `Axes3D`, `animation`, `HTML` |
| images | `imageio`, `imread`, `imwrite`, `mimsave`, `io` and `skio` (skimage.io), `resize`, `montage`, `rgb2hsv`, `view_as_windows`, `view_as_blocks` |
| PyTorch | `torch`, `nn`, `F`, `optim`, `lr_scheduler`, `Variable`, `Dataset`, `DataLoader`, `TensorDataset`, `random_split`, `matmul`, and every function in `torch.nn.functional` (`conv2d`, `relu`, `linear`, `max_pool2d`, ...) |
| torchvision | `torchvision`, `datasets`, `transforms`, `models`, `MNIST`, `save_image`, `normalize`, and every model constructor (`alexnet`, `resnet18`, `vgg16`, ...) |
| scikit-learn | `PCA`, `LR` (LogisticRegression), `confusion_matrix`, `roc_auc_score` |
| tools | `summary` (torchsummary), `wb` (wandb), `Backprop` and `apply_transforms` (flashtorch saliency maps), `Bar` (barbar progress bars), `cv2`, `nltk`, `drive` (Google Drive on Colab) |

wandb, nltk, cv2, flashtorch and barbar are imported the first time you use them, so `import mpcr`
stays quick (about 3 seconds, almost all of it PyTorch).

### The lab's helpers

| helper | what it does |
|---|---|
| `DEVICE`, `get_device()` | the best device: CUDA, then Apple MPS, then CPU. Set `MPCR_DEVICE=cpu` to choose one. |
| `GPU(x)`, `GPU_data(x)` | `x` as a float32 tensor on `DEVICE`: a weight (with gradients) or data (without) |
| `plot(x)` | show an image or tensor without axes (2-D, H×W×3, or 3×H×W) |
| `montage_plot(x)` | tile a stack of images with a thin border and show it |
| `montage(x)` | skimage's montage; also accepts the old `multichannel=True` |
| `vizvec(x)` | a small matrix drawn as coloured cells with the numbers written in |
| `one_hot(y)` | labels as rows of zeros with one 1 (10 classes unless you say otherwise) |
| `softmax(x)`, `cross_entropy(out, y)` | the textbook versions, written out, numerically safe |
| `acc(out, y)` | fraction of rows whose largest score is the label |
| `randn_trunc(shape)` | normal random numbers (σ = 0.1) cut off at two standard deviations |
| `scale(x)` | rescale so the minimum is 0 and the maximum is 1 |
| `get_batch('train')`, `get_batch('test')` | a random run of `c.b` examples from `X, Y` or `X_test, Y_test` |
| `gradient_step(w)` | `w ← w − c.h · grad` for every weight, then zero the gradients |
| `make_plots()` | log training and test accuracy to wandb |
| `log_arch()` | copy `c.f_s` and `c.f_n` into separate wandb config entries |
| `backprop('D')`, `backprop('G')` | one optimiser step for a GAN's discriminator or generator |
| `to_img(x)`, `show_image(x)` | show a tanh-range (−1 to 1) image tensor |
| `make_ani(A)` | animate a stack of frames |
| `Model`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, ..., `drawnet(...)` | draw a network's layers as SVG or PDF (from yu4u's convnet-drawer) |
| `tokenize`, `stem`, `bag_of_words` | bag-of-words features for a small chatbot |
| `get_uniprot_data(kw1, kw2, n)`, `process_strings(seqs)` | protein sequences from UniProt, as numbers |

`get_batch`, `gradient_step`, `make_plots`, `log_arch` and `backprop` read your notebook's
variables (`c`, `X`, `Y`, `X_test`, `Y_test`, `model`, `w`, `d_loss`, ...) as they always meant to.
You can also pass them in: `get_batch('train', b=64, X=X, Y=Y)`, `gradient_step(w, h=0.01)`.

## Upgrading from version 0

Notebooks written for version 0 run unchanged: every name it exported is still exported, and the
same ones win where two libraries share a name (`softmax`, `one_hot` and `cross_entropy` are the
lab's; `random` is the function; `io` is `skimage.io`). What changed:

- **Nothing is installed at import time.** pip installs the dependencies once; version 0 ran
  `pip install` three times and downloaded NLTK data on every import.
- **It works outside Colab.** `drive` is Google Drive on Colab and a stand-in that warns elsewhere.
- **No CUDA needed.** `GPU` and `GPU_data` use whatever device you have.
- **Fixed:** helpers defined twice, helpers that could never see your notebook's variables, the
  UniProt address (UniProt retired it), `montage(..., multichannel=True)` on new scikit-image,
  and an unstable `cross_entropy`.

See [CHANGELOG.md](CHANGELOG.md) for the details.

## Developing

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/pytest                    # add -m "not network" to skip UniProt and NLTK downloads
```

MIT licence. The network drawer is adapted from [convnet-drawer](https://github.com/yu4u/convnet-drawer)
by Yusuke Uchida (MIT).
