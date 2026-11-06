"""The lab's small helpers: devices, plotting, one-hot labels, softmax, accuracy, training steps.

Every helper runs on CUDA, Apple's MPS or the CPU, whichever is available. The helpers that used to
read notebook variables (`get_batch`, `gradient_step`, `make_plots`, `log_arch`, `backprop`) now
take those values as arguments, and fall back to the calling notebook's variables when they are
left out, so old notebooks keep working unchanged.
"""
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from skimage.util import montage as _sk_montage

__all__ = ['get_device', 'DEVICE', 'GPU', 'GPU_data', 'plot', 'montage', 'montage_plot', 'one_hot',
           'softmax', 'cross_entropy', 'randn_trunc', 'acc', 'get_batch', 'vizvec', 'gradient_step',
           'make_plots', 'log_arch', 'to_img', 'show_image', 'backprop', 'make_ani', 'scale']


# ------------------------------------------------------------------ devices

def get_device():
    """The best available device: CUDA, then Apple MPS, then CPU. Set MPCR_DEVICE (e.g. 'cpu') to choose."""
    if os.environ.get('MPCR_DEVICE'):
        return torch.device(os.environ['MPCR_DEVICE'])
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = get_device()


def _tensor(data, device):
    if isinstance(data, torch.Tensor):
        t = data.detach().to(dtype=torch.float32)
    else:
        t = torch.as_tensor(np.asarray(data, dtype=np.float32))
    return t.to(device if device is not None else DEVICE).clone()


def GPU(data, device=None):
    """A float32 tensor on the best device, with requires_grad=True (a weight)."""
    return _tensor(data, device).requires_grad_(True)


def GPU_data(data, device=None):
    """A float32 tensor on the best device, without gradients (data)."""
    return _tensor(data, device)


# ------------------------------------------------------------------ the caller's variables

def _caller(name, depth=2):
    """Look `name` up in the calling notebook: the caller's globals, then __main__."""
    frame = sys._getframe(depth)
    if name in frame.f_globals:
        return frame.f_globals[name]
    main = sys.modules.get('__main__')
    if main is not None and hasattr(main, name):
        return getattr(main, name)
    raise NameError(f'mpcr: pass {name!r} as an argument, or define it in your notebook first')


# ------------------------------------------------------------------ plotting

def _numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def plot(x, size=10, cmap='gray'):
    """Show an image (2-D, H x W x 3, or a 3 x H x W tensor) without axes."""
    x = _numpy(x)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    fig, ax = plt.subplots()
    ax.imshow(x, cmap=cmap)
    ax.axis('off')
    fig.set_size_inches(size, size)
    plt.show()


def montage(arr_in, *args, multichannel=None, **kwargs):
    """skimage's montage, also accepting the old `multichannel=True` (now `channel_axis=-1`)."""
    if multichannel is not None:
        kwargs.setdefault('channel_axis', -1 if multichannel else None)
    return _sk_montage(_numpy(arr_in), *args, **kwargs)


def montage_plot(x, size=10):
    """Tile a stack of images (N x H x W, or N x H x W x 3) with a one-pixel border and show it."""
    x = _numpy(x)
    if x.ndim == 4:
        x = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        plot(montage(x, channel_axis=-1), size)
    else:
        x = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
        plot(montage(x), size)


def vizvec(x):
    """Show a small matrix as coloured cells with each value written in."""
    x = _numpy(x)
    if x.ndim == 1:
        x = x[None, :]
    fig, ax = plt.subplots()
    ax.matshow(x, cmap=plt.cm.Blues)
    plt.axis('off')
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ax.text(j, i, str(x[i, j]), va='center', ha='center')


def to_img(x):
    """Map a tanh output in [-1, 1] to an image in [0, 1]."""
    return (0.5 * (x + 1)).clamp(0, 1)


def show_image(img):
    """Show a 3 x H x W tanh-range tensor as an image."""
    img = to_img(img.detach().cpu())
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))


def make_ani(A, interval=100):
    """Animate the frames A[0], A[1], ... (for Jupyter: `HTML(make_ani(A).to_jshtml())`)."""
    from matplotlib import animation
    A = _numpy(A)
    fig, ax = plt.subplots()
    im = ax.imshow(A[0])

    def animate(k):
        im.set_data(A[k])
        return (im,)
    return animation.FuncAnimation(fig, animate, frames=A.shape[0], interval=interval, repeat=True)


def scale(x, out_range=(0, 1)):
    """Linearly rescale x so its minimum and maximum land on out_range."""
    lo, hi = np.min(x), np.max(x)
    if hi == lo:
        return np.full_like(x, (out_range[0] + out_range[1]) / 2, dtype=float)
    y = (x - (hi + lo) / 2) / (hi - lo)
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


# ------------------------------------------------------------------ the maths

def one_hot(y, num_classes=10, device=None):
    """Labels 0..num_classes-1 as rows of zeros with a single one."""
    y = torch.as_tensor(_numpy(y)).long().reshape(-1)
    out = torch.zeros((y.shape[0], num_classes), device=device if device is not None else DEVICE)
    out[torch.arange(y.shape[0]), y.to(out.device)] = 1
    return out


def softmax(x, dim=1):
    """Exponentiate and normalise along `dim`; subtracting the maximum first keeps it from overflowing."""
    s = torch.exp(x - torch.max(x, dim, keepdim=True)[0])
    return s / s.sum(dim, keepdim=True)


def cross_entropy(outputs, labels):
    """Mean negative log probability of the right label (outputs are raw scores)."""
    shifted = outputs - torch.max(outputs, 1, keepdim=True)[0]
    log_p = shifted - torch.log(torch.exp(shifted).sum(1, keepdim=True))
    labels = torch.as_tensor(labels, device=outputs.device).long().reshape(-1)
    return -log_p[torch.arange(outputs.shape[0], device=outputs.device), labels].mean()


def randn_trunc(s, mu=0, sigma=0.1):
    """Normal random numbers cut off at two standard deviations."""
    return stats.truncnorm(-2, 2, loc=mu, scale=sigma).rvs(s)


def acc(out, y):
    """Fraction of rows whose largest score is the label."""
    with torch.no_grad():
        y = torch.as_tensor(y, device=out.device).reshape(-1)
        return (torch.argmax(out, 1) == y.long()).sum().item() / y.shape[0]


# ------------------------------------------------------------------ training-loop helpers

def get_batch(mode='train', b=None, X=None, Y=None, X_test=None, Y_test=None):
    """A random run of b consecutive examples from the training or test set.

    Anything left out is read from the notebook: `c.b`, `X`, `Y`, `X_test`, `Y_test`."""
    if b is None:
        b = _caller('c').b
    if mode == 'train':
        X = _caller('X') if X is None else X
        Y = _caller('Y') if Y is None else Y
    elif mode == 'test':
        X = _caller('X_test') if X_test is None else X_test
        Y = _caller('Y_test') if Y_test is None else Y_test
    else:
        raise ValueError("mode must be 'train' or 'test'")
    r = np.random.randint(X.shape[0] - b + 1)
    return X[r:r + b], Y[r:r + b]


def gradient_step(w, h=None):
    """w <- w - h * grad, then zero the gradients. h defaults to the notebook's `c.h`."""
    if h is None:
        h = _caller('c').h
    with torch.no_grad():
        for p in w:
            if p.grad is not None:
                p -= h * p.grad
                p.grad.zero_()


def make_plots(model=None, x=None, y=None, w=None, log=None, b=None, X_test=None, Y_test=None):
    """Training accuracy of model(x, w) on (x, y) and test accuracy on a random test batch, logged to
    wandb (or to `log`, any callable taking a dict). Anything left out is read from the notebook."""
    model = _caller('model') if model is None else model
    x = _caller('x') if x is None else x
    y = _caller('y') if y is None else y
    w = _caller('w') if w is None else w
    b = _caller('c').b if b is None else b
    X_test = _caller('X_test') if X_test is None else X_test
    Y_test = _caller('Y_test') if Y_test is None else Y_test
    acc_train = acc(model(x, w), y)
    xt, yt = get_batch('test', b=b, X_test=X_test, Y_test=Y_test)
    acc_test = acc(model(xt, w), yt)
    if log is None:
        import wandb
        log = wandb.log
    log({'acc_train': acc_train, 'acc_test': acc_test})
    return acc_train, acc_test


def log_arch(c=None):
    """Copy c.f_s[0..5] and c.f_n[0..6] into c.f_s0 ... c.f_n6 so wandb records each one."""
    c = _caller('c') if c is None else c
    for k, v in enumerate(c.f_s):
        setattr(c, f'f_s{k}', v)
    for k, v in enumerate(c.f_n):
        setattr(c, f'f_n{k}', v)


def backprop(model, loss=None, optimizer=None):
    """One optimiser step for the GAN's 'D' or 'G' (reads d_loss / d_optimizer or g_loss / g_optimizer)."""
    if model not in ('D', 'G'):
        raise ValueError("model must be 'D' or 'G'")
    p = model.lower()
    loss = _caller(f'{p}_loss') if loss is None else loss
    optimizer = _caller(f'{p}_optimizer') if optimizer is None else optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
