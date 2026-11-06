"""Tests for mpcr 1.x. Run with `pytest` (add `-m "not network"` to skip the ones that need the internet)."""
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPCR_DEVICE', 'cpu')

import numpy as np
import pytest
import torch
import torch.nn.functional as TF

import mpcr
from mpcr import *          # noqa: F401,F403  (the way notebooks use it)

HERE = Path(__file__).parent


# ------------------------------------------------------------------ importing

def test_import_installs_nothing_and_needs_no_colab():
    """A fresh interpreter imports mpcr without calling pip, downloading NLTK data, or google.colab."""
    code = (
        'import subprocess, sys\n'
        'calls = []\n'
        'def spy(real):\n'
        '    def f(*a, **k):\n'
        '        if "pip" in str(a):\n'
        '            calls.append(a)\n'
        '            return 0\n'
        '        return real(*a, **k)\n'
        '    return f\n'
        'subprocess.check_call = spy(subprocess.check_call)\n'
        'subprocess.run = spy(subprocess.run)\n'
        'import nltk\n'
        'nltk.download = lambda *a, **k: calls.append(("nltk.download",) + a)\n'
        'from mpcr import *\n'
        'assert not calls, calls\n'
        'assert "google.colab" not in sys.modules\n'
        'for heavy in ("wandb", "flashtorch", "barbar", "cv2"):\n'
        '    assert heavy not in sys.modules, heavy\n'
        'print("ok")\n'
    )
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith('ok')


def test_every_version_0_name_is_still_there():
    """Every name `from mpcr import *` gave in version 0 (tests/v0_names.txt) is still exported,
    except names a newer library itself no longer has."""
    import imageio
    import torchvision.models as tvm
    have = set(mpcr.__all__)
    missing = []
    for name in (HERE / 'v0_names.txt').read_text().split():
        if name in have:
            continue
        if any(hasattr(m, name) for m in (TF, tvm, imageio)):
            missing.append(name)
    assert not missing, missing


def test_names_the_course_notebooks_use():
    for name in ['np', 'torch', 'nn', 'optim', 'F', 'plt', 'datasets', 'transforms', 'models', 'alexnet',
                 'torchvision', 'io', 'skio', 'imread', 'imageio', 'stats', 'signal', 'montage', 'resize',
                 'view_as_windows', 'lr_scheduler', 'Variable', 'conv2d', 'conv_transpose2d', 'linear', 'relu',
                 'binary_cross_entropy', 'wb', 'drive', 'os', 'summary', 'DataLoader', 'TensorDataset',
                 'Backprop', 'apply_transforms', 'GPU', 'GPU_data', 'plot', 'montage_plot', 'one_hot']:
        assert name in mpcr.__all__, name


def test_version_0_quirks_are_kept():
    import random as random_module
    assert random is random_module.random          # `random` is the function, as in version 0
    assert io.__name__ == 'skimage.io' and skio is io
    assert softmax is mpcr.core.softmax and one_hot is mpcr.core.one_hot
    assert 'core' not in mpcr.__all__ and callable(drawnet)


def test_drive_stand_in_outside_colab():
    with pytest.warns(UserWarning, match='not running on Colab'):
        drive.mount('/content/drive')


def test_lazy_modules_load_on_first_use():
    assert 'not loaded' in repr(wb) or 'loaded' in repr(wb)
    assert hasattr(nltk, 'word_tokenize')
    assert Bar is not None and callable(Bar)


# ------------------------------------------------------------------ devices and tensors

def test_device_choice():
    assert DEVICE.type == 'cpu'                      # MPCR_DEVICE=cpu above
    assert mpcr.get_device().type in ('cpu', 'cuda', 'mps')


def test_GPU_and_GPU_data():
    w = GPU(np.ones((3, 2)))
    assert w.requires_grad and w.is_leaf and w.dtype == torch.float32 and w.device == DEVICE
    d = GPU_data(torch.arange(6.0, dtype=torch.float64).reshape(2, 3))
    assert not d.requires_grad and d.dtype == torch.float32
    (w.sum() * 2).backward()
    assert torch.equal(w.grad, torch.full((3, 2), 2.0))


def test_GPU_copies_its_input():
    a = np.zeros(3)
    t = GPU_data(a)
    t += 1
    assert a.sum() == 0


# ------------------------------------------------------------------ the maths

def test_softmax_matches_torch_and_does_not_overflow():
    x = torch.tensor([[1.0, 2.0, 3.0], [1000.0, 1000.0, 0.0]])
    assert torch.allclose(softmax(x), torch.softmax(x, 1))
    assert torch.isfinite(softmax(x)).all()


def test_cross_entropy_matches_torch():
    torch.manual_seed(0)
    out, y = torch.randn(8, 10) * 30, torch.randint(0, 10, (8,))
    assert torch.allclose(cross_entropy(out, y), TF.cross_entropy(out, y), atol=1e-5)
    assert torch.allclose(cross_entropy(out, y.float()), TF.cross_entropy(out, y), atol=1e-5)


def test_one_hot():
    y = one_hot(np.array([3, 0, 9]))
    assert y.shape == (3, 10) and y.sum() == 3 and y[0, 3] == 1 and y[2, 9] == 1
    assert one_hot(torch.tensor([1, 2]), num_classes=4).shape == (2, 4)


def test_acc():
    out = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    assert acc(out, torch.tensor([1, 0, 0])) == pytest.approx(2 / 3)
    assert acc(out, np.array([1.0, 0.0, 1.0])) == 1.0


def test_randn_trunc():
    r = randn_trunc((1000,))
    assert r.shape == (1000,) and np.abs(r).max() <= 0.2 + 1e-12


def test_scale():
    assert np.allclose(scale(np.array([2.0, 4.0, 6.0])), [0, 0.5, 1])
    assert np.allclose(scale(np.array([2.0, 4.0]), (-1, 1)), [-1, 1])
    assert np.allclose(scale(np.full(3, 5.0)), 0.5)


# ------------------------------------------------------------------ training-loop helpers

class _C:
    b, h = 4, 0.5


def test_get_batch_with_arguments_and_from_the_notebook():
    global c, X, Y, X_test, Y_test
    c = _C()
    X, Y = np.arange(40).reshape(10, 2, 2), np.arange(10)
    X_test, Y_test = -X, -Y
    x, y = get_batch('train')
    assert x.shape == (4, 2, 2) and np.array_equal(x[:, 0, 0] // 4, y)
    xt, yt = get_batch('test')
    assert (yt <= 0).all()
    x, y = get_batch('train', b=10, X=X, Y=Y)                 # b equal to the whole set is allowed
    assert np.array_equal(y, Y)
    with pytest.raises(ValueError):
        get_batch('validate')


def test_gradient_step():
    global c
    c = _C()
    w = [GPU(np.ones(2)), GPU(np.ones(3))]
    (w[0].sum() + 2 * w[1].sum()).backward()
    gradient_step(w)                                          # h from c.h = 0.5
    assert torch.allclose(w[0], torch.full((2,), 0.5)) and torch.allclose(w[1], torch.zeros(3))
    assert (w[0].grad == 0).all() and w[0].is_leaf
    (w[0].sum()).backward()
    gradient_step(w, h=1.0)                                   # w[1].grad is zero, w[0] steps by 1
    assert torch.allclose(w[0], torch.full((2,), -0.5))


def test_make_plots_logs_both_accuracies():
    w = [GPU(np.eye(2))]
    model = lambda x, w: x @ w[0]
    x = GPU_data([[1, 0], [0, 1]])
    y = torch.tensor([0, 1])
    logged = []
    a = make_plots(model, x, y, w, log=logged.append, b=2, X_test=x, Y_test=y)
    assert a == (1.0, 1.0) and logged == [{'acc_train': 1.0, 'acc_test': 1.0}]


def test_log_arch_and_backprop():
    class C:
        f_s = [3, 3, 3, 3, 3, 3]
        f_n = [1, 8, 8, 8, 8, 8, 10]
    cfg = C()
    log_arch(cfg)
    assert cfg.f_s5 == 3 and cfg.f_n6 == 10

    p = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.SGD([p], lr=1.0)
    backprop('D', loss=(p * 2).sum(), optimizer=opt)
    assert p.item() == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        backprop('X')


# ------------------------------------------------------------------ pictures

def test_plotting_helpers_run(tmp_path):
    import matplotlib.pyplot as plt
    plot(np.random.rand(8, 8))
    plot(torch.rand(3, 8, 8))                                 # channels-first colour
    plot(GPU(np.random.rand(8, 8)))                           # a tensor with gradients
    montage_plot(np.random.rand(4, 5, 5))
    montage_plot(np.random.rand(4, 5, 5, 3))                  # colour stacks
    vizvec(np.arange(6).reshape(2, 3))
    vizvec(np.arange(3))
    show_image(torch.rand(3, 8, 8) * 2 - 1)
    ani = make_ani(np.random.rand(3, 4, 4))
    assert ani is not None
    plt.close('all')


def test_montage_accepts_the_old_multichannel_argument():
    m = montage(np.random.rand(4, 5, 5, 3), multichannel=True)
    assert m.shape == (10, 10, 3)
    assert montage(torch.rand(4, 5, 5)).shape == (10, 10)


def test_to_img():
    assert torch.equal(to_img(torch.tensor([-3.0, -1.0, 0.0, 1.0])), torch.tensor([0.0, 0.0, 0.5, 1.0]))


def test_drawnet(tmp_path):
    m = Model(input_shape=(28, 28, 1))
    m.add(Conv2D(32, (3, 3), padding='same'))
    m.add(MaxPooling2D((2, 2)))
    m.add(Conv2D(64, (3, 3)))
    m.add(Flatten())
    m.add(Dense(10))
    m.save_fig(str(tmp_path / 'net.svg'))
    assert (tmp_path / 'net.svg').read_text().startswith('<svg')
    model = drawnet(28, [1, 8, 16, 32], [5, 3, 3, 3, 3, 3], filename=str(tmp_path / 'net.pdf'))
    assert (tmp_path / 'net.pdf').stat().st_size > 0 and len(model.layers) == 3


# ------------------------------------------------------------------ the saliency maps (flashtorch)

def test_flashtorch_still_works_with_current_torch():
    from PIL import Image
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
                              torch.nn.Flatten(), torch.nn.Linear(4, 5))
    img = apply_transforms(Image.fromarray((np.random.rand(40, 40, 3) * 255).astype(np.uint8)))
    assert tuple(img.shape) == (1, 3, 224, 224)
    g = Backprop(net).calculate_gradients(img.clone().requires_grad_(True), target_class=2)
    assert g.shape[-2:] == (224, 224)


# ------------------------------------------------------------------ text and proteins

def test_stem_and_bag_of_words():
    assert [stem(w) for w in ['organize', 'organizes', 'organizing']] == ['organ'] * 3
    bag = bag_of_words(['hello', 'how', 'are', 'you'], ['hi', 'hello', 'I', 'you', 'bye'])
    assert bag.tolist() == [0, 1, 0, 1, 0] and bag.dtype == np.float32


@pytest.mark.network
def test_tokenize():
    assert tokenize('Hello, how are you?') == ['Hello', ',', 'how', 'are', 'you', '?']


def test_process_strings():
    assert process_strings(['abc', 'a-z']) == [[0, 1, 2], [0, 0, 25]]


@pytest.mark.network
def test_uniprot():
    x, y = get_uniprot_data('insulin', '=collagen', 3)
    assert len(x) == 3 and len(y) == 3 and all(s.isalpha() and s.islower() for s in x + y)
