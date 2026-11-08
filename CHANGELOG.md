# Changelog

## 1.0.0 (2026-11-08)

A rewrite that keeps `from mpcr import *` working for every notebook written for version 0.

### Changed
- The package is split into modules: `core` (devices, plotting, maths, training helpers), `nlp`,
  `drawnet`, `bio`. `from mpcr import *` exports the same names as before (tested against the full
  list of version 0's 512 names in `tests/v0_names.txt`).
- Packaging moved from `setup.py` (named `ImportMPCR`, pointing at the old `mpcrlab/importmpcr`
  repo, with no dependencies) to `pyproject.toml` as `mpcr`, with real dependencies.
- `GPU`, `GPU_data` and `one_hot` use `DEVICE`: CUDA, then Apple MPS, then CPU (`MPCR_DEVICE`
  overrides). Version 0 required CUDA.
- `plot` handles tensors with gradients, channels-first colour images and a `size` argument.
  `montage_plot` handles colour stacks.
- `one_hot` is vectorised and takes `num_classes`. `softmax` takes `dim`.
- `drawnet` draws as many layers as you give it (version 0 always drew six) and takes `filename`.

### Fixed
- Importing no longer runs `pip install flashtorch`, `pip install barbar`, `pip install wandb`
  or `nltk.download('punkt')`. They are dependencies now, and are imported on first use.
- Importing no longer fails outside Colab (`from google.colab import drive`).
- `plot`, `montage_plot`, `GPU` and `GPU_data` were each defined twice; the second definition
  silently won. Each is now defined once (`plot` keeps the 10-inch size that won).
- `get_batch`, `gradient_step`, `make_plots`, `log_arch` and `backprop` looked up `c`, `X`, `model`,
  `w`, `d_loss`, ... in mpcr's own module, where they never exist, so they could only fail. They
  now take arguments and otherwise read the calling notebook's variables.
- `get_batch` could never return the last example; it also required 4-D arrays.
- `gradient_step` crashed on weights that had no gradient yet.
- `cross_entropy` took the log of a softmax, which gives `-inf` for confident wrong answers; it
  now uses the log-sum-exp form.
- `acc` failed when the labels were on a different device from the scores.
- `get_uniprot_data` used `uniprot.org/uniprot/?query`, which UniProt retired (and the query
  keyword had to start with `=`; both forms work now). It now uses `rest.uniprot.org`, with paging.
- `tokenize` downloads `punkt_tab`, which NLTK 3.8.2 and later need.
- `montage(..., multichannel=True)` raised a `TypeError` on scikit-image 0.19 and later.
- `imread` and friends come from `imageio.v2`, so they no longer warn about version 3.
- The network drawer's abstract classes used Python 2's `__metaclass__`; files are closed properly.

### Added
- `DEVICE`, `get_device()`, `view_as_windows`, `view_as_blocks`, `uniprot_sequences()`.
- Tests (`pytest`, 28 of them) and a GitHub Actions workflow (Python 3.9, 3.12 and 3.13).

### Checked
- All eight course notebooks that use mpcr were run end to end against version 0 and version 1
  on the same machine and compared cell by cell. Version 1 introduces no new errors; the cells that
  fail off Colab fail under both versions, for reasons of their own (hard-coded CUDA, Google Drive
  paths, image links that no longer work).
