# The Learned Kernel — companion code & notebooks

Runnable companion to the book **_The Learned Kernel: Geometric Discovery in Machine
Learning, from Gradient-Boosted Trees to In-Context Learning_** by Agus Sudjianto.

The book's thesis is that *learning is the discovery of geometry, the kernel carries it,
and the kernel should be **learned, not chosen**.* This repository lets you **run, see and
touch** that idea: one importable package (`lkbook`) and one interactive notebook per
chapter, on the same running examples the book uses. The code here is the same code the
book's figures are generated from — the notebook numbers match the page.

> This repo ships the **companion code and notebooks only**. The book text itself is a
> separate work and is not included here.

## Install

```bash
pip install "git+https://github.com/asudjianto-xml/Learned-Kernel.git"
```

For the interactive notebooks, include the extras:

```bash
pip install "learned-kernel[notebooks] @ git+https://github.com/asudjianto-xml/Learned-Kernel.git"
```

Or clone and develop against it (editable):

```bash
git clone https://github.com/asudjianto-xml/Learned-Kernel.git
cd Learned-Kernel
pip install -e ".[notebooks,dev]"
```

## Quickstart

```python
from lkbook import load_california
from lkbook.chapters import ch01

d = load_california()                      # fixed split + standardization
ridge, tree, knn = ch01.fit_models(d)
x = d.Xte[7]                               # one held-out Los Angeles block

# every model is a weighted vote over training labels: yhat(x) = sum_i w_i(x) y_i
w = ch01.tree_weights(tree, d.Xtr, x)
print(ch01.assert_weighted_vote(w, d.ytr, float(tree.predict(x[None])[0])))

ch01.make_influence_figure(d)              # the three geometries, over the map
```

## Notebooks

`notebooks/chNN_<slug>.ipynb` — one per chapter, runnable top to bottom. Each is authored
as a [jupytext](https://jupytext.readthedocs.io/) percent script (`.py`) paired to the
`.ipynb`; edit the `.py`, the `.ipynb` is the executed artifact.

```bash
jupyter lab notebooks/ch01_geometry_hidden.ipynb
```

| Chapter | Notebook | Run | What you explore |
|---|---|---|---|
| 1 — The geometry hidden in every model | [`ch01_geometry_hidden`](notebooks/ch01_geometry_hidden.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asudjianto-xml/Learned-Kernel/blob/main/notebooks/ch01_geometry_hidden.ipynb) | the implied weights of ridge / tree / k-NN; move the query and the knobs and watch the geometry redraw |
| 2 — The kernel as the normal form | [`ch02_normal_form`](notebooks/ch02_normal_form.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asudjianto-xml/Learned-Kernel/blob/main/notebooks/ch02_normal_form.ipynb) | KRR / GP / NW / SVM / attention as one machine; turn the kernel's bandwidth and ridge and watch all five predictions move |

*(rows added as chapters are published)*

## Running examples

- **California Housing** (regression) — fetched by scikit-learn on first use.
- **Taiwan Credit Default** (classification) — vendored in the package for offline
  reproducibility. Original data: I-Cheng Yeh and Che-hui Lien (2009), *The comparisons of
  data mining techniques for the predictive accuracy of probability of default of credit
  card clients*, Expert Systems with Applications; UCI Machine Learning Repository,
  "default of credit card clients."

## License

Code: MIT (see [LICENSE](LICENSE)). The book's text and figures are © Agus Sudjianto and
are not part of this repository.
