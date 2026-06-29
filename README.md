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
| 3 — From chosen to learned kernels | [`ch03_chosen_to_learned`](notebooks/ch03_chosen_to_learned.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asudjianto-xml/Learned-Kernel/blob/main/notebooks/ch03_chosen_to_learned.ipynb) | the bandwidth U-curve, ARD length-scales fit on a held-out fold, and the (K,λ) scale degeneracy; slide ℓ against the learned-geometry baseline |
| 4 — Trees and forests are kernels | [`ch04_trees_are_kernels`](notebooks/ch04_trees_are_kernels.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asudjianto-xml/Learned-Kernel/blob/main/notebooks/ch04_trees_are_kernels.ipynb) | the leaf kernel; the **exact** GNW recovery of a gradient-boosted forest from its leaf scores; raw-label vs leaf-score vs ridge values on one geometry; move the query to compare leaf-kernel and RBF similarity |
| 5 — Gaussian processes | [`ch05_gaussian_processes`](notebooks/ch05_gaussian_processes.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asudjianto-xml/Learned-Kernel/blob/main/notebooks/ch05_gaussian_processes.ipynb) | the marginal likelihood as the kernel's score; GP posterior mean = KRR exactly; an interior evidence optimum (Prop. F); move the noise and length scale and watch the evidence respond |
| 6 — Similarity, smoothing, case-based prediction | [`ch06_similarity_smoothing`](notebooks/ch06_similarity_smoothing.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asudjianto-xml/Learned-Kernel/blob/main/notebooks/ch06_similarity_smoothing.ipynb) | the per-prediction evidence ledger (N_eff, witnesses, local calibration); the blend ρ*→0 on California, ρ*→1 on Taiwan; pick a query and read its supporting cases |
| 7 — What makes a kernel learnable | [`ch07_what_makes_learnable`](notebooks/ch07_what_makes_learnable.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asudjianto-xml/Learned-Kernel/blob/main/notebooks/ch07_what_makes_learnable.ipynb) | in-sample fit vs leakage-free query-fold selection; SURE tracking the true risk; the leaf kernel over-credited in-sample; slide the ridge and watch in-sample mislead |
| 8 — Spectral kernels and Bochner | [`ch08_spectral_bochner`](notebooks/ch08_spectral_bochner.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asudjianto-xml/Learned-Kernel/blob/main/notebooks/ch08_spectral_bochner.ipynb) | the spectral-mixture (Bochner) kernel; recover a periodic+smooth signal a single RBF cannot and extrapolate it; the roughness ladder; move the frequency and bandwidth |
| 9 — Spectral kernels versus trees | [`ch09_spectral_vs_trees`](notebooks/ch09_spectral_vs_trees.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asudjianto-xml/Learned-Kernel/blob/main/notebooks/ch09_spectral_vs_trees.ipynb) | smooth/periodic vs sharp/axis-aligned structure; where the spectral basis wins and where the tree basis does; the capacity map across the synthetic suite |
| 10 — Fusing geometries | [`ch10_fusing_geometries`](notebooks/ch10_fusing_geometries.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asudjianto-xml/Learned-Kernel/blob/main/notebooks/ch10_fusing_geometries.ipynb) | convex fusion of a spectral bank and a tuned CatBoost leaf kernel; leakage-free fusion weights; bagging the leaf kernel toward the forest proximity |
| 11 — Symmetry suffices | [`ch11_symmetry_suffices`](notebooks/ch11_symmetry_suffices.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asudjianto-xml/Learned-Kernel/blob/main/notebooks/ch11_symmetry_suffices.ipynb) | the first-order asymmetry law; asymmetrizing a spectral kernel forfeits KRR; the earned asymmetry weight ρ*→0 exchangeable, >0 directed |
| 12 — Meta-learning a prior over kernels | [`ch12_meta_learning_prior`](notebooks/ch12_meta_learning_prior.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asudjianto-xml/Learned-Kernel/blob/main/notebooks/ch12_meta_learning_prior.ipynb) | the self-consistent kernel prior; an emitter meta-trained to infer the measure in one pass; regret over Bayes vs context size; zero-shot transfer to California |

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
