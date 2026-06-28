"""Running-example data loaders for *The Learned Kernel*.

One source of truth for the two datasets the book uses. The fixed split (seed=0,
20% test) and the train-fit standardization live here so every figure, script and
notebook sees identical numbers.

- **California Housing** is fetched by scikit-learn (cached locally on first use).
- **Taiwan Credit** is vendored inside the package (`lkbook/_data/taiwan_credit.npz`)
  so the notebooks reproduce offline. Original dataset: Yeh & Lien (2009), "default of
  credit card clients", UCI Machine Learning Repository. Set the environment variable
  ``LKBOOK_TAIWAN_NPZ`` to point at a different ``X``/``y`` npz if you prefer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TAIWAN_NAMES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


@dataclass
class RunningData:
    """A running-example dataset, prepared identically everywhere it is used."""
    Xtr: np.ndarray            # standardized train features
    Xte: np.ndarray            # standardized test features
    ytr: np.ndarray
    yte: np.ndarray
    Xtr_raw: np.ndarray        # raw (un-standardized) train features
    Xte_raw: np.ndarray
    names: list[str]
    scaler: StandardScaler
    task: str                  # "regression" | "classification"
    target_unit: str           # human-readable unit of y

    @property
    def n(self) -> int:
        return len(self.Xtr)

    @property
    def d(self) -> int:
        return self.Xtr.shape[1]

    def col(self, name: str) -> int:
        """Column index of a named feature."""
        return self.names.index(name)


def _prepare(X, y, names, task, target_unit, test_size, seed) -> RunningData:
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    sc = StandardScaler().fit(Xtr_raw)
    return RunningData(
        Xtr=sc.transform(Xtr_raw), Xte=sc.transform(Xte_raw),
        ytr=ytr, yte=yte, Xtr_raw=Xtr_raw, Xte_raw=Xte_raw,
        names=list(names), scaler=sc, task=task, target_unit=target_unit)


def load_california(test_size: float = 0.2, seed: int = 0) -> RunningData:
    """California Housing — regression. Target is median house value in $100,000s."""
    data = fetch_california_housing()
    return _prepare(data.data, data.target, data.feature_names,
                    "regression", "$100,000", test_size, seed)


@dataclass
class TabularData:
    """A mixed-type tabular dataset with cyclical features (Bike Sharing). Carries raw numeric
    features (tree-ready) plus a `cyclical` map naming the periodic columns and their periods, so
    a spectral channel can Fourier-encode them (sin/cos at the period)."""
    Xtr_raw: np.ndarray
    Xte_raw: np.ndarray
    ytr: np.ndarray
    yte: np.ndarray
    names: list[str]
    cyclical: dict          # {feature_name: period}
    task: str
    target_unit: str
    categorical: list[str] = None   # native-categorical columns (for the CatBoost channel)
    continuous: list[str] = None    # smooth numeric columns (temp, humidity, windspeed)

    @property
    def n(self) -> int:
        return len(self.Xtr_raw)

    @property
    def d(self) -> int:
        return self.Xtr_raw.shape[1]

    def col(self, name: str) -> int:
        return self.names.index(name)


def load_bikeshare(test_size: float = 0.2, seed: int = 0) -> TabularData:
    """Bike Sharing Demand — regression on hourly ride count (UCI; Fanaee-T & Gama 2014).

    A natural fusion example: periodic demand in the cyclical features (hour, month, weekday) is
    what the Fourier/spectral channel owns, while sharp categorical regimes and their interactions
    with time (a working-day morning is a commute peak; a weekend morning is not) are what the
    CatBoost channel owns. Redundant features are dropped — ``season`` (carried by ``month``) and
    ``feel_temp`` (collinear with ``temp``). The categorical regime features are kept as integer
    codes (not one-hot) for CatBoost's native categorical handling and named in ``categorical``;
    the cyclical integers are left raw and named in ``cyclical`` so the spectral channel can
    Fourier-encode them. The target is ``log1p(count)`` (the count is heavily right-skewed)."""
    from sklearn.datasets import fetch_openml
    df = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True).frame.copy()
    y = np.log1p(df.pop("count").to_numpy(dtype=float))
    df.pop("season")            # redundant with month (the cyclical channel already carries it)
    df.pop("feel_temp")         # nearly collinear with temp
    for b in ("holiday", "workingday"):
        df[b] = (df[b].astype(str) == "True").astype(int)
    # weather and year as integer category codes (native-categorical for CatBoost)
    df["weather"] = df["weather"].astype(str).astype("category").cat.codes.astype(int)
    df["year"] = df["year"].astype(int)
    df = df.astype({"weekday": float, "hour": float, "month": float})
    cyc = {"hour": 24.0, "month": 12.0, "weekday": 7.0}
    cont = ["temp", "humidity", "windspeed"]
    categorical = ["year", "holiday", "workingday", "weather"]
    order = list(cyc) + categorical + cont
    X = df[order].to_numpy(dtype=float)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    return TabularData(Xtr_raw, Xte_raw, ytr, yte, order, cyc, "regression", "log rides/hr",
                       categorical=categorical, continuous=cont)


def load_taiwan(test_size: float = 0.2, seed: int = 0) -> RunningData:
    """Taiwan Credit Default — binary classification (1 = default next month)."""
    override = os.environ.get("LKBOOK_TAIWAN_NPZ")
    if override:
        z = np.load(override)
    else:
        with resources.files("lkbook").joinpath("_data/taiwan_credit.npz").open("rb") as f:
            z = np.load(f)
            z = {"X": z["X"], "y": z["y"]}     # materialize before the file closes
    return _prepare(np.asarray(z["X"], float), np.asarray(z["y"], float), TAIWAN_NAMES,
                    "classification", "P(default)", test_size, seed)
