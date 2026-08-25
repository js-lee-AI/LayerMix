"""Linear probes over hidden states.

Three probes are enough to reproduce the paper's argument.

``full_probe``
    L2-regularized logistic regression on the whole activation vector. This is
    the probe the paper recommends and the one every architectural alternative
    is measured against.
``mean_shift_probe``
    Logistic regression on the single centroid-difference coordinate. It is the
    one-dimensional floor.
``shrinkage_lda_probe``
    Linear discriminant analysis with Ledoit-Wolf shrinkage. It sits between the
    two and separates the two candidate explanations for the gap: if shrinkage
    LDA closes most of it, the gap is covariance estimation under a small sample,
    not non-linear structure that a bigger probe could exploit.

Each probe follows the same signature, ``probe(train_x, train_y, test_x)``,
returning one score per test sample, so they can be passed straight to
:func:`layermix.geometry.decompose_signal`.
"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from layermix.geometry import mean_shift_direction, project_onto

DEFAULT_C = 1e-3


def full_probe(train_x, train_y, test_x, C: float = DEFAULT_C, max_iter: int = 1000, seed: int = 42):
    """L2-regularized logistic regression on standardized activations."""
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(train_x)
    x_te = scaler.transform(test_x)
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    clf.fit(x_tr, train_y)
    return clf.predict_proba(x_te)[:, 1]


def mean_shift_probe(train_x, train_y, test_x, max_iter: int = 2000, seed: int = 42):
    """Logistic regression on the centroid-difference coordinate alone.

    The direction is estimated on the training split only.
    """
    direction = mean_shift_direction(train_x, train_y)
    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(project_onto(train_x, direction), train_y)
    return clf.predict_proba(project_onto(test_x, direction))[:, 1]


def shrinkage_lda_probe(train_x, train_y, test_x):
    """Linear discriminant analysis with automatic Ledoit-Wolf shrinkage.

    Shrinkage is what makes LDA usable when the feature count is far larger than
    the sample count, which is the regime a 4096-dimensional residual stream and
    a few hundred labeled examples put us in.
    """
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(train_x)
    x_te = scaler.transform(test_x)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(x_tr, train_y)
    return lda.decision_function(x_te)


def cross_validated_auroc(features, labels, splitter, probe=full_probe):
    """Mean and per-fold AUROC of ``probe`` under ``splitter``."""
    from sklearn.metrics import roc_auc_score

    scores = []
    for train_idx, test_idx in splitter.split(features, labels):
        pred = probe(features[train_idx], labels[train_idx], features[test_idx])
        scores.append(roc_auc_score(labels[test_idx], pred))
    return float(np.mean(scores)), scores
