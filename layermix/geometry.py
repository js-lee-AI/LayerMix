"""Geometry of the hallucination signal in hidden states.

The paper's central claim is that the separation between truthful and
hallucinated responses is carried almost entirely by a single direction, the
difference between the two class centroids. The helpers here are the ones used
to establish that claim:

* ``mean_shift_direction`` returns the unit centroid-difference vector.
* ``cohens_d`` measures how far apart the classes are along it.
* ``project_out`` removes it from a set of activations.
* ``decompose_signal`` runs the three-way comparison that makes the point:
  the full representation, the single mean-shift coordinate alone, and the
  representation with that coordinate removed.

All functions take activations as a ``(n_samples, n_features)`` array and
binary labels as a ``(n_samples,)`` array. Nothing here depends on a specific
model or tokenizer.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-10


def class_means(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the centroid of the negative and of the positive class."""
    return features[labels == 0].mean(axis=0), features[labels == 1].mean(axis=0)


def mean_shift_direction(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Unit vector from the negative-class centroid to the positive-class centroid."""
    mu0, mu1 = class_means(features, labels)
    delta = mu1 - mu0
    return delta / (np.linalg.norm(delta) + _EPS)


def cohens_d(features: np.ndarray, labels: np.ndarray) -> float:
    """Effect size of the centroid separation, pooled over features.

    Used as the layer-selection criterion in :class:`layermix.layermix.LayerMix`,
    where a layer that separates the classes further is given more weight.
    """
    mu0, mu1 = class_means(features, labels)
    delta = mu1 - mu0
    pooled_var = 0.5 * (
        features[labels == 0].var(axis=0).mean() + features[labels == 1].var(axis=0).mean()
    )
    return float(np.linalg.norm(delta) / (np.sqrt(pooled_var) + _EPS))


def project_onto(features: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Coordinate of each sample along ``direction``, shaped ``(n_samples, 1)``."""
    return (features @ direction).reshape(-1, 1)


def project_out(features: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the component along ``direction`` from every sample."""
    return features - np.outer(features @ direction, direction)


def decompose_signal(features, labels, splitter, probe):
    """Compare the full representation against its mean-shift decomposition.

    ``probe`` is called as ``probe(train_features, train_labels, test_features)``
    and returns a score per test sample. ``splitter`` is any scikit-learn
    cross-validator. The direction is estimated on the training fold only, so
    the held-out fold never sees the class centroids it is scored against.

    Returns per-fold AUROC lists for three conditions:

    ``full``
        the untouched representation,
    ``mean_shift_only``
        the single coordinate along the centroid difference,
    ``mean_shift_removed``
        the representation with that coordinate projected out.

    Detection that survives in ``mean_shift_only`` and collapses in
    ``mean_shift_removed`` is the signature of a one-directional signal.
    """
    from sklearn.metrics import roc_auc_score

    out = {"full": [], "mean_shift_only": [], "mean_shift_removed": []}
    for train_idx, test_idx in splitter.split(features, labels):
        x_tr, y_tr = features[train_idx], labels[train_idx]
        x_te, y_te = features[test_idx], labels[test_idx]
        direction = mean_shift_direction(x_tr, y_tr)

        out["full"].append(roc_auc_score(y_te, probe(x_tr, y_tr, x_te)))
        out["mean_shift_only"].append(
            roc_auc_score(
                y_te,
                probe(project_onto(x_tr, direction), y_tr, project_onto(x_te, direction)),
            )
        )
        out["mean_shift_removed"].append(
            roc_auc_score(
                y_te,
                probe(project_out(x_tr, direction), y_tr, project_out(x_te, direction)),
            )
        )
    return out
