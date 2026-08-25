"""LayerMix: pick informative layers without an oracle, then combine them.

Probe accuracy depends sharply on which layer the activations come from, and the
best layer moves with the model and the dataset. Choosing it on the test set is
not something a deployed detector can do.

LayerMix replaces that choice with a criterion computed on training data alone.
Every layer is scored by the effect size of its centroid separation, the top
``top_k`` layers are kept, a probe is fit on each, and their scores are averaged
with weights proportional to the same effect size. Because the signal occupies a
contiguous band of layers rather than one, averaging over the band recovers
oracle-layer accuracy without oracle access.
"""

from __future__ import annotations

import numpy as np

from layermix.geometry import cohens_d
from layermix.probes import full_probe


class LayerMix:
    """Effect-size-weighted ensemble of per-layer probes.

    Parameters
    ----------
    top_k:
        Number of layers to keep. Five is the default used in the paper.
    probe:
        Probe factory with the ``probe(train_x, train_y, test_x)`` signature.
    """

    def __init__(self, top_k: int = 5, probe=full_probe):
        self.top_k = top_k
        self.probe = probe
        self.layers_: list = []
        self.weights_: np.ndarray | None = None

    def select_layers(self, layer_features: dict, labels: np.ndarray):
        """Rank layers by centroid effect size and keep the strongest ``top_k``.

        ``layer_features`` maps a layer identifier to a ``(n_samples, n_features)``
        array. Only training rows should be passed in.
        """
        scored = [(name, cohens_d(feats, labels)) for name, feats in layer_features.items()]
        scored.sort(key=lambda item: item[1], reverse=True)
        selected = scored[: self.top_k]
        self.layers_ = [name for name, _ in selected]
        weights = np.array([score for _, score in selected], dtype=np.float64)
        self.weights_ = weights / weights.sum()
        return self.layers_

    def score(self, layer_features: dict, labels: np.ndarray, train_idx, test_idx) -> np.ndarray:
        """Fit on ``train_idx`` and return one ensemble score per test sample.

        Layer selection, probe fitting and the weights are all derived from the
        training rows, so the held-out rows are untouched until scoring.
        """
        train_view = {
            name: np.asarray(feats, dtype=np.float32)[train_idx]
            for name, feats in layer_features.items()
        }
        self.select_layers(train_view, labels[train_idx])

        predictions = []
        for name in self.layers_:
            feats = np.asarray(layer_features[name], dtype=np.float32)
            predictions.append(
                self.probe(feats[train_idx], labels[train_idx], feats[test_idx])
            )
        return sum(w * p for w, p in zip(self.weights_, predictions))

    def cross_validate(self, layer_features: dict, labels: np.ndarray, splitter):
        """Mean and per-fold AUROC over ``splitter``."""
        from sklearn.metrics import roc_auc_score

        scores = []
        for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels):
            pred = self.score(layer_features, labels, train_idx, test_idx)
            scores.append(roc_auc_score(labels[test_idx], pred))
        return float(np.mean(scores)), scores
