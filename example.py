"""Runnable demo. No GPU, no model download, no dataset.

Synthetic activations are built with the geometry the paper reports: the two
classes differ by a shift along one direction, and the remaining dimensions are
shared noise carrying no class information. The three geometry conditions and
LayerMix then behave the way they do on real hidden states.

    python example.py
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold

from layermix import (
    LayerMix,
    decompose_signal,
    full_probe,
    mean_shift_probe,
    shrinkage_lda_probe,
)
from layermix.probes import cross_validated_auroc

N_SAMPLES, N_FEATURES = 400, 256


def make_layer(shift_strength, seed):
    """One layer of activations whose class signal is a single shift.

    The class-conditional covariance is identical for both classes, so every bit
    of separation lives in the centroid difference and none of it in the shape of
    the clouds. That is the structure the paper measures in real hidden states.
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], N_SAMPLES // 2)
    direction = rng.normal(size=N_FEATURES)
    direction /= np.linalg.norm(direction)
    noise = rng.normal(size=(N_SAMPLES, N_FEATURES))
    features = noise + np.outer(labels, shift_strength * direction)
    return features.astype(np.float32), labels


def main():
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    features, labels = make_layer(shift_strength=2.2, seed=0)

    print("Single layer, probe comparison")
    for name, probe in [
        ("full activations, L2-LR", full_probe),
        ("mean-shift coordinate only", mean_shift_probe),
        ("shrinkage LDA", shrinkage_lda_probe),
    ]:
        mean_auroc, _ = cross_validated_auroc(features, labels, splitter, probe)
        print(f"  {name:<32} AUROC {mean_auroc:.3f}")

    print("\nWhere the signal lives")
    parts = decompose_signal(features, labels, splitter, full_probe)
    for name in ("full", "mean_shift_only", "mean_shift_removed"):
        print(f"  {name:<32} AUROC {np.mean(parts[name]):.3f}")
    print("  removing one direction is enough to erase the signal")

    print("\nLayerMix over a band of layers")
    strengths = {2: 0.4, 5: 1.2, 8: 2.2, 11: 2.0, 14: 1.4, 17: 0.3}
    layer_features = {}
    shared_labels = None
    for offset, (layer, strength) in enumerate(strengths.items()):
        feats, labs = make_layer(strength, seed=100 + offset)
        layer_features[layer] = feats
        shared_labels = labs

    for layer in sorted(layer_features):
        mean_auroc, _ = cross_validated_auroc(
            layer_features[layer], shared_labels, splitter, full_probe
        )
        print(f"  layer {layer:>2} alone{'':<20} AUROC {mean_auroc:.3f}")

    model = LayerMix(top_k=3)
    mean_auroc, _ = model.cross_validate(layer_features, shared_labels, splitter)
    print(f"  LayerMix (top 3){'':<18} AUROC {mean_auroc:.3f}")
    print(f"  layers it chose from training data alone: {model.layers_}")


if __name__ == "__main__":
    main()
