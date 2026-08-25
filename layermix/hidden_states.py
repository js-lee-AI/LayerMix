"""Last-token hidden-state extraction.

Optional. The probes and the geometry analysis operate on plain arrays, so this
module is only needed to produce those arrays from a real model. It imports
torch and transformers lazily for that reason.

The paper reads the residual stream at the final prompt token, which is the
position that has attended to the whole response.
"""

from __future__ import annotations

import numpy as np


def extract_layer_states(model, tokenizer, texts, layers, device="cuda",
                         batch_size=4, max_length=512):
    """Return ``{layer_index: (n_texts, hidden_size) array}`` for ``layers``.

    ``layers`` are indices into the transformer blocks, so layer ``0`` is the
    output of the first block. ``hidden_states[0]`` is the embedding output and
    is skipped.
    """
    import torch

    model.eval()
    model.requires_grad_(False)
    collected = {layer: [] for layer in layers}

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        last_positions = inputs["attention_mask"].sum(dim=1) - 1
        for layer in layers:
            states = outputs.hidden_states[layer + 1]
            for row, position in enumerate(last_positions.tolist()):
                vector = states[row, position].float().cpu().numpy()
                collected[layer].append(np.nan_to_num(vector))

        del outputs, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {layer: np.asarray(rows) for layer, rows in collected.items()}
