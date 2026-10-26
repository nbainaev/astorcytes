"""Evaluator-only cell decoding of predicted input codes; no map reaches a model."""

import numpy as np
from astrorl.mlp import fingerprint

_CODEBOOKS = {}


def input_codebook(learner, spec, free):
    signature = fingerprint(
        [np.asarray(free), learner.input_coder.w.numpy(), learner.input_coder.bias.numpy()]
    )
    key = (signature, learner.c.input_action_gain, learner.c.input_probability_scale)
    if key not in _CODEBOOKS:
        cells = np.asarray(free, dtype=np.float32) / (spec["size"] - 1)
        codes = np.stack(
            [np.stack([learner.input_code(cell, a) for cell in cells]) for a in range(4)]
        )
        if len(_CODEBOOKS) >= 128:
            _CODEBOOKS.pop(next(iter(_CODEBOOKS)))
        _CODEBOOKS[key] = codes
    return _CODEBOOKS[key]


def nearest_cell(prediction, action, codebook):
    return int(np.square(codebook[action] - prediction).mean(axis=-1).argmin())
