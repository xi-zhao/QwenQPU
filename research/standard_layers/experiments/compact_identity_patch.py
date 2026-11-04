"""Versioned precision repair for the kind-0 digit-copy operation.

Only two first-layer MLP up-projection rows change. The standard forward,
controller, RAM and H/T arithmetic are unchanged. Original checkpoint files
remain untouched. This patch must be separately validated before promotion.
"""
import hashlib
import torch
from token_checkpoint import C, projection


def apply_identity_patch(model, layout):
    metadata=model.config.qwenqpu_added_layer_prototype
    if metadata['format'] not in ('compact-addressed-v7','compact-banked-v8'):
        raise ValueError('Unsupported compiler layout')
    if metadata.get('identity_digit_copy_patch'):
        raise ValueError('Patch already applied')
    weight=model.model.layers[0].mlp.up_proj.weight
    rows=[0,2] # kind 0, high/low z units; e units are rows 1/3.
    before=weight[rows].detach().cpu().clone()
    with torch.no_grad():
        for phase,row in enumerate(rows):
            weight[row].zero_()
            projection(weight[row],{layout.read+phase:1/256,0:-C/256})
    after=weight[rows].detach().cpu().clone()
    metadata['identity_digit_copy_patch']='direct-digits-r1'
    return dict(version='direct-digits-r1',parameter='model.layers.0.mlp.up_proj.weight',
        rows=rows,changed_values=int(torch.count_nonzero(before!=after)),
        before_sha256=hashlib.sha256(before.numpy().tobytes()).hexdigest(),
        after_sha256=hashlib.sha256(after.numpy().tobytes()).hexdigest(),
        reason='Copy stored integer digits directly; avoid reconstructing and requantizing identity amplitudes')
