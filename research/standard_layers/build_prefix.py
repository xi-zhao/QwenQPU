"""Construct the four-layer prefix using the published fixed-weight compiler."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent / 'experiments'))
from compact_checkpoint import compile_compact
from compact_identity_patch import apply_identity_patch
from compact_normalized_readout import apply_normalized_readout
from compact_sampling_digits import apply_digit_cdf_patch
from compact_sampling_rejection import apply_rejection_patch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--capacity', type=int, choices=range(1, 7), default=6)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new output directory')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(a.out.parent).free < 7 * 2**30:
        raise RuntimeError('Need 7 GiB free to retain a 5 GiB reserve')
    torch.set_num_threads(2)
    model, layout = compile_compact(a.capacity)
    apply_identity_patch(model, layout)
    apply_normalized_readout(model, layout)
    apply_digit_cdf_patch(model, layout)
    apply_rejection_patch(model, layout)
    model.save_pretrained(a.out, safe_serialization=True, max_shard_size='1GB')
    manifest = {f.name: {'bytes': f.stat().st_size,
                        'sha256': hashlib.file_digest(f.open('rb'), 'sha256').hexdigest()}
                for f in sorted(a.out.glob('*.safetensors'))}
    (a.out / 'constructed-files.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
