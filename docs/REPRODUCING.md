# Reproduce the small examples and media

The latest presentation-only publication checks are recorded in [VALIDATION.md](VALIDATION.md).

Local checks used Python 3.13.2 and existing PyTorch 2.9.1 on CPU. Historical
large-model/independent physics evidence used PyTorch 2.6.0+cu124,
Transformers 5.19.0 and Qiskit 2.6.1 on A100-SXM4-80GB. These are separate runs.
No large checkpoint was loaded for this showcase and no dependencies were installed.

```bash
python examples/replay_recorded.py
python examples/ghz.py --qubits 16 --shots 1024 --seed 123
python -m unittest discover -s tests -v
```

`replay_recorded.py` needs only Python. Core examples/tests require PyTorch.
The tests exercise the real core: historical GHZ16 and HTH counts, segmented
execution/reset/state isolation, invalid instructions, the 1,000-operation cap,
and weights-only export/reload with fixed-parameter validation.

To reconstruct the tiny fixed-module bundle in a fresh directory:

```bash
python tools/export_module.py --out ./review/new-module-bundle
```

For visuals, use existing Pillow/NumPy/Matplotlib/CairoSVG and ffmpeg with H.264:

```bash
python tools/build_trace.py
python tools/render_assets.py
python tools/render_benchmark.py
```

The current trace executes C16-4 gate-by-gate. Its event counts must match the
selected evaluation record before rendering proceeds. The renderer uses only that trace;
no language generation or network request occurs. Media sources do not copy or
depend on the reference animation. Fonts use local DIN/Avenir/Menlo when available,
with DejaVu fallbacks; font files themselves are not bundled.

The optional language entry requires an existing local checkpoint, the language
extra's packages and sufficient memory. It is provided for review and was not
large-model smoke-tested again in this stage. Missing checkpoints fail before
loading; there is no automatic download, retry, network change or credential creation.
