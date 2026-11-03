<p align="center"><img src="assets/banner.png" alt="QwenQPU — a language model with a fixed quantum network" width="100%"></p>

<p align="center">
  <a href="#try-it">Quickstart</a> ·
  <a href="docs/ARCHITECTURE.md">How it works</a> ·
  <a href="docs/EVIDENCE.md">Measurements & caveats</a> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

**QwenQPU pairs Qwen's language generation with a fixed-weight quantum state-vector network.** Qwen proposes a circuit, the network evolves its state and produces terminal samples, and Qwen answers from the observed counts. Supplied circuits can also be dispatched directly.

The implementation today is **ideal classical simulation with explicit host scheduling**. The quantum child runs in a composite module's `forward`; language generation is a separate call. Integration into Qwen's original decoder or KV state remains future work.

<p align="center"><img src="assets/demo.gif" alt="Recorded Qwen GHZ16 proposal, verified gate-by-gate state replay, and 1024 simulated terminal samples" width="100%"></p>

*An original replay of a real, previously verified A100 interaction. The Qwen circuit and reply are recorded; intermediate states are re-executed locally with the unchanged core. Playback speed is illustrative. One ideal circuit evolution supplies 1,024 terminal draws—not 1,024 physical-QPU jobs. [MP4](assets/demo.mp4) · [trace & provenance](assets/animation-provenance.json) · [animation source](tools/render_assets.py).*

## Try it

```bash
git clone https://github.com/xi-zhao/QwenQPU.git
cd QwenQPU
```

### Read the recorded interaction — Python only

```bash
python examples/replay_recorded.py
```

This prints the selected recorded Qwen request, circuit, network counts and reply. It does not run a language model.

### Execute the fixed network — PyTorch, no language weights

From this directory, with PyTorch already installed:

```bash
python examples/ghz.py --qubits 16 --shots 1024 --seed 123
```

The checked CPU example reproduces these recorded counts:

```json
{"0000000000000000": 534, "1111111111111111": 490}
```

For an editable package install, use `python -m pip install -e .`. The small-core example needs PyTorch; it never downloads a language checkpoint. [Tests and environment](docs/REPRODUCING.md).

### Use an existing local Qwen checkpoint — optional

```bash
python examples/qwen_local.py \
  --model-path /path/to/an/existing/Qwen3.5-4B-Qwenputer \
  --output ./review/new-local-dialogue.json
```

This local-only integration entry uses the standard HF class, `trust_remote_code=False`, one bounded planning call and a counts reply. It performs no downloads or automatic answer repairs. It was not rerun with the large checkpoint while preparing this showcase; the animation uses the historical verified dialogue. Historical end-to-end runs used an A100 80 GB. [Runtime boundaries](docs/ARCHITECTURE.md).

## What is inside

| Component | Current implementation |
|---|---|
| Register | 1–16 qubits; FP32 real/imaginary state `[2**n, 2]` |
| Fixed gates | H, X, T, CNOT; 48 fixed H/X/T coefficients; no training |
| Circuit bound | Up to 1,000 **sequential gate operations** between resets |
| Quantum graph | Addressing, local linear maps, masking, state writeback, CDF and bitstring selection |
| Host | JSON validation, bounded repeat expansion, RNG and counts aggregation |
| Measurement | 1–10,000 ideal terminal samples; `q0` is the least significant bit |
| Language model | Same pinned Qwenputer checkpoint used in the existing experiments; its neural CPU path is unused here |

<img src="assets/architecture.png" alt="Qwen generation, explicit host scheduling, fixed quantum module, counts, and Qwen reply" width="100%">

The scripted module uses local 4×4 real gate maps and an exponentially sized state vector. It does not allocate a global dense `2**n × 2**n` gate matrix. TorchScript integer control stays on CPU; state tensor operations run on the selected CPU/CUDA device. Qiskit is used only for independent offline verification.

## Evidence, with the scoring issue visible

The completed comparison used 40 questions × 2 conditions, the same model weights, a shared 16,384-generated-token cap per condition and 4,096 simulated samples. The 20 supplied-circuit calculation tasks are the primary subset; 20 planning tasks are auxiliary. The original questions had already been seen, so this is a same-question re-evaluation.

| 20 calculation tasks, each arm | Qwen | Qwen + fixed network |
|---|---:|---:|
| Original frozen automatic score **(parser affected)** | 1/20 | 19/20 |
| Post-hoc identical-JSON diagnostic | 8/20 | 19/20 |
| Diagnostic correct **and** naturally finished | 7/20 | 19/20 |

**The frozen parser rejected repeated copies of the same correct JSON as ambiguous.** We retain the original score and disclose the defect. The narrow post-hoc diagnostic merges identical candidate objects; it is not a complete natural-language semantic audit and does not replace the prespecified score. Planning's original joint score is **1/20 in each arm**; a planning improvement has not been established.

<img src="assets/computation-evidence.png" alt="Frozen parser scores, explicitly labelled post-hoc diagnostics, and all twenty paired computation outcomes" width="100%">

In the narrow diagnostic, there are 12 enhanced-only wins, 1 plain-only win and 7 both-correct pairs. The descriptive paired-bootstrap difference is 55 percentage points, with a 95% interval of 30–80 points. This small synthetic sample is not evidence of performance on all quantum tasks, a quantum advantage, or superiority over an external simulator. [Full interpretation, source data and limits](docs/EVIDENCE.md).

The existing independent core checks cover 1, 3, 8, 12 and 16 qubits with 1,000-gate random circuits and Qiskit complex128 references. In the later evaluation, all 38 actual network calls passed the offline execution audit. Correct execution alone does not guarantee a correct Qwen answer.

## Weights and releases

**Hugging Face model download: not published yet.** This repository includes the small fixed-module bundle; it does not host the language checkpoint.

The included [small module bundle](weight-release/module-bundle/qwenqpu_module.json) contains only the 48 fixed quantum gate parameters. Language weights are separate. The experiments used the official [`divagr1925/Qwen3.5-4B-Qwenputer`](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer) checkpoint at revision `2d9068508f2b680badb7abcded96fdd107e9ba5d`; no checkpoint shards are included here.

```python
from qwenqpu.weights import load_bundle
quantum = load_bundle("weight-release/module-bundle", n_qubits=16)
```

This is an auxiliary-module package, not a stand-alone `AutoModelForCausalLM` checkpoint. A [model-card draft and release checklist](weight-release/RELEASE_CHECKLIST.md) distinguish the language checkpoint, fixed module and host runtime.

## Rebuild the visuals

```bash
python tools/build_trace.py       # PyTorch: circuit/state replay, no language inference
python tools/render_assets.py     # Pillow + CairoSVG + local ffmpeg
python tools/render_benchmark.py  # Aggregate CSV only; no new evaluation
```

The reference GIF was inspected to understand its narrative, not copied into this repository. The banner, diagram, animation and render code here are original. [Media provenance](docs/MEDIA.md).

## Limits

Ideal pure states and terminal measurement only. No noise, mid-circuit measurement, unbounded registers, learned quantum solver, decoder/KV fusion or physical QPU. The 1,000-operation core bound is a gate count, not parallel circuit depth; the calculation evaluation itself contains 5–161 operations. The retained execution path is a host-scheduled composite model.

## Credits and licensing

Inspired by the presentation and broader compiled-computation idea of [Qwenputer](https://github.com/divagr18/qwenputer) and its [author's write-up](https://divagr.com/blog/qwenputer). Qwenputer's code is MIT; its pinned model metadata and Qwen's base checkpoint declare Apache-2.0. These upstream terms remain separate from QwenQPU's original module and artwork. No Qwenputer logo, animation frame or source asset is redistributed here.

No project-wide open-source license has been selected for the original QwenQPU code and artwork. Public availability does not add a license grant; upstream materials retain their own terms. [License and redistribution notes](docs/LICENSES.md).
