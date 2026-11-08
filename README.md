<p align="center">
  <img src="assets/banner-added-layers.gif" alt="QwenQPU — QPU inside a language model" width="100%">
</p>

[简体中文](README.zh-CN.md) · [Standard-layer implementation](docs/ADDED_LAYERS.md) · [Benchmark](docs/BENCHMARK_ADDED_LAYERS.md) · [Hugging Face](https://huggingface.co/xi-zhao/QwenQPU)

Language models approximate numerical computation; exact answers usually mean
handing the work to an external tool. QwenQPU puts the exact computation inside
the checkpoint. Four standard Qwen3.5 layers, with every weight set analytically
by a compiler, hold a quantum state vector in Gated DeltaNet memory and execute
the universal H/X/T/CNOT gate set in the forward pass. Qwen reads the task and
submits the circuit; the added layers return the measurement probabilities.

## Benchmark

![Qwen vs QwenQPU](assets/r5/benchmark-main.png)

**Qwen: 4/48 (8.3%). QwenQPU: 48/48 (100%).** Exact McNemar p = 1.1×10⁻¹³.

Each task starts from the all-zero state, applies a supplied circuit and asks for
two measurement-event probabilities. The 48 circuits span 3, 4 and 6 qubits and
12, 18 and 24 sequential gates. Tasks were selected on physical criteria alone
and frozen before any model inference. Both arms use the same language weights,
and every QPU forward prediction is charged to the same 16,384-prediction budget
as language tokens: the added layers get no extra compute.

| Calculation tasks | Qwen | QwenQPU |
|---|---:|---:|
| Non-Clifford circuits (36) | 2/36 | **36/36** |
| Clifford controls (12) | 2/12 | **12/12** |
| All paired tasks (48) | 4/48 | **48/48** |

A task passes when both probabilities are within `5e-5` of independent Qiskit
values. QwenQPU is held to a stricter standard: its submitted circuit must
reproduce the task's final state and its events must match the task. Its largest
error is `1.42e-5`.

The added layers fix all 44 of Qwen's errors and break none of its 4 correct
answers. On a six-qubit H3–T3–H3 circuit, QwenQPU returns `P(q3=1) = 0.14644647`,
within `1.4e-7` of Qiskit. On a six-qubit GHZ circuit it returns `0.5` for both
the all-zero and all-one outcomes.
[Tasks, methods and full results](docs/BENCHMARK_ADDED_LAYERS.md) ·
[Complete requests and answers](evidence/added-layers/completed-demonstrations.json)

**Deterministic cost.** Every task uses exactly 2,063 + 514 QPU predictions per
gate; the measured 543,120 predictions match this model on all 48 tasks. Per
correct answer, QwenQPU takes 520 s against Qwen's 2,785 s, and it uses 30× fewer
language tokens (8,007 vs 238,639). [Full cost table](docs/BENCHMARK_ADDED_LAYERS.md#execution-cost).

## How it works

![QPU in standard Qwen layers](assets/architecture-added-layers.png)

The language model and the QPU share one stock `Qwen3_5ForCausalLM` checkpoint and
the standard Transformers forward pass. Gated DeltaNet states hold amplitude
digits, instructions and the program counter. Fixed SwiGLU weights perform the
gate arithmetic and probability readout, reused at every amplitude address.
Numerical output tokens feed back into the model to write the next state.

Every amplitude update, Born probability, event sum and sample selection executes
in the weights. The host acts as the I/O bus: it parses the request, loads the
instruction ROM, schedules queries and decodes the output digits. It never
computes an amplitude, a probability or a sampled outcome.

QwenQPU builds on [Qwenputer](https://github.com/divagr18/qwenputer)'s compiled
layers and DeltaNet memory. It replaces Qwenputer's 11-layer CPU prefix with four
layers dedicated to amplitude arithmetic; the assembled model keeps all 32
language layers. [Implementation and completed tests](docs/ADDED_LAYERS.md).

**Scope.** QwenQPU performs exact classical state-vector simulation of supplied
circuits with ideal terminal measurement. The released constructor compiles a
six-qubit amplitude memory. As with any exact state-vector method, memory grows
as 2ⁿ; QwenQPU trades scale for exactness.

## Quickstart

```bash
git clone https://github.com/xi-zhao/QwenQPU && cd QwenQPU
python research/standard_layers/build_prefix.py --out ./models/qpu-prefix
python research/standard_layers/demo_repaired_full_model.py \
  --base ./models/Qwen3.5-4B-Qwenputer \
  --prefix ./models/qpu-prefix \
  --out ./reproduced/added-layer-demo
```

The constructor writes 1.42 GB of QPU weights with no training and no download.
Assembly takes the pinned Qwenputer checkpoint from a local directory
([setup](docs/ADDED_LAYERS.md#build-and-run)). The full model stays in FP32 so the
fixed-point arithmetic is exact, and runs on a single 80 GB A100 (peak 38.4 GB).
[Weight manifest](evidence/added-layers/weight-manifest.json).

## Citing

```bibtex
@software{qwenqpu2026,
  author = {xi-zhao},
  title  = {{QwenQPU}: QPU inside a language model},
  year   = {2026},
  url    = {https://github.com/xi-zhao/QwenQPU}
}
```

The language checkpoint and the compiled-layer approach come from
[Qwenputer](https://github.com/divagr18/qwenputer).
[Attribution and component terms](docs/LICENSES.md).
