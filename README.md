<p align="center">
  <img src="assets/banner.png" alt="QwenQPU: quantum circuits and measurement probabilities." width="100%">
</p>

QwenQPU weights: publication in progress  
Language checkpoint: [Hugging Face](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer)  
[简体中文](README.zh-CN.md) · [Architecture](docs/ARCHITECTURE.md) · [Evaluation method](docs/BENCHMARK.md)

QwenQPU integrates a fixed-weight QPU module with Qwen3.5-4B. Quantum gates are
encoded as neural-network weights, and the module executes circuits in its
forward pass. Qwen generates the circuit and answers from the returned
measurement counts.

<p align="center"><img src="assets/demo.gif" alt="Qwen, QwenQPU and Qiskit results for a 16-qubit, 161-gate calculation." width="100%"></p>

*Calculate `P(q14=1)` and `P(q14 xor q8 xor q10 xor q7=1)` after a 16-qubit,
161-gate circuit. [MP4](assets/demo.mp4) · [Full circuit and answers](docs/BENCHMARK.md#worked-example-c16-4).*

## Benchmarks

The calculation tasks start from `|0...0>`, apply a supplied H/X/T/CNOT circuit,
and calculate one or two specified parity-event probabilities. The 20 calculation
tasks use register sizes 1, 3, 8, 12 and 16, with four tasks at each size and
5–161 sequential gate operations. Another 20 tasks ask for a circuit that prepares
a specified product, phase-pair, controlled-S or graph state, followed by its
requested measurement probabilities.

Both conditions use the same language weights and greedy decoding. Each has a
total generation budget of 16,384 tokens across its calls. QwenQPU executes
the circuit and passes its 4,096-shot event counts to Qwen.

| Task and answer scoring | Qwen | **QwenQPU** |
|---|---:|---:|
| Calculation: identical repeated JSON merged (20 tasks) | 8/20 | **19/20** |
| Calculation: original answer parser (20 tasks) | 1/20 | **19/20** |
| Planning: joint circuit-and-answer success (20 tasks) | 1/20 | 1/20 |

With identical repeated answers merged, calculation success rises from 8/20 to
19/20: 12 QwenQPU-only successes, 1 Qwen-only success,
and 7 shared successes. The paired gain is 55 percentage points (bootstrap 95%
interval 30–80). The calculation score improves with the execution module;
the planning score stays at 1/20 in both conditions.

For the 16-qubit example above, Qwen answers `0.5, 0.5`; QwenQPU answers
`1, 0.4931640625`; Qiskit's reference is `1, 0.5`. QwenQPU's second result is
`2020/4096` from the measurement counts.

[Evaluation method, complete task list and worked examples](docs/BENCHMARK.md).

H, X and T are fixed real matrices acting on pairs of complex amplitudes.
CNOT applies X through a control mask. The same small gate network updates
different positions in the state vector, one instruction at a time.
`QwenQPU.forward` calls the quantum module for state evolution and terminal
sampling. The host parses circuit instructions, supplies random numbers,
collects counts and schedules language generation.

## Quickstart

With Python and PyTorch installed:

```bash
git clone https://github.com/xi-zhao/QwenQPU && cd QwenQPU
python examples/reproduce_benchmark_case.py --case C16-4
```

This runs the circuit shown above and reports its event counts and probabilities.
To run a Qwen-to-QPU dialogue using a local language checkpoint:

```bash
python examples/qwen_local.py \
  --model-path /path/to/an/existing/Qwen3.5-4B-Qwenputer \
  --output ./review/dialogue.json
```

[Setup and usage](docs/REPRODUCING.md) · [Architecture](docs/ARCHITECTURE.md) ·
[Evaluation](docs/BENCHMARK.md) · [Media source](tools/render_assets.py) ·
[Attribution and terms](docs/LICENSES.md)

## Citing

```bibtex
@software{qwenqpu2026,
  author = {xi-zhao},
  title  = {{QwenQPU}: Qwen with a fixed quantum state-vector network},
  year   = {2026},
  url    = {https://github.com/xi-zhao/QwenQPU}
}
```

The language checkpoint comes from [Qwenputer](https://github.com/divagr18/qwenputer).
That project also informed the fixed-computation approach.
Component terms are in [LICENSES.md](docs/LICENSES.md).
