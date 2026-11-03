<p align="center">
  <img src="assets/banner.png" alt="QwenQPU: quantum circuits and measurement probabilities." width="100%">
</p>

QwenQPU weights: publication in progress  
Language checkpoint: [Hugging Face](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer)  
[简体中文](README.zh-CN.md) · [Architecture](docs/ARCHITECTURE.md) · [Evaluation method](docs/BENCHMARK.md)

**QwenQPU pairs Qwen with a fixed quantum execution module.** Qwen handles the
question, the state-vector network executes the circuit and samples measurement
outcomes, and Qwen reports the result. H, X, T and CNOT are fixed network operations.

<p align="center"><img src="assets/demo.gif" alt="Qwen, QwenQPU and Qiskit results for a 16-qubit, 161-gate calculation." width="100%"></p>

*Calculate `P(q14=1)` and `P(q14 xor q8 xor q10 xor q7=1)` after a 16-qubit,
161-gate circuit. [MP4](assets/demo.mp4) · [Full circuit and answers](docs/BENCHMARK.md#worked-example-c16-4).*

## Benchmarks

**What is calculated.** Start from `|0...0>`, apply a supplied H/X/T/CNOT circuit,
and calculate one or two specified parity-event probabilities. The 20 calculation
tasks use register sizes 1, 3, 8, 12 and 16, with four tasks at each size and
5–161 sequential gate operations. Another 20 tasks ask for a circuit that prepares
a specified product, phase-pair, controlled-S or graph state, followed by its
requested measurement probabilities.

**Qwen versus QwenQPU.** Both use the same language weights and greedy decoding,
with a shared 16,384-generated-token budget per condition. QwenQPU executes
the circuit and passes its 4,096-shot event counts to Qwen.

| Task and answer scoring | Qwen | **QwenQPU** |
|---|---:|---:|
| Calculation: original answer parser (20 tasks) | 1/20 | **19/20** |
| Calculation: identical repeated JSON merged (20 tasks) | 8/20 | **19/20** |
| Planning: joint circuit-and-answer success (20 tasks) | 1/20 | 1/20 |

[Evaluation method, complete task list and worked examples](docs/BENCHMARK.md).

**What the results show.** With identical repeated answers merged, calculation
success rises from 8/20 to 19/20: 12 QwenQPU-only successes, 1 Qwen-only success,
and 7 shared successes. The paired gain is 55 percentage points (bootstrap 95%
interval 30–80). The execution module improves calculation reliability on these
circuits; designing a correct preparation circuit remains a separate challenge,
with planning at 1/20 for both conditions.

For the 16-qubit example above, Qwen answers `0.5, 0.5`; QwenQPU answers
`1, 0.4931640625`; Qiskit's reference is `1, 0.5`. QwenQPU's second result is
`2020/4096` from the measurement counts.

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

The runtime validates circuit instructions, calls the quantum child through
`QwenQPU.forward`, and passes counts to the next language call. Its register
stores the circuit's complex amplitudes; language generation and quantum
execution are scheduled by the host.

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

Built on Qwen and inspired by [Qwenputer](https://github.com/divagr18/qwenputer).
Component terms are in [LICENSES.md](docs/LICENSES.md).
