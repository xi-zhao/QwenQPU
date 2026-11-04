# Supplied-circuit probability benchmark (r5)

## Tasks

Every task starts from the all-zero state, supplies an ordered H/X/T/CNOT
circuit and asks for two computational-basis event probabilities. Event A
fixes one qubit; event B fixes two. All other qubits are marginalized.

The fixed set has 36 non-Clifford tasks: 3, 4 and 6 logical qubits × 12, 18
and 24 sequential gates × four circuits. Twelve Clifford controls use 18 gates,
four circuits at each register size. Gate count counts sequential operations.

Before model inference, candidates were selected by physical properties:
connected interaction graph, entangled final state, probabilities in
`[0.025, 0.975]`, and, for non-Clifford circuits, at least one requested
probability more than `0.001` from a multiple of `1/16`. Selection never used
model answers. Independent NumPy and Qiskit references agreed on all 48 tasks
and the two separate interface probes (maximum probability difference
`6.66e-16`). The probes are excluded from the benchmark.

[The complete fixed task list](../evidence/added-layers/r5-tasks.json) contains
the circuits, event conditions, prompts and generation seeds. Its SHA256 is
`b6d892f5441f7b66fcc63d3c4a194188b5b84c410376225ae9d779feb01d0759`.
The formal source/protocol manifest was frozen at 2026-11-03 18:43:02 UTC;
its SHA256 is `37f03134726b33a8c78785dfd1f4e41a067e71560fcc3c70f36ef624f3f00421`.

## Paired conditions

Both conditions use the same assembled checkpoint and supplied task. Plain
Qwen has quantum execution disabled. QwenQPU receives the native function
schema and its numerical QPU output is the final answer. The shared prompt
asks for immediate function submission when the function is available.
The [interface](../research/standard_layers/benchmark_native_r5/tool_interface.py)
contains the exact system text and schema.

Both have a total budget of 16,384 generated predictions. This counts language
tokens, function arguments and every native QPU forward prediction. Model
outputs are fed back for execution; host-supplied control inputs are recorded
separately. Expected QPU costs are 8,231, 11,315 and 14,399 predictions for
12, 18 and 24 gates. The budget does not reset at a function call.

Decoding uses the native non-thinking template, temperature 1, top-p 0.95,
top-k 20, presence penalty 1.5 and repetition penalty 1. Each task's recorded
generation seed is shared by its two conditions. Task order is fixed with
shuffle seed 2026110348; which arm runs first alternates. The per-condition
safety limit is 3,600 seconds; infrastructure failures pause scheduling.

This benchmark asks for numerical probabilities, so there is no shots-based
estimate. A task passes when both answers are within absolute `5e-5` of
independent complex128 Qiskit values. QwenQPU's submitted events must match,
and the submitted circuit must produce the same final state on the supplied
input (infidelity below `1e-10`). Identical repeated JSON answers are merged;
conflicting answers fail. Incorrect numbers, malformed calls, truncation and
time exhaustion remain in the denominator. Completed failures are not retried.

## Interim results — 27 of 48 paired tasks

Snapshot: **2026-11-04 00:26:20 UTC**, 54 terminal conditions. The full run is
still in progress; the table describes the completed pairs at this snapshot.

| Task family | Completed / planned tasks | Qwen correct | QwenQPU correct |
|---|---:|---:|---:|
| Non-Clifford (primary) | 20 / 36 | 1 / 20 | 20 / 20 |
| Clifford controls | 7 / 12 | 0 / 7 | 7 / 7 |
| All completed pairs | 27 / 48 | 1 / 27 | 27 / 27 |

There are 26 QwenQPU-only successes and one shared success. Plain Qwen has
25 incorrect probability answers and one conflicting-answer failure.
QwenQPU's maximum probability error is `1.326e-5`. All 27 frozen file hashes
matched, and scoring left the completed source records unchanged.
[Machine-readable snapshot](../evidence/added-layers/r5-interim.json).

The completed pairs show better probability calculation with the added QPU
layers. Final paired statistics, strata and figures will use all 48 tasks.
The [earlier composite benchmark](BENCHMARK.md) used a different execution
path, task set and measurement-count interface and is reported separately.
