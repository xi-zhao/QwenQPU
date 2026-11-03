# Quantum calculation benchmark

The benchmark asks for probabilities of specified measurement events after a
unitary gate program. The source data are the [40 frozen questions](../evidence/public/questions.json),
[complete prompts](../evidence/public/prompts.json), [protocol](../evidence/public/evaluation-protocol-summary.json),
[independent references](../evidence/public/reference_probabilities.json),
[calculation scores](../evidence/public/computation-pairs.csv),
[planning scores](../evidence/public/planning-pairs-original-score.csv), and
[condition status/cost data](../evidence/public/benchmark_trial_status.csv).

## Tasks

Every calculation starts in `|0>^n`. Apply the supplied ordered H/X/T/CNOT list
and report one or two probabilities

`p(S) = P(xor_{j in S} b_j = 1)`.

The corresponding diagonal Pauli-string expectation is `1 - 2*p(S)`, but the
requested outputs are event probabilities. All 20 calculation tasks use terminal
computational-basis readout. They do not ask for a 65,536-entry output distribution.

| Register width n | Calculation tasks | Gate operations in the four tasks | Planning tasks |
|---:|---:|---|---:|
|1|4|5, 13, 29, 61|4|
|3|4|7, 19, 47, 113|4|
|8|4|7, 19, 47, 113|4|
|12|4|11, 31, 79, 161|4|
|16|4|11, 31, 79, 161|4|

These numbers count sequential gate operations, not parallel circuit depth.
Register width and the active qubits for each task are recoverable from its full
gate list. The 1,000-operation core tests are a separate implementation test and
are not the language benchmark's circuit lengths.

The 20 planning tasks have the same width counts. They request 6 product states,
6 phase-pair states, 4 controlled-S states and 4 graph states. The model must
supply a legal preparation using the allowed gates and return the requested
probabilities after the specified readout. Some planning tasks include H readout
rotations. The goal specification also fixes unmentioned qubits to zero.

## Conditions and budget

Both conditions use `divagr1925/Qwen3.5-4B-Qwenputer` revision
`2d9068508f2b680badb7abcded96fdd107e9ba5d`, with greedy decoding (`do_sample=False`)
and `enable_thinking=False`. Plain Qwen has no quantum execution or upstream CPU
execution. The enhanced condition uses the delivered fixed state-vector module.

For a given circuit, the enhanced host executes that exact program once and
provides its terminal event counts to Qwen. For planning, Qwen first produces a
preparation; the host checks it, appends the task's readout, executes the network,
and gives Qwen the resulting counts. Qiskit is never available to either model
at runtime, and exact reference probabilities are not supplied as a substitute
for measurement counts.

Each question/condition has a cumulative cap of 16,384 newly generated token IDs
across all language calls. Prose or thinking-like text, circuit parameters,
formatting and the final answer all count. Tool results are input/context tokens,
not free model-generated tokens. A native EOS can stop early. A tool/plan pause
is not mistaken for the end of the whole condition. There are no hidden answer
repairs or selective model retries.

Each circuit supplies 4,096 terminal draws. The frozen question contains its
sampling seed, `8311100 + position_in_frozen_question_order`; the driver uses
`torch.Generator().manual_seed(seed)` with float64 uniforms. The two requested
events are computed from the same bitstring samples, so their sampling errors
may be correlated. Qwen receives the aggregated event counts, not 4,096 strings.
Language decoding is greedy; question arm order is frozen per question.

This v4 comparison reruns the original question set after earlier budget-limited
runs. It is a same-question re-evaluation, not an unseen test set.

## Independent reference and numerical criterion

Qiskit 2.6.1 complex128 state vectors provide the exact circuit/goal references.
All requested scalar probabilities must satisfy the same absolute criterion in
both conditions:

`abs(answer - p_reference) <= 0.03270826686567064`.

The tolerance was fixed from a Hoeffding union bound:

`epsilon = sqrt(log(2*K/alpha)/(2*shots))`, with `K=160`, `alpha=0.05`, `shots=4096`.

The bound covers the prespecified maximum scalar comparisons across both arms;
the actual 40 questions contain 72 probabilities per arm. The same threshold is
applied to the plain model's numerical answer, which does not itself draw shots.
The event counts' finite-sample deviation is distinct from the model returning
an incorrect probability or failing to produce an accepted answer.

For planning, joint success additionally requires a legal circuit, no more than
128 preparation gates, and independent target-state fidelity at least 0.99995,
up to global phase. A wrong circuit remains a failed planning task even if its
reported numbers happen to agree. Enhanced calculation success also requires
the requested circuit to have been executed correctly.

The quantum execution audit separately checks component/probability error at
5e-5 and norm drift at 1e-4, and reconstructs samples from saved uniforms. All
38 actual calls pass. These are numerical-network checks, not LLM answer scores.

## Answer parsing and denominators

The original answer parser extracts JSON candidates from model text. It marks
multiple candidate objects ambiguous even when repeated objects are identical.
It can accept a unique answer inside prose; its score is not identical to the
separate `strict_schema` field.

| Calculation scoring view, n=20 each | Qwen | QwenQPU |
|---|---:|---:|
| Original frozen joint task score | 1/20 | 19/20 |
| Post-hoc identical-JSON merge diagnostic | 8/20 | 19/20 |
| Merge diagnostic correct and naturally finished | 7/20 | 19/20 |

The second view was added after observing the repeated-identical-JSON defect.
It merges identical candidate answer objects and keeps conflicting objects
ambiguous. It is a narrow diagnostic rather than a complete manual semantic
review, and it does not replace the frozen score. A model can state useful
numbers in prose that this diagnostic still does not recognize. Original
planning joint scores are 1/20 in both conditions and were not recomputed using
the calculation diagnostic.

All 20 tasks remain in each category's denominator. Missing/illegal instructions,
missing or conflicting numerical answers, loop stops and length exhaustion are
not dropped. A normal EOS is a termination result, not proof of correctness.
Across 80 conditions there are 60 natural final answers, 15 repeated-token loop
stops, 3 length stops and 2 invalid planning outputs. The two invalid planning
outputs terminate their plan generation but never produce a valid final answer.
All loop and length stops are in the plain condition. The per-condition CSV
retains termination reasons and cost separately from correctness.

For the merged calculation diagnostic, paired outcomes are 12 enhanced-only,
1 plain-only, 7 both-correct and 0 both-wrong. The paired difference is 55 percentage
points; the descriptive percentile-bootstrap 95% interval is 30–80 points,
using 20,000 resamples and seed 7640301. Requiring natural final completion gives
13 enhanced-only, 1 plain-only and 6 both-correct, a 60-point difference with
interval 35–85. Per-rate intervals in the figures use Wilson 95% intervals.

## Worked example C16-4

Input: `|0>^16`; 161 ordered gate operations, including 39 H, 38 X, 38 T and
46 CNOT gates. All 16 qubits occur in the program. Requested events:

- `o1 = P(q14 = 1)`.
- `o2 = P(q14 xor q8 xor q10 xor q7 = 1)`.

The complete gate list is executed in the order shown:

```json
[
  "CNOT 4 6",
  "X 1",
  "X 5",
  "H 2",
  "T 9",
  "H 11",
  "CNOT 6 10",
  "CNOT 7 9",
  "H 9",
  "H 9",
  "T 7",
  "CNOT 10 4",
  "H 7",
  "CNOT 14 0",
  "T 2",
  "T 15",
  "H 3",
  "H 3",
  "CNOT 1 9",
  "CNOT 9 13",
  "CNOT 3 15",
  "X 5",
  "X 11",
  "CNOT 0 14",
  "X 12",
  "T 2",
  "H 2",
  "X 13",
  "H 0",
  "H 14",
  "T 12",
  "H 15",
  "X 15",
  "H 14",
  "X 15",
  "T 5",
  "T 3",
  "T 2",
  "X 3",
  "CNOT 1 11",
  "CNOT 9 10",
  "CNOT 8 2",
  "T 2",
  "T 6",
  "CNOT 3 15",
  "H 1",
  "CNOT 0 5",
  "X 4",
  "CNOT 10 11",
  "H 1",
  "CNOT 6 7",
  "H 12",
  "X 9",
  "H 11",
  "CNOT 12 10",
  "X 8",
  "T 1",
  "CNOT 12 9",
  "CNOT 1 15",
  "T 11",
  "CNOT 4 12",
  "T 13",
  "H 10",
  "T 14",
  "T 14",
  "X 6",
  "CNOT 6 8",
  "CNOT 7 6",
  "T 13",
  "H 13",
  "X 1",
  "T 12",
  "CNOT 10 6",
  "CNOT 6 12",
  "X 2",
  "X 8",
  "H 14",
  "H 0",
  "T 13",
  "H 4",
  "CNOT 1 12",
  "X 14",
  "X 13",
  "X 14",
  "T 9",
  "X 9",
  "CNOT 1 3",
  "T 7",
  "H 5",
  "CNOT 12 3",
  "X 2",
  "CNOT 4 13",
  "H 11",
  "X 2",
  "X 9",
  "H 10",
  "H 2",
  "CNOT 11 12",
  "CNOT 8 15",
  "T 6",
  "H 1",
  "CNOT 8 10",
  "T 1",
  "T 13",
  "H 1",
  "H 11",
  "T 10",
  "H 11",
  "H 14",
  "X 0",
  "T 9",
  "CNOT 9 12",
  "CNOT 6 13",
  "X 0",
  "X 10",
  "X 3",
  "H 12",
  "T 3",
  "CNOT 13 2",
  "H 8",
  "CNOT 2 11",
  "H 11",
  "CNOT 7 13",
  "H 0",
  "X 1",
  "T 9",
  "T 3",
  "CNOT 14 12",
  "CNOT 1 10",
  "T 2",
  "CNOT 6 2",
  "T 12",
  "T 9",
  "X 15",
  "X 5",
  "X 13",
  "T 8",
  "CNOT 3 10",
  "CNOT 14 4",
  "H 0",
  "T 3",
  "T 10",
  "CNOT 15 12",
  "X 6",
  "CNOT 11 13",
  "X 12",
  "X 0",
  "CNOT 15 12",
  "X 0",
  "CNOT 3 2",
  "CNOT 9 7",
  "H 8",
  "T 3",
  "X 4",
  "T 6",
  "X 8",
  "H 0",
  "H 7",
  "H 12",
  "T 12",
  "X 14"
]
```

Independent Qiskit reference: `o1=1`, `o2=0.5`.
The enhanced execution uses seed 8311126 and 4,096 shots:

| Event | Event=0 count | Event=1 count | Sample estimate |
|---|---:|---:|---:|
|o1|0|4096|1.0|
|o2|2076|2020|0.4931640625|

Final answer-object excerpts from the actual model outputs:

```json
{"Qwen": {"answers": {"o1": 0.5, "o2": 0.5}},
 "QwenQPU": {"answers": {"o1": 1.0, "o2": 0.4931640625}}}
```

Qwen's first probability is wrong by 0.5; QwenQPU's largest error is
0.0068359375. Both the original and merged scoring views mark Qwen incorrect and
QwenQPU correct. Both outputs finish naturally. The source-record hashes and
excerpts are in [benchmark_examples.json](../evidence/public/benchmark_examples.json).
The animation uses this one case; the aggregate table uses all 20 calculation tasks.

The trace also checks splitting the same program after gate 80 and continuing
with 81 more gates. Its final state equals uninterrupted execution, and its
4,096 event counts equal the original evaluation record.

## Phase-interference example C01-4

This one-qubit task applies 61 gates and asks for `P(q0=1)`. The full circuit is
also in `benchmark_examples.json`. Qiskit's reference is 0.6401650429449551.
The plain model's final answer object reports 0.8536. QwenQPU reports 0.6491,
from 2,659 ones in 4,096 samples (seed 8311139). The former numerical error is
about 0.2134 and the latter about 0.00893. The merged diagnostic marks the plain
answer wrong and the enhanced answer correct. The file also includes C08-1 and
the enhanced failure C01-2, where execution is correct but Qwen reports 1 instead
of the reference 0.5.

## Reproduce an example

With existing PyTorch:

```bash
python examples/reproduce_benchmark_case.py --case C16-4
python examples/reproduce_benchmark_case.py --case C01-4
```

These commands run the fixed network and verify its event counts against the
published records. They display the retained answers without generating new
language-model responses. With an existing Qiskit installation, recompute the
independent state-vector probabilities too:

```bash
python examples/reproduce_benchmark_case.py --case C16-4 --qiskit
```

The scripts do not download models or install dependencies. The original full
comparison ran on A100 80 GB with PyTorch 2.6.0+cu124 and Transformers 5.19.0.
Condition timing fields and CUDA-synchronized `network_seconds` are retained
separately from language token counts. GPU metrics
are PyTorch allocated-memory peaks/increments, not all process/device memory.
The existing calculation summary CSV and the raw result records retain slightly
different condition latency values: all 40 calculation rows differ, while their
generated-token counts agree. The raw-record field is named
`result_record_end_to_end_seconds` in the new status table. That timing-scope
difference is unresolved; this report uses neither table to claim a speed gain.
The correctness, circuits, event counts and answer comparisons do not use those
latency fields.
