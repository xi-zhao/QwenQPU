# What the measurements support

## Fixed-network physical checks

The existing A100 validation used Qiskit 2.6.1 complex128 as an independent
offline oracle. It checked1,000-gate random circuits at 1, 3, 8, 12 and 16 qubits,
target/control combinations, inverse circuits, reset, isolation, segmentation,
saved-state replay and invalid input. The selected public aggregate is
[`core-validation-summary.json`](../evidence/public/core-validation-summary.json).

The later40-question comparison contained38 actual quantum-module calls.
All 38 passed the retained independent state/sample/count audits. Maximum
component error was 4.531e-7, probability error 7.308e-8 and norm drift 1.621e-6.
This verifies the executor for those calls, not the correctness of every Qwen
plan or interpretation. In C01-2, execution passes but enhanced Qwen reports 1
instead of the reference 0.5; that remains a task failure.

## Same-model paired comparison

40 questions × 2 arms; supplied-circuit 20 primary, planning 20 auxiliary. Identical
language checkpoint, greedy decoding, cumulative 16,384 new tokens per condition,
including parameters and final text; tool returns count as input. shots = 4096.
Compute circuits contain 5–161 gate operations, with 4 tasks at each listed width.
The questions had been seen in an earlier run. This is not a new blind test.

|Calculation subset; denominator 20 per arm|Plain|Enhanced|
|---|---:|---:|
|Original frozen automatic joint score|1|19|
|Post-hoc identical-JSON diagnostic|8|19|
|Diagnostic correct AND natural final EOS|7|19|

The frozen parser treats repeated identical answer objects as ambiguous. A
correct answer can consequently be scored missing. That defect prevents an
unqualified5% versus 95% semantic-accuracy claim.

The narrow post-hoc diagnostic collapses structurally identical candidate JSON
objects, retains the original numeric fallback and rejects distinct candidates.
It does not pick the last/reference-matching answer. All 20 questions remain in
each denominator. It is still incomplete: prose or quoted example JSON can
remain unresolved. C08-4 and C03-3 contain correct values before looping but are
not counted by the narrow diagnostic. Do not call this a full manual semantic score.

Diagnostic pairs:12 enhanced-only, 1 plain-only, 7 both-correct, 0 both-wrong.
Difference 55 percentage points; descriptive paired-bootstrap 95% interval 30–80.
Bootstrap seed 7640301, 20,000 replicates. Post-hoc intervals are exploratory.
The natural-finish diagnostic difference is 60 points, interval 35–85.
Marginal error bars in the graphic are Wilson 95% intervals, not paired intervals.

Original planning joint scores remain 1/20 each. Planning fidelity was not recomputed
for this post-hoc extraction audit. No planning improvement or 4B-capacity cause
is established. The original 40-task overall automatic scores are 2/40 and 20/40,
also affected by the parser issue.

## Stops, recovery, and costs

All 80 conditions reached a terminal record: 60 natural replies, 15 external repeat
stops, 3 length stops, 2 invalid plans. No resource stops in this 16k run.
The first complete saved generation was recovered after a diagnostic-serialization
failure without regenerating it; two sequential model instances used the same
weights. Original failures and freezes are retained privately.

For 20 calculation tasks, actual generated-token medians were 4791 and 33.5.
Measured condition-plus-cleanup medians were 274.243 s and 2.577 s. These are the
specific eager-backend runs, with loading/repair/final packaging excluded.
No runtime speed is asserted by the 24-second presentation animation.

The small sample, bounded budget, external loop rule, seen questions and parser
defect limit the conclusions. This comparison does not demonstrate quantum
advantage, general quantum-task superiority, or superiority over a classical
external simulator. New prespecified holdout work would be needed for broader claims.

## Public data in this preview

The aggregate paired CSV and selected sanitized GHZ/phase replays contain no
account data, session credentials, full language weights or unrelated research.
Private raw evaluation logs are not included in this repository preview.
The original evaluation artifact SHA-256 is
`66da35539745d96c2e417426d1037455109340b3fb9cea787bcb65068b1762da`.
External publication of this candidate package has not been performed.

The synthetic question definitions and exact frozen prompt templates are also
included in`evidence/public/`, alongside a protocol/scoring summary. These are
test definitions, not private raw model logs. The large-model evaluation runner
is not invoked by any quickstart command in this preview.
