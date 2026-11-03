# Circuit comparison media

The current GIF/MP4 presents frozen calculation case C16-4: a 16-qubit register
with 161 H/X/T/CNOT operations, starting in the all-zero state. It requests
P(q14=1) and P(q14 xor q8 xor q10 xor q7=1). All 16 qubits occur in the program.

The plain Qwen answer is 0.5, 0.5. The QwenQPU answer is 1, 0.4931640625.
Independent Qiskit probabilities are 1, 0.5. The processor's event counts are
4096/4096 and 2020/4096 for the requested events, with seed 8311126.
The exact selected question, answer-object excerpts, record hashes and outcomes
are in [benchmark_examples.json](../evidence/public/benchmark_examples.json).

The language outputs are taken from the completed evaluation; no new language
generation is performed for the animation. The unchanged core executes the
full circuit locally to supply intermediate states and samples. Its counts
match the original evaluation. An 80-operation segment followed by 81 more
operations produces exactly the same final state as uninterrupted execution.

The visual timeline shows task, gate execution, continuation, progressive
terminal event counts and three-way answer comparison. It uses illustrative
presentation timing and contains no wall-clock performance display. Diagnostic
intermediate probabilities belong to the visualization; the model in the
evaluation consumed event counts. This is ideal state-vector simulation.

The original artwork and renderer are in [render_comparison.py](../tools/render_comparison.py),
with shared vector/typography helpers in [render_support.py](../tools/render_support.py).
The MP4 is H.264/yuv420p, 1280×720, 12 fps, 30 seconds. The GIF is 1024×576
with 360 frames. [demo-trace.json](../assets/demo-trace.json) records all gate
steps and event samples. [animation-provenance.json](../assets/animation-provenance.json)
records the source hashes.

The previous GHZ16 trace is retained as [ghz16-trace.json](../assets/ghz16-trace.json),
and its generator is [build_ghz_trace.py](../tools/build_ghz_trace.py). Earlier
media remain in repository history and the existing Space attachment.
No upstream animation frame, logo asset or private model-generation log is
used in the new video. The Qwenputer reference supplied presentation inspiration.
