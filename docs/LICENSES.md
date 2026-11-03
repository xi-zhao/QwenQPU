# Licensing and redistribution

The owner has authorized public publication of this repository. No project-wide
open-source license has been selected for the original QwenQPU code and artwork;
this publication does not add a license grant. Upstream materials retain their
own terms. Language-model weights are not redistributed here.

|Material|Verified source / handling|
|---|---|
|Qwenputer code|[MIT](https://github.com/divagr18/qwenputer/blob/main/LICENSE); credit Divyansh Agrawal and retain notices if code is redistributed. No source assets were copied for this showcase.|
|Pinned Qwenputer language checkpoint|[Official model](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer), revision`2d9068508f2b680badb7abcded96fdd107e9ba5d`; metadata declares Apache-2.0. Not bundled.|
|Qwen base weights|[Qwen3.5-4B LICENSE](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE), Apache-2.0. Preserve applicable license/attribution/NOTICE and mark modifications before any full redistribution.|
|Fixed QwenQPU gate module|48 constructed H/X/T coefficients; no training; original core source retained. No project-wide license grant has been selected.|
|New banner, diagrams, video and render source|Original for this project; no copied upstream branding or frames. Fonts are rendered locally, not redistributed as font files.|
|Aggregate evaluation and selected examples|Selected synthetic project evidence approved for publication. No private raw logs or unrelated data are included.|

The pinned official HF API was checked during preparation:8weight shards total
18,715,279,016bytes. The proposed lightweight module release avoids copying
them and instead records their official identity. If full language weights
are later redistributed, inspect and preserve that exact revision's license
and notices, include clear modification/provenance information, and verify all
shards. Repository metadata alone is not a completed redistribution checklist.

This project name describes an independent experiment using upstream components.
It does not claim upstream endorsement. See the[release checklist](../weight-release/RELEASE_CHECKLIST.md).
