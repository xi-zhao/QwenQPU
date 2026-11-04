<p align="center">
  <img src="assets/banner.png" alt="QwenQPU — QPU inside a language model" width="100%">
</p>

[English](README.md) · [新增层实现](docs/ADDED_LAYERS.md) · [评测方法](docs/BENCHMARK_ADDED_LAYERS.md) · [Hugging Face](https://huggingface.co/xi-zhao/QwenQPU)

QwenQPU 为 Qwen3.5-4B 加入固定权重 QPU。四个标准 Qwen 层存储量子振幅，执行 H/X/T/CNOT 线路，并输出测量事件的概率。Qwen 读取题目、提交线路，新增层在模型的正向传播中给出数值答案。

## Benchmark

从全零态出发，执行题目给定的线路，计算两个指定测量事件的概率。新版对照评测包含 48 条线路，使用 3、4、6 比特，以及 12、18、24 个顺序门操作。两组使用相同语言权重，每题总预算均为 16,384 次预测，QwenQPU 的原生执行预测也计入预算。

**中途结果：截至 11 月 4 日 00:26 UTC，完成 48 题中的 27 对。**

| 计算任务 | Qwen | QwenQPU |
|---|---:|---:|
| 非 Clifford 线路（已完成 20 题） | 1/20 | **20/20** |
| Clifford 对照（已完成 7 题） | 0/7 | **7/7** |
| 已完成配对合计（27 题） | 1/27 | **27/27** |

两个概率都须与独立 Qiskit 参考值相差不超过 `5e-5`。QwenQPU 提交的线路和测量事件也须与题目一致。上述已完成题中，QwenQPU 的最大概率误差为 `1.326e-5`。评测仍在运行。[题目、协议与中途数据](docs/BENCHMARK_ADDED_LAYERS.md)。

这些已完成题显示，新增层提高了线路概率计算的正确率。六比特 H3–T3–H3 示例中，QwenQPU 给出 `P(q3=1) = 0.1464464664`，参考值为 `0.1464466094`；六比特 GHZ 线路的全零和全一概率均为 `0.5`。[完整请求与结果](evidence/added-layers/completed-demonstrations.json)。

## 实现

语言模型与 QPU 使用同一个 `Qwen3_5ForCausalLM` checkpoint 布局。Gated DeltaNet 状态保存振幅数位、指令和程序计数器；固定 SwiGLU 权重执行小门运算和概率读出。同一套运算在不同振幅地址上复用，数值输出 token 回送到模型，完成下一状态写入。

宿主解析请求、加载指令、调度测量查询并解码输出数位。态更新与概率计算由模型完成。组装后的模型保留 32 个语言层，以四个 QPU 层替换继承的 CPU 前缀，使用标准 Transformers forward。[实现及已完成测试](docs/ADDED_LAYERS.md)。

## 运行

```bash
git clone https://github.com/xi-zhao/QwenQPU && cd QwenQPU
python research/standard_layers/build_prefix.py --out ./models/qpu-prefix
python research/standard_layers/demo_repaired_full_model.py \
  --base ./models/Qwen3.5-4B-Qwenputer \
  --prefix ./models/qpu-prefix \
  --out ./reproduced/added-layer-demo
```

使用已有本地语言 checkpoint，依赖见[运行说明](docs/ADDED_LAYERS.md#build-and-run)。构造器生成约 1.42 GB 固定权重；完整组装模型使用约 36.67 GB FP32 权重。新版完整序列化 checkpoint 尚待发布，HF 根目录目前仍是此前组合版。[权重清单](evidence/added-layers/weight-manifest.json)。

## 此前组合版

<p align="center"><img src="assets/demo.gif" alt="此前组合版：16 比特线路计算。" width="100%"></p>

此前版本在复合 forward 中调用固定小门网络。视频展示执行 16 比特、161 个门的线路后，计算两个测量事件的概率。
[视频](assets/demo.mp4) · [此前评测与完整例题](docs/BENCHMARK.md) · [组合版运行机制](docs/ARCHITECTURE.md)。

## 引用

```bibtex
@software{qwenqpu2026,
  author = {xi-zhao},
  title  = {{QwenQPU}: QPU inside a language model},
  year   = {2026},
  url    = {https://github.com/xi-zhao/QwenQPU}
}
```

继承的语言 checkpoint 和标准层构造思路来自 [Qwenputer](https://github.com/divagr18/qwenputer)。
[来源与各组件条款](docs/LICENSES.md)。
