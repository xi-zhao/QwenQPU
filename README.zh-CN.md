<p align="center">
  <img src="assets/banner-added-layers.gif" alt="QwenQPU — QPU inside a language model" width="100%">
</p>

[English](README.md) · [新增层实现](docs/ADDED_LAYERS.md) · [评测方法](docs/BENCHMARK_ADDED_LAYERS.md) · [Hugging Face](https://huggingface.co/xi-zhao/QwenQPU)

语言模型做数值计算靠近似，要拿到精确结果，通常得把计算交给外部工具。QwenQPU 把精确计算放进了 checkpoint 本身：四个标准 Qwen3.5 层，权重全部由编译器解析给出，用 Gated DeltaNet 存储量子态向量，在正向传播中执行通用门集 H/X/T/CNOT。Qwen 读题并提交线路，新增层返回测量概率。

## Benchmark

![Qwen vs QwenQPU](assets/r5/benchmark-main.png)

**Qwen：4/48（8.3%）。QwenQPU：48/48（100%）。** 精确 McNemar 检验 p = 1.1×10⁻¹³。

每道题从全零态出发，执行题目给定的线路，计算两个测量事件的概率。48 条线路覆盖 3、4、6 比特，以及 12、18、24 个顺序门。题目只按物理性质筛选，并在任何模型推理之前冻结。两组使用相同的语言权重；QPU 的每一次前向预测都和语言 token 计入同一个 16,384 次预算，新增层没有额外算力。

| 计算任务 | Qwen | QwenQPU |
|---|---:|---:|
| 非 Clifford 线路（36 题） | 2/36 | **36/36** |
| Clifford 对照（12 题） | 2/12 | **12/12** |
| 全部配对任务（48 题） | 4/48 | **48/48** |

两个概率都与独立 Qiskit 参考值相差不超过 `5e-5` 才算通过。QwenQPU 的标准更严：提交的线路必须复现题目的终态，测量事件也必须与题目一致。它的最大误差为 `1.42e-5`。

新增层修正了 Qwen 全部 44 道错题，原本答对的 4 道没有一道变错。六比特 H3–T3–H3 线路上，QwenQPU 给出 `P(q3=1) = 0.14644647`，与 Qiskit 相差 `1.4e-7`；六比特 GHZ 线路的全零和全一概率均为 `0.5`。
[题目、方法与完整结果](docs/BENCHMARK_ADDED_LAYERS.md) · [完整请求与答案](evidence/added-layers/completed-demonstrations.json)

**成本确定。** 每道题的 QPU 预测数恰好是 2,063 + 514 × 门数，实测共 543,120 次，48 题全部与此吻合。按每个正确答案计，QwenQPU 用时 520 秒，Qwen 用时 2,785 秒；QwenQPU 的语言 token 少 30 倍（8,007 对 238,639）。[完整成本表](docs/BENCHMARK_ADDED_LAYERS.md#execution-cost)。

## 实现

![QPU in standard Qwen layers](assets/architecture-added-layers.png)

语言模型与 QPU 共用一个标准 `Qwen3_5ForCausalLM` checkpoint 和 Transformers 原生正向传播。Gated DeltaNet 状态保存振幅数位、指令和程序计数器；固定 SwiGLU 权重执行门运算和概率读出，同一套运算在每个振幅地址上复用。数值输出 token 回送到模型，写入下一状态。

振幅更新、Born 概率、事件求和与采样选择全部在权重中执行。宿主只充当 I/O 总线：解析请求、加载指令 ROM、调度查询、解码输出数位，从不计算振幅、概率或采样结果。

QwenQPU 建立在 [Qwenputer](https://github.com/divagr18/qwenputer) 的编译层与 DeltaNet 内存之上，用四个专用于振幅运算的层取代其 11 层 CPU 前缀；组装后的模型保留全部 32 个语言层。[实现及已完成测试](docs/ADDED_LAYERS.md)。

**适用范围。** QwenQPU 对给定线路做精确的经典状态向量模拟，测量为理想终端测量。发布的构造器编译出六比特振幅内存。与所有精确状态向量方法一样，内存随比特数按 2ⁿ 增长；QwenQPU 以规模换取精确。

## 运行

```bash
git clone https://github.com/xi-zhao/QwenQPU && cd QwenQPU
python research/standard_layers/build_prefix.py --out ./models/qpu-prefix
python research/standard_layers/demo_repaired_full_model.py \
  --base ./models/Qwen3.5-4B-Qwenputer \
  --prefix ./models/qpu-prefix \
  --out ./reproduced/added-layer-demo
```

构造器生成 1.42 GB QPU 权重，无需训练，也无需下载。组装时从本地目录读取固定版本的 Qwenputer checkpoint（[运行说明](docs/ADDED_LAYERS.md#build-and-run)）。完整模型保持 FP32，使定点运算精确；单张 80 GB A100 即可运行（峰值 38.4 GB）。[权重清单](evidence/added-layers/weight-manifest.json)。

## 引用

```bibtex
@software{qwenqpu2026,
  author = {xi-zhao},
  title  = {{QwenQPU}: QPU inside a language model},
  year   = {2026},
  url    = {https://github.com/xi-zhao/QwenQPU}
}
```

语言 checkpoint 和编译层构造思路来自 [Qwenputer](https://github.com/divagr18/qwenputer)。
[来源与各组件条款](docs/LICENSES.md)。
