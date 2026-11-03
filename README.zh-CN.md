<p align="center">
  <img src="assets/banner.png" alt="QwenQPU：量子线路与测量概率" width="100%">
</p>

QwenQPU 权重：正在发布  
语言权重：[Hugging Face](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer)  
[English](README.md) · [架构](docs/ARCHITECTURE.md) · [评测方法](docs/BENCHMARK.md)

QwenQPU 为 Qwen3.5-4B 集成了一个固定权重的 QPU 模块。
量子门运算写成神经网络权重，在模块的正向传播中执行线路。
Qwen 生成线路，并根据返回的测量计数回答。

<p align="center"><img src="assets/demo.gif" alt="16比特161门线路上Qwen、QwenQPU与Qiskit的结果对照" width="100%"></p>

*给定一条16比特、161个门的线路，计算 `P(q14=1)` 和
`P(q14 xor q8 xor q10 xor q7=1)`。[MP4](assets/demo.mp4) ·
[完整线路与回答](docs/BENCHMARK.md#worked-example-c16-4)。*

## 基准测试

计算题从 `|0...0>` 出发，执行给定的 H/X/T/CNOT 线路，计算一至两个
指定奇偶事件的概率。20道计算题使用1、3、8、12、16比特寄存器，每种规模4题，
包含5–161个顺序门操作。另有20道规划题：构造指定的乘积态、带相位的两分量态、
controlled-S 态或图态，并给出要求的测量概率。

两组使用同一语言权重和贪心解码，每题每组跨调用累计生成预算为16384个 token。
QwenQPU 执行线路，并向 Qwen 提供4096次测量的事件计数。

|任务与答案评分方式|Qwen|**QwenQPU**|
|---|---:|---:|
|计算题：相同重复 JSON 合并评分（20题）|8/20|**19/20**|
|计算题：原答案解析评分（20题）|1/20|**19/20**|
|规划题：线路与答案联合成功（20题）|1/20|1/20|

合并相同重复答案后，计算题成功数从8/20提高到19/20：
12题仅 QwenQPU 正确，1题仅 Qwen 正确，7题双方均正确。配对提升为55个百分点，
bootstrap 95%区间为30–80个百分点。加入执行模块后，计算题成功率提高，
规划题仍为两组各1/20。

上面的16比特例题中，Qwen 回答 `0.5、0.5`，QwenQPU 回答 `1、0.4931640625`，
Qiskit 参考值为 `1、0.5`。QwenQPU 的第二个概率来自测量计数 `2020/4096`。

[评测方法、完整题集与例题](docs/BENCHMARK.md)。

H、X 和 T 使用固定实矩阵，作用于成对的复振幅；CNOT 通过控制掩码执行 X。
同一个小门网络作用于态向量的不同位置，逐条执行指令。
`QwenQPU.forward` 调用量子模块完成态演化和末端采样。
宿主解析线路、提供随机数、汇总 counts，并调度语言生成。

## 快速开始

已有 Python 和 PyTorch 时：

```bash
git clone https://github.com/xi-zhao/QwenQPU && cd QwenQPU
python examples/reproduce_benchmark_case.py --case C16-4
```

该命令执行上面的线路，输出事件计数与概率。
已有本地语言 checkpoint 时，可运行 Qwen 与量子模块的对话：

```bash
python examples/qwen_local.py \
  --model-path /path/to/an/existing/Qwen3.5-4B-Qwenputer \
  --output ./review/dialogue.json
```

[环境与用法](docs/REPRODUCING.md) · [架构](docs/ARCHITECTURE.md) ·
[评测](docs/BENCHMARK.md) · [媒体源码](tools/render_assets.py) ·
[来源与许可](docs/LICENSES.md)

## 引用

```bibtex
@software{qwenqpu2026,
  author = {xi-zhao},
  title  = {{QwenQPU}: Qwen with a fixed quantum state-vector network},
  year   = {2026},
  url    = {https://github.com/xi-zhao/QwenQPU}
}
```

语言权重来自 [Qwenputer](https://github.com/divagr18/qwenputer)，固定计算模块的思路也参考了该项目。
组件条款见[许可说明](docs/LICENSES.md)。
