# Qwen2.5-7B-Instruct LoRA 退化诊断

日期: 2026-07-12

## 结论先行

这不是一个可以简单归因为“LoRA 会破坏 Instruct reasoning”的结果。当前证据更准确地支持以下结论:

1. adapter 的加载、PEFT 包装和本评测脚本本身不会改变 base 输出。
2. 在本次 `MATH train2000`、SFT completion loss、rank 16 fused-QKV LoRA 的设置中，较大的学习率和累计更新会使 adapter 更新量过大，造成明显的生成行为漂移和准确率回退。
3. 将学习率降至 `1e-6` 后，严重回退基本消失，但 `eval100` 仍未出现明确提升。因此问题不是“只要对 Instruct 做 LoRA 就必然退化”，而是当前训练目标和优化强度尚未找到可稳定带来收益的组合。

这里的“更新量过大”是由学习率对照和 adapter 参数范数共同支持的工程性判断，不等价于已经从机制上证明了模型内部的 reasoning 能力如何被破坏。

## 问题与固定条件

研究对象是 `Qwen2.5-7B-Instruct`。训练和评测均使用 MATH L3/L4/core 切分:

| item | value |
|---|---|
| train | `/root/own/data/math_sft_l3_4_core_train2000_seed5151.jsonl`，2000 条 |
| eval | `/root/own/data/math_eval_l3_4_core_eval600_seed6060.jsonl`，600 条 |
| overlap | 0 条 |
| 训练方式 | Miles `debug_train_only` SFT，completion loss |
| 并行 | TP=2，两张 RTX 5090 |
| 评测 | HF + PEFT，chat template、greedy decoding、`max_new_tokens=1024` |

训练侧 CLI 写的是 `q_proj,v_proj`，但 Megatron 的 standard LoRA 在训练时落到 fused `linear_qkv`。导出 HF adapter 中实际可见 `q_proj/k_proj/v_proj` 的 LoRA 权重。因此本诊断控制的是同一条 Miles fused-QKV LoRA 路线，不应把它表述为严格的 q/v-only 消融。

## 现象复现

| model / adapter | train data | rank | lr | eval600 |
|---|---:|---:|---:|---:|
| Instruct base | - | - | - | 410/600 = 68.33% |
| LoRA smoke | 1000 | 8 | `1e-4` | 253/600 = 42.17% |
| LoRA v2 | 2000 | 16 | `5e-5` | 243/600 = 40.50% |

高学习率两组均显著低于 base，且 r16、更多数据、较低但仍偏高的 `5e-5` 没有改善结果。这说明不能把问题归结为 rank 过小或单纯训练样本不足。

## 控制实验 1: adapter 加载是否就是问题

构造了与 r16 adapter 结构相同、但所有 LoRA tensor 均置零的 adapter:

| setting | eval30 | 相比 base 前 30 |
|---|---:|---|
| base | 18/30 = 60.00% | - |
| zero LoRA adapter | 18/30 = 60.00% | fixed=0，regressed=0，30/30 response 完全一致 |

评测文件:

```text
/root/own/report/eval_7b_instruct_zero_lora_r16_30.jsonl
```

这排除了“PEFT attach 以后模型就变了”“adapter config 的包装逻辑本身造成掉分”这类解释。退化来自训练得到的非零 adapter 参数。

## 控制实验 2: 输出格式或过早终止是否是主因

高学习率 LoRA 不但没有更多抽取失败，反而减少了 `pred=null`，同时显著缩短输出:

| metric | base | r8, lr=`1e-4` | r16, lr=`5e-5` |
|---|---:|---:|---:|
| accuracy | 68.33% | 42.17% | 40.50% |
| `pred=null` | 55/600 = 9.2% | 10/600 = 1.7% | 16/600 = 2.7% |
| response mean words | 277.2 | 71.1 | 71.9 |
| response median words | 256 | 55 | 53 |

与 base 的逐题交叉结果:

| transition | r8, lr=`1e-4` | r16, lr=`5e-5` |
|---|---:|---:|
| base wrong -> LoRA correct | 18 | 26 |
| base correct -> LoRA wrong | 175 | 193 |

因此，“答案提取失败”不是主要解释。高学习率 adapter 确实让模型更常给出短且可解析的答案，但大量 base 原本正确的题目变错，说明主要问题是生成内容和解题行为发生了不利漂移。

## 控制实验 3: 降低更新强度

在相同 train2000、r16、alpha32、fused-QKV LoRA、TP=2 和 SFT 路线上，仅调整:

```text
lr:     1e-6
min_lr: 1e-7
save interval: 250
```

训练产物:

```text
/root/models/Qwen2.5-7B-Instruct_lora_sft_diag_r16_fusedqkv_lr1e-6
```

checkpoint quick eval:

| checkpoint | eval30 |
|---|---:|
| base | 18/30 = 60.00% |
| zero adapter | 18/30 = 60.00% |
| iter_0000249 | 18/30 = 60.00% |
| iter_0000999 | 18/30 = 60.00% |
| iter_0001999 | 19/30 = 63.33% |

final checkpoint 在更大的前 100 样本上:

| setting | eval100 | fixed | regressed |
|---|---:|---:|---:|
| base | 61/100 = 61.00% | - | - |
| low LR iter_0001999 | 60/100 = 60.00% | 6 | 7 |

低学习率最终模型的 response 并非与 base 相同: 只有 5/100 完全一致。其输出长度也接近 base，而不是高学习率组的异常短输出:

| metric | base eval100 | low LR eval100 |
|---|---:|---:|
| `pred=null` | 10/100 = 10.0% | 14/100 = 14.0% |
| response mean words | 299.4 | 291.6 |
| response median words | 270 | 275 |

这说明 `1e-6` 的 adapter 已经在更新和改变回答，但没有造成高学习率那种大面积回退；同时样本量 100 还不足以支持“低学习率带来提升”的结论。

## 参数更新幅度证据

LoRA 的 B 矩阵从全零初始化，训练后的 B 范数可以用作 adapter 更新幅度的直接代理。final checkpoint 的统计如下:

| adapter | B norm mean | B norm median | B norm max |
|---|---:|---:|---:|
| r16, lr=`1e-6` | 0.0421 | 0.0336 | 0.0899 |
| r16, lr=`5e-5` | 0.2144 | 0.1516 | 0.3729 |

高学习率组的 B 更新约为低学习率的 5 倍量级，而 A 范数接近。这与“高学习率导致过大的 adapter 改动，并伴随严重行为漂移”的观察一致。

## 已排除项与剩余变量

已排除或明显弱化:

- train/eval 数据重叠: overlap 为 0。
- 仅加载 adapter 的影响: 零增量对照与 base 输出逐条一致。
- 单纯 final-answer 抽取失败: 高学习率组 `pred=null` 更少。
- 训练样本普遍超过 context/token 上限: chat template 后训练样本 token length 的 median=270、mean=322.9、p95=704；超过 1024 的只有 25/2000 (1.2%)。这仍是应处理的边缘变量，但不是大规模掉分的充分解释。

仍未验证，不能作为既定事实:

- 是否是 completion-only loss 对这个已经 instruction-tuned 的模型不合适。
- 是否是 train2000 的数据质量、解题风格或答案格式与 base 能力冲突。
- strict split Q/V `canonical_lora` 是否优于当前 fused-QKV standard LoRA。
- 在 `1e-6` 与 `1e-5` 的中间学习率是否能取得正提升。
- 用 validation-based checkpoint selection 是否能避免某个训练阶段后的回退。

## 下一步实验

优先级从高到低:

1. 固定所有条件，仅跑 `2e-6` 和 `5e-6`，每 250 step 做 eval100，再只让最优 checkpoint 跑 eval600。
2. 使用 `--lora-type canonical_lora`，确认训练产物和导出的 target modules 确实是严格 split Q/V，再与 fused-QKV 结果比较。
3. 检查并清理长样本，或者提高 `max-tokens-per-gpu` 后在相同 token budget 下重复一个中间学习率。
4. 将训练/验证/测试严格拆开，按验证集选 checkpoint，测试集只评一次。

不建议继续把资源投入 `1e-4` 或 `5e-5` 的 all-linear/rank 堆叠: 当前证据已经显示这些更新强度下主要问题是稳定性，而不是 LoRA 容量不够。

## 增量实验: `lr=2e-6`

为检查 `1e-6` 是否只是“更新过小而没有退化”，在相同设置下新增了唯一变量为学习率的中间点:

```text
lr:     2e-6
min_lr: 2e-7
train:  2000 steps
rank/alpha: 16/32
adapter: fused-QKV standard LoRA
checkpoint interval: 250
```

训练已成功完成，产物位于:

```text
/root/models/Qwen2.5-7B-Instruct_lora_sft_diag_r16_fusedqkv_lr2e-6
```

首个筛选点结果:

| setting | eval30 | 相对 base 前30 |
|---|---:|---:|
| base | 18/30 = 60.00% | - |
| `lr=1e-6`, iter_0001999 | 19/30 = 63.33% | +1 |
| `lr=2e-6`, iter_0000499 | 19/30 = 63.33% | +1 |

该结果说明中间学习率没有立即复现高学习率的严重退化，并且确实形成了非零训练后的 adapter；但 `n=30` 的净增只有 1 题，不能视为有效提升。下一步应对这个 checkpoint 做 eval100，并与 base 的同一 100 题逐题比较，再决定是否值得做 eval600。

## 面试表述

可以这样回答:

> 我没有把 Instruct LoRA 掉分直接归因于“LoRA 破坏推理”。我先构造了零增量 adapter，对照发现 PEFT 加载与评测链路不会改变 base 输出。然后在相同数据上比较学习率，`1e-4`/`5e-5` 的 adapter 让回答显著变短，且大量 base 正确题回退；其 B 矩阵更新约为 `1e-6` 组的五倍。降到 `1e-6` 后，eval100 从 61/100 变为 60/100，严重回退消失但还没有显著提升。因此严谨结论是: 当前 MATH SFT 目标下，高学习率和累计 LoRA 更新造成了有害的行为漂移；后续要用中间学习率、验证集选 checkpoint，并检查 strict Q/V LoRA，而不是笼统说 Instruct 不能做 LoRA。
