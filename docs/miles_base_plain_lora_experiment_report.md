# Miles Qwen2.5-7B Base Plain LoRA 实验报告

日期: 2026-07-05

LoRA 机制补充文档:

```text
/root/own/report/miles_lora_internals_for_slime.md
```

这份补充文档说明 Miles 中 LoRA 如何通过 Megatron-Bridge 注入训练图、如何保存 adapter-only checkpoint、如何在 SGLang rollout 侧用 `miles_lora` 热加载 adapter，以及迁移到 Slime+LoRA 时需要对齐哪些接口。

## 1. 实验目的

前面在 Qwen2.5-7B-Instruct 上做 Miles LoRA SFT 时，LoRA 没有带来提升，反而明显退化:

| 设置 | 评测 | 结果 |
|---|---:|---:|
| Qwen2.5-7B-Instruct base | MATH eval600 | 410/600 = 68.33% |
| Qwen2.5-7B-Instruct LoRA r8 q/v | MATH eval600 | 253/600 = 42.17% |
| Qwen2.5-7B-Instruct LoRA r16 q/v lr=5e-5 | MATH eval600 | 243/600 = 40.50% |

判断: Instruct 模型本身已经有较强的数学/指令能力，小样本 MATH SFT 容易把模型拉向训练集解题风格，破坏原本的 reasoning distribution。继续盲目加 rank 或训练步数不是首选。

因此转向验证:

```text
Qwen2.5-7B base model + plain completion SFT + LoRA
```

核心问题:

```text
base 模型是否比 Instruct 模型更适合用 LoRA 做 MATH SFT？
base 模型是否必须用 plain prompt/completion，而不是复用 chat/messages pipeline？
```

## 2. 关键工程改动

### 2.1 plain SFT 数据

生成脚本:

```text
/root/own/scripts/create_math_plain_sft_eval.py
```

输入:

```text
/root/datasets/hendrycks-MATH-benchmark/data/train-00000-of-00001.parquet
/root/datasets/hendrycks-MATH-benchmark/data/test-00000-of-00001.parquet
```

过滤策略:

```text
level in {3, 4}
exclude geometry
solution must contain boxed answer
train seed = 5151
eval seed = 6060
```

输出:

```text
/root/own/data/math_plain_sft_l3_4_core_train1000_seed5151.jsonl
/root/own/data/math_plain_sft_l3_4_core_train2000_seed5151.jsonl
/root/own/data/math_plain_eval_l3_4_core_eval600_seed6060.jsonl
```

prompt/completion 格式:

```text
Prompt:
Problem:
{problem}

Solution:

Completion:
{solution}

Answer: \boxed{{answer}}
```

### 2.2 Miles plain rollout

新增 rollout:

```text
/root/own/rollout/plain_sft_rollout.py
```

核心逻辑:

```text
tokens = prompt_ids + completion_ids + eos
response_length = len(completion_ids)
loss_mask = [1] * response_length
```

含义:

```text
只在 completion 部分计算 SFT loss。
不使用 chat template。
不把 base 模型强行当成 Instruct/chat 模型。
```

### 2.3 plain eval

评测脚本:

```text
/root/own/scripts/eval_hf_math_plain_lora.py
```

评测方式:

```text
直接输入 plain prompt: "Problem:\n...\n\nSolution:\n"
不调用 apply_chat_template
从生成文本中抽取最后 boxed answer 做 correctness
```

## 3. 训练配置

训练脚本:

```text
/root/own/scripts/run-miles-qwen2.5-7B-base-plain-lora-sft-v2-lr1e-5.sh
```

主要参数:

| 项 | 值 |
|---|---|
| model | `/root/models/Qwen2.5-7B` |
| train data | `/root/own/data/math_plain_sft_l3_4_core_train2000_seed5151.jsonl` |
| method | Miles + Megatron-Bridge LoRA SFT |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| target modules | `q_proj,v_proj` |
| dropout | 0.0 |
| lr | 1e-5 |
| min lr | 1e-6 |
| batch | rollout batch 1, global batch 1 |
| tensor parallel | 2 |
| max tokens per GPU | 1024 |
| save interval | 500 |

输出:

```text
/root/models/Qwen2.5-7B_lora_sft_base_plain_v2_r16_qv_lr1e-5
```

训练结果:

```text
Ray job succeeded.
checkpoints: iter_0000499, iter_0000999, iter_0001499, iter_0001999
all checkpoints total size: 57M
single adapter checkpoint size: 15M
```

adapter config:

```json
{
  "peft_type": "LORA",
  "r": 16,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "v_proj"],
  "lora_dropout": 0.0,
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

## 4. 评测结果

### 4.1 raw base plain baseline

| 模型 | 数据 | 样本数 | 结果 | 输出 |
|---|---|---:|---:|---|
| Qwen2.5-7B base | plain eval | 30 | 6/30 = 20.00% | `/root/own/report/eval_7b_base_plain_raw_30.summary.json` |
| Qwen2.5-7B base | plain eval | 100 | 24/100 = 24.00% | `/root/own/report/eval_7b_base_plain_raw_100.summary.json` |
| Qwen2.5-7B base | plain eval | 600 | 201/600 = 33.50% | `/root/own/report/eval_7b_base_plain_raw_600.summary.json` |

### 4.2 LoRA checkpoint curve: quick eval30

| checkpoint | 样本数 | 结果 | 结论 |
|---|---:|---:|---|
| iter_0000499 | 30 | 13/30 = 43.33% | 最好 |
| iter_0000999 | 30 | 11/30 = 36.67% | 回落 |
| iter_0001499 | 30 | 11/30 = 36.67% | 回落 |
| iter_0001999 | 30 | 10/30 = 33.33% | 继续回落 |

判断:

```text
这个分支存在明显 early stopping 点。
小数据 LoRA 继续训练到 2000 samples 不一定更好，iter_0000499 反而最强。
```

### 4.3 最佳 checkpoint 扩评

最佳 checkpoint:

```text
/root/models/Qwen2.5-7B_lora_sft_base_plain_v2_r16_qv_lr1e-5/iter_0000499/adapter
```

| 模型 | checkpoint | 样本数 | 结果 | 输出 |
|---|---|---:|---:|---|
| Qwen2.5-7B base + LoRA | iter_0000499 | 100 | 43/100 = 43.00% | `/root/own/report/eval_7b_base_plain_lora_v2_r16_qv_lr1e-5_iter0499_100.summary.json` |
| Qwen2.5-7B base + LoRA | iter_0000499 | 600 | 271/600 = 45.17% | `/root/own/report/eval_7b_base_plain_lora_v2_r16_qv_lr1e-5_iter0499_600.summary.json` |

完整同 split 对照:

```text
在同一 plain eval600 上，LoRA 从 raw base 的 33.50% 提升到 45.17%。
绝对提升 11.67 个百分点，多做对 70 题。
在同一 plain eval 的 100 题快速对照上，LoRA 从 raw base 的 24.00% 提升到 43.00%。
这说明 base model + plain completion LoRA 路线是可行的，而且明显优于此前直接套 chat/messages pipeline 的 base LoRA v1。
```

## 5. 为什么之前 base LoRA v1 很差

之前的 base LoRA v1 使用的是 messages/chat-style 数据和评测:

```text
/root/models/Qwen2.5-7B_lora_sft_base_v1_r16_qv_lr1e-5
```

quick eval:

| checkpoint | 结果 |
|---|---:|
| iter_0000499 | 4/30 = 13.33% |
| iter_0000999 | 3/30 = 10.00% |

原因判断:

```text
Qwen2.5-7B base 不是 Instruct/chat 模型。
直接把 messages/chat template pipeline 套给 base model，会造成训练目标和推理输入不匹配。
base 模型更适合 plain completion SFT，即 prompt + solution continuation。
```

## 6. 面试可讲结论

可以这样讲:

```text
我没有把 LoRA 失败简单归因于 rank 不够，而是拆成了模型状态、数据格式和训练目标三个变量。
Instruct 模型上的 LoRA 退化，说明强指令模型做小样本 MATH SFT 容易破坏原有 reasoning distribution。
随后我换到 Qwen2.5-7B base，并且把 chat/messages 数据改成 plain prompt/completion。
为了适配 Miles，我写了一个 plain_sft_rollout，让 loss 只打在 completion 上。
这个改动之后，LoRA r16 q/v 在 100 题 fair contrast 上从 24% 提升到 43%，最佳 checkpoint 在 600 题上达到 45.17%。
补齐 raw base 600 后，同 split 对照是 33.50% 到 45.17%，绝对提升 11.67 个百分点。
所以结论不是“LoRA 一定有效”或“LoRA 没用”，而是 LoRA 是否有效高度依赖 model state 和 SFT 数据格式。
```

如果面试官问为什么 `iter_0499` 最好:

```text
这是典型小数据 SFT 的 early stopping 现象。
训练样本只有 2000 条，global batch 1，后续 checkpoint 的 eval30 从 43.33% 逐步回落到 33.33%。
这说明继续优化训练 loss 不等价于提升泛化 accuracy，后续应该加入 validation-based early stopping。
```

如果面试官问为什么只训 q/v:

```text
q_proj/v_proj 是 attention 里最常见的低成本 LoRA 注入点，参数少、显存压力低，也利于先验证工程链路。
当前 r16 q/v 已经产生正向结果，下一步如果追效果，再比较 q/k/v/o 或 all-linear，而不是一开始就扩大可训练参数。
```

如果面试官问为什么不用主 GRPO checkpoint:

```text
这个实验是主 GRPO 项目的附加实验，目的是验证 Miles/Megatron 下 LoRA SFT 的工程可行性和数据格式影响。
为了避免变量混杂，我没有挪用主 GRPO checkpoint。
这样可以把结论限定在 LoRA、base/Instruct、chat/plain 数据格式这些变量上。
```

## 7. 下一步

优先级从高到低:

1. 尝试更小训练量或 validation early stopping，例如 save interval 250。
2. 对比 `q_proj,v_proj` 和 `q_proj,k_proj,v_proj,o_proj`，但先保持 lr=1e-5。
3. 再考虑更高质量数据筛选，而不是优先增大 rank。

## 8. 错误分析

完整分析文件:

```text
/root/own/report/miles_base_plain_lora_error_analysis.md
/root/own/report/miles_base_plain_lora_error_analysis.json
```

整体交叉结果:

| category | count |
|---|---:|
| raw wrong -> LoRA correct | 113 |
| raw correct -> LoRA wrong | 43 |
| both correct | 158 |
| both wrong | 286 |

关键观察:

```text
LoRA 净增 70 题正确，主要不是单纯“推理变强”，而是显著改善了 base model 的答案格式稳定性。
raw base 的 pred=null 是 255/600 = 42.50%，LoRA 降到 42/600 = 7.00%。
raw base 平均/中位 response words 是 187.1/194.0，LoRA 是 79.5/62.0。
说明 plain LoRA 让 base model 更稳定地按训练格式给出 boxed final answer，并减少无效长输出。
```

按 subject 的收益:

| subject | raw | LoRA | delta |
|---|---:|---:|---:|
| Algebra | 80/137 = 58.39% | 99/137 = 72.26% | +13.87 pp |
| Counting & Probability | 18/73 = 24.66% | 25/73 = 34.25% | +9.59 pp |
| Intermediate Algebra | 23/129 = 17.83% | 35/129 = 27.13% | +9.30 pp |
| Number Theory | 23/79 = 29.11% | 25/79 = 31.65% | +2.53 pp |
| Prealgebra | 53/119 = 44.54% | 75/119 = 63.03% | +18.49 pp |
| Precalculus | 4/63 = 6.35% | 12/63 = 19.05% | +12.70 pp |

面试说法:

```text
错误分析显示 LoRA 最明显的收益是减少 format failure 和无效长输出。
它不是把所有数学能力都解决了: Number Theory 增益很小，Precalculus 虽然相对提升，但绝对正确率仍然低。
所以下一步应该按 subject 做数据增强或课程化采样，而不是只提高 rank。
```

## 9. Rank Ablation 更新

为了验证 LoRA 的典型调参变量，本项目补做了 rank 对照。固定条件如下:

```text
model: /root/models/Qwen2.5-7B
data: math_plain_sft_l3_4_core_train2000_seed5151.jsonl
eval: math_plain_eval_l3_4_core_eval600_seed6060.jsonl
target modules: q_proj,v_proj
lr: 1e-5
alpha: 2 * rank
```

quick eval30:

| rank | iter_0499 | iter_0999 | iter_1499 | iter_1999 | best |
|---:|---:|---:|---:|---:|---|
| 8 | 11/30 | 11/30 | 12/30 | 13/30 | iter_1999 |
| 16 | 13/30 | 11/30 | 11/30 | 10/30 | iter_0499 |
| 32 | 11/30 | 11/30 | 10/30 | 12/30 | iter_1999 |

完整 eval600:

| setting | adapter size | eval600 |
|---|---:|---:|
| raw base | 0 | 201/600 = 33.50% |
| LoRA r8 q/v | 29M | 275/600 = 45.83% |
| LoRA r16 q/v | 57M | 271/600 = 45.17% |

解释:

```text
r8 和 r16 的完整结果非常接近，r8 高 4 题但不能过度解读。
r32 在 quick eval 上没有超过 r8/r16，所以没有继续扩到 eval600。
这个结果支持一个更成熟的结论: LoRA rank 是容量/效率折中项，不是越大越好。
在当前小数据 SFT 里，r8 已经足够有效，r16 可以作为稳健默认，r32 没看到必要性。
```
