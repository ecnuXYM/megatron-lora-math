# Qwen2.5-7B Base Plain LoRA 错误分析

## 总览

- 样本数: 600
- raw base: 201/600 = 33.50%
- LoRA iter_0000499: 271/600 = 45.17%
- 绝对提升: 11.67 pp, 多做对 70 题
- raw wrong -> LoRA correct: 113
- raw correct -> LoRA wrong: 43
- both correct: 158
- both wrong: 286
- raw pred=null: 255/600 = 42.50%
- LoRA pred=null: 42/600 = 7.00%
- raw response words mean/median: 187.1/194.0
- LoRA response words mean/median: 79.5/62.0

## 按 Subject

| subject | total | raw | LoRA | delta | fixed | regressed | raw null | LoRA null |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 137 | 80/137 = 58.39% | 99/137 = 72.26% | +13.87 pp | 29 | 10 | 33 (24.09%) | 1 (0.73%) |
| Counting & Probability | 73 | 18/73 = 24.66% | 25/73 = 34.25% | +9.59 pp | 13 | 6 | 32 (43.84%) | 1 (1.37%) |
| Intermediate Algebra | 129 | 23/129 = 17.83% | 35/129 = 27.13% | +9.30 pp | 21 | 9 | 74 (57.36%) | 23 (17.83%) |
| Number Theory | 79 | 23/79 = 29.11% | 25/79 = 31.65% | +2.53 pp | 13 | 11 | 29 (36.71%) | 4 (5.06%) |
| Prealgebra | 119 | 53/119 = 44.54% | 75/119 = 63.03% | +18.49 pp | 28 | 6 | 37 (31.09%) | 6 (5.04%) |
| Precalculus | 63 | 4/63 = 6.35% | 12/63 = 19.05% | +12.70 pp | 9 | 1 | 50 (79.37%) | 7 (11.11%) |

## 按 Level

| level | total | raw | LoRA | delta | fixed | regressed | raw null | LoRA null |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| level 3 | 287 | 118/287 = 41.11% | 144/287 = 50.17% | +9.06 pp | 52 | 26 | 98 (34.15%) | 20 (6.97%) |
| level 4 | 313 | 83/313 = 26.52% | 127/313 = 40.58% | +14.06 pp | 61 | 17 | 157 (50.16%) | 22 (7.03%) |

## LoRA 修复样例

- `hendrycks_math_test_plain_1024` Algebra level 4: label `920.25`, raw `910.13`, LoRA `920.25`
  - How many dollars in interest are earned in two years on a deposit of $\$$10,000 invested at 4.5$\%$ and compounded an...
- `hendrycks_math_test_plain_1040` Algebra level 4: label `84`, raw `None`, LoRA `84`
  - The solutions of $x(2x-7)=3$ may be expressed in the form $\frac{m+\sqrt{n}}{p}$ and $\frac{m-\sqrt{n}}{p}$, where $m...
- `hendrycks_math_test_plain_1053` Algebra level 4: label `-2`, raw `None`, LoRA `-2`
  - Solve the following system of equations for $c$: \begin{align*} a - b &= 2 (c+d)\\ b &= a-2 \\ d &= c+5 \end{align*}
- `hendrycks_math_test_plain_1101` Algebra level 4: label `2`, raw `None`, LoRA `2`
  - For complex numbers $z$, let \[f(z) = \left\{ \begin{array}{cl} z^{2}&\text{ if }z\text{ is not real}, \\ z+2 &\text{...
- `hendrycks_math_test_plain_1135` Algebra level 3: label `0`, raw `None`, LoRA `0`
  - What is the greatest integer value of $x$ for which $6x^2 + x - 2 < 0$?

## LoRA 退化样例

- `hendrycks_math_test_plain_1150` Algebra level 4: label `2`, raw `2`, LoRA `1`
  - If $x^2 - x - 1 = 0$, what is the value of $x^3 - 2x + 1$?
- `hendrycks_math_test_plain_1158` Algebra level 4: label `\frac{\sqrt{2}}{20}`, raw `\frac{\sqrt{2}}{20}`, LoRA `\frac{5-2\sqrt{2}}{40}`
  - Rationalize the denominator: $\frac{1}{\sqrt{8}+\sqrt{50}+\sqrt{18}}$.
- `hendrycks_math_test_plain_1286` Counting & Probability level 3: label `720`, raw `720`, LoRA `360`
  - The Smith family has 4 sons and 3 daughters. In how many ways can they be seated in a row of 7 chairs such that all 3...
- `hendrycks_math_test_plain_1363` Counting & Probability level 3: label `12`, raw `12`, LoRA `24`
  - In how many ways can 5 different keys be placed on a keychain? (Two arrangements are considered the same if one can b...
- `hendrycks_math_test_plain_1428` Counting & Probability level 4: label `672`, raw `672`, LoRA `105`
  - How many ways are there to put five beads on a necklace if there are eight distinct beads to choose from, and rotatio...
