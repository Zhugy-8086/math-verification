<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 zhugy-8086
-->

# Integer Quantization: Rigorous Mathematical Verification

整数量化的严格数学验证——随机舍入、误差反馈、位拆分、精度分配、噪声整形、梯度恒等式、网络属性。

> 本仓库是对整数量化（integer quantization）相关数学结论的 **独立、可复现、双库互证** 的数值验证。
> 不引用任何框架/项目源码，仅依赖通用数值计算库（NumPy / PyTorch），clone 后开箱即跑。
> 每个结论用 **NumPy 与 PyTorch 两种独立计算库各实现一遍**，逐项比对，结果一致才算通过。
>
> ⚠️ **口径说明（2026-08-19 审计修正）**：
> 1. **双库一致 ≠ 理论成立**。"PASS" 仅代表 numpy 与 torch 算同一公式结果一致（排除单库实现 bug）；
>    理论是否成立由每个 numpy 脚本自身的 **效果量 + 统计容差判定** 决定（见「验证清单」的"理论判定"列）。
> 2. 本仓库是 **有限样本数值实验**（固定 seed + 有限 N），验证结论在容差内的自洽性，
>    不是数学证明；"严格/rigorous" 指判定方法学的严谨性，不指证明性质。
> 3. E 类经验标度律的数值（如 90327×、指数放大）**依赖具体配置**，报告中给出量级与配置范围，不当作普适常数。

## 声明（Acknowledgement）

本仓库的数学验证源于长期基于实践的探索与迭代沉淀，并非凭空生成。在推导与双库对拍的论证过程中，AI 以协作工具的身份参与了部分计算推演；而所有结论的灵感、方向与验证设计，均来自人类——AI 是让论证更严谨的辅助，而非结论的来源。如实披露这一协作过程，是对学术诚实的坚持，也坦然面向 AI 辅助研究走向常态的未来。

## 目录结构

```
.
├── formulas/                      # 核心数学公式（纯推导，自包含）
│   └── 整数量化核心公式.md
├── verification/                  # 方法论文档 + 验证脚本
│   ├── numpy_math_verification_plan_2026_08_13.md   # 验证方法论（T/S/E 分级标准）
│   ├── rigorous_math_verification_report_2026_08_03.md  # 严格数学验证报告（阶段 0-9）
│   ├── c1_c2_c3_math_verification_2026_08_01.md     # 相对误差均匀性三方向验证
│   └── validate_math_*.py         # 12 个验证脚本（6 主题 × numpy / torch 双库）
└── results/                       # 运行结果（JSON，含判定与耗时）
```

## 结论分级（T / S / E）

验证方法论见 `verification/numpy_math_verification_plan_2026_08_13.md`：

| 类别 | 含义 | 判定标准 |
|------|------|---------|
| **T** | 定理 / 恒等式 | 机器精度：allclose(atol≤1e-12)（非严格 bit-exact，用于排除实现误差） |
| **S** | 统计定律 | 假设检验 + 置信区间，多种子 |
| **E** | 经验标度律 | 验证关系/标度，报告 CI 与拟合优度 R² |

## 双库互证

| 库 | 角色 | 说明 |
|----|------|------|
| NumPy | 基线计算库 | 全部脚本必选 |
| PyTorch | 互证对照库 | 与 numpy 算同一公式，逐项比对（T 类机器精度 / S·E 类统计容差） |

每个主题提供 `validate_math_<topic>.py`（numpy 单库，含理论判定）与
`validate_math_<topic>_torch.py`（双库互证）两个脚本。

> **两类 PASS 的职责分工**：
> - `*_torch.py` 的 PASS = **双库一致**（numpy≈torch），只排除"单库实现 bug"，**不代表理论成立**；
> - `*.py`（numpy 单库）的 PASS = **理论判定**（效果量 + 统计容差/数值地板），才是"结论成立"的依据；
> - `test_math_verification.py` 同时校验两者：脚本退出码（理论判定）+ 结果 JSON 的 `pass` 字段。

## 快速开始

```bash
# 依赖
pip install numpy torch

# 单库验证（numpy）
python verification/validate_math_stage5_gradient_identities.py

# 双库互证（numpy + torch）
python verification/validate_math_stage5_gradient_identities_torch.py
```

所有脚本固定随机种子、公开参数，任何环境复跑可对拍；结果 JSON 写入 `results/`。

> **Windows 编码**：脚本输出含 Δ²/6 等非 ASCII 字符。Windows 控制台默认 GBK 时直接
> `python script.py` 可能报 UnicodeEncodeError；脚本已在头部 `reconfigure(encoding='utf-8')`，
> 若仍异常请先 `set PYTHONIOENCODING=utf-8` 或直接用 `verification/run_all.bat`。
> Linux/macOS 默认 UTF-8，无需处理。

## 验证清单

| # | 主题 | 结论示例 | 分类 | 理论判定 | 脚本 |
|---|------|---------|------|---------|------|
| 1 | 随机舍入（SR） | SR 噪声方差 = Δ²/6、无偏性 E[η]=0、clip 含 DC 分量 | S | 效果量+容差 | `stage1_sr.py` / `_torch.py` |
| 2 | 位操作代数 | 位拆分可逆 value=high·2^p+low、int16≡两个 int8、scale=max/(2^(b-1)-1) | T | 机器精度 | `stage2_bitsplit.py` / `_torch.py` |
| 3 | 精度分配 | 贪心 bits 分配 = 全局最优（gap=0）、误差∝1/(2^b-1)、16→32-bit 精度提升≈2^16 | T/E | 机器精度/拟合 | `stage3_precision_budget.py` / `_torch.py` |
| 4 | 噪声整形 | EF≡delta-sigma（NTF=(1-z⁻¹)^N, STF=1）、低频抑制、**clip 噪声放大 ~1e4-1e5×（配置相关，实测 9.0e4±3× 量级）**、分离残差修复 | S/E | 效果量+容差 | `stage4_ntf_noise_shaping.py` / `_torch.py` |
| 5 | 梯度恒等式 | 残差 skip 恒等梯度、损失梯度÷总元素数、CE 梯度=(softmax-one_hot)/B | T | 机器精度 | `stage5_gradient_identities.py` / `_torch.py` |
| 6 | 网络属性 | **超度量性违反率≈100%（= 证伪，非成立）**、深层残差放大（**量级/深度配置相关：4→8 层 R² 0.76→0.57 退化**） | S/E | 效果量+容差 | `stage6_network_properties.py` / `_torch.py` |

> **E 类数值的口径（2026-08-19 审计修正）**：
> - 90327× 是特定 T/幅度配置下的低频功率放大比；脚本 T∈[2e4,2e5] 扫描显示其随 T
>   在 log10 跨度 <0.5（约 3×）内波动（实测 107378× / 96894×）——结论是"~1e4-1e5×
>   量级放大"，不是精确常数 90327×。
> - 深层残差"指数放大"是经验标度律：每层放大>1.2 在 4/6/8 层成立，但 R² 随深度退化
>   （4 层 0.76 → 8 层 0.57），指数律不严格——脚本自身标注"标度律退化"。
> - 超度量"违反率 100%"是对"梯度满足超度量不等式"假设的**证伪**结果（定理 7.4 复现），
>   且依赖具体距离定义（|log₂|gᵢ|-log₂|gⱼ||）；不表述为超度量性"成立"。

> **双库一致性**：全部 6 个主题 numpy/torch 双库一致 22/22 项 PASS（2026-08-15，
> 含 stage1 逆推对照：反解 Δ 与 log-log 指数）。该数字只代表"双库实现一致"，理论成立与否
> 以每行"理论判定"列 + numpy 脚本退出码为准。

## 审计记录（2026-08-19）

针对"PASS 语义 / 统计方法 / 严格性宣称 / 配置相关结论 / 交付"五项问题做全量审计并修复：

1. **PASS 语义**：明确"双库一致 ≠ 理论成立"；numpy 单库脚本补上汇总理论判定 + 非零退出码
   （stage1 原 exit 恒 0），`test_math_verification.py` 改为同时解析结果 JSON 的 `pass` 字段。
2. **统计方法（stage1）**：`variance_estimates` 由"固定 x 复用 40 次"改为**每 trial 重采样全新 x**，
   SE 才包含完整统计不确定度（原实现 SE 低估导致 8-bit z=-7.73 的假拒绝）；判定改为
   **效果量 + max(2σ, 数值地板)**，数值地板 1e-3 覆盖 Δ=range/(2^b-1) 非二进制表示的 float64 伪影。
3. **严格性宣称**：README 顶部加"数值实验 ≠ 数学证明"口径说明。
4. **配置相关结论**：README 对 E 类数值（90327×、指数放大、超度量违反率）补量级范围与配置依赖。
5. **交付**：12 个脚本头部统一 `stdout.reconfigure(encoding='utf-8')`，Windows GBK 直接运行不再崩。

修复后 stage1 判定（2026-08-19 复跑）：#1-#5 全 PASS（8-bit ratio 0.99990、16-bit 1.00011、
无偏性 |E[noise]/Δ|=1.4e-5，均 ≤ max(2σ, 1e-3)）。

## 参考与灵感来源（References）

本仓库的验证主题多为通用数学结论，可独立证明；其中与外部工作存在直接灵感关联的如下：

| 主题 | 灵感来源 | 说明 |
|------|---------|------|
| 随机舍入（SR）· 阶段 1 | **Ghaffari, A., Tahaei, M. S., Tayaranian, M., Asgharian, M., & Partovi Nia, V.**, *Is Integer Arithmetic Enough for Deep Learning Training?*, NeurIPS 2022, arXiv:2207.08822（华为方舟实验室 / Noah's Ark Lab） | 本仓库 SR 量化形式、无偏梯度与整数训练框架的灵感直接源于该工作：其方法以 stochastic rounding 产生梯度的无偏估计，配合定点线性映射完成全整数训练管线 |
| 噪声整形 / 误差反馈（EF）· 阶段 4 | delta-sigma 调制（Delta-Sigma Modulation，信号处理经典框架） | NTF=(1-z⁻¹)^N 高通整形与 EF 闭环误差反馈即 delta-sigma 调制思想在梯度量化中的直接应用 |

其余主题（位操作代数、精度分配、梯度恒等式、超度量性）为自包含的数学推导与统计验证，无外部直接引用；SR 的方差/无偏性等基础性质亦可追溯到数值分析中的经典随机舍入方法。

## License

[MIT](LICENSE)
