# Integer Quantization: Rigorous Mathematical Verification

整数量化的严格数学验证——随机舍入、误差反馈、位拆分、精度分配、噪声整形、梯度恒等式、网络属性。

> 本仓库是对整数量化（integer quantization）相关数学结论的 **独立、可复现、双库互证** 的纯数据验证。
> 不引用任何框架/项目源码，仅依赖通用数值计算库（NumPy / PyTorch），clone 后开箱即跑。
> 每个结论用 **NumPy 与 PyTorch 两种独立计算库各实现一遍**，逐项比对，结果一致才算通过——排除"单库实现 bug"。

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
| **T** | 定理 / 恒等式 | bit-exact 或机器精度（零容差） |
| **S** | 统计定律 | 假设检验 + 置信区间，多种子 |
| **E** | 经验标度律 | 验证关系/标度，报告 CI 与拟合优度 R² |

## 双库互证

| 库 | 角色 | 说明 |
|----|------|------|
| NumPy | 基线计算库 | 全部脚本必选 |
| PyTorch | 互证对照库 | 与 numpy 算同一公式，逐项比对（T 类机器精度 / S·E 类统计容差） |

每个主题提供 `validate_math_<topic>.py`（numpy 单库）与 `validate_math_<topic>_torch.py`（双库互证）两个脚本。

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

## 验证清单

| # | 主题 | 结论示例 | 分类 | 脚本 |
|---|------|---------|------|------|
| 1 | 随机舍入（SR） | SR 噪声方差 = Δ²/6、无偏性 E[η]=0、clip 含 DC 分量 | S | `stage1_sr.py` / `_torch.py` |
| 2 | 位操作代数 | 位拆分严格可逆 value=high·2^p+low、bitsplit/concat 可逆、int16≡两个 int8、scale=max/(2^(b-1)-1) | T | `stage2_bitsplit.py` / `_torch.py` |
| 3 | 精度分配 | 贪心 bits 分配 = 全局最优（gap=0）、误差∝1/(2^b-1)、16→32-bit 精度提升≈2^16 | T/E | `stage3_precision_budget.py` / `_torch.py` |
| 4 | 噪声整形 | EF≡delta-sigma（NTF=(1-z⁻¹)^N, STF=1）、低频抑制、clip 噪声放大 ≈9.0×10⁴×（实测 90327×）、分离残差修复 | S/E | `stage4_ntf_noise_shaping.py` / `_torch.py` |
| 5 | 梯度恒等式 | 残差 skip 恒等梯度、损失梯度÷总元素数、CE 梯度=(softmax-one_hot)/B | T | `stage5_gradient_identities.py` / `_torch.py` |
| 6 | 网络属性 | 超度量性违反率 100%（证伪复核）、深层残差指数放大（量级配置相关） | S/E | `stage6_network_properties.py` / `_torch.py` |

> 全部 6 个主题双库互证 22/22 项 PASS（2026-08-15，含 stage1 逆推对照：反解 Δ 与 log-log 指数）。

## 参考与灵感来源（References）

本仓库的验证主题多为通用数学结论，可独立证明；其中与外部工作存在直接灵感关联的如下：

| 主题 | 灵感来源 | 说明 |
|------|---------|------|
| 随机舍入（SR）· 阶段 1 | **Ghaffari, A., Tahaei, M. S., Tayaranian, M., Asgharian, M., & Partovi Nia, V.**, *Is Integer Arithmetic Enough for Deep Learning Training?*, NeurIPS 2022, arXiv:2207.08822（华为方舟实验室 / Noah's Ark Lab） | 本仓库 SR 量化形式、无偏梯度与整数训练框架的灵感直接源于该工作：其方法以 stochastic rounding 产生梯度的无偏估计，配合定点线性映射完成全整数训练管线 |
| 噪声整形 / 误差反馈（EF）· 阶段 4 | delta-sigma 调制（Delta-Sigma Modulation，信号处理经典框架） | NTF=(1-z⁻¹)^N 高通整形与 EF 闭环误差反馈即 delta-sigma 调制思想在梯度量化中的直接应用 |

其余主题（位操作代数、精度分配、梯度恒等式、超度量性）为自包含的数学推导与统计验证，无外部直接引用；SR 的方差/无偏性等基础性质亦可追溯到数值分析中的经典随机舍入方法。

## License

[MIT](LICENSE)
