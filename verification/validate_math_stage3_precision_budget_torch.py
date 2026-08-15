# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zhugy-8086
"""
阶段 3：精度分配与信息论 — NumPy / PyTorch 双库互证
====================================================
对应计划 numpy_math_verification_plan_2026_08_13.md §3 阶段 3 的 #9-#12

目的：
  同一组精度分配结论用 NumPy 与 PyTorch 两种独立计算库各实现一遍，
  逐项比对输出，排除"单库实现 bug"——双库一致才算通过。

验证项（与 validate_math_stage3_precision_budget.py 对齐）：
  #9  [T] 贪心 bits 分配 = 全局最优（M-凸性，gap 精确为 0）
  #10 [E] 成本信号须含 in_dim 因子（c = grad_l2²·in_dim）
  #11 [E] 误差 ≈ 1/(2^b-1)（bits 主杠杆，log-log 斜率 ≈ -1）
  #12 [E] 16-bit→32-bit 精度提升 ≈ 2^16 ≈ 65536×

一致性判定：
  - T 类：贪心分配结果（bits 向量）双库完全一致 + 两边 gap 均精确为 0
  - E 类：统计容差（相对 5e-4）

用法（需安装 torch）：
    python validate_math_stage3_precision_budget_torch.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

report = {}


# ============================================================
# 双库贪心边际分配（误差模型 f_i(b) = c/(2^b-1)，M-凸性）
# ============================================================

def marginal_gain_np(c, b):
    """NumPy：Δ_i(b) = f_i(b) - f_i(b+1)"""
    if b == 0:
        return float(c)
    return float(c * (2 ** b) / ((2 ** b - 1) * (2 ** (b + 1) - 1)))


def marginal_gain_torch(c, b):
    """PyTorch：Δ_i(b) = f_i(b) - f_i(b+1)（torch 张量运算）"""
    cb = torch.as_tensor(c, dtype=torch.float64)
    if b == 0:
        return cb
    return cb * (2 ** b) / ((2 ** b - 1) * (2 ** (b + 1) - 1))


def greedy_allocate_np(costs, bmins, bmaxs, B):
    n = len(costs)
    bits = list(bmins)
    remaining = B - sum(bmins)
    while remaining > 0:
        best_gain = -1.0
        best_idx = -1
        for i in range(n):
            if bits[i] >= bmaxs[i]:
                continue
            gain = marginal_gain_np(costs[i], bits[i])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx < 0:
            break
        bits[best_idx] += 1
        remaining -= 1
    return bits


def greedy_allocate_torch(costs, bmins, bmaxs, B):
    n = len(costs)
    bits = list(bmins)
    remaining = B - sum(bmins)
    while remaining > 0:
        best_gain = torch.tensor(-1.0, dtype=torch.float64)
        best_idx = -1
        for i in range(n):
            if bits[i] >= bmaxs[i]:
                continue
            gain = marginal_gain_torch(costs[i], bits[i])
            if gain.item() > best_gain.item():
                best_gain = gain
                best_idx = i
        if best_idx < 0:
            break
        bits[best_idx] += 1
        remaining -= 1
    return bits


def layer_error(c, b):
    if b == 0:
        return c
    return c / (2 ** b - 1)


def total_error(costs, bmins, bmaxs, bits):
    return sum(layer_error(c, b) for c, b in zip(costs, bits))


def dp_optimal(costs, bmins, bmaxs, B):
    """DP 精确全局最优（库无关的精确整数规划参照，仅用于评估 gap）"""
    n = len(costs)
    smin = sum(bmins)
    smax = sum(bmaxs)
    if B <= smin:
        return list(bmins)
    if B >= smax:
        return list(bmaxs)
    INF = float("inf")
    dp = [[INF] * (smax + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    parent = {}
    for i in range(1, n + 1):
        c, lo, hi = costs[i - 1], bmins[i - 1], bmaxs[i - 1]
        for s in range(smax + 1):
            if dp[i - 1][s] == INF:
                continue
            for b in range(lo, hi + 1):
                ns = s + b
                if ns > smax:
                    continue
                e = dp[i - 1][s] + layer_error(c, b)
                if e < dp[i][ns]:
                    dp[i][ns] = e
                    parent[(i, ns)] = (i - 1, s, b)
    bits = [0] * n
    i, s = n, B
    while i > 0:
        pi, ps, b = parent[(i, s)]
        bits[i - 1] = b
        i, s = pi, ps
    return bits


# ============================================================
# #9 [T] 贪心 = 全局最优（M-凸性，gap 精确为 0）
# ============================================================
def verify_greedy_optimal():
    print("=" * 72)
    print("#9 [T] 贪心 bits 分配 = 全局最优（M-凸性，双库互证）")
    print("=" * 72)

    rng = np.random.default_rng(2026)
    n_scenarios = 300
    n_exact0 = 0
    n_same_bits = 0
    max_gap = 0.0
    for _ in range(n_scenarios):
        n = int(rng.integers(2, 13))
        costs = np.exp(rng.uniform(-8, 6, size=n)).tolist()
        bmins = [int(rng.integers(4, 9)) for _ in range(n)]
        bmaxs = [int(rng.integers(16, 25)) for _ in range(n)]
        smin, smax = sum(bmins), sum(bmaxs)
        B = int(rng.integers(smin + 1, smax))

        gbits_np = greedy_allocate_np(costs, bmins, bmaxs, B)
        gbits_t = greedy_allocate_torch(costs, bmins, bmaxs, B)
        dbits = dp_optimal(costs, bmins, bmaxs, B)

        gerr_np = total_error(costs, bmins, bmaxs, gbits_np)
        gerr_t = total_error(costs, bmins, bmaxs, gbits_t)
        derr = total_error(costs, bmins, bmaxs, dbits)

        same_bits = gbits_np == gbits_t
        n_same_bits += int(same_bits)
        gap_np = abs(gerr_np - derr) / max(derr, 1e-300)
        gap_t = abs(gerr_t - derr) / max(derr, 1e-300)
        max_gap = max(max_gap, gap_np, gap_t)
        if gap_np == 0.0 and gap_t == 0.0:
            n_exact0 += 1
        if not (same_bits and gap_np <= 1e-15 and gap_t <= 1e-15):
            print(f"  ✗ 场景 n={n} B={B}: bits_np={gbits_np} bits_t={gbits_t} "
                  f"gap_np={gap_np:.2e} gap_t={gap_t:.2e}")

    print(f"  随机场景 {n_scenarios} 个（n∈[2,12]，成本跨 14 个数量级）")
    print(f"  numpy 贪心 == torch 贪心（bits 完全一致）: {n_same_bits}/{n_scenarios} 例")
    print(f"  双库 gap 均精确为 0: {n_exact0}/{n_scenarios} 例")
    print(f"  所有场景 max gap = {max_gap:.3e}（≤1e-15 即 1 ulp 内 = 达到全局最优）")
    ok = (n_same_bits == n_scenarios) and (n_exact0 == n_scenarios) and max_gap <= 1e-15
    print(f"  判定: {'✓ 双库一致且 gap=0（M-凸性成立）' if ok else '✗'}")
    report["#9"] = {"scenarios": n_scenarios, "same_bits": n_same_bits,
                    "exact0": n_exact0, "max_gap": max_gap, "pass": ok}
    return ok


# ============================================================
# #10 [E] 成本信号须含 in_dim 因子（c = grad_l2²·in_dim）
# ============================================================
def verify_in_dim_factor():
    print("\n" + "=" * 72)
    print("#10 [E] 成本信号须含 in_dim 因子（双库互证）")
    print("=" * 72)

    rng = np.random.default_rng(7)
    in_dims = [3, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    B, bmin, bmax = 124, 4, 20
    grad_l2_sets = [
        [1.0] * 10,
        [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 0.001],
        np.exp(rng.uniform(-3, 0, 10)).tolist(),
    ]
    all_pass = True
    for gi, grad_l2 in enumerate(grad_l2_sets):
        c_correct = [g * g * d for g, d in zip(grad_l2, in_dims)]
        c_wrong = [g * g for g in grad_l2]

        bits_c_np = greedy_allocate_np(c_correct, [bmin]*10, [bmax]*10, B)
        bits_w_np = greedy_allocate_np(c_wrong, [bmin]*10, [bmax]*10, B)
        bits_c_t = greedy_allocate_torch(c_correct, [bmin]*10, [bmax]*10, B)
        bits_w_t = greedy_allocate_torch(c_wrong, [bmin]*10, [bmax]*10, B)

        err_c = total_error(c_correct, [bmin]*10, [bmax]*10, bits_c_np)
        err_w = total_error(c_correct, [bmin]*10, [bmax]*10, bits_w_np)
        err_c_t = total_error(c_correct, [bmin]*10, [bmax]*10, bits_c_t)
        err_w_t = total_error(c_correct, [bmin]*10, [bmax]*10, bits_w_t)

        ok = (err_c < err_w and err_c_t < err_w_t
              and bits_c_np == bits_c_t and bits_w_np == bits_w_t)
        all_pass &= ok
        print(f"  grad_l2 组 {gi}: numpy 正确分配 err={err_c:.4e} < 错误分配 err={err_w:.4e} "
              f"(劣化比={err_w/err_c:.2f}x)")
        print(f"      torch 正确分配 err={err_c_t:.4e} < 错误分配 err={err_w_t:.4e} "
              f"(劣化比={err_w_t/err_c_t:.2f}x) 双库 bits 一致={'✓' if bits_c_np == bits_c_t and bits_w_np == bits_w_t else '✗'}")

    print(f"  判定: {'✓ 成本必须含 in_dim 因子（双库一致）' if all_pass else '✗'}")
    report["#10"] = {"pass": all_pass}
    return all_pass


# ============================================================
# 双库对称量化 round-trip
# ============================================================

def quantize_sym_np(x, bits):
    x64 = np.asarray(x, dtype=np.float64)
    max_abs = float(np.abs(x64).max()) if x64.size else 0.0
    if max_abs == 0.0:
        return x64.copy()
    qmax = 2 ** (bits - 1) - 1
    scale = max_abs / qmax
    q = np.clip(np.round(x64 / scale), -(2 ** (bits - 1)), qmax).astype(np.int64)
    return q.astype(np.float64) * scale


def quantize_sym_torch(x, bits):
    xt = torch.as_tensor(x, dtype=torch.float64)
    max_abs = float(xt.abs().max()) if xt.numel() else 0.0
    if max_abs == 0.0:
        return xt.clone()
    qmax = 2 ** (bits - 1) - 1
    scale = max_abs / qmax
    q = torch.clamp(torch.round(xt / scale), -(2 ** (bits - 1)), qmax).to(torch.int64)
    return q.to(torch.float64) * scale


def rel_l2(diff, ref):
    den = float(torch.norm(torch.as_tensor(ref, dtype=torch.float64)))
    return float(torch.norm(torch.as_tensor(diff, dtype=torch.float64))) / den if den > 0 else 0.0


# ============================================================
# #11 [E] 误差 ≈ 1/(2^b-1)（bits 主杠杆）
# ============================================================
def verify_bits_lever():
    print("\n" + "=" * 72)
    print("#11 [E] 误差 ≈ 1/(2^b-1)（log-log 斜率 ≈ -1，双库互证）")
    print("=" * 72)

    rng = np.random.default_rng(11)
    g = rng.normal(0.0, 1.0, size=(4096,)).astype(np.float64)
    bits_list = list(range(4, 25))

    g_t = torch.as_tensor(g, dtype=torch.float64)
    errs_np, errs_t = [], []
    for b in bits_list:
        errs_np.append(rel_l2(quantize_sym_np(g, b) - g, g))
        errs_t.append(rel_l2(quantize_sym_torch(g, b) - g_t, g_t))
    errs_np, errs_t = np.array(errs_np), np.array(errs_t)

    X = np.log2(2.0 ** np.array(bits_list) - 1.0)
    # numpy 拟合
    A = np.vstack([X, np.ones_like(X)]).T
    k_np, logA_np = np.linalg.lstsq(A, np.log2(errs_np), rcond=None)[0]
    Yhat = A @ np.array([k_np, logA_np])
    r2_np = 1 - np.sum((np.log2(errs_np) - Yhat) ** 2) / np.sum((np.log2(errs_np) - np.log2(errs_np).mean()) ** 2)
    # torch 拟合（torch.linalg.lstsq）
    A_t = torch.from_numpy(A).to(torch.float64)
    Y_t = torch.from_numpy(np.log2(errs_t)).to(torch.float64)
    sol, _, _, _ = torch.linalg.lstsq(A_t, Y_t)
    k_t, logA_t = float(sol[0]), float(sol[1])
    Yhat_t = (A_t @ sol).numpy()
    r2_t = 1 - np.sum((np.log2(errs_t) - Yhat_t) ** 2) / np.sum((np.log2(errs_t) - np.log2(errs_t).mean()) ** 2)

    print(f"  numpy 拟合: 斜率 k={k_np:.4f}  R²={r2_np:.6f}")
    print(f"  torch 拟合: 斜率 k={k_t:.4f}  R²={r2_t:.6f}")
    ok_k = abs(k_np - k_t) < 1e-4
    ok = ok_k and abs(k_np + 1.0) < 0.02 and r2_np > 0.999 and r2_t > 0.999
    print(f"  判定: 双库斜率差={abs(k_np-k_t):.2e}，斜率≈-1，R²→1 "
          f"→ {'✓ bits 主杠杆成立（双库一致）' if ok else '✗'}")
    report["#11"] = {"k_np": k_np, "k_torch": k_t, "r2_np": r2_np,
                     "r2_torch": r2_t, "pass": ok}
    return ok


# ============================================================
# #12 [E] 16-bit→32-bit 精度提升 ≈ 2^16 ≈ 65536×
# ============================================================
def verify_16_32_scaling():
    print("\n" + "=" * 72)
    print("#12 [E] 16-bit→32-bit 精度提升 ≈ 2^16 ≈ 65536×（双库互证）")
    print("=" * 72)

    seeds = [1, 2, 3, 4, 5]
    amplitudes = [0.01, 0.1, 1.0]
    n = 20000
    ratios_16_32_np, ratios_16_32_t = [], []
    for amp in amplitudes:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            g = rng.normal(0.0, amp, size=n).astype(np.float64)
            g_t = torch.as_tensor(g, dtype=torch.float64)
            e8_np = rel_l2(quantize_sym_np(g, 8) - g, g)
            e16_np = rel_l2(quantize_sym_np(g, 16) - g, g)
            e32_np = rel_l2(quantize_sym_np(g, 32) - g, g)
            e8_t = rel_l2(quantize_sym_torch(g, 8) - g_t, g_t)
            e16_t = rel_l2(quantize_sym_torch(g, 16) - g_t, g_t)
            e32_t = rel_l2(quantize_sym_torch(g, 32) - g_t, g_t)
            ratios_16_32_np.append(e16_np / e32_np if e32_np > 0 else float("inf"))
            ratios_16_32_t.append(e16_t / e32_t if e32_t > 0 else float("inf"))

    m_np = float(np.mean(ratios_16_32_np))
    m_t = float(np.mean(ratios_16_32_t))
    theory = 2.0 ** 16
    lo, hi = 0.97 * theory, 1.03 * theory
    ok_band = lo <= m_np <= hi and lo <= m_t <= hi
    ok_agree = abs(m_np - m_t) / theory < 1e-3
    print(f"  16-bit→32-bit 放大比 numpy={m_np:.0f}x  torch={m_t:.0f}x（理论 2^16={theory:.0f}）")
    print(f"  双库比值差 = {abs(m_np-m_t):.0f}x（相对 {abs(m_np-m_t)/theory:.2e}）")
    ok = ok_band and ok_agree
    print(f"  判定: 双库均落在 65536±3% 且互相一致 → "
          f"{'✓ 位宽翻倍→误差 2^-b 缩放' if ok else '✗'}")
    report["#12"] = {"ratio_np": m_np, "ratio_torch": m_t, "theory": theory, "pass": ok}
    return ok


def _json_default(o):
    """numpy/torch 标量 → JSON 可序列化类型"""
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, torch.Tensor):
        return float(o.item())
    return str(o)


def main():
    t0 = time.time()
    r9 = verify_greedy_optimal()
    r10 = verify_in_dim_factor()
    r11 = verify_bits_lever()
    r12 = verify_16_32_scaling()
    dt = time.time() - t0

    print("\n" + "=" * 72)
    print("阶段 3 双库互证汇总")
    print("=" * 72)
    print(f"  耗时 {dt:.1f}s")
    print(f"  #9  [T] 贪心=全局最优 gap=0（双库一致）: {'✓' if r9 else '✗'}")
    print(f"  #10 [E] 成本含 in_dim 因子（双库一致） : {'✓' if r10 else '✗'}")
    print(f"  #11 [E] 误差∝1/(2^b-1)（双库一致）     : {'✓' if r11 else '✗'}")
    print(f"  #12 [E] 16-bit→32-bit ≈65536×（双库一致）  : {'✓' if r12 else '✗'}")
    overall = r9 and r10 and r11 and r12
    print(f"\n  总体判定: {'✅ 双库全部一致通过' if overall else '❌ 存在失败'}")
    if not overall:
        return 1

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "math_stage3_precision_budget_results_torch.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stage": 3, "backend": "numpy+torch", "kind": "T/E",
                   "elapsed_s": round(dt, 2), "report": report},
                  f, ensure_ascii=False, indent=2,
                  default=_json_default)
    print(f"  结果已写入 {os.path.normpath(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
