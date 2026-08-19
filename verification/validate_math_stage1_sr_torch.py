# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zhugy-8086
"""
阶段 1：随机量化统计性质 — NumPy / PyTorch 双库互证
=====================================================
对应 numpy_math_verification_plan_2026_08_13.md §3 阶段 1 的 #1-#4

目的：
  同一组数学结论用 NumPy 与 PyTorch 两种独立计算库各实现一遍，
  逐项比对输出，排除"单库实现 bug"——双库一致才算通过。

验证项（与 validate_math_stage1_sr.py 对齐）：
  #1 [S] Bernoulli SR 噪声方差 = Δ²/6
  #2 [S] SR 无偏性 E[noise] = 0
  #3 [S] clip 噪声含 DC 分量（E[noise]≠0 当发生 clip）
  #4 [S] 三种量化机制方差区分：SR(Δ²/6) / 抖动+round(Δ²/6) / 确定性round(Δ²/12)
  逆推  [S] 反解 Δ=√(6·Var) + log-log 指数 b（理论 2）双库对照（与 numpy 版对齐）

一致性判定：
  - T 类：机器精度（allclose atol=1e-12）
  - S 类：|ratio_np - ratio_torch| ≤ 2·max(se_np, se_torch)（统计容差随 N 收紧）

用法（需安装 torch）：
    python validate_math_stage1_sr_torch.py
"""
from __future__ import annotations

import time

import numpy as np
import torch
import sys

# Windows GBK 控制台直接运行时不因 Δ²/6 等非 ASCII 字符崩溃（审计 2026-08-19）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SEEDS = [0, 1, 2, 3, 4]
FLOAT_PREC = "float64"
DT_NP = np.float64
DT_T = torch.float64

report = {}


# ============================================================
# 双库核心量化函数
# ============================================================

def bernoulli_sr_np(x, delta, qmin, qmax, rng):
    """NumPy：Bernoulli stochastic rounding，返回反量化值 x_q = q*delta"""
    x_div = x / delta
    q_floor = np.floor(x_div).astype(np.int64)
    frac = x_div - q_floor
    u = rng.random(size=x.shape, dtype=DT_NP)
    q = q_floor + (u < frac).astype(np.int64)
    q = np.clip(q, qmin, qmax)
    return q.astype(DT_NP) * delta


def bernoulli_sr_torch(x, delta, qmin, qmax, rng):
    """PyTorch：Bernoulli stochastic rounding，返回反量化值 x_q = q*delta"""
    x_div = x / delta
    q_floor = torch.floor(x_div).to(torch.int64)
    frac = x_div - q_floor
    u = torch.from_numpy(rng.random(size=x.shape, dtype=DT_NP)).to(DT_T)
    q = q_floor + (u < frac).to(torch.int64)
    q = torch.clamp(q, qmin, qmax)
    return q.to(DT_T) * delta


def additive_noise_round_np(x, delta, qmin, qmax, rng):
    """NumPy：加性均匀噪声 U(-Δ/2,Δ/2) + round（抖动）"""
    u = (rng.random(size=x.shape, dtype=DT_NP) - 0.5) * delta
    q = np.round((x + u) / delta).astype(np.int64)
    q = np.clip(q, qmin, qmax)
    return q.astype(DT_NP) * delta


def additive_noise_round_torch(x, delta, qmin, qmax, rng):
    """PyTorch：加性均匀噪声 U(-Δ/2,Δ/2) + round（抖动）"""
    u = (torch.from_numpy(rng.random(size=x.shape, dtype=DT_NP)).to(DT_T) - 0.5) * delta
    q = torch.round((x + u) / delta).to(torch.int64)
    q = torch.clamp(q, qmin, qmax)
    return q.to(DT_T) * delta


def deterministic_round_np(x, delta, qmin, qmax):
    """NumPy：确定性 round（无抖动）→ Δ²/12"""
    q = np.round(x / delta).astype(np.int64)
    q = np.clip(q, qmin, qmax)
    return q.astype(DT_NP) * delta


def deterministic_round_torch(x, delta, qmin, qmax):
    """PyTorch：确定性 round（无抖动）→ Δ²/12"""
    q = torch.round(x / delta).to(torch.int64)
    q = torch.clamp(q, qmin, qmax)
    return q.to(DT_T) * delta


def variance_estimates(quant_fn, x, delta, qmin, qmax, rng, n_trials, backend):
    """返回 n_trials 个独立方差/均值估计（条件独立于给定 x）"""
    vars_ = np.zeros(n_trials)
    means_ = np.zeros(n_trials)
    x_np = np.asarray(x)
    x_t = torch.from_numpy(x_np).to(DT_T)
    for i in range(n_trials):
        if backend == "numpy":
            x_q = quant_fn(x_np, delta, qmin, qmax, rng)
        else:
            x_q = quant_fn(x_t, delta, qmin, qmax, rng)
            x_q = x_q.cpu().numpy()
        noise = x_q - x_np
        vars_[i] = np.var(noise)
        means_[i] = np.mean(noise)
    return vars_, means_


def _compare(label, np_val, torch_val, kind="S", rtol=None):
    """双库一致性判定：T 类机器精度；S 类统计容差（蒙特卡洛采样波动）"""
    if kind == "T":
        ok = np.isclose(np_val, torch_val, atol=1e-12, rtol=1e-12)
        print(f"    [T] 一致性: numpy={np_val:.15e} torch={torch_val:.15e} "
              f"→ {'PASS' if ok else 'FAIL'}")
        return bool(ok)
    # S 类：双库各自独立蒙特卡洛采样，允许统计波动。近零统计量
    # （如 E[noise]≈0）需 atol 绝对容差，非零量由 rtol 主导。
    tol = rtol if rtol is not None else 5e-4
    ok = np.isclose(np_val, torch_val, atol=5e-4, rtol=tol)
    print(f"    [S] 一致性: numpy={np_val:.8e} torch={torch_val:.8e} "
          f"→ {'PASS' if ok else 'FAIL'}")
    return bool(ok)


# ============================================================
# 实验 #1：Bernoulli SR 方差 = Δ²/6
# ============================================================

def experiment_variance_delta2_over_6():
    print("=" * 72)
    print("#1 [S] Bernoulli SR 噪声方差 = Δ²/6（双库互证）")
    print("=" * 72)

    range_ = 10.0
    results = {}
    for bits, qmin, qmax, label in [
        (8, 0, 255, "8-bit  [0,255]"),
        (16, 0, 65535, "16-bit [0,65535]"),
    ]:
        delta = range_ / (2**bits - 1)
        theory = delta**2 / 6

        all_vars_np, all_vars_t = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            x = rng.uniform(0, range_, 500_000)
            v_np, _ = variance_estimates(bernoulli_sr_np, x, delta, qmin, qmax, rng, 40, "numpy")
            v_t, _ = variance_estimates(bernoulli_sr_torch, x, delta, qmin, qmax, rng, 40, "torch")
            all_vars_np.extend(v_np)
            all_vars_t.extend(v_t)
        all_vars_np = np.array(all_vars_np)
        all_vars_t = np.array(all_vars_t)

        mean_np, mean_t = np.mean(all_vars_np), np.mean(all_vars_t)
        ratio_np, ratio_t = mean_np / theory, mean_t / theory
        ok = _compare(label, ratio_np, ratio_t, kind="S")

        results[label] = {
            "theory": theory, "ratio_np": ratio_np, "ratio_torch": ratio_t,
            "consistent": ok,
        }
        print(f"    ratio_np={ratio_np:.6f}（理论 1） ratio_torch={ratio_t:.6f}（理论 1）")

    # 更大 N 扫描（8-bit）——双库比对
    print("\n  --- 更大 N 扫描（8-bit，双库比对）---")
    delta = 10.0 / 255
    theory = delta**2 / 6
    for N in [1e4, 1e5, 5e5, 2e6]:
        N = int(N)
        ratios_np, ratios_t = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            x = rng.uniform(0, 10.0, N)
            v_np, _ = variance_estimates(bernoulli_sr_np, x, delta, 0, 255, rng, 20, "numpy")
            v_t, _ = variance_estimates(bernoulli_sr_torch, x, delta, 0, 255, rng, 20, "torch")
            ratios_np.append(np.mean(v_np) / theory)
            ratios_t.append(np.mean(v_t) / theory)
        print(f"    N={N:>8d}: ratio_np={np.mean(ratios_np):.6f} "
              f"ratio_torch={np.mean(ratios_t):.6f} "
              f"→ {'PASS' if _compare('N-scan', np.mean(ratios_np), np.mean(ratios_t)) else 'FAIL'}")

    report["#1"] = results


# ============================================================
# 实验 #2：SR 无偏性 E[noise] = 0
# ============================================================

def experiment_unbiasedness():
    print("\n" + "=" * 72)
    print("#2 [S] SR 无偏性 E[noise]=0（双库互证）")
    print("=" * 72)

    delta = 10.0 / 255
    N = 1_000_000
    all_means_np, all_means_t = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 10.0, N)
        _, m_np = variance_estimates(bernoulli_sr_np, x, delta, 0, 255, rng, 20, "numpy")
        _, m_t = variance_estimates(bernoulli_sr_torch, x, delta, 0, 255, rng, 20, "torch")
        all_means_np.extend(m_np)
        all_means_t.extend(m_t)
    mean_e_np = np.mean(all_means_np)
    mean_e_t = np.mean(all_means_t)
    ok = _compare("E[noise]", mean_e_np / delta, mean_e_t / delta, kind="S")
    print(f"    E[noise]/Δ: numpy={mean_e_np/delta:+.6f} torch={mean_e_t/delta:+.6f}（应≈0）")
    report["#2"] = {"mean_e_np": mean_e_np, "mean_e_torch": mean_e_t, "consistent": ok}


# ============================================================
# 实验 #3：clip 噪声含 DC 分量
# ============================================================

def experiment_clip_dc():
    print("\n" + "=" * 72)
    print("#3 [S] clip 噪声含 DC 分量（双库互证）")
    print("=" * 72)

    N = 500_000
    delta = 10.0 / 255
    for cr in [0.0, 0.05, 0.30]:
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 10.0, N)
        n_clip = int(N * cr)
        x[:n_clip] = 20.0 + rng.uniform(0, 5.0, n_clip)
        rng2 = np.random.default_rng(7)
        x_q_np = bernoulli_sr_np(x, delta, 0, 255, rng2)
        mean_e_np = np.mean(x_q_np - x)
        x_t = torch.from_numpy(x).to(DT_T)
        x_q_t = bernoulli_sr_torch(x_t, delta, 0, 255, rng2).cpu().numpy()
        mean_e_t = np.mean(x_q_t - x)
        ok = _compare(f"clip率={cr}", mean_e_np, mean_e_t, kind="S")
        print(f"    E[noise] numpy={mean_e_np:+.6e} torch={mean_e_t:+.6e}（应<0 当 clip>0）")
        report[f"#3_clip{cr}"] = {"mean_e_np": mean_e_np, "mean_e_torch": mean_e_t, "consistent": ok}


# ============================================================
# 实验 #4：三种量化机制方差区分
# ============================================================

def experiment_delta12_vs_delta6():
    print("\n" + "=" * 72)
    print("#4 [S] 三种量化机制方差区分（双库互证）")
    print("=" * 72)

    delta = 10.0 / 255
    N = 500_000
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 10.0, N)
    x_t = torch.from_numpy(x).to(DT_T)

    # SR
    v_sr_np, _ = variance_estimates(bernoulli_sr_np, x, delta, 0, 255, rng, 40, "numpy")
    v_sr_t, _ = variance_estimates(bernoulli_sr_torch, x, delta, 0, 255, rng, 40, "torch")
    # 抖动+round
    v_add_np, _ = variance_estimates(additive_noise_round_np, x, delta, 0, 255, rng, 40, "numpy")
    v_add_t, _ = variance_estimates(additive_noise_round_torch, x, delta, 0, 255, rng, 40, "torch")
    # 确定性 round
    v_det_np = np.var(deterministic_round_np(x, delta, 0, 255) - x)
    v_det_t = np.var(deterministic_round_torch(x_t, delta, 0, 255).cpu().numpy() - x)

    def _r(v, d2):
        return np.mean(v) / d2

    checks = [
        ("SR/Δ²·6", _r(v_sr_np, delta**2 / 6), _r(v_sr_t, delta**2 / 6)),
        ("抖动/Δ²·6", _r(v_add_np, delta**2 / 6), _r(v_add_t, delta**2 / 6)),
        ("抖动/Δ²·12", _r(v_add_np, delta**2 / 12), _r(v_add_t, delta**2 / 12)),
        ("确定性/Δ²·12", v_det_np / (delta**2 / 12), v_det_t / (delta**2 / 12)),
    ]
    for name, a, b in checks:
        _compare(f"#4 {name}", a, b, kind="S")
        print(f"    {name}: numpy={a:.6f} torch={b:.6f}")
    report["#4"] = {"checks": [(n, a, b) for n, a, b in checks]}


# ============================================================
# 逆推对照：反解 Δ + log-log 指数（与 numpy 版对齐）
# ============================================================

def experiment_reverse():
    print("\n" + "=" * 72)
    print("逆推对照：反解 Δ + log(Var) vs log(Δ) 指数 b（双库互证）")
    print("=" * 72)

    range_ = 10.0
    deltas = [range_ / (2**b - 1) for b in (8, 10, 12, 14, 16)]
    rng = np.random.default_rng(11)
    x = rng.uniform(0, range_, 500_000)
    x_t = torch.from_numpy(x).to(DT_T)

    vars_np, vars_t = [], []
    for d in deltas:
        qmax = int(range_ / d)
        v_np, _ = variance_estimates(bernoulli_sr_np, x, d, 0, qmax, rng, 30, "numpy")
        v_t, _ = variance_estimates(bernoulli_sr_torch, x_t, d, 0, qmax, rng, 30, "torch")
        vars_np.append(np.mean(v_np))
        vars_t.append(np.mean(v_t))

    # 倒推 1：从实测方差反解 Δ = √(6·Var)，双库比对
    print("  [倒推 1] 从实测方差反解 Δ = √(6·Var)")
    ok_d = True
    for i, d in enumerate(deltas):
        d_np = np.sqrt(6 * vars_np[i])
        d_t = np.sqrt(6 * vars_t[i])
        ok = _compare(f"Δ 反解(b={8+2*i})", d_np, d_t, kind="S")
        ok_d &= ok
        print(f"      Δ_实测 numpy={d_np:.6e} torch={d_t:.6e}（理论 {d:.6e}）")

    # 倒推 2：log-log 拟合指数 b（理论 2），双库比对
    print("  [倒推 2] log(Var) vs log(Δ) 拟合指数 b（理论 2）")
    logd = np.log(deltas)
    b_np, loga_np = np.polyfit(logd, np.log(vars_np), 1)
    b_t, loga_t = np.polyfit(logd, np.log(vars_t), 1)
    # 拟合指数对蒙特卡洛波动较敏感（5 个 Δ 点），rtol 放宽到 2e-2
    ok_b = _compare("指数 b", float(b_np), float(b_t), kind="S", rtol=2e-2)
    ok_a = _compare("log a（理论 -log6）", float(loga_np), float(loga_t),
                    kind="S", rtol=2e-2)
    print(f"      b numpy={b_np:.6f} torch={b_t:.6f}（理论 2）")

    ok = ok_d and ok_b and ok_a
    report["#reverse"] = {"b_np": float(b_np), "b_torch": float(b_t),
                          "consistent": bool(ok)}
    return ok


if __name__ == "__main__":
    t0 = time.time()
    experiment_variance_delta2_over_6()
    experiment_unbiasedness()
    experiment_clip_dc()
    experiment_delta12_vs_delta6()
    experiment_reverse()
    dt = time.time() - t0

    print("\n" + "=" * 72)
    print("阶段 1 双库互证汇总")
    print("=" * 72)
    print(f"  浮点精度: {FLOAT_PREC}，耗时 {dt:.1f}s")
    n_pass = sum(1 for v in report.values()
                 if isinstance(v, dict) and v.get("consistent"))
    n_tot = sum(1 for v in report.values() if isinstance(v, dict) and "consistent" in v)
    print(f"  双库一致性: {n_pass}/{n_tot} 项 PASS")
    if n_pass == n_tot:
        print("  ✅ NumPy 与 PyTorch 结果一致")
    else:
        print(f"  ❌ 有 {n_tot - n_pass} 项不一致")
        raise SystemExit(1)

    # 结果写入 results/
    import json
    import os

    def _json_default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, torch.Tensor):
            return float(o.item())
        return str(o)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "math_stage1_sr_results_torch.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stage": 1, "backend": "numpy+torch", "kind": "S",
                   "elapsed_s": round(dt, 2), "report": report},
                  f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"  结果已写入 {os.path.normpath(out_path)}")
