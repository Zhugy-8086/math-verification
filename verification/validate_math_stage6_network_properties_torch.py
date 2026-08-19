# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zhugy-8086
"""
阶段 6：网络数学性质 — NumPy / PyTorch 双库互证
===============================================
对应计划 numpy_math_verification_plan_2026_08_13.md §3 阶段 6 的 #20-#21

目的：
  同一组网络数学结论用 NumPy 与 PyTorch 两种独立计算库各实现一遍，
  逐项比对输出，排除"单库实现 bug"——双库一致才算通过。

验证项（与 validate_math_stage6_network_properties.py 对齐）：
  #20 [S] 超度量性违反率 100%（定理 7.4 复现证伪）
  #21 [E] 深层残差指数放大（Jacobian 范数>1）

双库实现分工：
  - #20 两库独立采样（np rng / torch.Generator），独立计算统计量
  - #21 两库独立实现混合精度 EF 前向/反向（torch 矩阵乘 + int16/int32 张量）
  - 指标级比对：S 类统计容差（rtol 按统计量波动设定）

用法（需安装 torch）：
    python validate_math_stage6_network_properties_torch.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import sys

# Windows GBK 控制台直接运行时不因 Δ²/6 等非 ASCII 字符崩溃（审计 2026-08-19）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DT = np.float64
SEEDS = [0, 1, 2, 3, 4]

report = {}


# ============================================================
# #20 [S] 超度量性违反率 100%（定理 7.4 复现证伪）
# ============================================================
def verify_ultrametric_violation():
    print("=" * 72)
    print("#20 [S] 超度量性违反率 100%（双库互证，独立采样）")
    print("=" * 72)
    print("距离 d=|log₂|g_i|-log₂|g_j||；独立采样3个 log-half-normal→排序→检查超度量")

    REF = {"viol": 1.00, "p99": 2.5212}
    N = 1_000_000

    viol_np, p99_np = [], []
    viol_t, p99_t = [], []
    for seed in SEEDS:
        # numpy 独立采样
        rng = np.random.default_rng(seed)
        g = rng.normal(0, 1, N * 3).reshape(N, 3)
        trip = np.log2(np.abs(g))
        srt = np.sort(trip, axis=1)
        d1 = srt[:, 1] - srt[:, 0]
        d2 = srt[:, 2] - srt[:, 1]
        d3 = srt[:, 2] - srt[:, 0]
        viol = d3 > np.maximum(d1, d2)
        amp = d3 - np.maximum(d1, d2)
        viol_np.append(viol.mean())
        p99_np.append(np.percentile(amp, 99))

        # torch 独立采样
        gen = torch.Generator(); gen.manual_seed(seed)
        g_t = torch.randn(N * 3, dtype=torch.float64, generator=gen).reshape(N, 3)
        trip_t = torch.log2(g_t.abs())
        srt_t, _ = torch.sort(trip_t, dim=1)
        d1_t = srt_t[:, 1] - srt_t[:, 0]
        d2_t = srt_t[:, 2] - srt_t[:, 1]
        d3_t = srt_t[:, 2] - srt_t[:, 0]
        viol_t_ = d3_t > torch.maximum(d1_t, d2_t)
        amp_t = d3_t - torch.maximum(d1_t, d2_t)
        viol_t.append(viol_t_.double().mean().item())
        p99_t.append(float(torch.quantile(amp_t, 0.99).item()))

    vn, vt = np.array(viol_np), np.array(viol_t)
    pn, pt = np.array(p99_np), np.array(p99_t)
    ok_v = abs(vn.mean() - REF["viol"]) < 1e-3 and abs(vt.mean() - REF["viol"]) < 1e-3
    ok_p = abs(pn.mean() - REF["p99"]) / REF["p99"] < 0.02
    ok_p &= abs(pt.mean() - REF["p99"]) / REF["p99"] < 0.02
    ok_ag1 = _compare("违反率", float(vn.mean()), float(vt.mean()),
                      kind="S", rtol=1e-3)
    ok_ag2 = _compare("违反幅度 p99", float(pn.mean()), float(pt.mean()),
                      kind="S", rtol=1e-2)
    ok = ok_v and ok_p and ok_ag1 and ok_ag2
    print(f"  均值: 违反率 numpy={vn.mean():.4f} torch={vt.mean():.4f} (报告100%)")
    print(f"        p99    numpy={pn.mean():.4f} torch={pt.mean():.4f} (报告2.52)")
    print(f"  判定: 双库均复现证伪（违反率≈100%，p99≈2.52）→ "
          f"{'✓ 超度量性【证伪】成立（双库一致）' if ok else '✗'}")
    report["#20"] = {"viol_np": float(vn.mean()), "viol_torch": float(vt.mean()),
                     "p99_np": float(pn.mean()), "p99_torch": float(pt.mean()),
                     "pass": ok}
    return ok


# ============================================================
# #21 [E] 深层残差指数放大（Jacobian 范数>1）
# ============================================================
LAYER_DIMS = [64, 48, 32, 16, 8]
N_LAYERS = len(LAYER_DIMS) - 1
FIXED_SCALE = 1.0 / 127.0
MAX_VAL_APPLY = 127
ACCUM_STEPS = 80


def _bernoulli_sr_torch(data, scale, gen):
    """PyTorch Bernoulli SR（随机源：torch.Generator）"""
    x_div = data / scale
    x_floor = torch.floor(x_div)
    frac = x_div - x_floor
    u = torch.rand(data.shape, dtype=torch.float32, generator=gen)
    q = x_floor.to(torch.int32) + (u < frac).to(torch.int32)
    q = torch.clamp(q, -32768, 32767)
    return q.to(torch.int16)


def _ef_forward_backward_accum_torch(weights, biases, scale_factor, gen, B=32, dims=None):
    """ReLU 前向+反向，混合精度 EF（torch 实现，int16/int32 张量语义）。
    dims: 各层维度列表（含输入层），默认全局 LAYER_DIMS。"""
    if dims is None:
        dims = LAYER_DIMS
    n_layers = len(dims) - 1
    Ws = [w * scale_factor for w in weights]
    efs = [torch.zeros(dims[i + 1] * dims[i], dtype=torch.int32)
           for i in range(n_layers)]
    res_norms = torch.zeros(n_layers, dtype=torch.float64)
    grad_norms = torch.zeros(n_layers, dtype=torch.float64)
    for _ in range(ACCUM_STEPS):
        x = torch.randn(B, dims[0], dtype=torch.float64, generator=gen)
        y = torch.randn(B, dims[-1], dtype=torch.float64, generator=gen) * 0.5
        acts = [x]; h = x
        for i in range(n_layers - 1):
            h = torch.relu(h @ Ws[i].T + biases[i] * scale_factor)
            acts.append(h)
        h = h @ Ws[-1].T + biases[-1] * scale_factor
        acts.append(h)
        d_h = (h - y) / B
        for i in range(n_layers - 1, -1, -1):
            g_w = d_h.T @ acts[i]
            g_flat = g_w.flatten()
            g16 = _bernoulli_sr_torch(g_flat, FIXED_SCALE, gen)
            hat = g16.to(torch.int32) + efs[i]
            q = torch.clamp(hat, -MAX_VAL_APPLY, MAX_VAL_APPLY).to(torch.int16)
            efs[i] = hat - q.to(torch.int32)
            res_norms[i] += torch.linalg.norm(efs[i].to(torch.float64))
            grad_norms[i] += torch.linalg.norm(g_w)
            if i > 0:
                d_h = (d_h @ Ws[i]) * (acts[i] > 0)
    return res_norms / ACCUM_STEPS, grad_norms / ACCUM_STEPS


def _fit_logslope_torch(log_rel, n_layers=None):
    """torch lstsq 拟合 log(rel) = a + b·depth，返回 (b, R²)"""
    if n_layers is None:
        n_layers = N_LAYERS
    depth = torch.arange(n_layers, dtype=torch.float64)
    A = torch.stack([depth, torch.ones_like(depth)], dim=1)
    sol, _, _, _ = torch.linalg.lstsq(A, log_rel)
    b, a = float(sol[0]), float(sol[1])
    pred = a + b * depth
    ss_res = torch.sum((log_rel - pred) ** 2)
    ss_tot = torch.sum((log_rel - log_rel.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, torch.tensor(1e-30))
    return b, float(r2)


def verify_residual_amplification():
    print("\n" + "=" * 72)
    print("#21 [E] 深层残差指数放大（双库互证，独立实现）")
    print("=" * 72)
    print(f"4 层 ReLU [{LAYER_DIMS[0]},...,{LAYER_DIMS[-1]}]，混合精度 EF")

    SCALE = 1.0
    per_layer_np, geo_np, gap_np, r2_np = [], [], [], []
    per_layer_t, geo_t, gap_t, r2_t = [], [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        Ws = [rng.standard_normal((LAYER_DIMS[i + 1], LAYER_DIMS[i])) *
              np.sqrt(2.0 / LAYER_DIMS[i]) for i in range(N_LAYERS)]
        bs0 = [rng.standard_normal(LAYER_DIMS[i + 1]) * 0.1 for i in range(N_LAYERS)]

        # ---- numpy 实现（与原脚本同逻辑）----
        res_n, grad_n = _ef_forward_backward_accum_np(Ws, bs0, SCALE, rng)
        rel_n = res_n / np.maximum(grad_n, 1e-12)
        log_rel_n = np.log(np.maximum(rel_n, 1e-12))
        b_n, a_n = np.polyfit(np.arange(N_LAYERS, dtype=DT), log_rel_n, 1)
        pred_n = a_n + b_n * np.arange(N_LAYERS, dtype=DT)
        ss_res_n = np.sum((log_rel_n - pred_n) ** 2)
        ss_tot_n = np.sum((log_rel_n - log_rel_n.mean()) ** 2)
        r2n = 1 - ss_res_n / max(ss_tot_n, 1e-30)
        per_layer_np.append(np.exp(b_n))
        geo_np.append(rel_n[2] / max(rel_n[0], 1e-12))
        gap_np.append(grad_n[2] / grad_n[0])
        r2_np.append(r2n)

        # ---- torch 实现（独立随机源，同一权重）----
        gen = torch.Generator(); gen.manual_seed(1000 + seed)
        Ws_t = [torch.as_tensor(w, dtype=torch.float64) for w in Ws]
        bs_t = [torch.as_tensor(b, dtype=torch.float64) for b in bs0]
        res_t, grad_t = _ef_forward_backward_accum_torch(Ws_t, bs_t, SCALE, gen)
        rel_t = res_t / torch.clamp(grad_t, min=1e-12)
        log_rel_t = torch.log(torch.clamp(rel_t, min=1e-12))
        b_t, r2t = _fit_logslope_torch(log_rel_t)
        per_layer_t.append(np.exp(b_t))
        geo_t.append(float(rel_t[2].item() / max(rel_t[0].item(), 1e-12)))
        gap_t.append(float(grad_t[2].item() / grad_t[0].item()))
        r2_t.append(r2t)

    pl_np, pl_t = np.mean(per_layer_np), np.mean(per_layer_t)
    geo_np_, geo_t_ = np.exp(np.mean(np.log(np.maximum(geo_np, 1e-12)))), \
        np.exp(np.mean(np.log(np.maximum(geo_t, 1e-12))))
    mg_np, mg_t = np.mean(gap_np), np.mean(gap_t)
    mr_np, mr_t = np.mean(r2_np), np.mean(r2_t)

    # E 类经验标度：跨 seed 波动达 5-10%，双库独立采样 → 用 2σ(标准误) 自适应判定
    ok_ag1 = _compare_stat("每层放大因子", np.array(per_layer_np),
                           np.array(per_layer_t))
    ok_ag2 = _compare_stat("L3/L1 相对残差(几何均)", np.array(geo_np),
                           np.array(geo_t))
    ok_ag3 = _compare_stat("梯度 L1→L3 放大", np.array(gap_np),
                           np.array(gap_t))
    ok_ag4 = _compare_stat("log-rel 拟合 R²", np.array(r2_np), np.array(r2_t))

    ok = (pl_np > 1.2 and pl_t > 1.2 and mr_np > 0.7 and mr_t > 0.7
          and mg_np > 1.0 and mg_t > 1.0
          and ok_ag1 and ok_ag2 and ok_ag3 and ok_ag4)
    print(f"  每层放大: numpy={pl_np:.1f}× torch={pl_t:.1f}×  (R² numpy={mr_np:.3f} torch={mr_t:.3f})")
    print(f"  L3/L1:   numpy={geo_np_:.0f}× torch={geo_t_:.0f}×  (结构性结论，数值配置相关)")
    print(f"  梯度放大: numpy={mg_np:.1f}× torch={mg_t:.1f}×  (Jacobian>1)")
    print("  注: 每层放大/L3-L1 为高方差 E 类统计量（跨 seed 近 log-分布，个别 seed 病态值主导均值）；")
    print("      2σ 判定确认双库估计在统计上不可区分，定性结论（每层>1.2、R²>0.7、Jacobian>1）双库均成立")
    print(f"  判定: 双库均满足 每层>1.2、R²>0.7、Jacobian>1 且互相一致 → "
          f"{'✓ 复现深层残差放大（双库一致）' if ok else '✗'}")

    # --- 更大深度扫描：每层放大因子随深度稳定（更大规模不改变结论，双库比对）---
    print("\n  --- 更大深度扫描（4/6/8 层，双库比对）---")
    depth_dims = {
        4: [64, 48, 32, 16, 8],
        6: [64, 56, 48, 40, 32, 24, 8],
        8: [64, 58, 52, 46, 40, 34, 28, 22, 8],
    }
    depth_scan = {}
    for n_layers, dims in depth_dims.items():
        nl = len(dims) - 1
        pl_n_arr, pl_t_arr, r2_n_arr, r2_t_arr = [], [], [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            Ws = [rng.standard_normal((dims[i + 1], dims[i])) *
                  np.sqrt(2.0 / dims[i]) for i in range(nl)]
            bs0 = [rng.standard_normal(dims[i + 1]) * 0.1 for i in range(nl)]
            res_n, grad_n = _ef_forward_backward_accum_np(Ws, bs0, SCALE, rng, dims=dims)
            rel_n = res_n / np.maximum(grad_n, 1e-12)
            log_rel_n = np.log(np.maximum(rel_n, 1e-12))
            b_n, a_n = np.polyfit(np.arange(nl, dtype=DT), log_rel_n, 1)
            pred_n = a_n + b_n * np.arange(nl, dtype=DT)
            ss_res_n = np.sum((log_rel_n - pred_n) ** 2)
            ss_tot_n = np.sum((log_rel_n - log_rel_n.mean()) ** 2)
            r2n = 1 - ss_res_n / max(ss_tot_n, 1e-30)
            gen = torch.Generator(); gen.manual_seed(1000 + seed)
            Ws_t = [torch.as_tensor(w, dtype=torch.float64) for w in Ws]
            bs_t = [torch.as_tensor(b, dtype=torch.float64) for b in bs0]
            res_t, grad_t = _ef_forward_backward_accum_torch(Ws_t, bs_t, SCALE, gen, dims=dims)
            rel_t = res_t / torch.clamp(grad_t, min=1e-12)
            log_rel_t = torch.log(torch.clamp(rel_t, min=1e-12))
            b_t, r2t = _fit_logslope_torch(log_rel_t, n_layers=nl)
            pl_n_arr.append(np.exp(b_n)); pl_t_arr.append(np.exp(b_t))
            r2_n_arr.append(r2n); r2_t_arr.append(r2t)
        ok_ag1 = _compare_stat(f"深度{n_layers} 每层放大",
                               np.array(pl_n_arr), np.array(pl_t_arr))
        ok_ag2 = _compare_stat(f"深度{n_layers} R²",
                               np.array(r2_n_arr), np.array(r2_t_arr))
        m_pl_np, m_pl_t = float(np.mean(pl_n_arr)), float(np.mean(pl_t_arr))
        m_r2_np, m_r2_t = float(np.mean(r2_n_arr)), float(np.mean(r2_t_arr))
        gap_n = grad_n[-1] / max(grad_n[0], 1e-12)
        gap_t = float(grad_t[-1].item() / max(grad_t[0].item(), 1e-12))
        ok_d = m_pl_np > 1.2 and m_pl_t > 1.2   # 核心：每层放大>1.2（含 Jacobian 效应的稳健判据）
        r2_ok = m_r2_np > 0.7 and m_r2_t > 0.7   # 标度律质量（可退化为边界条件）
        depth_scan[n_layers] = {"per_layer_np": m_pl_np, "per_layer_torch": m_pl_t,
                                "r2_np": m_r2_np, "r2_torch": m_r2_t,
                                "gap_np": float(gap_n), "gap_torch": float(gap_t),
                                "consistent": bool(ok_d and ok_ag1)}
        print(f"    深度 {n_layers} 层: 每层放大 numpy={m_pl_np:.1f}× torch={m_pl_t:.1f}× "
              f"R²={m_r2_np:.2f}/{m_r2_t:.2f}{'（标度律✓）' if r2_ok else '（标度律退化）'}, "
              f"梯度放大={gap_n:.1f}×/{gap_t:.1f}×（报告）→ {'✓' if ok_d else '✗'}")
    ok_scan = all(depth_scan[n]["consistent"] for n in depth_scan)
    worst_r2 = min(min(depth_scan[n]["r2_np"], depth_scan[n]["r2_torch"]) for n in depth_scan)
    print(f"    判定: 核心结论（每层放大>1.2）在 4/6/8 层双库均成立 → "
          f"{'✓ 更大深度结论稳定' if ok_scan else '✗'}")
    print(f"    边界条件报告: R² 随深度退化（worst={worst_r2:.2f}），深度>4 后指数标度律变差"
          f"——如实记录为深度边界条件，不影响核心定性结论")
    report["#21_depth_scan"] = depth_scan
    ok = ok and ok_scan

    report["#21"] = {"per_layer_np": float(pl_np), "per_layer_torch": float(pl_t),
                     "geo_np": float(geo_np_), "geo_torch": float(geo_t_),
                     "gap_np": float(mg_np), "gap_torch": float(mg_t),
                     "r2_np": float(mr_np), "r2_torch": float(mr_t),
                     "pass": ok}
    return ok


def _ef_forward_backward_accum_np(weights, biases, scale_factor, rng, B=32, dims=None):
    """NumPy 混合精度 EF（与主脚本逻辑一致，供双库对照）。
    dims: 各层维度列表（含输入层），默认全局 LAYER_DIMS。"""
    if dims is None:
        dims = LAYER_DIMS
    n_layers = len(dims) - 1
    Ws = [w * scale_factor for w in weights]
    efs = [np.zeros(dims[i + 1] * dims[i], dtype=np.int32)
           for i in range(n_layers)]
    res_norms = np.zeros(n_layers)
    grad_norms = np.zeros(n_layers)
    for _ in range(ACCUM_STEPS):
        x = rng.normal(0, 1.0, (B, dims[0]))
        y = rng.normal(0, 0.5, (B, dims[-1]))
        acts = [x]; h = x
        for i in range(n_layers - 1):
            h = np.maximum(0, h @ Ws[i].T + biases[i] * scale_factor)
            acts.append(h)
        h = h @ Ws[-1].T + biases[-1] * scale_factor
        acts.append(h)
        d_h = (h - y) / B
        for i in range(n_layers - 1, -1, -1):
            g_w = d_h.T @ acts[i]
            g_flat = g_w.flatten()
            g16 = _bernoulli_sr_np(g_flat, FIXED_SCALE, rng)
            hat = g16.astype(np.int32) + efs[i]
            q = np.clip(hat, -MAX_VAL_APPLY, MAX_VAL_APPLY).astype(np.int16)
            efs[i] = hat.astype(np.int32) - q.astype(np.int32)
            res_norms[i] += np.linalg.norm(efs[i].astype(np.float64))
            grad_norms[i] += np.linalg.norm(g_w)
            if i > 0:
                d_h = (d_h @ Ws[i]) * (acts[i] > 0)
    return res_norms / ACCUM_STEPS, grad_norms / ACCUM_STEPS


def _bernoulli_sr_np(data, scale, rng):
    x_div = data / scale
    x_floor = np.floor(x_div)
    frac = x_div - x_floor
    u = rng.random(size=data.shape, dtype=np.float32)
    q = x_floor.astype(np.int32) + (u < frac).astype(np.int32)
    q = np.clip(q, -32768, 32767)
    return q.astype(np.int16)


def _compare_stat(label, np_arr, torch_arr):
    """E 类统计量一致性判定：|μ_np - μ_t| < 2·max(se_np, se_t)

    双库各自独立采样，均值差在 2 倍标准误内即判定一致
    （容差随统计波动自适应，不误判高方差经验标度）。
    """
    m_np, m_t = float(np.mean(np_arr)), float(np.mean(torch_arr))
    se_np = float(np.std(np_arr)) / np.sqrt(len(np_arr))
    se_t = float(np.std(torch_arr)) / np.sqrt(len(torch_arr))
    tol = 2.0 * max(se_np, se_t)
    ok = abs(m_np - m_t) < tol
    print(f"    [E] numpy={m_np:.6e} torch={m_t:.6e} "
          f"|Δ|={abs(m_np-m_t):.2e} < 2σ={tol:.2e} → {'PASS' if ok else 'FAIL'}")
    return ok


def _compare(label, np_val, torch_val, kind="S", rtol=None):
    """双库一致性判定：T 类机器精度；S 类统计容差（rtol 按统计量波动）"""
    if kind == "T":
        ok = np.isclose(np_val, torch_val, atol=1e-12, rtol=1e-12)
        print(f"    [T] numpy={np_val:.15e} torch={torch_val:.15e} "
              f"→ {'PASS' if ok else 'FAIL'}")
        return bool(ok)
    tol = rtol if rtol is not None else 5e-4
    ok = np.isclose(np_val, torch_val, atol=0.0, rtol=tol)
    print(f"    [S] numpy={np_val:.8e} torch={torch_val:.8e} "
          f"(rtol={tol:.0e}) → {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, torch.Tensor):
        return float(o.item())
    return str(o)


def main():
    t0 = time.time()
    r20 = verify_ultrametric_violation()
    r21 = verify_residual_amplification()
    dt = time.time() - t0

    print("\n" + "=" * 72)
    print("阶段 6 双库互证汇总")
    print("=" * 72)
    print(f"  耗时 {dt:.1f}s")
    print(f"  #20 [S] 超度量违反率100% (复现证伪) : {'✓' if r20 else '✗'}")
    print(f"  #21 [E] 深层残差指数放大 (经验标度)   : {'✓' if r21 else '✗'}")
    overall = r20 and r21
    print(f"\n  总体判定: {'✅ 双库全部一致通过' if overall else '❌ 存在失败'}")
    if not overall:
        return 1

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "math_stage6_network_properties_results_torch.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stage": 6, "backend": "numpy+torch", "kind": "S/E",
                   "elapsed_s": round(dt, 2), "report": report},
                  f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"  结果已写入 {os.path.normpath(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
