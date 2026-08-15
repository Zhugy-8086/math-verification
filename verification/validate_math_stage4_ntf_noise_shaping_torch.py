# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zhugy-8086
"""
阶段 4：频域 / NTF 噪声整形 — NumPy / PyTorch 双库互证
=======================================================
对应计划 numpy_math_verification_plan_2026_08_13.md §3 阶段 4 的 #13-#16

目的：
  同一组噪声整形结论用 NumPy 与 PyTorch 两种独立计算库各实现一遍，
  逐项比对输出，排除"单库实现 bug"——双库一致才算通过。

验证项（与 validate_math_stage4_ntf_noise_shaping.py 对齐）：
  #13 [S] EF ≡ delta-sigma：NTF=(1-z⁻¹)^N, STF=1
  #14 [S] 1 阶 EF 低频抑制（梯度=低通，EF=高通互补）
  #15 [E] clip 噪声 NTF 放大 90327×（低频功率）
  #16 [S] 分离残差 EF 修复 clip 场景（不反馈 clip 误差）

双库实现分工：
  - 信号生成：numpy rng（同 seed，保证输入一致）
  - 量化器随机源：numpy 版用 np rng；torch 版用 torch.Generator（独立随机流）
  - EF 时序递推：numpy 版与 torch 版独立实现（run_ef_np / run_ef_torch）
  - 频谱分析：numpy 版用 np.fft；torch 版用 torch.fft（FFT 算法实现不同）
  - 指标级比对：S 类统计容差（相对 5e-4），T 类机器精度

用法（需安装 torch）：
    python validate_math_stage4_ntf_noise_shaping_torch.py
"""
from __future__ import annotations

import json
import os
import time
from math import comb

import numpy as np
import torch

SEEDS = [0, 1, 2, 3, 4]     # 多随机种子，§2.5

report = {}


# ============================================================
# 核心工具：EF 递推（库无关标量时序）+ 双库 NTF 滤波 / PSD
# ============================================================

def ef_coefficients(N: int):
    """N 阶 EF 反馈系数 a_k = (-1)^{k+1} C(N,k)"""
    return [(-1) ** (k + 1) * comb(N, k) for k in range(1, N + 1)]


def run_ef_np(g_seq, quantize, scheme: str, N: int = 1):
    """NumPy EF 时序递推（numpy 标量路径，独立于 torch 路径）"""
    T = len(g_seq)
    if scheme == 'no_ef':
        n = np.zeros(T)
        for t in range(T):
            q, _ = quantize(g_seq[t])
            n[t] = q - g_seq[t]
        return n, None
    e_buffers = [0.0 for _ in range(N)]
    coeffs = ef_coefficients(N)
    n = np.zeros(T)
    eta = np.zeros(T)
    for t in range(T):
        g = g_seq[t]
        g_hat = float(g)
        for k in range(N):
            g_hat = g_hat + coeffs[k] * e_buffers[k]
        q, q_trunc = quantize(g_hat)
        q = float(q)
        q_trunc = float(q_trunc)
        n[t] = q - g
        eta[t] = q - g_hat
        if scheme == 'standard':
            e_new = g_hat - q
        else:
            e_new = g_hat - q_trunc
        for k in range(N - 1, 0, -1):
            e_buffers[k] = e_buffers[k - 1]
        e_buffers[0] = e_new
    return n, eta


def run_ef_torch(g_seq, quantize, scheme: str, N: int = 1):
    """PyTorch EF 时序递推（torch 标量张量累积，与 numpy 路径独立实现）"""
    T = len(g_seq)
    if scheme == 'no_ef':
        n = torch.zeros(T, dtype=torch.float64)
        for t in range(T):
            q, _ = quantize(float(g_seq[t]))
            n[t] = float(q) - float(g_seq[t])
        return n.numpy(), None
    e_buffers = [torch.zeros((), dtype=torch.float64) for _ in range(N)]
    coeffs = ef_coefficients(N)
    n = torch.zeros(T, dtype=torch.float64)
    eta = torch.zeros(T, dtype=torch.float64)
    for t in range(T):
        g = float(g_seq[t])
        g_hat = torch.full((), g, dtype=torch.float64)
        for k in range(N):
            g_hat = g_hat + coeffs[k] * e_buffers[k]
        q_t, q_trunc_t = quantize(g_hat)            # torch 标量张量
        q = float(q_t)
        q_trunc = float(q_trunc_t)
        n[t] = q - g
        eta[t] = q - g_hat.item()
        if scheme == 'standard':
            e_new = g_hat - q_t
        else:
            e_new = g_hat - q_trunc_t
        for k in range(N - 1, 0, -1):
            e_buffers[k] = e_buffers[k - 1]
        e_buffers[0] = e_new
    return n.numpy(), eta.numpy()


def make_sr_quantizer_np(delta, clamp, rng):
    """NumPy SR 量化器（随机源：np rng）"""
    def quantize(x):
        if np.ndim(x) == 0:
            u = rng.random(dtype=np.float32)
            return _sr_step(float(x), u)
        u = rng.random(size=x.shape, dtype=np.float32)
        return _sr_step_np_array(x, u)

    def _sr_step(x, u):
        x_div = x / delta
        q_floor = np.floor(x_div)
        frac = x_div - q_floor
        q_trunc = (q_floor + (u < frac).astype(np.float64)) * delta
        q_clip = np.clip(q_trunc, -clamp, clamp) if clamp is not None else q_trunc
        return q_clip, q_trunc

    def _sr_step_np_array(x, u):
        return _sr_step(x, u)
    return quantize


def make_sr_quantizer_torch(delta, clamp, gen):
    """PyTorch SR 量化器（纯 torch 计算：torch.floor/torch.clamp；
    随机源：torch.Generator 独立随机流）"""
    delta_t = torch.tensor(delta, dtype=torch.float64)

    def quantize(x):
        u = torch.rand((), dtype=torch.float32, generator=gen)
        x_t = torch.as_tensor(float(x), dtype=torch.float64)
        x_div = x_t / delta_t
        q_floor = torch.floor(x_div)
        frac = x_div - q_floor
        q_trunc = (q_floor + (u.to(torch.float64) < frac).to(torch.float64)) * delta_t
        if clamp is not None:
            q_clip = torch.clamp(q_trunc, -clamp, clamp)
        else:
            q_clip = q_trunc
        return q_clip, q_trunc
    return quantize


def apply_ntf_filter_np(eta, N: int):
    """NumPy 向量化 NTF=(1-z⁻¹)^N"""
    T = len(eta)
    out = np.zeros(T)
    for k in range(N + 1):
        coeff = (-1) ** k * comb(N, k)
        if k == 0:
            out += coeff * eta
        else:
            out[k:] += coeff * eta[:-k]
    return out


def apply_ntf_filter_torch(eta, N: int):
    """PyTorch 向量化 NTF=(1-z⁻¹)^N"""
    e = torch.as_tensor(np.asarray(eta), dtype=torch.float64)
    T = e.shape[0]
    out = torch.zeros(T, dtype=torch.float64)
    for k in range(N + 1):
        coeff = (-1) ** k * comb(N, k)
        if k == 0:
            out = out + coeff * e
        else:
            out[k:] = out[k:] + coeff * e[:-k]
    return out


def psd_of_np(x):
    T = len(x)
    fft = np.fft.rfft(x)
    return np.abs(fft) ** 2 / T, np.fft.rfftfreq(T, d=1.0)


def psd_of_torch(x):
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    T = xt.shape[0]
    fft = torch.fft.rfft(xt)
    return (fft.abs() ** 2 / T).numpy(), np.fft.rfftfreq(T, d=1.0)


def welch_psd_np(x, nperseg=4096):
    T = len(x)
    nseg = T // nperseg
    w = np.hanning(nperseg)
    acc = np.zeros(nperseg // 2 + 1)
    for i in range(nseg):
        seg = x[i * nperseg:(i + 1) * nperseg]
        seg = seg - seg.mean()
        fft = np.fft.rfft(seg * w)
        acc += np.abs(fft) ** 2 / (nperseg * (w ** 2).mean())
    return acc / nseg, np.fft.rfftfreq(nperseg)


def welch_psd_torch(x, nperseg=4096):
    """PyTorch Welch 平均 PSD（torch.fft + torch.hann_window）"""
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    T = xt.shape[0]
    nseg = T // nperseg
    w = torch.hann_window(nperseg, periodic=False, dtype=torch.float64)
    acc = torch.zeros(nperseg // 2 + 1, dtype=torch.float64)
    for i in range(nseg):
        seg = xt[i * nperseg:(i + 1) * nperseg]
        seg = seg - seg.mean()
        fft = torch.fft.rfft(seg * w)
        acc = acc + fft.abs() ** 2 / (nperseg * (w ** 2).mean())
    return (acc / nseg).numpy(), np.fft.rfftfreq(nperseg)


def lowfreq_power_np(x, cutoff=0.05):
    psd, freqs = psd_of_np(x)
    omegas = 2 * np.pi * freqs
    return float(np.mean(psd[omegas < cutoff]))


def lowfreq_power_torch(x, cutoff=0.05):
    psd, freqs = psd_of_torch(x)
    omegas = 2 * np.pi * freqs
    return float(np.mean(psd[omegas < cutoff]))


def ar1_signal(T, rho, sigma, rng):
    """AR(1) 平稳信号（numpy rng 生成，双库共用同一输入）"""
    g = np.zeros(T)
    g[0] = float(rng.normal(0, sigma))
    for t in range(1, T):
        g[t] = rho * g[t - 1] + float(rng.normal(0, sigma * np.sqrt(1 - rho ** 2)))
    return g


def _compare(label, np_val, torch_val, kind="S", rtol=None):
    """双库一致性判定：T 类机器精度；S 类统计容差。

    S 类默认 rtol=5e-4；高方差统计量（低频平均 PSD 等，波动达 1e-2）
    需显式传更大 rtol——容差应反映"两库独立蒙特卡洛采样的波动"。
    """
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


# ============================================================
# #13 [S] EF ≡ delta-sigma：NTF=(1-z⁻¹)^N, STF=1
# ============================================================
def verify_ntf_identity():
    print("=" * 72)
    print("#13 [S] EF ≡ delta-sigma：NTF=(1-z⁻¹)^N, STF=1（双库互证）")
    print("=" * 72)

    T = 100_000
    DELTA = 0.1
    all_ok = True

    # ---- 1) 确定性 NTF 恒等式：n_t = NTF(η)_t（两库各自验证）----
    print("  [1] 确定性恒等式 n = NTF(η)（标准 EF，机器精度）")
    for N in [1, 2, 3]:
        rng_n = np.random.default_rng(2026)
        gen_t = torch.Generator(); gen_t.manual_seed(2026)
        g = ar1_signal(T, 0.99, 0.5, rng_n)
        q_n = make_sr_quantizer_np(DELTA, None, rng_n)
        q_t = make_sr_quantizer_torch(DELTA, None, gen_t)
        n_n, eta_n = run_ef_np(g, q_n, 'standard', N=N)
        n_t, eta_t = run_ef_torch(g, q_t, 'standard', N=N)
        ident_np = float(np.max(np.abs(n_n - apply_ntf_filter_np(eta_n, N))))
        ident_t = float(torch.max(torch.abs(
            torch.as_tensor(n_t) - apply_ntf_filter_torch(eta_t, N))).item())
        ok_np = ident_np < 1e-9
        ok_t = ident_t < 1e-9
        ok = ok_np and ok_t
        all_ok &= ok
        print(f"    N={N}: max|n-NTF(η)| numpy={ident_np:.2e} torch={ident_t:.2e}"
              f"  → {'✓' if ok else '✗'}")

    # ---- 2) STF=1：信号分量无损通过（两库指标比对）----
    print("  [2] STF=1：信号路径无损（纯正弦输入，双库比对）")
    fs = 2 * np.pi * 0.05
    rng_n = np.random.default_rng(2026)
    gen_t = torch.Generator(); gen_t.manual_seed(2026)
    sig = 1.0 * np.sin(fs * np.arange(T)) + 0.3
    q_n = make_sr_quantizer_np(DELTA, None, rng_n)
    q_t = make_sr_quantizer_torch(DELTA, None, gen_t)
    n_n, _ = run_ef_np(sig, q_n, 'standard', N=1)
    n_t, _ = run_ef_torch(sig, q_t, 'standard', N=1)
    probe = np.exp(-1j * fs * np.arange(T))
    sig_amp = np.abs(np.dot(sig, probe))
    stf_np = np.abs(np.dot(sig + n_n, probe)) / sig_amp
    stf_t = np.abs(np.dot(sig + n_t, probe)) / sig_amp
    ok2 = abs(stf_np - 1.0) < 1e-3 and abs(stf_t - 1.0) < 1e-3
    ok_ag = _compare("STF 幅度", float(stf_np), float(stf_t), kind="S")
    all_ok &= ok2 and ok_ag
    print(f"      STF numpy={stf_np:.6f} torch={stf_t:.6f}")

    # ---- 3) PSD ratio（多 seed，常量信号 + Welch，中频段，双库比对）----
    print("  [3] PSD ratio = 实测/理论 |NTF|²（常量信号 + Welch）")
    ratios_np = {1: [], 2: [], 3: []}
    ratios_t = {1: [], 2: [], 3: []}
    for seed in SEEDS:
        rng_s = np.random.default_rng(seed)
        gen_s = torch.Generator(); gen_s.manual_seed(seed)
        g_const = np.full(T, 0.0037)
        for N in [1, 2, 3]:
            q_n = make_sr_quantizer_np(DELTA, None, rng_s)
            q_t = make_sr_quantizer_torch(DELTA, None, gen_s)
            n_n, _ = run_ef_np(g_const, q_n, 'standard', N=N)
            n_t, _ = run_ef_torch(g_const, q_t, 'standard', N=N)
            psd_n, freqs = welch_psd_np(n_n)
            psd_t, _ = welch_psd_torch(n_t)
            omegas = 2 * np.pi * freqs
            theory = (DELTA ** 2 / 6) * (4 * np.sin(omegas / 2) ** 2) ** N
            mid = (omegas > 0.05) & (omegas < 0.4)
            ratios_np[N].append(float(np.mean(psd_n[mid] / theory[mid])))
            ratios_t[N].append(float(np.mean(psd_t[mid] / theory[mid])))
    for N in [1, 2, 3]:
        rn = np.array(ratios_np[N]); rt = np.array(ratios_t[N])
        okN = abs(rn.mean() - 1.0) < 0.05 and abs(rt.mean() - 1.0) < 0.05
        # 低频平均 PSD 高方差（ratio 波动 ~1e-2），双库独立采样 → rtol=2e-2
        ok_ag = _compare(f"N={N} ratio", float(rn.mean()), float(rt.mean()),
                         kind="S", rtol=2e-2)
        all_ok &= okN and ok_ag
        print(f"      ratio numpy={rn.mean():.4f}±{rn.std():.4f} "
              f"torch={rt.mean():.4f}±{rt.std():.4f}")

    print(f"  结论: {'✓ EF ≡ delta-sigma（NTF/STF 双库一致）' if all_ok else '✗ 有失败'}")
    report["#13"] = {"pass": all_ok}
    return all_ok


# ============================================================
# #14 [S] 1 阶 EF 低频抑制（梯度=低通，EF=高通互补）
# ============================================================
def verify_lowfreq_suppression():
    print("\n" + "=" * 72)
    print("#14 [S] 1 阶 EF 低频抑制（双库互证）")
    print("=" * 72)

    T = 100_000
    DELTA = 0.1
    theory_var = DELTA ** 2 / 6
    all_ok = True

    # ---- 1) NTF=(1-z⁻¹) 高通：低频被抑制（多 seed，双库比对）----
    print("  [1] NTF=(1-z⁻¹) 高通整形：低频 PSD 抑制")
    supp_np, supp_t = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        gen = torch.Generator(); gen.manual_seed(seed)
        g = ar1_signal(T, 0.99, 0.5, rng)
        q_n = make_sr_quantizer_np(DELTA, None, rng)
        q_t = make_sr_quantizer_torch(DELTA, None, gen)
        n_n, _ = run_ef_np(g, q_n, 'standard', N=1)
        n_t, _ = run_ef_torch(g, q_t, 'standard', N=1)
        supp_np.append(theory_var / max(lowfreq_power_np(n_n), 1e-30))
        supp_t.append(theory_var / max(lowfreq_power_torch(n_t), 1e-30))
    sn, st = np.array(supp_np), np.array(supp_t)
    ok1 = np.all(sn > 10) and np.all(st > 10)
    # 低频功率高方差（多 seed 均值 ~1e-2 波动），双库独立采样 → rtol=2e-2
    ok_ag = _compare("低频抑制×", float(sn.mean()), float(st.mean()),
                     kind="S", rtol=2e-2)
    all_ok &= ok1 and ok_ag
    print(f"      抑制 numpy={sn.mean():.0f}× torch={st.mean():.0f}×")

    # ---- 2) 权重噪声 TF=(1-z⁻¹)^{N-1}（多阶，双库比对）----
    print("  [2] 权重噪声 TF=(1-z⁻¹)^{N-1}：SGD 积分器与 NTF 互补")
    for N in [1, 2, 3]:
        rng = np.random.default_rng(2026)
        gen = torch.Generator(); gen.manual_seed(2026)
        eta = rng.uniform(-DELTA / 2, DELTA / 2, T)
        shaped_np = apply_ntf_filter_np(eta, N)
        shaped_t = apply_ntf_filter_torch(eta, N)
        wnoise_np = np.cumsum(shaped_np)
        wnoise_t = torch.cumsum(shaped_t, dim=0).numpy()
        psd_n, freqs = psd_of_np(wnoise_np)
        psd_t, _ = psd_of_torch(wnoise_t)
        omegas = 2 * np.pi * freqs
        uvar = DELTA ** 2 / 12
        theory = uvar * (4 * np.sin(omegas / 2) ** 2) ** (N - 1)
        mid = (omegas > 0.02) & (omegas < 0.4)
        r_np = float(np.mean(psd_n[mid] / theory[mid]))
        r_t = float(np.mean(psd_t[mid] / theory[mid]))
        okN = abs(r_np - 1.0) < 0.1 and abs(r_t - 1.0) < 0.1
        ok_ag = _compare(f"N={N} 权重TF ratio", r_np, r_t, kind="S")
        all_ok &= okN and ok_ag
        print(f"      ratio numpy={r_np:.3f} torch={r_t:.3f}")

    # ---- 3) 高阶 DC 抑制对比（1阶→2阶 权重 DC 功率比，双库比对）----
    rng = np.random.default_rng(2026)
    eta = rng.uniform(-DELTA / 2, DELTA / 2, T)
    dc1_np = float(psd_of_np(np.cumsum(apply_ntf_filter_np(eta, 1)))[0][0])
    dc2_np = float(psd_of_np(np.cumsum(apply_ntf_filter_np(eta, 2)))[0][0])
    e1 = torch.cumsum(apply_ntf_filter_torch(eta, 1), dim=0)
    e2 = torch.cumsum(apply_ntf_filter_torch(eta, 2), dim=0)
    dc1_t = float(psd_of_torch(e1)[0][0])
    dc2_t = float(psd_of_torch(e2)[0][0])
    r12_np = dc1_np / max(dc2_np, 1e-30)
    r12_t = dc1_t / max(dc2_t, 1e-30)
    ok_ag = _compare("1阶→2阶 DC 比", r12_np, r12_t, kind="S")
    all_ok &= ok_ag
    print(f"      1阶→2阶 DC 功率比 numpy={r12_np:.0f}× torch={r12_t:.0f}×（报告 657740×）")

    print(f"  结论: {'✓ 1 阶 EF 低频抑制 + 高低通互补（双库一致）' if all_ok else '✗ 有失败'}")
    report["#14"] = {"pass": all_ok}
    return all_ok


# ============================================================
# #15 [E] clip 噪声 NTF 放大 90327×（低频功率）
# ============================================================
def verify_clip_amplification():
    print("\n" + "=" * 72)
    print("#15 [E] clip 噪声 NTF 放大 90327×（双库互证）")
    print("=" * 72)

    T = 100_000
    DELTA = 1e-3
    CLAMP = 0.3
    RHO = 0.99

    ratios_np, ratios_t = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        gen = torch.Generator(); gen.manual_seed(seed)
        g = ar1_signal(T, RHO, 0.5, rng)
        q_n = make_sr_quantizer_np(DELTA, CLAMP, rng)
        q_t = make_sr_quantizer_torch(DELTA, CLAMP, gen)
        n_noef_n, _ = run_ef_np(g, q_n, 'no_ef')
        n_noef_t, _ = run_ef_torch(g, q_t, 'no_ef')
        _, eta_std_n = run_ef_np(g, q_n, 'standard', N=1)
        _, eta_std_t = run_ef_torch(g, q_t, 'standard', N=1)
        lp_noef_n = lowfreq_power_np(n_noef_n)
        lp_std_n = lowfreq_power_np(eta_std_n)
        lp_noef_t = lowfreq_power_torch(n_noef_t)
        lp_std_t = lowfreq_power_torch(eta_std_t)
        ratios_np.append(lp_std_n / max(lp_noef_n, 1e-30))
        ratios_t.append(lp_std_t / max(lp_noef_t, 1e-30))

    rn = np.array(ratios_np); rt = np.array(ratios_t)
    mean_np, mean_t = float(rn.mean()), float(rt.mean())
    theory = 90327.0
    ok_np = theory / 3 <= mean_np <= theory * 3
    ok_t = theory / 3 <= mean_t <= theory * 3
    ok_ag = _compare("低频功率放大比", mean_np, mean_t, kind="S")
    ok = ok_np and ok_t and ok_ag
    print(f"      放大比 numpy={mean_np:.0f}× torch={mean_t:.0f}×（冻结结论 90327×）")

    # --- 更大 T 扫描：放大比量级随 T 稳定（更大规模不改变结论，双库比对）---
    print("\n  --- 更大 T 扫描（放大比随 T 稳定性，双库比对）---")
    scan = {}
    for T_s in [2e4, 1e5, 2e5]:
        T_s = int(T_s)
        r_np, r_t = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            gen = torch.Generator(); gen.manual_seed(seed)
            g = ar1_signal(T_s, RHO, 0.5, rng)
            q_n = make_sr_quantizer_np(DELTA, CLAMP, rng)
            q_t = make_sr_quantizer_torch(DELTA, CLAMP, gen)
            n_noef_n, _ = run_ef_np(g, q_n, 'no_ef')
            n_noef_t, _ = run_ef_torch(g, q_t, 'no_ef')
            _, eta_std_n = run_ef_np(g, q_n, 'standard', N=1)
            _, eta_std_t = run_ef_torch(g, q_t, 'standard', N=1)
            r_np.append(lowfreq_power_np(eta_std_n) / max(lowfreq_power_np(n_noef_n), 1e-30))
            r_t.append(lowfreq_power_torch(eta_std_t) / max(lowfreq_power_torch(n_noef_t), 1e-30))
        m_np, m_t = float(np.mean(r_np)), float(np.mean(r_t))
        ok_sc = (theory / 3 <= m_np <= theory * 3) and (theory / 3 <= m_t <= theory * 3)
        ok_ag = _compare(f"T={T_s} 放大比", m_np, m_t, kind="S", rtol=2e-2)
        scan[T_s] = {"np": m_np, "torch": m_t, "consistent": bool(ok_sc and ok_ag)}
        print(f"    T={T_s:>6d}: 放大比 numpy={m_np:.0f}× torch={m_t:.0f}×")
    log_span = abs(np.log10(scan[2e4]["np"]) - np.log10(scan[2e5]["np"]))
    stable = log_span < 0.5
    print(f"    T∈[2e4,2e5] 放大比 log10 跨度 = {log_span:.3f} < 0.5 → "
          f"{'✓ 更大规模结论稳定' if stable else '✗ 结论随规模漂移'}")
    report["#15_scan"] = {"scan": scan, "log_span": log_span, "stable": stable}
    ok = ok and stable

    print(f"  结论: {'✓ clip 噪声 NTF 放大（双库一致）' if ok else '✗'}")
    report["#15"] = {"ratio_np": mean_np, "ratio_torch": mean_t,
                     "theory": theory, "pass": ok}
    return ok


# ============================================================
# #16 [S] 分离残差 EF 修复 clip 场景
# ============================================================
def verify_separated_ef():
    print("\n" + "=" * 72)
    print("#16 [S] 分离残差 EF 修复 clip 场景（双库互证）")
    print("=" * 72)

    T = 100_000
    DELTA = 1e-3
    CLAMP = 0.3
    RHO = 0.99

    sep_np, std_np = [], []
    sep_t, std_t = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        gen = torch.Generator(); gen.manual_seed(seed)
        g = ar1_signal(T, RHO, 0.5, rng)
        q_n = make_sr_quantizer_np(DELTA, CLAMP, rng)
        q_t = make_sr_quantizer_torch(DELTA, CLAMP, gen)
        n_noef_n, _ = run_ef_np(g, q_n, 'no_ef')
        _, eta_std_n = run_ef_np(g, q_n, 'standard', N=1)
        _, eta_sep_n = run_ef_np(g, q_n, 'separated', N=1)
        n_noef_t, _ = run_ef_torch(g, q_t, 'no_ef')
        _, eta_std_t = run_ef_torch(g, q_t, 'standard', N=1)
        _, eta_sep_t = run_ef_torch(g, q_t, 'separated', N=1)
        lp_noef_n = lowfreq_power_np(n_noef_n)
        lp_noef_t = lowfreq_power_torch(n_noef_t)
        sep_np.append(lowfreq_power_np(eta_sep_n) / max(lp_noef_n, 1e-30))
        std_np.append(lowfreq_power_np(eta_std_n) / max(lp_noef_n, 1e-30))
        sep_t.append(lowfreq_power_torch(eta_sep_t) / max(lp_noef_t, 1e-30))
        std_t.append(lowfreq_power_torch(eta_std_t) / max(lp_noef_t, 1e-30))

    s_np, st_np = np.array(sep_np), np.array(std_np)
    s_t, st_t = np.array(sep_t), np.array(std_t)
    ok_sep_np = np.all(np.abs(s_np - 1.0) < 0.1)
    ok_sep_t = np.all(np.abs(s_t - 1.0) < 0.1)
    ok_std_np = np.all(st_np > 10)
    ok_std_t = np.all(st_t > 10)
    ok_ag1 = _compare("分离 EF / 无 EF", float(s_np.mean()), float(s_t.mean()), kind="S")
    ok_ag2 = _compare("标准 EF / 无 EF", float(st_np.mean()), float(st_t.mean()), kind="S")
    ok = ok_sep_np and ok_sep_t and ok_std_np and ok_std_t and ok_ag1 and ok_ag2
    print(f"      分离 EF 低频/无EF: numpy={s_np.mean():.3f} torch={s_t.mean():.3f}（无放大）")
    print(f"      标准 EF 低频/无EF: numpy={st_np.mean():.0f} torch={st_t.mean():.0f}（放大）")
    print(f"  结论: {'✓ 分离残差 EF 修复 clip 场景（双库一致）' if ok else '✗ 有失败'}")
    report["#16"] = {"sep_np": float(s_np.mean()), "sep_torch": float(s_t.mean()),
                     "std_np": float(st_np.mean()), "std_torch": float(st_t.mean()),
                     "pass": ok}
    return ok


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
    r13 = verify_ntf_identity()
    r14 = verify_lowfreq_suppression()
    r15 = verify_clip_amplification()
    r16 = verify_separated_ef()
    dt = time.time() - t0

    print("\n" + "=" * 72)
    print("阶段 4 双库互证汇总")
    print("=" * 72)
    print(f"  耗时 {dt:.1f}s")
    print(f"  #13 [S] EF≡delta-sigma NTF/STF : {'✓' if r13 else '✗'}")
    print(f"  #14 [S] 1阶EF低频抑制+高低通互补 : {'✓' if r14 else '✗'}")
    print(f"  #15 [E] clip 噪声 NTF 放大 90327× : {'✓' if r15 else '✗'}")
    print(f"  #16 [S] 分离残差 EF 修复 clip   : {'✓' if r16 else '✗'}")
    overall = r13 and r14 and r15 and r16
    print(f"\n  总体判定: {'✅ 双库全部一致通过' if overall else '❌ 存在失败'}")
    if not overall:
        return 1

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "math_stage4_ntf_noise_shaping_results_torch.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stage": 4, "backend": "numpy+torch", "kind": "S/E",
                   "elapsed_s": round(dt, 2), "report": report},
                  f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"  结果已写入 {os.path.normpath(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
