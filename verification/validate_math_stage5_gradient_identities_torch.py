# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zhugy-8086
"""
阶段 5：梯度恒等式 — NumPy / PyTorch 双库互证（T 类）
=====================================================
对应计划 numpy_math_verification_plan_2026_08_13.md §3 阶段 5 的 #17-#19

目的：
  同一组梯度恒等式用 NumPy 与 PyTorch 两种独立计算库各实现一遍，
  逐项比对输出，排除"单库实现 bug"——双库一致才算通过。

验证项（与 validate_math_stage5_gradient_identities.py 对齐）：
  #17 [T] 残差反向传播恒等梯度 dy = dy_act·W^T + dy_skip
  #18 [T] 损失梯度除以总元素数 y_pred.size（而非 batch_size）
  #19 [T] CE 梯度 = (softmax - one_hot) / batch_size

一致性判定（§2.4 T 类）：
  - 双库解析梯度逐元素比对：机器精度（rtol=1e-12）
  - 各自库内：解析梯度 vs 数值梯度（中心差分）确认实现正确

用法（需安装 torch）：
    python validate_math_stage5_gradient_identities_torch.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

SEEDS = [0, 1, 2, 3, 4]
DT = np.float64

report = {}


# ============================================================
# 数值梯度工具（中心差分，双库各一）
# ============================================================

def numerical_gradient_np(loss_fn, x, eps=1e-6):
    """NumPy：对 x 逐元素中心差分"""
    grad = np.zeros_like(x, dtype=DT)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = loss_fn()
        x[idx] = orig - eps
        f_minus = loss_fn()
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def numerical_gradient_torch(loss_fn, x, eps=1e-6):
    """PyTorch：对 torch 张量 x 逐元素中心差分"""
    grad = torch.zeros_like(x)
    for idx in np.ndindex(tuple(x.shape)):
        orig = x[idx].item()
        x[idx] = orig + eps
        f_plus = loss_fn()
        x[idx] = orig - eps
        f_minus = loss_fn()
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def _compare(label, np_val, torch_val):
    """双库一致性判定（T 类：机器精度）"""
    if np.ndim(np_val) == 0:
        ok = np.isclose(np_val, torch_val, atol=1e-12, rtol=1e-12)
    else:
        ok = np.max(np.abs(np.asarray(np_val) - np.asarray(torch_val))) < 1e-12
    print(f"    [T] max|numpy-torch|={np.max(np.abs(np.asarray(np_val) - np.asarray(torch_val))):.2e} "
          f"→ {'PASS' if ok else 'FAIL'}")
    return bool(ok)


# ============================================================
# #18 [T] 损失梯度除以总元素数 y_pred.size
# ============================================================
def verify_mse_size_division():
    print("=" * 72)
    print("#18 [T] 损失梯度除以总元素数 y_pred.size（双库互证）")
    print("=" * 72)

    all_ok = True
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        B, D = int(rng.integers(2, 9)), int(rng.integers(2, 9))
        y = rng.normal(0, 1, (B, D)).astype(DT)
        y_true = rng.normal(0, 1, (B, D)).astype(DT)
        N = y.size

        # ---- numpy 库 ----
        loss_np = lambda: 0.5 * np.mean((y - y_true) ** 2)
        ana_np = (y - y_true) / N
        num_np = numerical_gradient_np(loss_np, y)

        # ---- torch 库 ----
        y_t = torch.from_numpy(y).to(torch.float64).clone()
        yt_t = torch.from_numpy(y_true).to(torch.float64)
        loss_t = lambda: 0.5 * torch.mean((y_t - yt_t) ** 2).item()
        ana_t = (y_t - yt_t) / N
        num_t = numerical_gradient_torch(loss_t, y_t)

        # 库内校验 + 双库互证
        ok_np = np.max(np.abs(ana_np - num_np)) < 1e-7
        ok_t = torch.max(torch.abs(ana_t - num_t)).item() < 1e-7
        ok_ag = _compare(f"seed{seed} 解析梯度", ana_np, ana_t.numpy())
        ok = ok_np and ok_t and ok_ag
        all_ok &= ok
        print(f"      B={B},D={D}: 数值校验 numpy={ok_np} torch={ok_t}")
    print(f"  结论: {'✓ 损失梯度必须除以 y_pred.size（双库一致）' if all_ok else '✗ 有失败'}")
    report["#18"] = {"pass": all_ok}
    return all_ok


# ============================================================
# #19 [T] CE 梯度 = (softmax - one_hot) / batch_size
# ============================================================
def softmax_rows_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def softmax_rows_torch(z):
    z = z - z.max(dim=1, keepdim=True).values
    e = torch.exp(z)
    return e / e.sum(dim=1, keepdim=True)


def verify_ce_gradient():
    print("\n" + "=" * 72)
    print("#19 [T] CE 梯度 = (softmax - one_hot) / batch_size（双库互证）")
    print("=" * 72)

    all_ok = True
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        B, C = int(rng.integers(2, 9)), int(rng.integers(3, 10))
        logits = rng.normal(0, 1, (B, C)).astype(DT)
        idx = rng.integers(0, C, size=B)
        one_hot = np.eye(C)[idx].astype(DT)

        # ---- numpy 库（类别索引标签）----
        def ce_loss_np():
            p = softmax_rows_np(logits)
            return float(-np.mean(np.log(p[np.arange(B), idx] + 1e-15)))
        p_np = softmax_rows_np(logits)
        ana_np = (p_np - one_hot) / B
        num_np = numerical_gradient_np(ce_loss_np, logits)

        # ---- torch 库 ----
        logits_t = torch.from_numpy(logits).to(torch.float64).clone()
        idx_t = torch.from_numpy(idx).to(torch.int64)
        oh_t = torch.from_numpy(one_hot).to(torch.float64)
        def ce_loss_t():
            p = softmax_rows_torch(logits_t)
            return float(-torch.mean(torch.log(p[torch.arange(B), idx_t] + 1e-15)).item())
        p_t = softmax_rows_torch(logits_t)
        ana_t = (p_t - oh_t) / B
        num_t = numerical_gradient_torch(ce_loss_t, logits_t)

        ok_np = np.max(np.abs(ana_np - num_np)) < 1e-7
        ok_t = torch.max(torch.abs(ana_t - num_t)).item() < 1e-7
        ok_ag = _compare(f"seed{seed} CE梯度", ana_np, ana_t.numpy())
        ok = ok_np and ok_t and ok_ag
        all_ok &= ok
        print(f"      B={B},C={C}: 数值校验 numpy={ok_np} torch={ok_t}")
    print(f"  结论: {'✓ CE 梯度 = (softmax - one_hot)/batch_size（双库一致）' if all_ok else '✗ 有失败'}")
    report["#19"] = {"pass": all_ok}
    return all_ok


# ============================================================
# #17 [T] 残差反向传播恒等梯度 dy = dy_act·W^T + dy_skip
# ============================================================
def residual_forward_torch(Ws, bs, x):
    """残差网络前向（torch 实现），返回 (activations, pre-activations)"""
    a = [x]
    z = []
    for W, b in zip(Ws, bs):
        zl = a[-1] @ W.T + b
        z.append(zl)
        a.append(torch.relu(zl) + a[-1])
    return a, z


def residual_backward_torch(Ws, bs, x, a, z, dy):
    """残差网络反向（torch 实现），含 skip 恒等梯度 + dA[l+1]"""
    L = len(Ws)
    dA = [None] * (L + 1)
    dA[L] = dy
    dW = [None] * L
    dB = [None] * L
    for l in range(L - 1, -1, -1):
        dZ = dA[l + 1] * (z[l] > 0)
        dW[l] = dZ.T @ a[l]
        dB[l] = dZ.sum(0)
        dA[l] = dZ @ Ws[l] + dA[l + 1]   # skip 路径恒等梯度
    return dW, dB, dA


def verify_residual_skip_gradient():
    print("\n" + "=" * 72)
    print("#17 [T] 残差反向传播恒等梯度（双库互证）")
    print("=" * 72)

    all_ok = True
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        B = int(rng.integers(2, 6))
        D = int(rng.integers(2, 6))
        L = int(rng.integers(2, 5))
        dims = [D] * (L + 1)

        # numpy 网络（与原脚本一致的结构）
        Ws_np = [rng.normal(0, 0.5, (dims[i + 1], dims[i])).astype(DT) for i in range(L)]
        bs_np = [rng.normal(0, 0.1, dims[i + 1]).astype(DT) for i in range(L)]
        x_np = rng.normal(0, 1, (B, D)).astype(DT)
        y_true_np = rng.normal(0, 1, (B, D)).astype(DT)

        # ---- numpy 前向/反向（解析）----
        a_np = [x_np]
        z_np = []
        for i in range(L):
            zl = a_np[-1] @ Ws_np[i].T + bs_np[i]
            z_np.append(zl)
            a_np.append(np.maximum(zl, 0.0) + a_np[-1])
        y_np = a_np[L]
        dy_np = (y_np - y_true_np) / y_np.size
        dA_np = [None] * (L + 1)
        dA_np[L] = dy_np
        dW_np = [None] * L
        for l in range(L - 1, -1, -1):
            dZ = dA_np[l + 1] * (z_np[l] > 0)
            dW_np[l] = dZ.T @ a_np[l]
            dA_np[l] = dZ @ Ws_np[l] + dA_np[l + 1]
        # 数值梯度会原地扰动 W，loss 闭包必须实时重算前向
        def make_loss_np(Ws, bs, x):
            def f():
                aa = [x]
                for i in range(len(Ws)):
                    zl = aa[-1] @ Ws[i].T + bs[i]
                    aa.append(np.maximum(zl, 0.0) + aa[-1])
                return float(0.5 * np.mean((aa[L] - y_true_np) ** 2))
            return f
        lf_np = make_loss_np(Ws_np, bs_np, x_np)

        # ---- torch 前向/反向（解析，同一初始权重）----
        Ws_t = [torch.from_numpy(w).to(torch.float64).clone() for w in Ws_np]
        bs_t = [torch.from_numpy(b).to(torch.float64).clone() for b in bs_np]
        x_t = torch.from_numpy(x_np).to(torch.float64).clone()
        yt_t = torch.from_numpy(y_true_np).to(torch.float64)
        a_t, z_t = residual_forward_torch(Ws_t, bs_t, x_t)
        y_t = a_t[L]
        dy_t = (y_t - yt_t) / y_t.numel()
        dW_t, dB_t, dA_t = residual_backward_torch(Ws_t, bs_t, x_t, a_t, z_t, dy_t)
        def make_loss_t(Ws, bs, x):
            def f():
                aa, _ = residual_forward_torch(Ws, bs, x)
                return float(0.5 * torch.mean((aa[L] - yt_t) ** 2).item())
            return f
        lf_t = make_loss_t(Ws_t, bs_t, x_t)

        max_dw_ag = 0.0
        ok_all = True
        for l in range(L):
            num_np = numerical_gradient_np(lf_np, Ws_np[l])
            num_t = numerical_gradient_torch(lf_t, Ws_t[l])
            ok_np = np.max(np.abs(dW_np[l] - num_np)) < 1e-7
            ok_t = torch.max(torch.abs(dW_t[l] - num_t)).item() < 1e-7
            max_dw_ag = max(max_dw_ag,
                            np.max(np.abs(dW_np[l] - dW_t[l].numpy())))
            ok_all &= ok_np and ok_t
        ok_ag = _compare(f"seed{seed} 含skip dL/dW", max_dw_ag, 0.0)
        all_ok &= ok_all and ok_ag
        print(f"      L={L}, dims={dims}: 数值校验双库均={ok_all}")
    print(f"  结论: {'✓ 残差反向传播必须含 skip 恒等梯度（双库一致）' if all_ok else '✗ 有失败'}")
    report["#17"] = {"pass": all_ok}
    return all_ok


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
    r17 = verify_residual_skip_gradient()
    r18 = verify_mse_size_division()
    r19 = verify_ce_gradient()
    dt = time.time() - t0

    print("\n" + "=" * 72)
    print("阶段 5 双库互证汇总（T 类）")
    print("=" * 72)
    print(f"  耗时 {dt:.1f}s")
    print(f"  #17 [T] 残差 skip 恒等梯度（双库一致） : {'✓' if r17 else '✗'}")
    print(f"  #18 [T] 损失梯度÷y_pred.size（双库一致） : {'✓' if r18 else '✗'}")
    print(f"  #19 [T] CE 梯度=(softmax-onehot)/B（双库一致） : {'✓' if r19 else '✗'}")
    overall = r17 and r18 and r19
    print(f"\n  总体判定: {'✅ 双库全部一致通过' if overall else '❌ 存在失败'}")
    if not overall:
        return 1

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "math_stage5_gradient_identities_results_torch.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stage": 5, "backend": "numpy+torch", "kind": "T",
                   "elapsed_s": round(dt, 2), "report": report},
                  f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"  结果已写入 {os.path.normpath(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
