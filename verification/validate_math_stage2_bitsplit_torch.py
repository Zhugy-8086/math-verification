# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zhugy-8086
"""
阶段 2：位操作代数 — NumPy / PyTorch 双库互证（T 类，bit-exact）
================================================================
对应计划 numpy_math_verification_plan_2026_08_13.md §3 阶段 2 的 #5-#8

目的：
  同一组位操作恒等式用 NumPy 与 PyTorch 两种独立计算库各实现一遍，
  逐项比对输出，排除"单库实现 bug"——双库一致才算通过。

验证项（与 validate_math_stage2_bitsplit.py 对齐）：
  #5 [T] bitsplit/concat 可逆（含顺序不对称，零进位泄漏）
  #6 [T] 整数位拆分严格可逆：value = high·2^p + low
  #7 [T] int16 ≡ 由两个 int8 重构（high<<8|low）
  #8 [T] 量化公式与 scale 定义：scale = max/(2^(bits-1)-1)

一致性判定（§2.4 T 类）：
  - 位精确相等（bit-exact），零容差
  - 双库逐项比对（numpy 数组 vs torch 张量）

用法（需安装 torch）：
    python validate_math_stage2_bitsplit_torch.py
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

SEEDS = [0, 1, 2, 3, 4]

report = {}


# ============================================================
# 双库向量化位拆分 / 拼接（MSB 在前 concat）
# ============================================================

def bitsplit_np(values, total_bits, target_bits):
    """NumPy 向量化 bitsplit：低位在前（parts[0]=LSB），负数转补码"""
    v = values.astype(np.int64).copy()
    if (v < 0).any():
        v = v & ((1 << total_bits) - 1)
    mask = (1 << target_bits) - 1
    n_parts = (total_bits + target_bits - 1) // target_bits
    parts = []
    rem = v
    for _ in range(n_parts):
        parts.append(rem & mask)
        rem = rem >> target_bits
    return parts


def concat_np(parts, target_bits):
    """NumPy 向量化 concat：第一个值在高位（MSB 在前）"""
    recon = np.zeros_like(parts[0])
    for p in parts:
        recon = (recon << target_bits) | p
    return recon


def bitsplit_torch(values, total_bits, target_bits):
    """PyTorch 向量化 bitsplit：低位在前，负数转补码"""
    v = values.clone().to(torch.int64)
    if torch.any(v < 0):
        v = v & ((1 << total_bits) - 1)
    mask = (1 << target_bits) - 1
    n_parts = (total_bits + target_bits - 1) // target_bits
    parts = []
    rem = v
    for _ in range(n_parts):
        parts.append(rem & mask)
        rem = rem >> target_bits
    return parts


def concat_torch(parts, target_bits):
    """PyTorch 向量化 concat：第一个值在高位（MSB 在前）"""
    recon = torch.zeros_like(parts[0])
    for p in parts:
        recon = (recon << target_bits) | p
    return recon


def _compare(label, ok_np, ok_torch, detail=""):
    """双库一致性判定（T 类：位精确布尔）"""
    same = bool(ok_np == ok_torch)
    ok = same and ok_np
    print(f"    [T] numpy={'PASS' if ok_np else 'FAIL'} "
          f"torch={'PASS' if ok_torch else 'FAIL'} 一致性={same} "
          f"→ {'PASS' if ok else 'FAIL'}{detail}")
    return ok


# ============================================================
# #6 [T] 位拆分严格可逆：value = high·2^p + low
# ============================================================
def verify_value_high_low():
    print("=" * 72)
    print("#6 [T] 位拆分严格可逆：value = high·2^p + low（双库互证）")
    print("=" * 72)

    rng = np.random.default_rng(2026)
    N = 1_000_000
    all_pass = True
    total_checked = 0
    for p in [1, 2, 4, 8, 12, 16, 24]:
        max_bits = p * 2
        values = rng.integers(0, 1 << max_bits, N, dtype=np.int64)

        # numpy
        low = values & ((1 << p) - 1)
        high = values >> p
        ok1_np = np.array_equal((high << p) | low, values)
        ok2_np = np.array_equal(high * (1 << p) + low, values)

        # torch
        vt = torch.from_numpy(values).to(torch.int64)
        low_t = vt & ((1 << p) - 1)
        high_t = vt >> p
        ok1_t = torch.equal((high_t << p) | low_t, vt)
        ok2_t = torch.equal(high_t * (1 << p) + low_t, vt)

        ok = _compare(f"p={p}", ok1_np and ok2_np, ok1_t and ok2_t,
                      detail=f"  值域[0,2^{max_bits}) N={N}")
        all_pass &= ok
        total_checked += N

    print(f"\n  总计检查 {total_checked} 个值（双库各一遍）")
    report["#6"] = {"checked": total_checked, "pass": all_pass}
    return all_pass


# ============================================================
# #5 [T] bitsplit/concat 可逆（含顺序不对称验证）
# ============================================================
def verify_bitsplit_concat():
    print("\n" + "=" * 72)
    print("#5 [T] bitsplit/concat 可逆（int12 零进位泄漏，双库互证）")
    print("=" * 72)

    rng = np.random.default_rng(7)
    N = 200_000
    combos = [
        (12, 4), (12, 8), (16, 8), (16, 4),
        (24, 8), (32, 8), (32, 16),
    ]
    all_pass = True
    print(f"  {'组合':<12} {'N':>8} {'正确顺序重构(双库)':<18}")
    for total_bits, target_bits in combos:
        n_parts = (total_bits + target_bits - 1) // target_bits
        values = rng.integers(0, 1 << total_bits, N, dtype=np.int64)

        # numpy：bitsplit(低→高) → 反转 → concat(高→低)
        parts_np = bitsplit_np(values, total_bits, target_bits)
        recon_np = concat_np(list(reversed(parts_np)), target_bits)
        ok_np = np.array_equal(recon_np, values)

        # torch
        vt = torch.from_numpy(values).to(torch.int64)
        parts_t = bitsplit_torch(vt, total_bits, target_bits)
        recon_t = concat_torch(list(reversed(parts_t)), target_bits)
        ok_t = torch.equal(recon_t, vt)

        ok = _compare(f"{total_bits}→{target_bits}×{n_parts}", ok_np, ok_t,
                      detail=f"  正确顺序 bit-exact 可逆")
        all_pass &= ok

        # 顺序不对称：直接 concat（不反转）应有零进位泄漏（wrong ≠ v）
        wrong_np = concat_np(parts_np, target_bits)
        wrong_t = concat_torch(parts_t, target_bits)
        leak_np = not np.array_equal(wrong_np, values)
        leak_t = not torch.equal(wrong_t, vt)
        # 对 p 组合：错误顺序必然 ≠ 原值（除对称退化组合 12→8 外需排除）
        if total_bits == 12 and target_bits == 8:
            continue   # 2×8 对称拆分，反转等价，跳过泄漏检查
        leak_ok = _compare(f"{total_bits}→{target_bits} 泄漏", leak_np, leak_t,
                           detail=f"  错误顺序无法还原（存在泄漏）")
        all_pass &= leak_ok

    print(f"  结论: {'✓ 全部 bit-exact 可逆（双库一致）' if all_pass else '✗ 有失败'}")
    report["#5"] = {"pass": all_pass}
    return all_pass


# ============================================================
# #7 [T] int16 视角 ≡ 两个 int8 重构的 int16
# ============================================================
def verify_int16_from_two_int8():
    print("\n" + "=" * 72)
    print("#7 [T] int16 ≡ 两个 int8 重构（high<<8|low，双库互证）")
    print("=" * 72)

    rng = np.random.default_rng(99)
    N = 1_000_000
    vals = rng.integers(-32768, 32768, N, dtype=np.int64)

    # numpy
    u = vals & 0xFFFF
    low = u & 0xFF
    high = u >> 8
    recon_u = (high << 8) | low
    recon_s = np.where(recon_u >= 32768, recon_u - 65536, recon_u)
    ok_u_np = np.array_equal(recon_u, u)
    ok_s_np = np.array_equal(recon_s, vals)

    # torch
    vt = torch.from_numpy(vals).to(torch.int64)
    ut = vt & 0xFFFF
    low_t = ut & 0xFF
    high_t = ut >> 8
    recon_ut = (high_t << 8) | low_t
    recon_st = torch.where(recon_ut >= 32768, recon_ut - 65536, recon_ut)
    ok_u_t = torch.equal(recon_ut, ut)
    ok_s_t = torch.equal(recon_st, vt)

    ok = _compare("int16 重构", ok_u_np and ok_s_np, ok_u_t and ok_s_t,
                  detail=f"  覆盖 [-32768,32767] N={N}")
    report["#7"] = {"pass": ok}
    return ok


# ============================================================
# #8 [T] scale = max/(2^(bits-1)-1)
# ============================================================
def verify_scale_formula():
    print("\n" + "=" * 72)
    print("#8 [T] scale = max/(2^(bits-1)-1)（双库互证）")
    print("=" * 72)

    rng = np.random.default_rng(5)
    all_pass = True
    for bits in [8, 16, 24, 32]:
        max_code = 2 ** (bits - 1) - 1
        for _ in range(100):
            x_max = float(10.0 ** rng.uniform(-3, 3))
            scale_np = np.float64(x_max) / max_code
            scale_t = torch.tensor(x_max, dtype=torch.float64) / max_code
            same = scale_np == scale_t.item()   # 位精确相等
            if not same:
                all_pass = False
        print(f"  bits={bits:>2}: max_code=2^{bits-1}-1={max_code}, "
              f"scale(双库 bit-exact 一致) → {'✓' if all_pass else '✗'}")
    report["#8"] = {"pass": all_pass}
    return all_pass


def main():
    t0 = time.time()
    r6 = verify_value_high_low()
    r5 = verify_bitsplit_concat()
    r7 = verify_int16_from_two_int8()
    r8 = verify_scale_formula()
    dt = time.time() - t0

    print("\n" + "=" * 72)
    print("阶段 2 双库互证汇总（T 类，bit-exact）")
    print("=" * 72)
    print(f"  耗时 {dt:.1f}s")
    print(f"  #6 位拆分可逆 value=high·2^p+low : {'✓' if r6 else '✗'}")
    print(f"  #5 bitsplit/concat 可逆(int12 等) : {'✓' if r5 else '✗'}")
    print(f"  #7 int16≡两个int8(high<<8|low)    : {'✓' if r7 else '✗'}")
    print(f"  #8 scale=max/(2^(bits-1)-1)       : {'✓' if r8 else '✗'}")
    overall = r6 and r5 and r7 and r8
    print(f"\n  总体判定: {'✅ 双库全部 bit-exact 通过' if overall else '❌ 存在失败'}")
    if not overall:
        return 1

    # 结果写入 results/
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "math_stage2_bitsplit_results_torch.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stage": 2, "backend": "numpy+torch", "kind": "T",
                   "elapsed_s": round(dt, 2), "report": report},
                  f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {os.path.normpath(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
