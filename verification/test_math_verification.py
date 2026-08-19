# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zhugy-8086
"""
pytest 包装：subprocess 逐个运行 12 个验证脚本，断言：
  1) exit code == 0（numpy 脚本理论判定失败 / torch 双库不一致时返回非零）
  2) 结果 JSON 顶层 pass 字段为 True（若存在）

设计约束：
  - 结果 JSON 由脚本自身写入 `../results/`（与脚本位置相对，cwd 无关）
  - 源目录中依赖 C++ 扩展的 msint 脚本不属发布验证范围，已排除
  - 2026-08-19 审计：原实现只断言 returncode==0，而 stage1 原 exit 恒 0；
    现 stage1 已补非零退出码 + JSON pass 字段，测试改为"退出码 + JSON 判定"双通道

用法：
    python -m pytest verification/test_math_verification.py -q
    # 或在仓库根目录：
    python -m pytest verification -q
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

# 排除 msint（硬依赖 C++ 扩展，不属发布验证范围）
SCRIPTS = sorted(
    p for p in glob.glob(os.path.join(HERE, "validate_math_stage*.py"))
    if "msint" not in os.path.basename(p)
)
SCRIPT_IDS = [os.path.basename(s) for s in SCRIPTS]


def _results_json_for(script):
    """由脚本名推导结果 JSON 路径。
    validate_math_stage1_sr.py → math_stage1_sr_results.json
    validate_math_stage1_sr_torch.py → math_stage1_sr_results_torch.json
    """
    stem = os.path.basename(script)[len("validate_math_"):]
    stem = stem[:-3]  # 去 .py
    return os.path.join(HERE, "..", "results", "math_" + stem + "_results.json")


def test_script_inventory():
    """确认被测脚本恰好 12 个（stage1-6 × numpy/torch）。"""
    assert len(SCRIPTS) == 12, (
        f"expected 12 stage scripts, got {len(SCRIPTS)}: {SCRIPT_IDS}"
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=SCRIPT_IDS)
def test_stage_script(script: str):
    """逐脚本回归：
    1) exit code 必须为 0——numpy 脚本理论判定失败返回非零（stage1 已修，
       stage2-6 原本即 sys.exit(main())）；torch 脚本双库不一致返回非零。
    2) 若结果 JSON 存在且含顶层 pass 字段，必须为 True（理论判定）。
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, script],
        cwd=HERE, env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=3600,
    )
    tail_out = (proc.stdout or "")[-3000:]
    tail_err = (proc.stderr or "")[-3000:]
    assert proc.returncode == 0, (
        f"{os.path.basename(script)} exited with {proc.returncode}\n"
        f"--- stdout (tail) ---\n{tail_out}\n"
        f"--- stderr (tail) ---\n{tail_err}"
    )

    rj = _results_json_for(script)
    if os.path.exists(rj):
        with open(rj, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "pass" in data:
            assert data["pass"] is True, (
                f"{os.path.basename(script)}: 结果 JSON pass=False "
                f"（理论判定失败）: {rj}"
            )
