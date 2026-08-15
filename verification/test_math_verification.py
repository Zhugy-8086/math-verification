# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zhugy-8086
"""
pytest 包装：subprocess 逐个运行 12 个验证脚本，断言 exit code == 0。

设计约束：
  - 不改 12 个主脚本（保持"开箱即跑"，`python validate_math_stageX_*.py` 直接可用）
  - 结果 JSON 由脚本自身写入 `../results/`（与脚本位置相对，cwd 无关）
  - 源目录中依赖 C++ 扩展的 msint 脚本不属发布验证范围，已排除

用法：
    python -m pytest verification/test_math_verification.py -q
    # 或在仓库根目录：
    python -m pytest verification -q
"""
from __future__ import annotations

import glob
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


def test_script_inventory():
    """确认被测脚本恰好 12 个（stage1-6 × numpy/torch）。"""
    assert len(SCRIPTS) == 12, (
        f"expected 12 stage scripts, got {len(SCRIPTS)}: {SCRIPT_IDS}"
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=SCRIPT_IDS)
def test_stage_script(script: str):
    """逐脚本回归：exit code 必须为 0（脚本内部自带全部数值判定）。"""
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
