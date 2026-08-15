@echo off
rem -- one-click runner: all 12 stage verification scripts -------------
rem -- usage: double-click, or run in cmd. exit code 0 = all passed ----
chcp 65001 >nul
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ==============================================================
echo  math-verification: run all stage scripts (numpy + torch)
echo  results JSON will be written to ..\results\
echo ==============================================================

set PASS=0
set FAIL=0
set "FAILED_LIST="

for %%f in (validate_math_stage*.py) do (
    echo.
    echo [RUN] %%f ...
    python "%%f"
    if !errorlevel! equ 0 (
        echo [PASS] %%f
        set /a PASS+=1
    ) else (
        echo [FAIL] %%f - exit code !errorlevel!
        set /a FAIL+=1
        set "FAILED_LIST=!FAILED_LIST! %%f"
    )
)

echo.
echo ==============================================================
echo  Summary: PASS=!PASS!  FAIL=!FAIL!
if defined FAILED_LIST echo  Failed:!FAILED_LIST!
echo ==============================================================
if !FAIL! equ 0 (
    echo  ALL SCRIPTS PASSED
    exit /b 0
) else (
    exit /b 1
)
