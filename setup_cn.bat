@echo off
setlocal enabledelayedexpansion
set "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"

echo ============================================
echo   SlimeMold Artist Explorer - Setup
echo ============================================
echo.

:: Find Python
set PYTHON=
for %%p in (python3 python) do (
    where %%p >nul 2>&1
    if !errorlevel!==0 (
        %%p --version >nul 2>&1
        if !errorlevel!==0 set PYTHON=%%p
    )
)
if "%PYTHON%"=="" (
    echo [ERROR] Python not found. Install Python 3.10+ first.
    pause
    exit /b 1
)
echo [CHECK] Python:
%PYTHON% --version

:: Check/create venv
if exist "venv\Scripts\activate.bat" (
    echo [SKIP] venv already exists
    set "SKIP_INSTALL=1"
    goto :activate
)

echo.
echo [1/4] Creating venv ...
%PYTHON% -m venv venv --clear
if !errorlevel! neq 0 (
    echo [ERROR] Failed to create venv
    pause
    exit /b 1
)

:activate
call "venv\Scripts\activate.bat"
if !errorlevel! neq 0 (
    echo [ERROR] Failed to activate venv
    pause
    exit /b 1
)

if "%SKIP_INSTALL%"=="1" goto :menu

:: Upgrade pip
echo.
echo [2/4] Upgrading pip ...
python -m pip install --upgrade pip

:: GPU detection + torch (torch must use official index, not mirror)
echo.
echo [3/4] Installing torch ...
nvidia-smi >nul 2>&1
if !errorlevel!==0 (
    echo   NVIDIA GPU found - installing CUDA torch ...
    set "PIP_INDEX_URL="
    pip install torch --index-url https://mirrors.aliyun.com/pytorch-wheels/cu130
    if !errorlevel! neq 0 (
        echo   [WARN] CUDA torch failed, fallback CPU ...
        pip install torch
    )
    set "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"
) else (
    echo   No NVIDIA GPU - installing CPU torch ...
    set "PIP_INDEX_URL="
    pip install torch
    set "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"
)

:: Dependencies
echo.
echo [4/4] Installing dependencies ...
pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install dependencies. Check your network.
    pause
    exit /b 1
)

:menu
echo.
echo ============================================
echo   Launch:
echo ============================================
echo.
echo   [1] SDXL Explorer        (webui/webui.py)
echo   [2] Anima Explorer       (webui/anima_webui.py)
echo   [3] SDXL Build Vectors   (webui/build_webui.py)
echo   [4] Anima Build Vectors  (webui/build_webui_anima.py)
echo   [0] Exit
echo.
set /p CHOICE="Enter number (0-4): "

if "%CHOICE%"=="1" python webui/webui.py
if "%CHOICE%"=="2" python webui/anima_webui.py
if "%CHOICE%"=="3" python webui/build_webui.py
if "%CHOICE%"=="4" python webui/build_webui_anima.py
if "%CHOICE%"=="0" exit /b 0

echo.
echo   Run this script again to restart.
pause
