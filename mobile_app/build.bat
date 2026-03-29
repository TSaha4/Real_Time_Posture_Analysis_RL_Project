@echo off
REM UPRYT Mobile - Windows Build Script
REM Builds Android APK on Windows

echo ============================================
echo UPRYT Mobile - Android Build
echo ============================================
echo.

REM Check for Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Install buildozer if not present
python -c "import buildozer" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing Buildozer...
    pip install buildozer cython==0.29.33
)

REM Parse arguments
set BUILD_MODE=debug
set CLEAN=

if "%1"=="release" set BUILD_MODE=release
if "%1"=="clean" (
    echo Cleaning build directory...
    if exist .buildozer rmdir /s /q .buildozer
    exit /b 0
)

echo Build mode: %BUILD_MODE%
echo.

REM Check for ANDROID_HOME
if "%ANDROID_HOME%"=="" (
    echo WARNING: ANDROID_HOME not set.
    echo The build may fail without Android SDK.
    echo.
)

REM Build
echo Starting build...
echo.

if "%BUILD_MODE%"=="release" (
    buildozer android release
) else (
    buildozer android debug
)

REM Find APK
echo.
echo ============================================
echo Build complete!
echo.

for %%F in (bin\*.apk) do (
    echo APK: %%F
    echo Size: %%~zF bytes
    echo.
    echo To install on device:
    echo   adb install -r "%%F"
)

if not exist "bin\*.apk" (
    echo WARNING: APK not found. Check build logs.
)

echo ============================================
pause
