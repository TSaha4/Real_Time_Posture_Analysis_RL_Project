@echo off
echo ========================================
echo   UPRYT - Command Line Mode
echo ========================================
echo.
echo Available modes:
echo   1 - Real-Time Monitoring (PPO)
echo   2 - Real-Time Monitoring (DQN)
echo   3 - Real-Time Monitoring (Rule-Based)
echo   4 - Train PPO Agent
echo   5 - Train DQN Agent
echo   6 - Compare All Algorithms
echo.
set /p choice="Select mode (1-6): "

if "%choice%"=="1" goto realtime_ppo
if "%choice%"=="2" goto realtime_dqn
if "%choice%"=="3" goto realtime_rule
if "%choice%"=="4" goto train_ppo
if "%choice%"=="5" goto train_dqn
if "%choice%"=="6" goto compare

echo Invalid choice
pause
exit /b

:realtime_ppo
python "%~dp0main.py" --mode realtime --algorithm ppo
pause
exit /b

:realtime_dqn
python "%~dp0main.py" --mode realtime --algorithm dqn
pause
exit /b

:realtime_rule
python "%~dp0main.py" --mode realtime --algorithm rule
pause
exit /b

:train_ppo
python "%~dp0main.py" --mode train --algorithm ppo --episodes 500
pause
exit /b

:train_dqn
python "%~dp0main.py" --mode train --algorithm dqn --episodes 500
pause
exit /b

:compare
python "%~dp0main.py" --mode compare
pause
exit /b
