@echo off
setlocal

pushd "%~dp0" >nul 2>&1

if "%~1"=="" (
	echo Usage: %~nx0 ^<number of workers^>
	popd >nul 2>&1
	exit /b 1
)

set "N=%~1"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"') do set "DATE=%%I"

if not exist "logs" mkdir "logs"
if not exist "logs\%DATE%" mkdir "logs\%DATE%"

start "" /B cmd /c ""%~dp0bin\learner" 1^> "logs\%DATE%\learner.log" 2^>^&1"

for /L %%i in (1,1,%N%) do (
	start "" /B cmd /c ""%~dp0bin\worker" %%i 1^> "logs\%DATE%\worker_%%i.log" 2^>^&1"
)

echo Started learner and %N% workers. Logs are in the .\logs\%DATE% directory.
echo To stop all processes, use: taskkill /IM learner.exe /F ^&^& taskkill /IM worker.exe /F
echo To see running processes, use: tasklist ^| findstr /I "learner worker"

popd >nul 2>&1
endlocal
exit /b 0

