@echo off
echo Start installing virtual environment and dependence.

REM you can use py command to select python version
REM e.g. py -3.12 -m venv venv
REM or you can use python specified in your environment
REM python -m venv venv

python -m venv venv
.\venv\Scripts\python.exe -m pip install -U pip setuptools
.\venv\Scripts\python.exe -m pip install -r requirements.txt

echo Finish installing virtual environment and dependence.
echo Press any key to close...
pause >nul