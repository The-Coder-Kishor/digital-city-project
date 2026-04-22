@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all_tender_ids.ps1" %*
exit /b %ERRORLEVEL%
