@echo off
setlocal DisableDelayedExpansion
if "%~1"=="" goto usage
python "%~dp0gemma_locate.py" %*
set "locate_exit=%errorlevel%"
echo.
pause
exit /b %locate_exit%

:usage
echo Drag an image folder onto this BAT file.
echo LM Studio must be running at http://127.0.0.1:1234 with model gemma3 loaded.
echo Creates locations.json and PNG crops in the folder's crops\NC subfolder.
echo Locates visible fastenings and estimates their positions when obscured.
echo Records visible, estimated, or unavailable. Estimated crops default to 256 pixels.
echo Runs 4 concurrent image jobs by default. Use --workers to change concurrency.
echo Completed unchanged images are skipped silently. Only new or changed images call Gemma.
echo Missing crops are restored from saved locations. Use --crop-only to rebuild crops.
echo Existing results from the previous format require --redo once before resuming.
echo Thinking is off by default. Optional: gemma_locate.bat "folder" --think
echo.
pause
exit /b 1
