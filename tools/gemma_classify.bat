@echo off
setlocal DisableDelayedExpansion
if "%~1"=="" goto usage
python -B -X utf8 "%~dp0gemma_classify.py" %*
set "classify_exit=%errorlevel%"
echo.
pause
exit /b %classify_exit%

:usage
echo Drag an image folder onto this BAT file.
echo LM Studio must be running at http://127.0.0.1:1234 with model gemma3 loaded.
echo Sends one image with the configured prompt. No reference image is needed.
echo Keeps matching images in place and permanently deletes nonmatching images.
echo Edit PROMPT in gemma_classify.py or use: gemma_classify.bat "folder" --prompt "requirement"
echo Only top-level images are processed. All subfolders are skipped.
echo Runs 4 requests in parallel by default. Optional: --workers 4
echo Thinking is on by default. Optional: gemma_classify.bat "folder" --no-think
echo.
pause
exit /b 1
