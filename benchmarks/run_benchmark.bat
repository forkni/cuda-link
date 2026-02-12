@echo off
echo ================================================================================
echo CUDA IPC Benchmark
echo ================================================================================
echo.
echo Running benchmark with 1000 frames at 512x512 resolution with IPC events...
echo.
python benchmark_cuda_ipc.py --frames 1000 --events --resolution 512x512
echo.
echo ================================================================================
echo Benchmark Complete
echo ================================================================================
pause
