@echo off
chcp 65001 >nul
echo ============================================================
echo           牙齿影像异常区域检测系统 - 性能测试
echo ============================================================
echo.
echo 提示：请确保后端服务已启动！
echo 如果未启动，请先在另一个终端运行：
echo   cd d:\文档\毕业设计\teeth\backend
echo   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
pause
echo.
echo 开始运行性能测试...
echo.

cd /d "d:\文档\毕业设计\teeth"
python performance_test.py

echo.
echo ============================================================
echo                      测试完成！
echo ============================================================
echo.
pause
