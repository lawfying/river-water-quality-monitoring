param(
    [switch]$NoBrowser = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AI水质安全监测预测系统 - 一键启动" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$VENV_PYTHON = "c:\water-llm\conference\Scripts\python"
$STREAMLIT_PORT = 8501
$DJANGO_PORT = 8000

# 检测端口是否被占用
function Test-PortInUse($port) {
    $connections = netstat -ano | Select-String "LISTENING" | Select-String ":$port "
    return ($connections -ne $null)
}

# 1. 启动 Django 后端（新窗口）
Write-Host "[1/2] 启动 Django 后端 (端口 $DJANGO_PORT)..." -ForegroundColor Yellow
if (Test-PortInUse $DJANGO_PORT) {
    Write-Host "  [!] 端口 $DJANGO_PORT 已被占用，跳过 Django 启动" -ForegroundColor DarkYellow
    Write-Host "  如需重启请先关闭已有进程" -ForegroundColor DarkYellow
} else {
    $djangoJob = Start-Process powershell -ArgumentList "-NoExit", "-Command", "
        cd 'c:\water-llm\iot_backend'
        Write-Host '[Django] 后端启动中...' -ForegroundColor Green
        c:\water-llm\conference\Scripts\python manage.py runserver 0.0.0.0:$DJANGO_PORT
    " -WindowStyle Normal -PassThru
    Start-Sleep -Seconds 2
    Write-Host "  [OK] Django 后端窗口已打开" -ForegroundColor Green
}

# 2. 启动 Streamlit 前端
Write-Host "[2/2] 启动 Streamlit 前端 (端口 $STREAMLIT_PORT)..." -ForegroundColor Yellow
if (Test-PortInUse $STREAMLIT_PORT) {
    Write-Host "  [!] 端口 $STREAMLIT_PORT 已被占用，跳过 Streamlit 启动" -ForegroundColor DarkYellow
} else {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "
        cd 'c:\water-llm'
        Write-Host '[Streamlit] 前端启动中...' -ForegroundColor Green
        c:\water-llm\conference\Scripts\python -m streamlit run src/app.py --server.port $STREAMLIT_PORT --server.headless true
    " -WindowStyle Normal
    Start-Sleep -Seconds 3
    Write-Host "  [OK] Streamlit 前端窗口已打开" -ForegroundColor Green
}

# 3. 输出访问地址
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  启动完成！访问地址：" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Streamlit 前端 : http://localhost:$STREAMLIT_PORT" -ForegroundColor White
Write-Host "  Django API     : http://127.0.0.1:$DJANGO_PORT/api/" -ForegroundColor White
Write-Host "  Django Admin   : http://127.0.0.1:$DJANGO_PORT/admin/" -ForegroundColor White
Write-Host ""

if (-not $NoBrowser) {
    Start-Process "http://localhost:$STREAMLIT_PORT"
}

Write-Host "  按任意键关闭此窗口（后端和前端将继续运行）..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
