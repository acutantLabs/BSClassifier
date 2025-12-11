# dev-start.ps1
# Development Environment Startup Script

Write-Host "Starting Development Environment..." -ForegroundColor Green
Write-Host "Production instance on port 3333 will remain running" -ForegroundColor Yellow
Write-Host ""

# Backend
Write-Host "Starting Backend on port 3334..." -ForegroundColor Yellow
$backendPath = Join-Path $PSScriptRoot "backend"
cd $backendPath
$env:APP_ENV="development"
$env:POSTGRES_URL="postgresql+asyncpg://bsclassifier_user:lHL1NkxI4RJGur5xOqAq@localhost:5432/bsclassifier_db"
$env:DATABASE_URL="postgresql+asyncpg://bsclassifier_user:lHL1NkxI4RJGur5xOqAq@localhost:5432/bsclassifier_db"
$env:CORS_ORIGINS="http://localhost:3001,http://localhost:3334"
# Add other environment variables from your production config here
# Example: $env:OPENAI_API_KEY="your_key_here"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendPath'; .\venv\Scripts\Activate.ps1; uvicorn server:app --host 0.0.0.0 --port 3334 --reload"

# Wait a moment for backend to start
Start-Sleep -Seconds 2

# Frontend
Write-Host "Starting Frontend on port 3001..." -ForegroundColor Yellow
$frontendPath = Join-Path $PSScriptRoot "frontend"
cd $frontendPath
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; `$env:PORT='3001'; `$env:REACT_APP_API_URL='http://localhost:3334'; npm start"

Write-Host ""
Write-Host "Development servers starting in separate windows..." -ForegroundColor Green
Write-Host "Backend: http://localhost:3334" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3001" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit this script (servers will continue running)..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

