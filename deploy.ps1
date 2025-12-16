Write-Host "Starting Deployment Process..." -ForegroundColor Cyan

# 1. Run Database Migrations
Write-Host "Running Database Migrations..." -ForegroundColor Yellow
Set-Location backend
# Activate virtual environment and run alembic
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
    alembic upgrade head
    deactivate
} else {
    Write-Host "Warning: Virtual environment not found. Skipping migrations." -ForegroundColor Yellow
}
Set-Location ..

# 2. Build Frontend
Write-Host "Building Frontend..." -ForegroundColor Yellow
Set-Location frontend
npm run build
Set-Location ..

# 3. CRITICAL FIX: Transfer Build Files
Write-Host "Deploying Build Files to Backend..." -ForegroundColor Yellow

# Define paths
$FrontendBuildPath = ".\frontend\build"
$BackendBuildPath = ".\backend\build"

# Remove old backend build folder if it exists
if (Test-Path $BackendBuildPath) {
    Write-Host "Removing old build files..."
    Remove-Item -Path $BackendBuildPath -Recurse -Force
}

# Copy new build folder from frontend to backend
if (Test-Path $FrontendBuildPath) {
    Write-Host "Copying new build files..."
    Copy-Item -Path $FrontendBuildPath -Destination ".\backend" -Recurse -Force
} else {
    Write-Host "ERROR: Frontend build failed. Directory not found." -ForegroundColor Red
    Exit
}

# 4. Restart PM2
Write-Host "Restarting PM2 Service..." -ForegroundColor Yellow

# Check if PM2 daemon is running, restart it if needed
try {
    $pm2Status = pm2 status 2>&1
    if ($LASTEXITCODE -ne 0 -or $pm2Status -match "error|EPERM|connect") {
        Write-Host "PM2 daemon appears to be down. Resetting daemon..." -ForegroundColor Yellow
        pm2 kill
        Start-Sleep -Seconds 2
        pm2 resurrect 2>&1 | Out-Null
        Start-Sleep -Seconds 1
    }
} catch {
    Write-Host "Resetting PM2 daemon..." -ForegroundColor Yellow
    pm2 kill
    Start-Sleep -Seconds 2
}

# Check if BSClassifier process exists
# Use text-based pm2 list instead of JSON to avoid duplicate key issues
$pm2ListText = pm2 list 2>&1
if ($LASTEXITCODE -eq 0) {
    if ($pm2ListText -match "BSClassifier") {
        Write-Host "Restarting existing BSClassifier process..." -ForegroundColor Cyan
        pm2 restart BSClassifier
    } else {
        Write-Host "BSClassifier process not found. Starting from ecosystem.config.js..." -ForegroundColor Cyan
        Set-Location backend
        pm2 start ecosystem.config.js
        Set-Location ..
    }
} else {
    Write-Host "Warning: Could not check PM2 processes. Attempting restart anyway..." -ForegroundColor Yellow
    pm2 restart BSClassifier 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Restart failed. Trying to start from config..." -ForegroundColor Yellow
        Set-Location backend
        pm2 start ecosystem.config.js
        Set-Location ..
    }
}

# Verify PM2 status
Start-Sleep -Seconds 2
Write-Host "`nPM2 Status:" -ForegroundColor Cyan
pm2 status

Write-Host "`nDeployment Complete!" -ForegroundColor Green
Write-Host "To view logs: pm2 logs BSClassifier" -ForegroundColor Gray
Write-Host "To monitor: pm2 monit" -ForegroundColor Gray