Write-Host "Starting Deployment Process..." -ForegroundColor Cyan

# 1. Run Database Migrations
Write-Host "Running Database Migrations..." -ForegroundColor Yellow
Set-Location backend
alembic upgrade head
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
pm2 restart BSClassifier

Write-Host "Deployment Complete!" -ForegroundColor Green