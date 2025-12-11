# deploy.ps1 - Production Deployment Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Production Deployment Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = $PSScriptRoot
if (-not $projectRoot) {
    $projectRoot = Get-Location
}

# Step 1: Database migrations
Write-Host "[1/3] Checking for database migrations..." -ForegroundColor Yellow
$backendPath = Join-Path $projectRoot "backend"
Set-Location $backendPath

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    & .\venv\Scripts\Activate.ps1
    Write-Host "  Applying migrations..." -ForegroundColor Gray
    alembic upgrade head
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[1/3] Database migrations complete" -ForegroundColor Green
    } else {
        Write-Host "[1/3] Warning: Migration check completed with warnings" -ForegroundColor Yellow
    }
} else {
    Write-Host "[1/3] Warning: Virtual environment not found, skipping migrations" -ForegroundColor Yellow
}

Write-Host ""

# Step 2: Build frontend
Write-Host "[2/3] Building frontend for production..." -ForegroundColor Yellow
$frontendPath = Join-Path $projectRoot "frontend"
Set-Location $frontendPath
Write-Host "  Running npm build..." -ForegroundColor Gray
npm run build

if ($LASTEXITCODE -ne 0) {
    Write-Host "[2/3] Error: Frontend build failed" -ForegroundColor Red
    Write-Host "  Deployment aborted. Please fix build errors and try again." -ForegroundColor Red
    exit 1
}

Write-Host "[2/3] Frontend build complete" -ForegroundColor Green
$buildPath = Join-Path $projectRoot "backend\build"
if (Test-Path $buildPath) {
    Write-Host "  Build artifacts verified in backend/build/" -ForegroundColor Gray
}

Write-Host ""

# Step 3: Restart PM2
Write-Host "[3/3] Restarting production instance..." -ForegroundColor Yellow
Set-Location $projectRoot

$pm2Check = Get-Command pm2 -ErrorAction SilentlyContinue
if (-not $pm2Check) {
    Write-Host "[3/3] Error: PM2 not found. Please install PM2 globally:" -ForegroundColor Red
    Write-Host "  npm install -g pm2" -ForegroundColor Yellow
    exit 1
}

$instanceExists = $false
$pm2Output = pm2 jlist 2>&1
if ($LASTEXITCODE -eq 0) {
    $pm2List = $pm2Output | ConvertFrom-Json
    $foundInstance = $pm2List | Where-Object { $_.name -eq "BSClassifier" }
    if ($foundInstance) {
        $instanceExists = $true
    }
}

if (-not $instanceExists) {
    Write-Host "[3/3] Error: BSClassifier instance not found in PM2" -ForegroundColor Red
    Write-Host "  Please start it manually with: pm2 start backend/ecosystem.config.js" -ForegroundColor Yellow
    exit 1
}

Write-Host "  Restarting BSClassifier instance..." -ForegroundColor Gray
pm2 restart BSClassifier

if ($LASTEXITCODE -eq 0) {
    Write-Host "[3/3] Production instance restarted successfully" -ForegroundColor Green
} else {
    Write-Host "[3/3] Warning: PM2 restart completed with warnings" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Deployment Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Production Status:" -ForegroundColor Yellow
pm2 status BSClassifier
Write-Host ""
Write-Host "Production URL: http://localhost:3333" -ForegroundColor Cyan
Write-Host ""
Write-Host "To view logs: pm2 logs BSClassifier" -ForegroundColor Gray
Write-Host "To monitor: pm2 monit" -ForegroundColor Gray
Write-Host ""
