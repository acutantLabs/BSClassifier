# PM2 Daemon Fix Script
# This script fixes PM2 daemon connection issues on Windows

Write-Host "Fixing PM2 Daemon Issues..." -ForegroundColor Cyan

# Step 1: Kill all PM2 processes (requires admin if processes are protected)
Write-Host "`nStep 1: Stopping PM2 processes..." -ForegroundColor Yellow
pm2 kill 2>&1 | Out-Null

# Step 2: Clean up PM2 socket files
Write-Host "Step 2: Cleaning PM2 socket files..." -ForegroundColor Yellow
$pm2Home = "$env:USERPROFILE\.pm2"
if (Test-Path $pm2Home) {
    Get-ChildItem -Path $pm2Home -Filter "*.sock" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    if (Test-Path "$pm2Home\pids") {
        Remove-Item -Path "$pm2Home\pids" -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host "  Cleaned PM2 socket files" -ForegroundColor Green
}

# Step 3: Wait for processes to fully terminate
Write-Host "Step 3: Waiting for processes to terminate..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Step 4: Start PM2 daemon fresh
Write-Host "Step 4: Starting PM2 daemon..." -ForegroundColor Yellow
pm2 ping 2>&1 | Out-Null

# Step 5: Check PM2 status
Write-Host "`nStep 5: Checking PM2 status..." -ForegroundColor Yellow
$pm2Status = pm2 status 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nPM2 is now running!" -ForegroundColor Green
    pm2 status
    
    # Check if BSClassifier process exists
    Write-Host "`nChecking for BSClassifier process..." -ForegroundColor Cyan
    $pm2List = pm2 jlist 2>&1
    if ($LASTEXITCODE -eq 0) {
        $pm2Json = $pm2List | ConvertFrom-Json
        $existingProcess = $pm2Json | Where-Object { $_.name -eq "BSClassifier" }
        
        if ($existingProcess) {
            Write-Host "BSClassifier process found. You can restart it with: pm2 restart BSClassifier" -ForegroundColor Green
        } else {
            Write-Host "BSClassifier process not found. Start it with: pm2 start backend/ecosystem.config.js" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "`nPM2 daemon still having issues. You may need to:" -ForegroundColor Red
    Write-Host "  1. Close all terminal windows" -ForegroundColor Yellow
    Write-Host "  2. Run PowerShell as Administrator" -ForegroundColor Yellow
    Write-Host "  3. Run: taskkill /F /IM node.exe" -ForegroundColor Yellow
    Write-Host "  4. Run this script again" -ForegroundColor Yellow
}

Write-Host "`nFix script complete!" -ForegroundColor Cyan

