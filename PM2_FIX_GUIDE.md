# PM2 Daemon Fix Guide (Windows)

## Quick Fix (If PM2 won't connect)

### Option 1: Reset PM2 Completely (Recommended)
1. **Close all terminal/PowerShell windows**
2. **Open PowerShell as Administrator** (Right-click → Run as Administrator)
3. Run these commands:
   ```powershell
   # Kill all Node processes
   taskkill /F /IM node.exe
   
   # Clean PM2 directory
   Remove-Item -Path "$env:USERPROFILE\.pm2" -Recurse -Force -ErrorAction SilentlyContinue
   
   # Wait a moment
   Start-Sleep -Seconds 2
   
   # Start PM2 fresh
   pm2 ping
   
   # Check status
   pm2 status
   ```

4. **Restart your application**:
   ```powershell
   cd "D:\Custom Tools\BSClassifier\backend"
   pm2 start ecosystem.config.js
   ```

### Option 2: Use the Fix Script
Run the provided fix script:
```powershell
cd "D:\Custom Tools\BSClassifier"
.\fix-pm2.ps1
```

If that doesn't work, follow Option 1 above.

## Prevention

The updated `deploy.ps1` script now includes automatic PM2 daemon recovery. It will:
- Detect PM2 daemon issues
- Attempt to reset the daemon
- Fallback to starting from config if needed

## Troubleshooting

### Error: "connect EPERM //./pipe/rpc.sock"
This means PM2's IPC socket is corrupted or locked.

**Solution**: Run Option 1 above (requires admin privileges)

### Error: "Access is denied" when killing processes
Some Node processes are protected or running under a different user.

**Solution**: 
1. Open Task Manager (Ctrl+Shift+Esc)
2. Go to "Details" tab
3. Find all `node.exe` processes
4. Right-click → End Task
5. Then run the fix commands

### PM2 won't start after fix
1. Verify Node.js is installed: `node --version`
2. Verify PM2 is installed: `pm2 --version`
3. Try reinstalling PM2: `npm install -g pm2`
4. Check Windows firewall isn't blocking PM2

## After Fixing

Once PM2 is working again:
```powershell
# Start your application
cd "D:\Custom Tools\BSClassifier\backend"
pm2 start ecosystem.config.js

# Save PM2 process list (so it can auto-resurrect after reboot)
pm2 save
pm2 startup
```

