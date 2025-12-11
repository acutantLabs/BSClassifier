# Development Environment Setup

This guide explains how to run the development environment alongside your production PM2 instance.

## Port Configuration

- **Production (PM2)**: Backend on port `3333`, Frontend served from backend
- **Development**: Backend on port `3334`, Frontend dev server on port `3001`

## Quick Start

### Option 1: Using the Startup Script (Recommended)

Simply run:
```powershell
.\dev-start.ps1
```

This will start both backend and frontend in separate PowerShell windows.

### Option 2: Manual Start

**Terminal 1 - Backend:**
```powershell
cd backend
.\venv\Scripts\Activate.ps1
$env:APP_ENV="development"
$env:POSTGRES_URL="postgresql+asyncpg://bsclassifier_user:lHL1NkxI4RJGur5xOqAq@localhost:5432/bsclassifier_db"
$env:DATABASE_URL="postgresql+asyncpg://bsclassifier_user:lHL1NkxI4RJGur5xOqAq@localhost:5432/bsclassifier_db"
$env:CORS_ORIGINS="http://localhost:3001,http://localhost:3334"
# Add other environment variables as needed (e.g., OPENAI_API_KEY)
uvicorn server:app --host 0.0.0.0 --port 3334 --reload
```

**Terminal 2 - Frontend:**
```powershell
cd frontend
$env:PORT="3001"
$env:REACT_APP_API_URL="http://localhost:3334"
npm start
```

## Important Notes

1. **Don't stop your PM2 instance** - The development environment uses different ports, so both can run simultaneously.

2. **Environment Variables**: Make sure to set all required environment variables in the backend terminal (especially `OPENAI_API_KEY` if needed). You can copy them from your production `ecosystem.config.js`.

3. **Database**: By default, both environments share the same database. If you want to use a separate dev database, update the `DATABASE_URL` and `POSTGRES_URL` environment variables.

4. **Auto-reload**: The backend uses `--reload` flag, so it will automatically restart when you make code changes.

5. **Access URLs**:
   - Development Frontend: http://localhost:3001
   - Development Backend API: http://localhost:3334
   - Production (unchanged): http://localhost:3333

## Using PM2 for Development (Optional)

If you prefer to use PM2 for development as well:

```powershell
cd backend
pm2 start ecosystem.dev.config.js
```

**Note**: Remember to update `OPENAI_API_KEY` in `ecosystem.dev.config.js` with your actual API key before using PM2.

## Stopping Development Servers

- If using the startup script: Close the PowerShell windows
- If using PM2: `pm2 stop BSClassifier-Dev`
- If running manually: Press `Ctrl+C` in each terminal

## Deploying Changes to Production

Once you've tested your changes in the development environment and are ready to deploy:

### Quick Deployment (Automated)

Simply run the deployment script from the project root:

```powershell
.\deploy.ps1
```

This script will:
1. Check and apply any pending database migrations
2. Build the frontend for production
3. Restart the PM2 production instance
4. Display the deployment status

### Manual Deployment

If you prefer to deploy manually:

1. **Apply database migrations** (if any):
   ```powershell
   cd backend
   .\venv\Scripts\Activate.ps1
   alembic upgrade head
   ```

2. **Build frontend**:
   ```powershell
   cd frontend
   npm run build
   ```

3. **Restart production**:
   ```powershell
   pm2 restart BSClassifier
   ```

### Post-Deployment

- Check production logs: `pm2 logs BSClassifier`
- Monitor production: `pm2 monit`
- Access production: http://localhost:3333

