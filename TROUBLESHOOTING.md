# BSClassifier Production Deployment Troubleshooting Guide

## Tool Overview

**BSClassifier** is a full-stack web application for processing and classifying bank statements. It uses machine learning and rule-based classification to automatically categorize transactions and match them with accounting ledgers.

### Architecture

- **Backend**: FastAPI (Python) with async SQLAlchemy
- **Frontend**: React (Create React App with CRACO)
- **Database**: PostgreSQL with asyncpg driver
- **Process Manager**: PM2
- **Deployment**: Windows Server environment

### Key Components

1. **Backend (`backend/server.py`)**
   - FastAPI application serving REST API endpoints
   - Serves static React build files in production mode
   - Uses environment variable `APP_ENV` to determine mode:
     - `APP_ENV=production`: Serves static files from `backend/build/`
     - `APP_ENV=development`: API-only mode (no static file serving)

2. **Frontend (`frontend/`)**
   - React application built with Create React App + CRACO
   - Build command: `npm run build` (outputs to `frontend/build/`)
   - Production build must be copied/moved to `backend/build/` for serving

3. **Database**
   - PostgreSQL database with Alembic migrations
   - Connection via `POSTGRES_URL` environment variable
   - Both dev and prod environments share the same database

## Development vs Production Setup

### Development Environment
- **Backend Port**: 3334
- **Frontend Port**: 3001 (React dev server)
- **Backend Command**: `uvicorn server:app --host 0.0.0.0 --port 3334 --reload`
- **Frontend Command**: `npm start` (with `PORT=3001`)
- **Environment**: `APP_ENV=development`
- **Auto-reload**: Enabled for both frontend and backend

### Production Environment
- **Port**: 3333 (single port, backend serves frontend)
- **Process Manager**: PM2
- **Backend Command**: `uvicorn server:app --host 0.0.0.0 --port 3333`
- **Environment**: `APP_ENV=production`
- **Static Files**: Served from `backend/build/` directory

## Current Issue

**Problem**: Code changes made during development are not appearing in the production environment, even after:
1. Building the frontend (`npm run build`)
2. Restarting PM2 instance
3. Verifying PM2 status shows "online"

**Symptoms**:
- Production instance at `http://localhost:3333` shows old code
- New features/changes visible in dev environment (port 3001/3334) but not in production
- PM2 logs show no errors, application appears to be running normally

## Deployment Configuration Files

### PM2 Production Config (`backend/ecosystem.config.js`)

```javascript
module.exports = {
  apps: [{
    name: 'BSClassifier',
    script: 'uvicorn',
    args: 'server:app --host 0.0.0.0 --port 3333',
    cwd: 'D:\\Custom Tools\\BSClassifier\\backend',
    interpreter: 'D:\\Custom Tools\\BSClassifier\\backend\\venv\\Scripts\\python.exe',
    env: {
      "APP_ENV": "production",
      "PYTHONPATH": ".",
      "OPENAI_API_KEY": "your_openai_api_key_here",
      "DATABASE_URL": "postgresql+asyncpg://bsclassifier_user:lHL1NkxI4RJGur5xOqAq@localhost:5432/bsclassifier_db",
      "POSTGRES_URL": "postgresql+asyncpg://bsclassifier_user:lHL1NkxI4RJGur5xOqAq@localhost:5432/bsclassifier_db",
      "CORS_ORIGINS": "http://localhost:3333,http://192168.79.10:3333",
    },
  }],
};
```

**Key Settings**:
- `cwd`: Sets working directory to backend folder (critical for finding `server.py`)
- `APP_ENV`: Set to "production" to enable static file serving
- `interpreter`: Points to Python virtual environment

### Static File Serving Logic (`backend/server.py`)

The application conditionally serves static files based on `APP_ENV`:

```python
# Lines 2073-2085 in server.py
if os.environ.get("APP_ENV") == "production":
    # Mount the static files directory for the built React app
    app.mount("/static", StaticFiles(directory="build/static"), name="static")

    @app.get("/{catchall:path}", response_class=FileResponse)
    def read_root(catchall: str):
        """
        Catch-all route to serve the React index.html file for any non-API, non-static path.
        This is essential for the React Router to handle client-side routing in production.
        """
        return "build/index.html"
```

**Important**: The path `"build/index.html"` and `"build/static"` are relative to the `cwd` (which is `backend/`), so files must be in `backend/build/`.

### Frontend Build Process

**Build Command**: `npm run build` (from `frontend/` directory)

**Build Script** (from `frontend/package.json`):
```json
"build": "ren .env .env.temp && craco build && ren .env.temp .env"
```

**Build Output Location**: `frontend/build/` (by default)

**Required Location for Production**: `backend/build/` (for FastAPI to serve)

## Deployment Workflow

### Current Deployment Script (`deploy.ps1`)

The deployment script performs these steps:

1. **Database Migrations**
   ```powershell
   cd backend
   alembic upgrade head
   ```

2. **Frontend Build**
   ```powershell
   cd frontend
   npm run build
   ```

3. **PM2 Restart**
   ```powershell
   pm2 restart BSClassifier
   ```

### Issue Identified

**Critical Problem**: The build process creates files in `frontend/build/`, but the production server expects them in `backend/build/`. 

**Current State**:
- Frontend builds to: `frontend/build/`
- Backend expects: `backend/build/`
- **Missing Step**: Copy/move build files from `frontend/build/` to `backend/build/`

## File Structure

```
BSClassifier/
├── backend/
│   ├── build/              # ← Production static files should be here
│   │   ├── index.html
│   │   └── static/
│   ├── server.py           # FastAPI app (serves from build/ when APP_ENV=production)
│   ├── ecosystem.config.js # PM2 production config
│   ├── ecosystem.dev.config.js
│   ├── venv/
│   └── ...
├── frontend/
│   ├── build/              # ← Build output goes here (needs to be copied to backend/build/)
│   │   ├── index.html
│   │   └── static/
│   ├── src/
│   ├── package.json
│   └── ...
└── deploy.ps1              # Deployment script
```

## Recent Changes Made

1. **Created Development Environment Setup**
   - `dev-start.ps1`: Script to start dev environment (ports 3001/3334)
   - `backend/ecosystem.dev.config.js`: PM2 config for dev (optional)
   - Updated `frontend/package.json`: Added PORT configuration

2. **Fixed PM2 Configuration**
   - Added `cwd` setting to `ecosystem.config.js` (fixes import errors)
   - Added `POSTGRES_URL` environment variable (required by `database.py`)

3. **Created Deployment Script**
   - `deploy.ps1`: Automated deployment script
   - Handles migrations, frontend build, and PM2 restart

## Troubleshooting Steps Taken

1. ✅ Verified PM2 is running (`pm2 status` shows "online")
2. ✅ Verified frontend build completes successfully
3. ✅ Verified `backend/build/` directory exists
4. ✅ Verified `APP_ENV=production` is set in PM2 config
5. ✅ Restarted PM2 multiple times
6. ❌ **NOT VERIFIED**: Whether `backend/build/` contains the latest build files

## Root Cause Hypothesis

The most likely issue is that:

1. **Build files are not being copied** from `frontend/build/` to `backend/build/`
2. **Old build files remain** in `backend/build/` from a previous deployment
3. **Browser cache** may be serving old JavaScript files (less likely if PM2 was restarted)

## Required Fix

The deployment script (`deploy.ps1`) needs to copy the build files from `frontend/build/` to `backend/build/` after building.

**Proposed Fix**:
```powershell
# After npm run build in frontend/
# Copy build files to backend/build/
Copy-Item -Path "frontend\build\*" -Destination "backend\build\" -Recurse -Force
```

Or update the frontend build script to output directly to `backend/build/`.

## Additional Context

### Environment Variables

**Required for Production**:
- `APP_ENV=production` (enables static file serving)
- `POSTGRES_URL` (database connection)
- `DATABASE_URL` (may be used by some components)
- `OPENAI_API_KEY` (if using OpenAI features)
- `CORS_ORIGINS` (allowed origins for CORS)

### Port Configuration

- **Production**: Single port 3333 (backend serves frontend)
- **Development**: 
  - Backend: 3334
  - Frontend: 3001 (React dev server)

### Database

- Both environments use the same PostgreSQL database
- Migrations are applied via Alembic
- Database URL: `postgresql+asyncpg://bsclassifier_user:lHL1NkxI4RJGur5xOqAq@localhost:5432/bsclassifier_db`

### PM2 Status

- PM2 daemon is running
- BSClassifier instance shows "online" status
- No errors in PM2 logs
- Application responds on port 3333

## Next Steps for Resolution

1. **Verify build file locations**:
   - Check if `frontend/build/` contains new files
   - Check if `backend/build/` contains old files
   - Compare file modification dates

2. **Fix deployment script**:
   - Add copy step to move build files from `frontend/build/` to `backend/build/`
   - Or modify build process to output directly to `backend/build/`

3. **Clear browser cache**:
   - Hard refresh (Ctrl+Shift+R) or clear cache
   - Check browser DevTools Network tab for file versions

4. **Verify static file serving**:
   - Check if `http://localhost:3333/static/js/main.*.js` serves the latest file
   - Compare file hashes/versions between `frontend/build/` and what's served

5. **Check file permissions**:
   - Ensure PM2 process can read files in `backend/build/`
   - Check if files are locked or read-only

## Files to Review

- `backend/server.py` (lines 2073-2085): Static file serving logic
- `backend/ecosystem.config.js`: PM2 production configuration
- `deploy.ps1`: Deployment script (missing build file copy step)
- `frontend/package.json`: Build script configuration
- `backend/build/`: Current production build files (may be outdated)
- `frontend/build/`: Latest build output (may not be in production location)

## Questions for Gemini

1. Should the build files be copied from `frontend/build/` to `backend/build/` during deployment?
2. Is there a better way to configure the build output location?
3. Could browser caching be causing old files to be served even after PM2 restart?
4. Are there any FastAPI/StaticFiles caching mechanisms that might serve old files?
5. Should we verify the file paths in `server.py` are correct relative to the `cwd`?

