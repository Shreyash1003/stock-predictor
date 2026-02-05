# TODO: Stock Predictor Performance Fixes

## Phase 1: Backend Optimizations (main.py) ✅ DONE
- [x] 1.1 Add model caching - train ONCE at startup, reuse for all predictions
- [x] 1.2 Extend cache duration from 30 seconds to 5 minutes
- [x] 1.3 Add market status detection (open/closed)
- [x] 1.4 Implement background prediction scheduler
- [x] 1.5 Add Redis-like fast in-memory cache
- [x] 1.6 Fix Sensex/NIFTY price extraction bug

## Phase 2: Frontend Improvements ✅ DONE
- [x] 2.1 Show cached/stale data immediately on load
- [x] 2.2 Add loading skeleton UI
- [x] 2.3 Display "Last Updated" timestamp
- [x] 2.4 Show market status indicator
- [x] 2.5 Add visual warning when using stale data
- [x] 2.6 Configurable API URL for deployment

## Phase 3: Netlify Deployment ✅ DONE
- [x] 3.1 Create _redirects file for SPA behavior
- [x] 3.2 Configure API_BASE variable

## Deployment Instructions

### Frontend (Netlify):
1. Go to https://app.netlify.com
2. Click "Add new site" → "Import an existing project"
3. Select your GitHub repo
4. Build settings:
   - Build command: (leave empty - plain HTML)
   - Publish directory: `frontend`
5. Click "Deploy site"

### Backend (Render/Railway/Heroku):
Recommended: **Render.com** (free tier available)

1. Go to https://render.com and sign up
2. Click "New" → "Web Service"
3. Connect your GitHub repo
4. Settings:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Create environment variables:
   - `PORT`: 8001
6. Click "Create Web Service"

After backend deploys, update frontend:
```javascript
// In frontend/index.html, change:
const API_BASE = "https://your-backend-name.onrender.com";  // Your deployed backend URL
```

## Project Structure
```
Stocks/
├── backend/
│   ├── main.py          # FastAPI backend
│   ├── model.pkl        # Trained ML model
│   └── ...
├── frontend/
│   ├── index.html       # Main UI
│   ├── _redirects      # Netlify SPA config
│   └── ...
└── requirements.txt      # Backend dependencies
```

