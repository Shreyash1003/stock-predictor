# Deployment Plan: Stock Predictor (Full Stack)

## Overview
Deploy backend to **Render.com** and frontend to **Netlify**

## Prerequisites
- GitHub account
- Git installed
- GitHub repository with your code

---

## Part 1: Deploy Backend to Render.com

### Step 1.1: Create GitHub Repository (if not already created)
```bash
cd /Users/shreyashwaghmare/Desktop/Coding/Python/Stocks
git init
git add .
git commit -m "Initial commit - Stock Predictor"
# Create repo on GitHub and push
```

### Step 1.2: Create Render Web Service

1. Go to https://render.com and sign up (free tier)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `stock-predictor-backend`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free ($0/month)

5. Add Environment Variables:
   - `ALPHA_VANTAGE_API_KEY` (optional - get free key from alphavantage.co)
   - `FINNHUB_API_KEY` (optional - get free key from finnhub.io)

6. Click "Create Web Service"

### Step 1.3: Get Backend URL
After deployment (2-5 minutes), Render will give you a URL like:
```
https://stock-predictor-backend.onrender.com
```

---

## Part 2: Deploy Frontend to Netlify

### Step 2.1: Update Frontend API Configuration

Edit `frontend/script.js` and change:
```javascript
// OLD (localhost)
const API_BASE = "http://127.0.0.1:8001";

// NEW (Render backend URL)
const API_BASE = "https://stock-predictor-backend.onrender.com";
```

### Step 2.2: Commit and Push Changes
```bash
git add frontend/script.js
git commit -m "Update API URL for production"
git push
```

### Step 2.3: Deploy to Netlify

**Option A: Via Netlify Dashboard**
1. Go to https://netlify.com and sign up
2. Click "Add new site" → "Import an existing project"
3. Connect GitHub and select your repository
4. Configure:
   - **Base directory**: `frontend`
   - **Build command**: (leave empty)
   - **Publish directory**: `frontend`
5. Click "Deploy site"

**Option B: Via Netlify CLI**
```bash
cd frontend
npm install netlify-cli -g
netlify login
netlify deploy --prod --dir=.
```

### Step 2.4: Get Frontend URL
Netlify will give you a URL like:
```
https://stock-predictor.netlify.app
```

---

## Part 3: Configure CORS on Backend (Render)

The backend is already configured for CORS with `allow_origins=["*"]`, but for security, you can update it to only allow your Netlify domain:

In `backend/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-netlify-site.netlify.app",
        "http://localhost:3000"  # For local testing
    ],
    allow_methods=["*"],
    allow_headers=["*"]
)
```

After updating, commit and push - Render will auto-redeploy.

---

## Part 4: Testing

1. Open your Netlify URL: `https://stock-predictor.netlify.app`
2. The app should connect to Render backend
3. Market status and predictions should load
4. Stock search should work

If issues:
- Check browser console (F12) for errors
- Verify backend health: `https://stock-predictor-backend.onrender.com/health`
- Check backend logs in Render dashboard

---

## Estimated Costs
- **Render**: Free tier (750 hours/month)
- **Netlify**: Free tier (100GB bandwidth/month)

---

## Important Notes

1. **Free Tier Limits**:
   - Render: Backend goes to sleep after 15 min of inactivity, takes ~30 sec to wake
   - First request after sleep may timeout - refresh page

2. **API Rate Limits**:
   - yfinance: Rate limited by Yahoo
   - Alpha Vantage: 25 requests/day (free)
   - Finnhub: 500 requests/day (free)

3. **Cold Start**:
   - First load may be slow (30-60 sec) while backend wakes up
   - Subsequent requests are fast

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "CORS error" in console | Update `allow_origins` in main.py and redeploy |
| "Backend not responding" | Check Render logs, ensure service is running |
| No predictions showing | Check browser network tab for failed requests |
| Market closed message | Expected behavior - backend uses stale data |

---

## Next Steps (Optional)

1. **Custom Domain**: Add custom domain in Netlify/Render settings
2. **SSL**: Both services provide free SSL automatically
3. **Monitoring**: Set up uptime monitoring with https://healthchecks.io
4. **Alerts**: Configure alerts for service downtime

