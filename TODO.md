# TODO: Advanced News Sentiment Integration

## Goal: Improve prediction accuracy by incorporating news sentiment analysis

### Step 1: Install Required Dependencies
- `newsapi-python` or use requests for Alpha Vantage
- `textblob` or `vaderSentiment` for sentiment analysis

### Step 2: Create News Sentiment Service
- [x] Create `news_sentiment.py` module
- [x] Integrate Alpha Vantage News API (free tier: 25 requests/day)
- [x] Implement sentiment analysis using VADER/TextBlob
- [x] Cache news data to reduce API calls

### Step 3: Update Prediction Model
- [x] Add sentiment score as additional feature
- [x] Retrain model with sentiment-weighted predictions
- [x] Adjust prediction confidence based on news sentiment

### Step 4: Update Backend API
- [x] Add `/news/{symbol}` endpoint to fetch news sentiment
- [x] Update `/predict` endpoint to include sentiment data
- [x] Add sentiment caching (update every 15 minutes)

### Step 5: Update Frontend UI
- [x] Add sentiment indicator to stock list (📈/📉/➡️)
- [x] Show news sentiment section in full report
- [x] Display recent news headlines
- [x] Add confidence adjustment indicator

### Step 6: Testing & Optimization
- [ ] Test with various stocks
- [ ] Monitor API usage limits
- [ ] Optimize caching strategy

---

# DEPLOYMENT CHECKLIST

## Phase 1: Backend Deployment (Render.com)
- [ ] 1.1 Create GitHub repository and push code
- [ ] 1.2 Sign up at https://render.com
- [ ] 1.3 Create Web Service:
      - Name: `stock-predictor-backend`
      - Root Directory: `backend`
      - Build Command: `pip install -r requirements.txt`
      - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- [ ] 1.4 Add Environment Variables (optional):
      - `ALPHA_VANTAGE_API_KEY`
      - `FINNHUB_API_KEY`
- [ ] 1.5 Get backend URL (e.g., `https://stock-predictor-backend.onrender.com`)
- [ ] 1.6 Test backend health: `https://your-render-url/health`

## Phase 2: Frontend Deployment (Netlify)
- [ ] 2.1 Update `frontend/script.js`:
      - Change `API_BASE` from `http://127.0.0.1:8001` to your Render URL
- [ ] 2.2 Commit and push changes
- [ ] 2.3 Sign up at https://netlify.com
- [ ] 2.4 Deploy frontend:
      - Base directory: `frontend`
      - Publish directory: `frontend`
- [ ] 2.5 Get frontend URL (e.g., `https://stock-predictor.netlify.app`)

## Phase 3: Configuration & Testing
- [ ] 3.1 Update CORS in `backend/main.py` to allow Netlify domain
- [ ] 3.2 Redeploy backend (push to GitHub for auto-redeploy)
- [ ] 3.3 Test full application:
      - [ ] Open Netlify URL
      - [ ] Check market status loads
      - [ ] Search for a stock (e.g., RELIANCE)
      - [ ] Verify predictions display
      - [ ] Check news sentiment shows
- [ ] 3.4 Configure custom domain (optional)

## Troubleshooting
- [ ] Check browser console for CORS errors
- [ ] Check Render logs for backend issues
- [ ] Verify `health` endpoint works
- [ ] Note: Free tier = backend sleeps after 15 min inactivity

## Estimated Time: 15-20 minutes
- Backend deploy: ~3-5 minutes
- Frontend deploy: ~1-2 minutes
- Configuration & testing: ~10 minutes

