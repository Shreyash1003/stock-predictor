from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
import threading
import time
import json
import os
import asyncio
import concurrent.futures
from datetime import datetime, timedelta
import pytz

# Import news sentiment module
try:
    from news_sentiment import get_news_sentiment, adjust_prediction_with_sentiment, get_sentiment_for_all_symbols
    NEWS_SENTIMENT_ENABLED = True
    print("News sentiment module loaded successfully")
except ImportError as e:
    NEWS_SENTIMENT_ENABLED = False
    print(f"News sentiment module not available: {e}")

# --------------------------------
# APP
# --------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --------------------------------
# CONFIG
# --------------------------------
CACHE_DURATION = 300  # 5 minutes instead of 30 seconds
PREDICTION_BATCH_SIZE = 100  # Max stocks per batch
MAX_WORKERS = 30  # Increased for faster parallel fetching

# --------------------------------
# LOAD BASE MODEL
# --------------------------------
try:
    base_model = joblib.load("model.pkl")
    print("Base model loaded successfully")
except Exception as e:
    base_model = None
    print(f"No base model found or error loading: {e}")

# --------------------------------
# STORAGE
# --------------------------------
active_alerts = []
# In-memory prediction cache (simulating Redis)
prediction_cache = {}
cache_timestamp = 0
market_status_cache = {"is_open": False, "last_checked": None}
scheduler_lock = threading.Lock()

# --------------------------------
# MARKET STATUS
# --------------------------------
def is_market_open():
    """Check if NSE market is currently open (IST timezone)"""
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        
        # NSE market hours: 9:15 AM to 3:30 PM IST, Mon-Fri
        if now.weekday() > 5:  # Saturday or Sunday
            return False
        
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # Also check if it's a holiday (simplified - you can add more holidays)
        # Example holidays can be added here
        
        return market_start <= now <= market_end
    except Exception as e:
        print(f"Error checking market status: {e}")
        return False

def get_market_status():
    """Get cached market status"""
    global market_status_cache
    
    # Check if cache is still valid (5 minutes)
    if market_status_cache["last_checked"]:
        last_checked = market_status_cache["last_checked"]
        if (datetime.now() - last_checked).total_seconds() < 300:
            return market_status_cache
    
    # Fresh check
    is_open = is_market_open()
    market_status_cache = {
        "is_open": is_open,
        "last_checked": datetime.now(),
        "next_update": datetime.now() + timedelta(minutes=5)
    }
    
    return market_status_cache

# --------------------------------
# PRICE FETCH (Optimized)
# --------------------------------
def get_prices(symbol, interval="5m"):
    symbol = symbol.upper()

    if symbol in ["NIFTY", "BANKNIFTY", "SENSEX"]:
        symbol_map = {
            "NIFTY": "^NSEI",
            "BANKNIFTY": "^NSEBANK",
            "SENSEX": "^BSESN"
        }
        yf_symbol = symbol_map[symbol]
    elif not symbol.endswith(".NS"):
        yf_symbol = symbol + ".NS"
    else:
        yf_symbol = symbol

    try:
        # Use appropriate interval based on market status
        market_status = get_market_status()
        
        if market_status["is_open"]:
            # During market hours: use intraday data
            data = yf.download(yf_symbol, period="1d", interval="5m", 
                             progress=False, timeout=5)
        else:
            # After market hours: use daily data
            data = yf.download(yf_symbol, period="5d", interval="1d", 
                             progress=False, timeout=5)
        
        if data.empty:
            return {"prices": [], "type": "none", "is_stale": True}
        
        # Debug: print column structure for first fetch
        if symbol in ["SENSEX", "NIFTY", "BANKNIFTY"]:
            print(f"DEBUG {symbol}: columns={data.columns.tolist()[:3]}...")
        
        # Extract close prices - handle both single-level and multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-level columns: data['Close']['^BSESN'] or data[('Close', '^BSESN')]
            # The correct way is to access as data['Close']['TICKER'] 
            # or data[('Close', 'TICKER')] which returns a Series
            close_col = None
            
            # Try to get the Close column properly
            try:
                # Check if columns have the expected tuple structure
                first_col = data.columns[0]
                if isinstance(first_col, tuple) and len(first_col) >= 2:
                    # MultiIndex like ('Close', '^BSESN')
                    ticker = first_col[1]
                    close_col = data['Close'][ticker]
                else:
                    close_col = data['Close']
            except:
                close_col = data.iloc[:, 0]
        else:
            # Single-level columns
            if "Close" in data.columns:
                close_col = data["Close"]
            else:
                close_col = data.iloc[:, 0]
        
        prices = np.array(close_col).flatten().tolist()
        prices = [float(p) for p in prices if p and not np.isnan(p)]
        
        if len(prices) < 5:
            return {"prices": [], "type": "none", "is_stale": True}
        
        # Debug: print first and last price
        if symbol in ["SENSEX", "NIFTY", "BANKNIFTY"]:
            print(f"DEBUG {symbol}: first_price={prices[0] if prices else 'N/A'}, last_price={prices[-1] if prices else 'N/A'}")
        
        data_type = "intraday" if market_status["is_open"] else "daily"
        
        return {
            "prices": prices[-60:], 
            "type": data_type, 
            "is_stale": not market_status["is_open"]
        }
        
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return {"prices": [], "type": "none", "is_stale": True}

# --------------------------------
# CACHED MODEL
# --------------------------------
_live_model = None
_live_model_lock = threading.Lock()

def get_live_model():
    """Get or create cached live model"""
    global _live_model
    
    if _live_model is not None:
        return _live_model
    
    with _live_model_lock:
        # Double-check after acquiring lock
        if _live_model is not None:
            return _live_model
        
        # Create a fresh model (will be trained on first use)
        _live_model = LogisticRegression(max_iter=300)
        return _live_model

def reset_live_model():
    """Reset the live model (call daily or when needed)"""
    global _live_model
    _live_model = None

# --------------------------------
# FEATURES
# --------------------------------
def extract_features(w):
    w = np.array(w)
    r = np.diff(w)
    
    # Handle edge case where r might be empty
    if len(r) == 0:
        r = np.array([0])
    
    return [
        float(np.mean(r)) if len(r) > 0 else 0,
        float(w[-1] - w[0]),
        float(np.std(w)) if len(w) > 0 else 0,
        float(w[-1] - np.mean(w)) if len(w) > 0 else 0
    ]

# --------------------------------
# HYBRID MODEL (Optimized - trains once)
# --------------------------------
def hybrid_predict(prices, model=None):
    """Predict using hybrid model with optional pre-trained model"""
    
    if len(prices) < 30:
        return 0.5

    # Use provided model or get cached one
    if model is not None:
        # Use pre-trained model for fast prediction
        f = extract_features(prices[-20:])
        return model.predict_proba([f])[0][1]
    
    # Fallback: train if no model provided
    X, y = [], []
    
    for i in range(20, len(prices)-1):
        X.append(extract_features(prices[i-20:i]))
        y.append(1 if prices[i+1] > prices[i] else 0)
    
    live = LogisticRegression(max_iter=300)
    live.fit(X, y)
    
    f = extract_features(prices[-20:])
    p_live = live.predict_proba([f])[0][1]
    
    if base_model:
        p_base = base_model.predict_proba([f])[0][1]
    else:
        p_base = 0.5
    
    return (p_live * 0.6) + (p_base * 0.4)

# --------------------------------
# PREDICT SINGLE STOCK
# --------------------------------
@app.get("/predict")
def predict(symbol: str, interval: str = "5m"):
    """Get prediction for a single stock with news sentiment analysis"""
    
    data = get_prices(symbol, interval)
    prices = data["prices"]
    data_type = data.get("type", "none")
    is_stale = data.get("is_stale", False)

    if len(prices) < 5:
        return {
            "error": "Not enough market data", 
            "suggestion": "Market may be closed or data unavailable",
            "is_stale": True
        }

    current = prices[-1]
    prob = hybrid_predict(prices)
    
    # Get live model for predictions
    model = get_live_model()
    
    # Retrain model with recent data for better predictions
    X, y = [], []
    for i in range(20, len(prices)-1):
        X.append(extract_features(prices[i-20:i]))
        y.append(1 if prices[i+1] > prices[i] else 0)
    
    if len(X) > 10:  # Only retrain if we have enough data
        try:
            model.fit(X, y)
        except Exception as e:
            print(f"Error retraining model: {e}")
    
    # Get fresh prediction
    f = extract_features(prices[-20:])
    p_live = model.predict_proba([f])[0][1]
    
    # Combine with base model
    if base_model:
        p_base = base_model.predict_proba([f])[0][1]
    else:
        p_base = 0.5
    
    prob = (p_live * 0.6) + (p_base * 0.4)
    
    # Get news sentiment and adjust prediction
    sentiment_data = {
        'sentiment': 'neutral',
        'score': 0.0,
        'confidence': 'low',
        'headlines': [],
        'source': 'disabled'
    }
    
    if NEWS_SENTIMENT_ENABLED:
        try:
            sentiment_data = get_news_sentiment(symbol)
            # Adjust probability based on sentiment
            prob, adjustment = adjust_prediction_with_sentiment(prob, sentiment_data)
        except Exception as e:
            print(f"Error getting sentiment for {symbol}: {e}")

    # Adjust prediction based on data type
    if data_type == "intraday":
        move_pct = (0.01 + prob * 0.04)
        label = "Intraday"
    else:
        # For next-day prediction, reduce confidence slightly
        move_pct = (0.02 + prob * 0.06)
        label = "Next Day Prediction (Stale Data)" if is_stale else "Next Day"

    expected = current + (current * move_pct) if prob > 0.5 else current - (current * move_pct)
    direction = "UP" if expected > current else "DOWN"
    confidence = round(min(abs(expected-current)/current*100, 99), 2)
    
    high = expected * 1.01
    low = expected * 0.99
    safe = (expected + current) / 2

    return {
        "symbol": symbol,
        "current_price": round(current, 2),
        "direction": direction,
        "confidence": confidence,
        "expected_price": round(expected, 2),
        "high_target": round(high, 2),
        "low_target": round(low, 2),
        "safe_price": round(safe, 2),
        "prices": prices[-60:],
        "prediction_type": label,
        "is_stale": is_stale,
        "market_status": get_market_status(),
        "sentiment": sentiment_data
    }

# --------------------------------
# PREDICT ALL (Cached)
# --------------------------------
@app.get("/predict/all")
def predict_all():
    """Get predictions for all stocks (cached for 5 minutes)"""
    global prediction_cache, cache_timestamp
    
    current_time = time.time()
    market_status = get_market_status()
    
    # Return cached predictions if still valid and market is closed
    # (Don't refresh cache if market is closed to save API calls)
    if prediction_cache and (current_time - cache_timestamp) < CACHE_DURATION:
        return {
            "predictions": prediction_cache,
            "cached": True,
            "seconds_left": int(CACHE_DURATION - (current_time - cache_timestamp)),
            "market_status": market_status,
            "last_updated": datetime.fromtimestamp(cache_timestamp).isoformat()
        }
    
    # If market is closed, use cached data even if expired
    if not market_status["is_open"] and prediction_cache:
        return {
            "predictions": prediction_cache,
            "cached": True,
            "seconds_left": 0,
            "market_status": market_status,
            "last_updated": datetime.fromtimestamp(cache_timestamp).isoformat(),
            "warning": "Using cached data from market close"
        }
    
    # Fetch new predictions
    results = []
    symbols_to_fetch = WORKING_SYMBOLS[:PREDICTION_BATCH_SIZE]
    
    # Use ThreadPoolExecutor for concurrent fetching
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {
            executor.submit(get_prediction_for_symbol, sym): sym 
            for sym in symbols_to_fetch
        }
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if result:
                results.append(result)
    
    # Add index symbols
    index_results = []
    for idx_sym, idx_name in [("NIFTY", "NIFTY 50"), ("BANKNIFTY", "NIFTY Bank"), ("SENSEX", "BSE Sensex")]:
        try:
            result = get_prediction_for_symbol(idx_sym)
            if result:
                index_results.append(result)
        except Exception as e:
            print(f"Error predicting {idx_sym}: {e}")
            continue
    
    # Combine: index symbols first, then stocks
    final_results = index_results + results
    
    # Update cache
    prediction_cache = final_results
    cache_timestamp = current_time
    
    return {
        "predictions": final_results,
        "cached": False,
        "seconds_left": CACHE_DURATION,
        "market_status": market_status,
        "last_updated": datetime.fromtimestamp(cache_timestamp).isoformat()
    }

def get_prediction_for_symbol(symbol):
    """Get prediction for a single symbol (thread-safe) with sentiment"""
    try:
        data = get_prices(symbol, "5m")
        prices = data["prices"]
        
        if len(prices) < 5:
            return None
        
        current = prices[-1]
        prob = hybrid_predict(prices)
        
        # Get sentiment and adjust probability
        sentiment_info = {
            'sentiment': 'neutral',
            'score': 0.0,
            'confidence': 'low'
        }
        
        if NEWS_SENTIMENT_ENABLED:
            try:
                sentiment_data = get_news_sentiment(symbol)
                prob, _ = adjust_prediction_with_sentiment(prob, sentiment_data)
                sentiment_info = {
                    'sentiment': sentiment_data.get('sentiment', 'neutral'),
                    'score': sentiment_data.get('score', 0.0),
                    'confidence': sentiment_data.get('confidence', 'low')
                }
            except Exception as e:
                pass
        
        # Adjust move percentage based on data type
        data_type = data.get("type", "daily")
        if data_type == "intraday":
            move_pct = (0.01 + prob * 0.04)
        else:
            move_pct = (0.02 + prob * 0.06)
        
        expected = current + (current * move_pct) if prob > 0.5 else current - (current * move_pct)
        direction = "UP" if expected > current else "DOWN"
        
        return {
            "symbol": symbol,
            "name": SYMBOL_NAME_MAP.get(symbol, symbol),
            "current_price": round(current, 2),
            "direction": direction,
            "expected_price": round(expected, 2),
            "is_stale": data.get("is_stale", False),
            "sentiment": sentiment_info
        }
    except Exception as e:
        print(f"Error predicting {symbol}: {e}")
        return None

# --------------------------------
# INVEST & ALERTS
# --------------------------------
@app.post("/invest")
def invest(data: dict):
    data["triggered"] = False
    active_alerts.append(data)
    return {"status": "saved"}

@app.get("/alerts")
def alerts():
    return [a for a in active_alerts if a["triggered"]]

# --------------------------------
# ALERT WATCHER THREAD
# --------------------------------
def watcher():
    while True:
        for a in active_alerts:
            prices = get_prices(a["symbol"])
            if len(prices) == 0:
                continue
            
            p = prices[-1]
            
            if a["direction"] == "UP" and p < a["safe"]:
                a["triggered"] = True
                a["message"] = "Price dropped below SAFE level"
            
            if a["direction"] == "DOWN" and p > a["safe"]:
                a["triggered"] = True
                a["message"] = "Price crossed SAFE level"
        
        time.sleep(30)

threading.Thread(target=watcher, daemon=True).start()

# --------------------------------
# BACKGROUND SCHEDULER
# --------------------------------
def scheduler():
    """Background scheduler to refresh predictions periodically"""
    while True:
        try:
            # Refresh every 5 minutes during market hours
            market_status = get_market_status()
            if market_status["is_open"]:
                # Trigger prediction refresh in background
                try:
                    predict_all()
                    print(f"Scheduled prediction refresh completed at {datetime.now()}")
                except Exception as e:
                    print(f"Scheduled refresh failed: {e}")
            
            # Sleep for 5 minutes
            for _ in range(300):  # 5 minutes = 300 seconds
                time.sleep(1)
                # Check market status periodically
                if get_market_status()["is_open"] == False:
                    break
                    
        except Exception as e:
            print(f"Scheduler error: {e}")
            time.sleep(60)

# Start scheduler in background
threading.Thread(target=scheduler, daemon=True, name="PredictionScheduler").start()

# --------------------------------
# SYMBOL LIST
# --------------------------------
SYMBOL_NAME_MAP = {}
WORKING_SYMBOLS = []

def load_symbol_names():
    global SYMBOL_NAME_MAP, WORKING_SYMBOLS
    
    map_file = "symbol_name_map.json"
    if os.path.exists(map_file):
        with open(map_file, "r", encoding="utf-8") as f:
            SYMBOL_NAME_MAP = json.load(f)
        WORKING_SYMBOLS = list(SYMBOL_NAME_MAP.keys())
        print(f"Loaded {len(WORKING_SYMBOLS)} symbol names")

# Load on startup
load_symbol_names()

@app.get("/symbols")
def symbols():
    """Return list of symbols with company names"""
    index_symbols = [
        {"symbol": "NIFTY", "name": "NIFTY 50 Index"},
        {"symbol": "BANKNIFTY", "name": "NIFTY Bank Index"},
        {"symbol": "SENSEX", "name": "BSE Sensex Index"}
    ]
    
    stock_symbols = [
        {"symbol": s, "name": SYMBOL_NAME_MAP.get(s, s)}
        for s in WORKING_SYMBOLS
    ]
    
    return index_symbols + stock_symbols

@app.get("/symbols/simple")
def symbols_simple():
    """Return simple list of symbols"""
    return WORKING_SYMBOLS

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "market_status": get_market_status(),
        "predictions_cached": len(prediction_cache),
        "cache_age_seconds": int(time.time() - cache_timestamp) if cache_timestamp > 0 else None
    }

@app.get("/market-status")
def market_status():
    """Get current market status"""
    return get_market_status()

@app.get("/")
def root():
    """Root endpoint to verify server is running"""
    return {
        "status": "ok",
        "message": "AI Stock Predictor Backend is running",
        "version": "2.0.0",
        "news_sentiment_enabled": NEWS_SENTIMENT_ENABLED,
        "endpoints": {
            "market_status": "/market-status",
            "predict": "/predict?symbol=RELIANCE",
            "predict_all": "/predict/all",
            "symbols": "/symbols",
            "alerts": "/alerts",
            "invest": "/invest",
            "health": "/health",
            "news": "/news?symbol=RELIANCE"
        }
    }

@app.get("/news")
def news_sentiment_endpoint(symbol: str):
    """Get news sentiment for a stock"""
    if not NEWS_SENTIMENT_ENABLED:
        return {
            "error": "News sentiment is not enabled",
            "message": "Install required dependencies: pip install vaderSentiment requests"
        }
    
    try:
        sentiment_data = get_news_sentiment(symbol)
        return {
            "symbol": symbol,
            "sentiment": sentiment_data
        }
    except Exception as e:
        return {"error": str(e)}

# Run with: uvicorn main:app --host 0.0.0.0 --port 8001
