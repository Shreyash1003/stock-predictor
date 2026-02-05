"""
News Sentiment Analysis Service
Fetches stock-related news and performs sentiment analysis
"""

import requests
import json
import os
import time
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor
import threading

# --------------------------------
# CONFIG
# --------------------------------
# Free News API options:
# 1. Alpha Vantage (demo key available, 25 requests/day)
# 2. NewsAPI (requires API key, 100 requests/day for development)
# 3. Finnhub (500 requests/day free)

NEWS_CACHE_DURATION = 900  # 15 minutes
MAX_NEWS_ITEMS = 10

# --------------------------------
# SENTIMENT ANALYZER
# --------------------------------
analyzer = SentimentIntensityAnalyzer()

# --------------------------------
# CACHE
# --------------------------------
news_cache = {}
sentiment_cache = {}
cache_lock = threading.Lock()

# --------------------------------
# NEWS API FUNCTIONS
# --------------------------------

def get_alpha_vantage_news(symbol):
    """Fetch news from Alpha Vantage"""
    try:
        # API key - Get free key from https://www.alphavantage.co/support/#api-key
        API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', 'Y9ABJ3ZNQF6STZIV')
        
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'feed' in data:
            return data['feed'][:MAX_NEWS_ITEMS]
        
        # If no tickers field, might be searching by keyword instead
        if 'best_matches' not in data:
            return []
            
        return []
    except Exception as e:
        print(f"Alpha Vantage API error: {e}")
        return []

def get_finnhub_news(symbol):
    """Fetch news from Finnhub"""
    try:
        API_KEY = os.environ.get('FINNHUB_API_KEY', '')
        
        # Get company news
        end_date = int(time.time())
        start_date = end_date - (7 * 24 * 60 * 60)  # Last 7 days
        
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={datetime.fromtimestamp(start_date).strftime('%Y-%m-%d')}&to={datetime.fromtimestamp(end_date).strftime('%Y-%m-%d')}&token={API_KEY}"
        
        if not API_KEY:
            return []
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if isinstance(data, list):
            return data[:MAX_NEWS_ITEMS]
        return []
    except Exception as e:
        print(f"Finnhub API error: {e}")
        return []

def get_news_from_web_search(symbol):
    """Fallback: Get news by searching web (simplified)"""
    try:
        # This is a simplified fallback - in production use proper news API
        return []
    except Exception as e:
        print(f"Web search error: {e}")
        return []

def get_stock_news(symbol):
    """Get news for a stock symbol"""
    global news_cache
    
    # Check cache
    current_time = time.time()
    if symbol in news_cache:
        cached_data, timestamp = news_cache[symbol]
        if current_time - timestamp < NEWS_CACHE_DURATION:
            return cached_data
    
    # Try multiple sources
    news_items = []
    
    # Try Alpha Vantage first
    if symbol in ["NIFTY", "BANKNIFTY", "SENSEX"]:
        # For indices, use broader search
        news_items = get_alpha_vantage_news("INDEX:" + symbol)
    else:
        news_items = get_alpha_vantage_news(symbol)
    
    # Update cache
    with cache_lock:
        news_cache[symbol] = (news_items, current_time)
    
    return news_items

# --------------------------------
# SENTIMENT ANALYSIS
# --------------------------------

def analyze_sentiment(text):
    """Analyze sentiment of text using VADER"""
    try:
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        
        # Classify sentiment
        if compound >= 0.05:
            return 'positive', compound
        elif compound <= -0.05:
            return 'negative', compound
        else:
            return 'neutral', compound
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return 'neutral', 0.0

def get_news_sentiment(symbol):
    """Get aggregated sentiment for a stock"""
    global sentiment_cache
    
    current_time = time.time()
    
    # Check cache
    if symbol in sentiment_cache:
        cached_data, timestamp = sentiment_cache[symbol]
        if current_time - timestamp < NEWS_CACHE_DURATION:
            return cached_data
    
    # Get news
    news_items = get_stock_news(symbol)
    
    if not news_items:
        # Return neutral sentiment if no news found
        result = {
            'sentiment': 'neutral',
            'score': 0.0,
            'confidence': 'low',
            'headlines': [],
            'source': 'none'
        }
        with cache_lock:
            sentiment_cache[symbol] = (result, current_time)
        return result
    
    # Analyze sentiment of each headline
    sentiments = []
    headlines = []
    
    for item in news_items:
        # Get headline/title
        if 'title' in item:
            title = item['title']
        elif 'headline' in item:
            title = item['headline']
        else:
            continue
        
        sentiment, score = analyze_sentiment(title)
        sentiments.append(score)
        headlines.append({
            'title': title,
            'sentiment': sentiment,
            'score': score,
            'source': item.get('source', 'Unknown'),
            'url': item.get('url', ''),
            'datetime': item.get('time_published', item.get('datetime', ''))
        })
    
    if not sentiments:
        result = {
            'sentiment': 'neutral',
            'score': 0.0,
            'confidence': 'low',
            'headlines': [],
            'source': 'no_data'
        }
        with cache_lock:
            sentiment_cache[symbol] = (result, current_time)
        return result
    
    # Calculate aggregate sentiment
    avg_score = sum(sentiments) / len(sentiments)
    
    # Determine overall sentiment
    if avg_score >= 0.05:
        overall_sentiment = 'positive'
    elif avg_score <= -0.05:
        overall_sentiment = 'negative'
    else:
        overall_sentiment = 'neutral'
    
    # Determine confidence based on agreement
    positive_count = sum(1 for s in sentiments if s >= 0.05)
    negative_count = sum(1 for s in sentiments if s <= -0.05)
    max_agreement = max(positive_count, negative_count)
    agreement_ratio = max_agreement / len(sentiments)
    
    if agreement_ratio >= 0.7:
        confidence = 'high'
    elif agreement_ratio >= 0.5:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    result = {
        'sentiment': overall_sentiment,
        'score': round(avg_score, 3),
        'confidence': confidence,
        'headlines': headlines[:5],  # Top 5 headlines
        'article_count': len(news_items),
        'source': 'alpha_vantage'
    }
    
    # Update cache
    with cache_lock:
        sentiment_cache[symbol] = (result, current_time)
    
    return result

def get_sentiment_for_all_symbols(symbols):
    """Get sentiment for multiple symbols (parallel)"""
    results = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {
            executor.submit(get_news_sentiment, sym): sym 
            for sym in symbols[:50]  # Limit to 50 symbols
        }
        
        for future in future_to_symbol:
            symbol = future_to_symbol[future]
            try:
                results[symbol] = future.result()
            except Exception as e:
                print(f"Error getting sentiment for {symbol}: {e}")
                results[symbol] = {
                    'sentiment': 'neutral',
                    'score': 0.0,
                    'confidence': 'low',
                    'headlines': [],
                    'source': 'error'
                }
    
    return results

# --------------------------------
# HELPER FUNCTIONS
# --------------------------------

def adjust_prediction_with_sentiment(prob, sentiment_data):
    """
    Adjust prediction probability based on news sentiment
    
    Args:
        prob: Base prediction probability (0-1)
        sentiment_data: Sentiment analysis result
        
    Returns:
        Adjusted probability and adjustment factor
    """
    sentiment = sentiment_data['sentiment']
    score = sentiment_data['score']
    confidence = sentiment_data['confidence']
    
    # Base adjustment factor
    if confidence == 'high':
        adjustment_factor = 0.15
    elif confidence == 'medium':
        adjustment_factor = 0.08
    else:
        adjustment_factor = 0.0
    
    # Adjust probability based on sentiment
    if sentiment == 'positive':
        # Boost probability of upward movement
        adjusted_prob = prob + (score * adjustment_factor)
    elif sentiment == 'negative':
        # Reduce probability of upward movement
        adjusted_prob = prob + (score * adjustment_factor)
    else:
        # Neutral sentiment - no adjustment
        adjusted_prob = prob
    
    # Clamp to valid range
    adjusted_prob = max(0.0, min(1.0, adjusted_prob))
    
    # Calculate adjustment applied
    adjustment = adjusted_prob - prob
    
    return adjusted_prob, adjustment

# Test function
if __name__ == "__main__":
    # Test with a symbol
    test_symbol = "RELIANCE"
    print(f"Fetching news sentiment for {test_symbol}...")
    
    sentiment = get_news_sentiment(test_symbol)
    print(f"\nSentiment: {sentiment['sentiment']}")
    print(f"Score: {sentiment['score']}")
    print(f"Confidence: {sentiment['confidence']}")
    print(f"Articles: {sentiment['article_count']}")
    
    if sentiment['headlines']:
        print("\nTop Headlines:")
        for h in sentiment['headlines'][:3]:
            print(f"  [{h['sentiment'].upper()}] {h['title'][:60]}...")

