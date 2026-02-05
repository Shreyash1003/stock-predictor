import yfinance as yf
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier

WINDOW=20

def load_symbols():
    with open("nse_symbols.txt") as f:
        return [x.strip()+".NS" for x in f if x.strip()]

def features(w):
    w=np.array(w,dtype=float)
    r=np.diff(w)
    return [
        float(np.mean(r)),
        float(w[-1]-w[0]),
        float(np.std(w)),
        float(w[-1]-np.mean(w))
    ]

X=[]
y=[]

for s in load_symbols():
    d=yf.download(s,period="60d",interval="5m")
    if len(d)==0:
        continue

    p=d["Close"].dropna().values

    for i in range(WINDOW,len(p)-1):
        X.append(features(p[i-WINDOW:i]))
        y.append(1 if p[i+1]>p[i] else 0)

X=np.array(X)
y=np.array(y)

print("Samples:",len(X))

model=GradientBoostingClassifier()
model.fit(X,y)

joblib.dump(model,"model.pkl")
print("Model saved")
