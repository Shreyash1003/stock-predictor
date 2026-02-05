import pandas as pd

URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

df = pd.read_csv(URL)

symbols = df["SYMBOL"].dropna().unique()

with open("nse_symbols.txt","w") as f:
    for s in symbols:
        f.write(s.strip()+"\n")

print("Saved",len(symbols),"symbols")
