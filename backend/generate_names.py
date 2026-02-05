#!/usr/bin/env python3
"""
Generate symbol_name_map.json with company names from NSE data
"""
import pandas as pd
import json

URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

print("Fetching NSE data...")
df = pd.read_csv(URL)

# Read failed symbols
with open("failed_symbols.txt", "r") as f:
    failed_symbols = set()
    for line in f:
        if line.strip() and not line.startswith("#"):
            symbol = line.split("|")[0].strip()
            failed_symbols.add(symbol)

print(f"Failed symbols to exclude: {failed_symbols}")

# Create mapping: SYMBOL -> COMPANY NAME
symbol_name_map = {}

for _, row in df.iterrows():
    symbol = str(row.get('SYMBOL', '')).strip()
    company_name = str(row.get('NAME OF COMPANY', '')).strip()
    
    if symbol and company_name and symbol not in failed_symbols:
        # Clean up company name
        company_name = company_name.replace('"', "'")
        symbol_name_map[symbol] = company_name

# Save to JSON
with open("symbol_name_map.json", "w", encoding="utf-8") as f:
    json.dump(symbol_name_map, f, indent=2, ensure_ascii=False)

print(f"Saved {len(symbol_name_map)} symbol-to-name mappings")

# Also create a working_symbols.txt (just symbols, no failed ones)
working_symbols = [s for s in symbol_name_map.keys()]

with open("working_symbols.txt", "w") as f:
    for s in sorted(working_symbols):
        f.write(s + "\n")

print(f"Saved {len(working_symbols)} working symbols to working_symbols.txt")

# Print sample
print("\nSample entries:")
for i, (symbol, name) in enumerate(list(symbol_name_map.items())[:5]):
    print(f"  {symbol}: {name}")

