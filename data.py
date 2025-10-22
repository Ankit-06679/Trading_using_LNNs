import yfinance as yf
import pandas as pd

# Fetch TCS stock data from 2020 till now
tcs = yf.download("TCS.NS", start="2020-01-01", end="2025-09-23", interval="1d")

# Reset index for easier handling
tcs.reset_index(inplace=True)
print(tcs.head())

# Save for later use
tcs.to_csv("TCS_2020_present.csv", index=False)
