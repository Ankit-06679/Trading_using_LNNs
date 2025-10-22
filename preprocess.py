# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def add_technical_indicators(df):
    # 1. SMA (window=14)
    df['SMA_14'] = df['Close'].rolling(window=14).mean()

    # 2. EMA (span=14)
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()

    # 3. RSI (14)
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 4. MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 5. Bollinger Bands (20-day SMA ± 2*std)
    df['BB_MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_MA20'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_MA20'] - 2 * df['BB_std']

    # 6. ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    # 7. Momentum (10-day)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

    # 8. OBV (On Balance Volume)
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # Drop intermediate columns for Bollinger Bands
    df = df.drop(columns=['BB_MA20', 'BB_std'])

    return df

def load_and_preprocess(filepath, seq_length=60):
    """
    Loads TCS OHLCV data, adds technical indicators, scales it,
    creates sequences, and saves as NumPy files.
    """
    # Load CSV, skip redundant second row
    df = pd.read_csv("C:\\Users\\ankit\\OneDrive\\Desktop\\Capstone\\trial_05\\TCS_2020_present.csv", skiprows=[1])

    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Keep OHLCV columns initially
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Add technical indicators
    df = add_technical_indicators(df)

    # Fill missing values
    df = df.fillna(method='ffill')

    # Optionally, log-scale volume
    df['Volume'] = np.log1p(df['Volume'])

    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Create sequences for model
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i + seq_length, 3])  # Predict Close price

    X = np.array(X)
    y = np.array(y)

    # Save preprocessed data
    np.save("X_preprocessed.npy", X)
    np.save("y_preprocessed.npy", y)
    print("Preprocessed data with extended technical indicators saved as NumPy files")

    return X, y, scaler

if __name__ == "__main__":
    filepath = "C:\\Users\\ankit\\OneDrive\\Desktop\\Capstone\\trial_05\\TCS_2020_present.csv"  # replace with your CSV path
    X, y, scaler = load_and_preprocess(filepath)
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
