import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Model Definition (ODE-free)
# ------------------------------
class StackedLTC(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        h = self.net(x)
        return self.head(h)

# ------------------------------
# Load Checkpoint Safely
# ------------------------------
def load_checkpoint(model_path):
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        input_dim = 300  # 60 * 5 (seq_len * features: Close, High, Low, Open, Volume)
        hidden_dims = [64, 64, 32]
        output_dim = 1
        model = StackedLTC(input_dim, hidden_dims, output_dim)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model loaded with ODE-free inference.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# ------------------------------
# Load and Prepare Data
# ------------------------------
def load_data(csv_path, seq_len=60):
    try:
        # Read CSV, skipping the second row (index 1) with 'TCS.NS'
        df = pd.read_csv(csv_path, skiprows=[1])

        # Verify and keep only numeric columns
        required_columns = ["Close", "High", "Low", "Open", "Volume"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV missing required columns: {required_columns}")

        df = df[required_columns].astype(float)

        X_seq, y_seq = [], []
        for i in range(len(df) - seq_len):
            X_seq.append(df.iloc[i:i+seq_len].values)
            y_seq.append(df["Close"].iloc[i+seq_len])  # Predict next close price

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)

        print(f"✅ Data loaded: {len(X_seq)} sequences, shape {X_seq.shape}")
        return X_seq, y_seq, df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# ------------------------------
# Backtest
# ------------------------------
def backtest(model, X_seq, y_seq, prices):
    model.eval()
    pnl = []
    position = 0
    entry_price = 0

    for i in range(len(X_seq)):
        x = torch.tensor(X_seq[i], dtype=torch.float32).flatten().unsqueeze(0)  # [1, seq_len*features]
        with torch.no_grad():
            pred = model(x).item()

        actual = y_seq[i]
        price = prices[i]

        if pred > price and position == 0:
            position = 1
            entry_price = price
        elif pred < price and position == 1:
            pnl.append(price - entry_price)
            position = 0

    total_pnl = sum(pnl)
    print(f"Total PnL: {total_pnl:.2f}")
    return pnl

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    # File paths (update these to your local paths)
    model_path = "C:\\Users\\ankit\\OneDrive\\Desktop\\Capstone\\trial_05\\lnn_final_model.pth"
    csv_path = "C:\\Users\\ankit\\OneDrive\\Desktop\\Capstone\\trial_05\\TCS_2020_present.csv"

    try:
        model = load_checkpoint(model_path)
        X_seq, y_seq, df = load_data(csv_path, seq_len=60)

        # Run backtest
        pnl = backtest(model, X_seq, y_seq, df["Close"].values[60:])

        # Plot cumulative PnL
        plt.plot(np.cumsum(pnl))
        plt.title("Backtest Cumulative PnL")
        plt.xlabel("Trades")
        plt.ylabel("Cumulative PnL")
        plt.show()
    except Exception as e:
        print(f"Main execution error: {e}")