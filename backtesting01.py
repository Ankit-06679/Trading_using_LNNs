import backtrader as bt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import joblib
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
# Load Checkpoint + Scaler
# ------------------------------
def load_checkpoint_and_scaler(model_path, scaler_path):
    print(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    try:
        state_dict = torch.load(model_path, map_location="cpu")
        input_dim = 300  # 60 * 5
        hidden_dims = [64, 64, 32]
        output_dim = 1
        model = StackedLTC(input_dim, hidden_dims, output_dim)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        scaler = joblib.load(scaler_path)

        print("✅ Model and Scaler loaded.")
        return model, scaler
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        raise


# ------------------------------
# Custom Strategy
# ------------------------------
class MLStrategy(bt.Strategy):
    params = (
        ("seq_len", 60),
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low
        self.data_open = self.datas[0].open
        self.data_volume = self.datas[0].volume

        model_path = "C:\\Users\\ankit\\OneDrive\\Desktop\\Capstone\\trial_05\\lnn_final_model.pth"
        scaler_path = "C:\\Users\\ankit\\OneDrive\\Desktop\\Capstone\\trial_05\\scaler.pkl"

        self.model, self.scaler = load_checkpoint_and_scaler(model_path, scaler_path)

        self.seq_data = []
        self.next_index = self.params.seq_len
        self.trade_count = 0
        self.pnl = []
        self.y_true = []
        self.y_pred = []
        self.entry_price = None

    def next(self):
        if len(self.data) > self.next_index:
            close_seq = self.data_close.get(size=self.params.seq_len)
            high_seq = self.data_high.get(size=self.params.seq_len)
            low_seq = self.data_low.get(size=self.params.seq_len)
            open_seq = self.data_open.get(size=self.params.seq_len)
            volume_seq = self.data_volume.get(size=self.params.seq_len)

            if len(close_seq) == self.params.seq_len:
                df_seq = pd.DataFrame({
                    "Close": close_seq,
                    "High": high_seq,
                    "Low": low_seq,
                    "Open": open_seq,
                    "Volume": volume_seq
                })
                scaled = self.scaler.transform(df_seq)
                x = torch.tensor(scaled.flatten(), dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    pred_normalized = self.model(x).item()

                pred_price = self.scaler.inverse_transform(
                    [[pred_normalized, pred_normalized, pred_normalized,
                      pred_normalized, pred_normalized]]
                )[0, 0]

                current_price = self.data_close[0]

                self.y_true.append(current_price)
                self.y_pred.append(pred_price)

                print(f"Step {len(self.data)-1}: pred={pred_price:.2f}, actual={current_price:.2f}, position={self.position.size}")

                # --- Trading logic ---
                if pred_price > current_price * 1.002 and not self.position:
                    self.buy()
                    self.trade_count += 1
                    self.entry_price = current_price
                    print(f"  BUY at {self.entry_price:.2f}")

                elif pred_price < current_price * 0.998 and self.position:
                    self.sell()
                    profit = current_price - self.entry_price
                    self.pnl.append(profit)
                    self.trade_count += 1
                    print(f"  SELL at {current_price:.2f}, profit={profit:.2f}")

            self.next_index += 1

    def stop(self):
        total_pnl = sum(self.pnl) if self.pnl else 0
        win_trades = [p for p in self.pnl if p > 0]
        win_ratio = len(win_trades) / len(self.pnl) if self.pnl else 0

        returns = pd.Series(self.pnl)
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 else 0

        correct = sum(np.sign(np.diff(self.y_true)) == np.sign(np.diff(self.y_pred)))
        accuracy = correct / (len(self.y_true) - 1) if len(self.y_true) > 1 else 0

        print("\n📊 Backtest Summary")
        print(f"Total PnL: {total_pnl:.2f}")
        print(f"Number of trades: {self.trade_count}")
        print(f"Win Ratio: {win_ratio:.2%}")
        print(f"Directional Accuracy: {accuracy:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Final Portfolio Value: {self.broker.getvalue():.2f}")

        # --- Plot predictions vs actual ---
        plt.figure(figsize=(12, 6))
        plt.plot(self.y_true, label="Actual Price")
        plt.plot(self.y_pred, label="Predicted Price")
        plt.title("Actual vs Predicted Prices")
        plt.legend()
        plt.show()


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    csv_path = "C:\\Users\\ankit\\OneDrive\\Desktop\\Capstone\\trial_05\\TCS_2020_present.csv"
    df = pd.read_csv(csv_path, skiprows=[1])
    df["datetime"] = pd.date_range(start="2020-01-01", periods=len(df), freq="B")
    df = df[["datetime", "Open", "High", "Low", "Close", "Volume"]].copy()

    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(
        dataname=df,
        datetime='datetime', open='Open', high='High', low='Low', close='Close', volume='Volume'
    )
    cerebro.adddata(data)
    cerebro.addstrategy(MLStrategy)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
