import torch
import torch.nn as nn
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import time

# ------------------------------
# Model Definition (same as before)
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
# Load Model
# ------------------------------
def load_checkpoint(model_path):
    state_dict = torch.load(model_path, map_location="cpu")
    input_dim = 300  # 60 * 5 (OHLCV)
    hidden_dims = [64, 64, 32]
    output_dim = 1
    model = StackedLTC(input_dim, hidden_dims, output_dim)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# ------------------------------
# Send Webhook to Tradetron
# ------------------------------
WEBHOOK_URL = "https://api.tradetron.tech/api?auth-token=ca3eabf0-7a4e-4f42-9cfc-046533a1ca17&key=api_buy&value=1"  # Replace with your Tradetron webhook

def send_signal(symbol, action, qty):
    payload = {
        "symbol": symbol,
        "action": action,
        "quantity": qty
    }
    r = requests.post(WEBHOOK_URL, json=payload)
    print(f"📤 Signal sent: {payload} | Response: {r.status_code}")

# ------------------------------
# Trading Loop
# ------------------------------
if __name__ == "__main__":
    model_path = r"C:\Users\ankit\OneDrive\Desktop\Capstone\trial_05\lnn_final_model.pth"
    csv_path = r"C:\Users\ankit\OneDrive\Desktop\Capstone\trial_05\TCS_2020_present.csv"

    model = load_checkpoint(model_path)
    scaler = MinMaxScaler()

    # Load recent data (you’ll need live feed for production)
    df = pd.read_csv(csv_path, skiprows=[1])
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    SEQ_LEN = 60
    for i in range(SEQ_LEN, len(df)):
        seq = df.iloc[i-SEQ_LEN:i]
        scaled = scaler.fit_transform(seq)
        x = torch.tensor(scaled.flatten(), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred = model(x).item()

        actual = df.iloc[i]["Close"]

        # Simple rule
        if pred > actual * 1.01:  # Buy if prediction > 1% above actual
            send_signal("NSE:TCS", "BUY", 10)
        elif pred < actual * 0.99:  # Sell if prediction < 1% below actual
            send_signal("NSE:TCS", "SELL", 10)

        time.sleep(1)  # delay between signals (for demo)
