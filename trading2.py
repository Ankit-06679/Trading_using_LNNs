# streamlit_app.py
import streamlit as st
import pandas as pd
import torch
import time


# -----------------------------
# Load historical data
# -----------------------------
data = pd.read_csv("C:\\Users\\ankit\\OneDrive\\Desktop\\Capstone\\trial_05\\TCS_2020_present.csv")  # Make sure CSV has Date, Open, High, Low, Close, Volume

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


model = StackedLTC()
model.load_state_dict(torch.load("C:\\Users\\ankit\\OneDrive\\Desktop\\Capstone\\trial_05\\lnn_final_model.pth", map_location=torch.device('cpu')))
model.eval()


# -----------------------------
# Initialize session state
# -----------------------------
if "idx" not in st.session_state:
    st.session_state.idx = 0
    st.session_state.cash = 100000  # Starting cash
    st.session_state.shares = 0
    st.session_state.portfolio_values = []
    st.session_state.trade_log = []

# -----------------------------
# Streamlit UI placeholders
# -----------------------------
st.title("📈 TCS Stock Paper Trading Simulator")
st.write("Simulated live trading using LNN predictions")

portfolio_val_display = st.empty()
cash_display = st.empty()
shares_display = st.empty()
chart_display = st.empty()
trade_log_display = st.empty()

# -----------------------------
# Simulate one step per refresh
# -----------------------------
i = st.session_state.idx
if i < len(data)-1:
    # Prepare input for model
    x = torch.tensor(data.iloc[i][['Open','High','Low','Close','Volume']].values, dtype=torch.float32)
    with torch.no_grad():
        signal_probs = model(x.unsqueeze(0))  # Example: output 3 probabilities (Hold, Buy, Sell)
        signal = torch.argmax(signal_probs, dim=1).item()  # 0=Hold, 1=Buy, 2=Sell

    price = data.iloc[i]['Close']
    cash = st.session_state.cash
    shares = st.session_state.shares
    trade_action = "Hold"

    # Execute trade based on signal
    if signal == 1 and cash >= price:
        shares_to_buy = cash // price
        cash -= shares_to_buy * price
        shares += shares_to_buy
        trade_action = f"Buy {shares_to_buy} shares"
    elif signal == 2 and shares > 0:
        cash += shares * price
        trade_action = f"Sell {shares} shares"
        shares = 0

    # Update portfolio
    portfolio_value = cash + shares * price
    st.session_state.cash = cash
    st.session_state.shares = shares
    st.session_state.portfolio_values.append(portfolio_value)
    st.session_state.trade_log.append({
        "Date": data.iloc[i]['Date'],
        "Price": price,
        "Action": trade_action,
        "PortfolioValue": portfolio_value
    })

    # Update UI
    portfolio_val_display.metric("Portfolio Value", f"₹{portfolio_value:,.2f}")
    cash_display.metric("Cash", f"₹{cash:,.2f}")
    shares_display.metric("Shares Held", shares)
    chart_display.line_chart(pd.DataFrame({
        "Portfolio Value": st.session_state.portfolio_values,
        "Stock Price": data['Close'][:i+1].tolist()
    }))
    trade_log_display.dataframe(pd.DataFrame(st.session_state.trade_log).tail(10))

    # Increment index for next refresh
    st.session_state.idx += 1

    # Auto-refresh every 1 second
    st.experimental_rerun()
else:
    st.success("Simulation completed!")
    st.write("Final Portfolio Value: ₹{:,.2f}".format(st.session_state.portfolio_values[-1]))
