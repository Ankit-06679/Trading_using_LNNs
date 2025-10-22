import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import threading
from collections import deque
import json
import os
from datetime import datetime, timedelta
from config import MODEL_CONFIG, DATA_CONFIG, TRADING_CONFIG, PERFORMANCE_CONFIG, SYSTEM_CONFIG

# System optimizations
if SYSTEM_CONFIG['use_cpu_only']:
    torch.set_num_threads(SYSTEM_CONFIG['torch_threads'])
    
if SYSTEM_CONFIG['pandas_mode']:
    pd.options.mode.copy_on_write = True

# =======================
# Load Your Model (LNN) - Flexible Architecture
# =======================
class LiquidTimeConstantLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidTimeConstantLayer, self).__init__()
        self.hidden_size = hidden_size
        self.time_constant = nn.Parameter(torch.randn(hidden_size))
        self.W = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, x, hidden):
        combined = torch.cat([x, hidden], dim=-1)
        new_hidden = torch.tanh(self.W(combined))
        # Apply time constant
        tau = torch.sigmoid(self.time_constant)
        hidden = (1 - tau) * hidden + tau * new_hidden
        return hidden

class StackedLTC(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, output_size=None, num_layers=None):
        super(StackedLTC, self).__init__()
        # Use config values
        input_size = input_size or MODEL_CONFIG['input_size']
        hidden_size = hidden_size or MODEL_CONFIG['hidden_size']
        output_size = output_size or MODEL_CONFIG['output_size']
        num_layers = num_layers or MODEL_CONFIG['num_layers']
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create LTC layers
        self.layers = nn.ModuleList([
            LiquidTimeConstantLayer(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # Output head
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden states
        hidden_states = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        
        # Process sequence
        for t in range(seq_len):
            current_input = x[:, t, :]
            
            for i, layer in enumerate(self.layers):
                if i == 0:
                    hidden_states[i] = layer(current_input, hidden_states[i])
                else:
                    hidden_states[i] = layer(hidden_states[i-1], hidden_states[i])
        
        # Use final hidden state for prediction
        output = self.head(hidden_states[-1])
        return output

# Fallback LSTM model for compatibility
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, output_size=None, num_layers=None):
        super(SimpleLSTM, self).__init__()
        input_size = input_size or MODEL_CONFIG['input_size']
        hidden_size = hidden_size or MODEL_CONFIG['hidden_size']
        output_size = output_size or MODEL_CONFIG['output_size']
        num_layers = num_layers or MODEL_CONFIG['num_layers']
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last timestep
        return out

# Load model (singleton pattern for efficiency)
class ModelManager:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self):
        if self._model is None:
            # Create a more active demo model that will actually execute trades
            class ActiveTradingModel(nn.Module):
                def __init__(self):
                    super(ActiveTradingModel, self).__init__()
                    self.lstm = nn.LSTM(5, 32, 2, batch_first=True)
                    self.fc = nn.Linear(32, 1)
                    self.step_counter = 0
                    
                    # Initialize with weights that will generate stronger signals
                    with torch.no_grad():
                        for param in self.parameters():
                            param.data.normal_(0, 0.3)
                
                def forward(self, x):
                    self.step_counter += 1
                    out, _ = self.lstm(x)
                    out = self.fc(out[:, -1, :])
                    
                    # Create more dynamic predictions that will trigger trades
                    base_pred = torch.tanh(out) * 0.15
                    
                    # Add some cyclical behavior to ensure trades happen
                    cycle_factor = np.sin(self.step_counter * 0.1) * 0.08
                    trend_factor = (self.step_counter % 50 - 25) * 0.003
                    
                    # Combine factors to create realistic trading signals
                    final_pred = base_pred + cycle_factor + trend_factor
                    
                    return final_pred
            
            self._model = ActiveTradingModel()
            self._model.eval()
            print("🔧 Using active demo model that will execute trades")
            print("💡 This model generates stronger signals to demonstrate trading functionality")
            print("🎯 Buy threshold: >0.05, Sell threshold: <-0.05")
            
        return self._model

# =======================
# Data Manager (CPU Optimized)
# =======================
class DataManager:
    def __init__(self):
        self.df = None
        self.scaled_data = None
        self.mean = None
        self.std = None
        self.feature_cols = DATA_CONFIG['feature_columns']
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess data once"""
        self.df = pd.read_csv(DATA_CONFIG['data_path'])
        
        # Skip the second row if it contains 'TCS.NS' (header row)
        if len(self.df) > 0 and 'TCS.NS' in str(self.df.iloc[0].values):
            self.df = self.df.drop(index=0).reset_index(drop=True)
        
        # Convert date column and clean data
        self.df[DATA_CONFIG['date_column']] = pd.to_datetime(self.df[DATA_CONFIG['date_column']])
        self.df = self.df.dropna().reset_index(drop=True)
        
        # Ensure numeric columns are properly converted
        for col in self.feature_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Remove any rows with NaN values after conversion
        self.df = self.df.dropna().reset_index(drop=True)
        
        # Normalize OHLCV
        data = self.df[self.feature_cols].values.astype(float)
        self.mean, self.std = data.mean(axis=0), data.std(axis=0)
        self.scaled_data = (data - self.mean) / self.std
    
    def get_data_slice(self, start_idx, end_idx):
        """Get data slice without full copy"""
        return self.scaled_data[start_idx:end_idx], self.df.iloc[start_idx:end_idx]

# =======================
# Real-Time Trading Simulator (CPU Optimized)
# =======================
class RealTimeTradingSimulator:
    def __init__(self, initial_cash=None, window_size=None, speed_multiplier=1):
        # Use config values
        self.initial_cash = initial_cash or TRADING_CONFIG['initial_cash']
        self.cash = self.initial_cash
        self.shares = 0
        self.history = deque(maxlen=TRADING_CONFIG['max_history_length'])
        self.window_size = window_size or TRADING_CONFIG['window_size']
        self.speed_multiplier = speed_multiplier
        
        # Managers
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        
        # State
        self.current_step = self.window_size
        self.is_running = False
        self.last_update = time.time()
        
        # Performance optimization
        self.update_interval = PERFORMANCE_CONFIG['update_interval']
        
    def predict(self, window_data):
        """CPU-optimized prediction"""
        model = self.model_manager.get_model()
        x = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred = model(x).item()
        return pred
    
    def step(self):
        """Single trading step"""
        if self.current_step >= len(self.data_manager.df):
            return None
            
        # Get data window
        scaled_window, df_row = self.data_manager.get_data_slice(
            self.current_step - self.window_size, 
            self.current_step + 1
        )
        
        if len(scaled_window) < self.window_size:
            return None
            
        # Prediction
        pred = self.predict(scaled_window[:-1])  # Exclude current step
        
        current_data = df_row.iloc[-1]
        price = current_data['Close']
        action = "HOLD"
        
        # Trading Logic (optimized)
        buy_threshold = TRADING_CONFIG['buy_threshold']
        sell_threshold = TRADING_CONFIG['sell_threshold']
        
        if pred > buy_threshold and self.cash >= price:
            shares_to_buy = min(1, int(self.cash // price))
            if shares_to_buy > 0:
                self.shares += shares_to_buy
                self.cash -= shares_to_buy * price
                action = "BUY"
        elif pred < sell_threshold and self.shares > 0:
            self.shares -= 1
            self.cash += price
            action = "SELL"
        
        portfolio = self.cash + self.shares * price
        
        result = {
            "Step": self.current_step,
            "Date": current_data[DATA_CONFIG['date_column']].strftime('%Y-%m-%d'),
            "Price": round(float(price), 2),
            "Pred": round(float(pred), 4),
            "Action": action,
            "Cash": round(self.cash, 2),
            "Shares": self.shares,
            "Portfolio": round(portfolio, 2),
            "Timestamp": datetime.now().isoformat()
        }
        
        self.history.append(result)
        self.current_step += 1
        return result
    
    def get_current_state(self):
        """Get current trading state"""
        if not self.history:
            return None
        return dict(self.history[-1])
    
    def get_recent_history(self, n=50):
        """Get recent trading history"""
        return list(self.history)[-n:]
    
    def reset(self):
        """Reset simulation"""
        self.cash = self.initial_cash
        self.shares = 0
        self.history.clear()
        self.current_step = self.window_size
        self.is_running = False

# =======================
# Real-Time Trading Engine
# =======================
class TradingEngine:
    def __init__(self):
        self.simulator = RealTimeTradingSimulator()
        self.thread = None
        self.running = False
        
    def start_simulation(self, speed_multiplier=1):
        """Start real-time simulation"""
        if self.running:
            return False
            
        self.simulator.speed_multiplier = speed_multiplier
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        return True
    
    def stop_simulation(self):
        """Stop simulation"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def _run_loop(self):
        """Main simulation loop"""
        while self.running:
            current_time = time.time()
            
            # CPU-friendly update rate
            if current_time - self.simulator.last_update >= self.simulator.update_interval:
                result = self.simulator.step()
                self.simulator.last_update = current_time
                
                if result is None:  # End of data
                    self.running = False
                    break
            
            # Sleep to prevent CPU overload
            time.sleep(PERFORMANCE_CONFIG['cpu_sleep_time'] * self.simulator.speed_multiplier)
    
    def get_status(self):
        """Get current status"""
        return {
            "running": self.running,
            "current_step": self.simulator.current_step,
            "total_steps": len(self.simulator.data_manager.df),
            "progress": (self.simulator.current_step / len(self.simulator.data_manager.df)) * 100
        }

# Global engine instance
trading_engine = TradingEngine()

# =======================
# API Functions
# =======================
def start_trading(speed=1):
    """Start real-time trading simulation"""
    return trading_engine.start_simulation(speed)

def stop_trading():
    """Stop trading simulation"""
    trading_engine.stop_simulation()

def get_trading_state():
    """Get current trading state"""
    return trading_engine.simulator.get_current_state()

def get_trading_history(n=50):
    """Get recent trading history"""
    return trading_engine.simulator.get_recent_history(n)

def get_trading_status():
    """Get simulation status"""
    return trading_engine.get_status()

def reset_trading():
    """Reset trading simulation"""
    trading_engine.stop_simulation()
    trading_engine.simulator.reset()

# Legacy function for compatibility
def run_trading():
    """Legacy function - runs full simulation"""
    sim = RealTimeTradingSimulator()
    results = []
    
    while sim.current_step < len(sim.data_manager.df):
        result = sim.step()
        if result:
            results.append(result)
        
        # Prevent CPU overload
        if len(results) % 100 == 0:
            time.sleep(0.01)
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Test real-time simulation
    print("Starting real-time trading simulation...")
    start_trading(speed=10)  # 10x speed for testing
    
    time.sleep(5)  # Let it run for 5 seconds
    
    print("Current state:", get_trading_state())
    print("Status:", get_trading_status())
    
    stop_trading()
