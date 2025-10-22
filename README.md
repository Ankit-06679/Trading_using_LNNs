# Real-Time TCS Trading Simulation

A CPU-optimized real-time trading simulation using a Liquid Neural Network (LNN) model for TCS stock prediction and automated trading.

## 🚀 Features

- **Real-time simulation** with configurable speed
- **CPU-optimized** performance to prevent system overload
- **Interactive dashboard** with live charts and metrics
- **Performance monitoring** with CPU and memory tracking
- **Configurable parameters** for easy customization
- **Memory-efficient** data handling with limited history
- **Threaded execution** for smooth real-time updates

## 📋 Requirements

- Python 3.7+
- PyTorch
- Streamlit
- Pandas
- NumPy
- Plotly
- psutil (for performance monitoring)

## 🛠️ Installation

1. **Install required packages:**
```bash
pip install torch streamlit pandas numpy plotly psutil
```

2. **Ensure you have the required files:**
- `lnn_final_model.pth` - Your trained LNN model
- `TCS_2020_present.csv` - TCS stock data (2020-2025)

## 🎮 Usage

### Quick Start
```bash
python run_simulation.py
```

This will automatically:
- Check all requirements
- Launch the Streamlit interface
- Open your browser to `http://localhost:8501`

### Manual Start
```bash
streamlit run frontend.py
```

## 🎛️ Controls

### Simulation Controls
- **▶️ Start**: Begin real-time trading simulation
- **⏸️ Stop**: Pause the simulation
- **🔄 Reset**: Reset all trading data and start fresh

### Performance Settings
- **Simulation Speed**: 0.1x to 10x real-time speed
- **Auto Refresh**: Enable/disable automatic UI updates
- **Refresh Interval**: How often to update the display (0.5-5 seconds)
- **Performance Stats**: Show CPU and memory usage

## 📊 Dashboard Features

### Real-Time Metrics
- Current stock price with prediction delta
- Available cash and shares owned
- Total portfolio value with profit/loss

### Interactive Charts
- **Price Chart**: Real-time price movements with buy/sell signals
- **Portfolio Chart**: Portfolio value over time
- **Prediction Chart**: Model predictions with buy/sell thresholds

### Performance Monitoring
- CPU usage tracking
- Memory consumption monitoring
- Trading speed (steps per second)
- Performance warnings and recommendations

## ⚙️ Configuration

Edit `config.py` to customize:

### Trading Parameters
```python
TRADING_CONFIG = {
    'initial_cash': 100000,      # Starting cash
    'window_size': 30,           # Data window for predictions
    'buy_threshold': 0.05,       # Buy signal threshold
    'sell_threshold': -0.05,     # Sell signal threshold
}
```

### Performance Optimization
```python
PERFORMANCE_CONFIG = {
    'update_interval': 0.1,      # Minimum update interval (seconds)
    'cpu_sleep_time': 0.01,      # CPU throttling sleep time
    'max_chart_points': 200,     # Limit chart points for performance
    'refresh_interval': 1.0,     # UI refresh rate
}
```

### System Settings
```python
SYSTEM_CONFIG = {
    'use_cpu_only': True,        # Force CPU usage (no GPU)
    'torch_threads': 2,          # Limit PyTorch threads
    'memory_limit_mb': 512       # Memory usage limit
}
```

## 🔧 CPU Optimization Features

1. **Threaded Execution**: Simulation runs in background thread
2. **Memory Management**: Limited history with deque data structures
3. **CPU Throttling**: Configurable sleep intervals to prevent overload
4. **Efficient Updates**: Minimal UI refresh rates
5. **Batch Processing**: Data processed in optimized batches
6. **Resource Monitoring**: Real-time CPU and memory tracking

## 📈 Trading Logic

The simulation uses your trained LNN model to:

1. **Analyze** the last 30 data points (configurable)
2. **Predict** price movement direction
3. **Execute trades** based on thresholds:
   - **Buy** when prediction > 0.05
   - **Sell** when prediction < -0.05
   - **Hold** otherwise

## 🚨 Performance Tips

### For Better Performance:
- Reduce simulation speed if CPU usage is high
- Increase refresh interval to reduce UI load
- Limit chart points for smoother rendering
- Close other applications to free resources

### For Faster Results:
- Increase simulation speed (up to 10x)
- Reduce refresh interval for more responsive UI
- Enable performance monitoring to track efficiency

## 📁 File Structure

```
├── backend.py              # Core trading simulation engine
├── frontend.py             # Streamlit web interface
├── config.py               # Configuration settings
├── performance_monitor.py  # Performance tracking
├── run_simulation.py       # Easy launcher script
├── lnn_final_model.pth    # Your trained model
├── TCS_2020_present.csv   # Stock data
└── README.md              # This file
```

## 🐛 Troubleshooting

### High CPU Usage
- Reduce simulation speed
- Increase `cpu_sleep_time` in config
- Reduce `torch_threads` setting

### Memory Issues
- Reduce `max_history_length` in config
- Limit `max_chart_points`
- Close other applications

### Slow Performance
- Check if other applications are using resources
- Reduce chart update frequency
- Enable performance monitoring to identify bottlenecks

## 📝 Notes

- The simulation processes historical data in real-time fashion
- All trades are simulated - no real money involved
- Performance monitoring helps optimize system resource usage
- Configuration changes require restart to take effect

## 🤝 Support

If you encounter issues:
1. Check the performance monitor for resource usage
2. Adjust configuration settings as needed
3. Ensure all required files are present
4. Verify Python package versions are compatible

---

**Happy Trading! 📈**