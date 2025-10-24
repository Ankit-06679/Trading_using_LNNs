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

