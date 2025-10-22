"""
Configuration settings for the TCS Trading Simulation
Adjust these parameters to optimize performance for your system
"""

# ======================
# Model Configuration
# ======================
MODEL_CONFIG = {
    'model_path': 'lnn_final_model.pth',
    'input_size': 5,
    'hidden_size': 64,
    'output_size': 1,
    'num_layers': 2
}

# ======================
# Data Configuration
# ======================
DATA_CONFIG = {
    'data_path': 'TCS_2020_present.csv',
    'feature_columns': ["Open", "High", "Low", "Close", "Volume"],
    'date_column': 'Date'
}

# ======================
# Trading Configuration
# ======================
TRADING_CONFIG = {
    'initial_cash': 100000,
    'window_size': 30,
    'buy_threshold': 0.03,    # Lowered to make trades more likely
    'sell_threshold': -0.03,  # Lowered to make trades more likely
    'max_history_length': 1000  # Limit memory usage
}

# ======================
# Performance Configuration
# ======================
PERFORMANCE_CONFIG = {
    'update_interval': 0.1,  # Minimum seconds between updates
    'cpu_sleep_time': 0.01,  # Sleep time to prevent CPU overload
    'max_chart_points': 200,  # Limit chart points for performance
    'refresh_interval': 1.0,  # Frontend refresh interval
    'batch_size': 100  # Process data in batches
}

# ======================
# UI Configuration
# ======================
UI_CONFIG = {
    'page_title': 'Real-Time TCS Trading Simulation',
    'default_speed': 1.0,
    'max_speed': 10.0,
    'min_speed': 0.1,
    'chart_height': 500,
    'mini_chart_height': 300
}

# ======================
# System Optimization
# ======================
SYSTEM_CONFIG = {
    'use_cpu_only': True,  # Force CPU usage to prevent GPU overhead
    'torch_threads': 2,    # Limit PyTorch threads
    'pandas_mode': 'copy_on_write',  # Optimize pandas memory usage
    'enable_caching': True,  # Enable data caching
    'memory_limit_mb': 512   # Memory limit in MB
}