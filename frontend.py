import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from backend import (
    start_trading, stop_trading, get_trading_state, 
    get_trading_history, get_trading_status, reset_trading
)
from performance_monitor import (
    start_performance_monitoring, stop_performance_monitoring,
    get_performance_stats, get_performance_report
)
from config import UI_CONFIG, PERFORMANCE_CONFIG, TRADING_CONFIG, TRADING_CONFIG

# Page config
st.set_page_config(
    page_title=UI_CONFIG['page_title'],
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-running { background-color: #28a745; }
    .status-stopped { background-color: #dc3545; }
    .status-warning { background-color: #ffc107; }
    
    .trade-signal-buy {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .trade-signal-sell {
        background-color: #dc3545;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .trade-signal-hold {
        background-color: #6c757d;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header with status indicator
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">🚀 TCS Trading Simulation Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Powered by Liquid Neural Network • Real-time Analytics</p>', unsafe_allow_html=True)

# Helper functions for advanced analytics
def calculate_technical_indicators(df):
    """Calculate technical indicators for the price data"""
    if len(df) < 20:
        return df
    
    # Simple Moving Averages
    df['SMA_10'] = df['Price'].rolling(window=10).mean()
    df['SMA_20'] = df['Price'].rolling(window=20).mean()
    
    # RSI calculation
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Price'].rolling(window=20).mean()
    bb_std = df['Price'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def calculate_performance_metrics(history):
    """Calculate advanced performance metrics"""
    if not history or len(history) < 2:
        return {}
    
    df = pd.DataFrame(history)
    df['Returns'] = df['Portfolio'].pct_change()
    
    # Basic metrics
    total_return = (df['Portfolio'].iloc[-1] / df['Portfolio'].iloc[0] - 1) * 100
    
    # Risk metrics
    volatility = df['Returns'].std() * np.sqrt(252) * 100  # Annualized
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = (df['Returns'].mean() / df['Returns'].std()) * np.sqrt(252) if df['Returns'].std() > 0 else 0
    
    # Maximum drawdown
    rolling_max = df['Portfolio'].expanding().max()
    drawdown = (df['Portfolio'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    # Win rate
    trades = df[df['Action'].isin(['BUY', 'SELL'])]
    if len(trades) > 1:
        trade_returns = trades['Portfolio'].diff().dropna()
        win_rate = (trade_returns > 0).sum() / len(trade_returns) * 100
    else:
        win_rate = 0
    
    return {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': len(trades)
    }

def create_status_indicator(is_running):
    """Create a status indicator with color coding"""
    if is_running:
        return '<span class="status-indicator status-running"></span>Running'
    else:
        return '<span class="status-indicator status-stopped"></span>Stopped'

# Initialize session state
if 'simulation_started' not in st.session_state:
    st.session_state.simulation_started = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'performance_monitoring' not in st.session_state:
    st.session_state.performance_monitoring = False
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Overview"

# ======================
# Enhanced Sidebar Controls
# ======================
st.sidebar.markdown("### 🎮 Trading Controls")

# Status display with indicator
status = get_trading_status()
if status:
    status_html = create_status_indicator(status['running'])
    st.sidebar.markdown(f"**Status:** {status_html}", unsafe_allow_html=True)
    
    # Progress bar
    progress = status['progress'] / 100
    st.sidebar.progress(progress)
    st.sidebar.caption(f"Progress: {status['progress']:.1f}% ({status['current_step']}/{status['total_steps']})")

st.sidebar.markdown("---")

# Speed control with better styling
speed = st.sidebar.slider(
    "🚀 Simulation Speed", 
    UI_CONFIG['min_speed'], 
    UI_CONFIG['max_speed'], 
    UI_CONFIG['default_speed'], 
    0.1,
    help="Adjust simulation speed (1x = real-time, 10x = 10x faster)"
)

# Control buttons with better layout
st.sidebar.markdown("**Controls:**")
col1, col2, col3 = st.sidebar.columns(3)

with col1:
    if st.button("▶️ Start", disabled=st.session_state.simulation_started):
        if start_trading(speed):
            st.session_state.simulation_started = True
            if not st.session_state.performance_monitoring:
                start_performance_monitoring()
                st.session_state.performance_monitoring = True
            st.success("Simulation started!")
        else:
            st.error("Failed to start simulation")

with col2:
    if st.button("⏸️ Stop", disabled=not st.session_state.simulation_started):
        stop_trading()
        st.session_state.simulation_started = False
        if st.session_state.performance_monitoring:
            stop_performance_monitoring()
            st.session_state.performance_monitoring = False
        st.info("Simulation stopped")

with col3:
    if st.button("🔄 Reset"):
        reset_trading()
        st.session_state.simulation_started = False
        if st.session_state.performance_monitoring:
            stop_performance_monitoring()
            st.session_state.performance_monitoring = False
        st.info("Simulation reset")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=True)
refresh_interval = st.sidebar.slider(
    "Refresh Interval (seconds)", 
    0.5, 5.0, 
    PERFORMANCE_CONFIG['refresh_interval'], 
    0.1,
    help="How often to update the dashboard"
)

# Performance monitoring toggle
show_performance = st.sidebar.checkbox("📊 Performance Stats", value=True)
show_advanced = st.sidebar.checkbox("🔬 Advanced Analytics", value=True)

# Trading parameters display
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Trading Parameters")
st.sidebar.info(f"""
**Buy Threshold:** {TRADING_CONFIG['buy_threshold']}  
**Sell Threshold:** {TRADING_CONFIG['sell_threshold']}  
**Initial Cash:** ₹{TRADING_CONFIG['initial_cash']:,}  
**Window Size:** {TRADING_CONFIG['window_size']} periods
""")

# Manual refresh button
if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
    st.rerun()

# ======================
# Enhanced Status Display
# ======================
current_state = get_trading_state()

# Performance metrics in sidebar
if show_performance and st.session_state.performance_monitoring:
    perf_stats = get_performance_stats()
    if perf_stats:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚡ System Performance")
        
        # CPU usage with color coding
        cpu_color = "🟢" if perf_stats['cpu_usage'] < 50 else "🟡" if perf_stats['cpu_usage'] < 80 else "🔴"
        st.sidebar.metric("CPU Usage", f"{perf_stats['cpu_usage']:.1f}%", delta=None, help=f"{cpu_color} System CPU utilization")
        
        # Memory usage with color coding
        mem_color = "🟢" if perf_stats['memory_usage'] < 70 else "🟡" if perf_stats['memory_usage'] < 85 else "🔴"
        st.sidebar.metric("Memory Usage", f"{perf_stats['memory_usage']:.1f}%", delta=None, help=f"{mem_color} System memory utilization")
        
        st.sidebar.metric("Trading Speed", f"{perf_stats['trading_speed']:.1f} steps/sec", help="Processing speed of the simulation")
        
        # Performance warnings with better styling
        if perf_stats['cpu_usage'] > 80:
            st.sidebar.error("⚠️ High CPU usage detected!")
        if perf_stats['memory_usage'] > 80:
            st.sidebar.error("⚠️ High memory usage detected!")

# ======================
# Enhanced Main Dashboard with Tabs
# ======================

# Create tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Analytics", "🎯 Trading", "📋 Reports"])

with tab1:  # Overview Tab
    if current_state:
        # Enhanced metrics with better styling
        st.markdown("### 💼 Portfolio Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Price with prediction indicator
            pred_color = "🟢" if current_state['Pred'] > 0 else "🔴" if current_state['Pred'] < 0 else "⚪"
            st.metric(
                "Current Price", 
                f"₹{current_state['Price']:,.2f}", 
                delta=f"{current_state['Pred']:.4f}" if current_state['Pred'] != 0 else None,
                help=f"{pred_color} Model prediction: {current_state['Pred']:.4f}"
            )
        
        with col2:
            st.metric("Available Cash", f"₹{current_state['Cash']:,.2f}", help="Cash available for trading")
        
        with col3:
            st.metric("Shares Held", current_state['Shares'], help="Number of TCS shares currently owned")
        
        with col4:
            portfolio_change = current_state['Portfolio'] - TRADING_CONFIG['initial_cash']
            change_pct = (portfolio_change / TRADING_CONFIG['initial_cash']) * 100
            st.metric(
                "Portfolio Value", 
                f"₹{current_state['Portfolio']:,.2f}",
                delta=f"₹{portfolio_change:,.2f} ({change_pct:+.2f}%)",
                help="Total portfolio value including cash and shares"
            )
        
        with col5:
            # Last action with styled indicator
            action = current_state['Action']
            if action == 'BUY':
                action_display = "🟢 BUY"
            elif action == 'SELL':
                action_display = "🔴 SELL"
            else:
                action_display = "⚪ HOLD"
            
            st.metric("Last Action", action_display, help=f"Most recent trading decision")
        
        # Advanced performance metrics
        if show_advanced:
            history = get_trading_history(100)
            if history and len(history) > 10:
                metrics = calculate_performance_metrics(history)
                
                st.markdown("---")
                st.markdown("### 📊 Performance Analytics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Return", f"{metrics['total_return']:+.2f}%", 
                             help="Total portfolio return since start")
                
                with col2:
                    st.metric("Volatility", f"{metrics['volatility']:.2f}%", 
                             help="Annualized portfolio volatility")
                
                with col3:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}", 
                             help="Risk-adjusted return measure")
                
                with col4:
                    st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%", 
                             help="Maximum peak-to-trough decline")
                
                with col5:
                    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%", 
                             help="Percentage of profitable trades")
    
    else:
        st.info("🚀 Start the simulation to see real-time portfolio metrics!")
        st.markdown("""
        ### 🎯 Quick Start Guide:
        1. **Click ▶️ Start** in the sidebar to begin simulation
        2. **Adjust speed** for faster results (1x-10x)
        3. **Monitor** real-time charts and metrics
        4. **Analyze** performance in the Analytics tab
        """)

with tab2:  # Analytics Tab
    st.markdown("### 📈 Technical Analysis")
    
    if current_state:
        history = get_trading_history(PERFORMANCE_CONFIG['max_chart_points'])
        
        if history and len(history) > 20:
            df_history = pd.DataFrame(history)
            df_history['Date'] = pd.to_datetime(df_history['Date'])
            df_history = calculate_technical_indicators(df_history)
            
            # Advanced charting with technical indicators
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Technical Indicators', 'RSI', 'Model Predictions'),
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Main price chart with indicators
            fig.add_trace(go.Scatter(x=df_history['Date'], y=df_history['Price'], 
                                   name='Price', line=dict(color='blue', width=2)), row=1, col=1)
            
            # Moving averages
            if 'SMA_10' in df_history.columns:
                fig.add_trace(go.Scatter(x=df_history['Date'], y=df_history['SMA_10'], 
                                       name='SMA 10', line=dict(color='orange', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_history['Date'], y=df_history['SMA_20'], 
                                       name='SMA 20', line=dict(color='red', width=1)), row=1, col=1)
            
            # Trading signals
            buy_signals = df_history[df_history['Action'] == 'BUY']
            sell_signals = df_history[df_history['Action'] == 'SELL']
            
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Price'],
                                       mode='markers', marker=dict(symbol="triangle-up", color="green", size=12),
                                       name="Buy Signal"), row=1, col=1)
            
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Price'],
                                       mode='markers', marker=dict(symbol="triangle-down", color="red", size=12),
                                       name="Sell Signal"), row=1, col=1)
            
            # RSI
            if 'RSI' in df_history.columns:
                fig.add_trace(go.Scatter(x=df_history['Date'], y=df_history['RSI'], 
                                       name='RSI', line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Model predictions
            fig.add_trace(go.Scatter(x=df_history['Date'], y=df_history['Pred'], 
                                   name='Predictions', line=dict(color='orange')), row=3, col=1)
            fig.add_hline(y=TRADING_CONFIG['buy_threshold'], line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=TRADING_CONFIG['sell_threshold'], line_dash="dash", line_color="red", row=3, col=1)
            
            fig.update_layout(height=800, showlegend=True, title="Advanced Technical Analysis")
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="Prediction", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Portfolio performance chart
            col1, col2 = st.columns(2)
            
            with col1:
                portfolio_fig = go.Figure()
                portfolio_fig.add_trace(go.Scatter(
                    x=df_history['Date'],
                    y=df_history['Portfolio'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='green', width=3)
                ))
                portfolio_fig.add_hline(
                    y=TRADING_CONFIG['initial_cash'], 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Initial Investment"
                )
                portfolio_fig.update_layout(
                    title="Portfolio Performance",
                    xaxis_title="Date",
                    yaxis_title="Value (₹)",
                    height=400
                )
                st.plotly_chart(portfolio_fig, use_container_width=True)
            
            with col2:
                # Returns distribution
                returns = df_history['Portfolio'].pct_change().dropna()
                if len(returns) > 10:
                    fig_hist = px.histogram(
                        x=returns * 100,
                        nbins=20,
                        title="Returns Distribution",
                        labels={'x': 'Returns (%)', 'y': 'Frequency'}
                    )
                    fig_hist.update_layout(height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("📊 Collecting data for technical analysis... Please wait for more data points.")
    else:
        st.info("📈 Start the simulation to view technical analysis charts!")

with tab3:  # Trading Tab
    st.markdown("### 🎯 Trading Activity")
    
    if current_state:
        history = get_trading_history(50)
        
        if history:
            df_history = pd.DataFrame(history)
            
            # Trading summary
            col1, col2 = st.columns(2)
            
            with col1:
                # Recent trades table with enhanced styling
                st.markdown("#### 📋 Recent Trades")
                recent_trades = df_history[df_history['Action'].isin(['BUY', 'SELL'])].tail(10)
                
                if not recent_trades.empty:
                    st.dataframe(
                        recent_trades[['Date', 'Action', 'Price', 'Pred', 'Cash', 'Portfolio']],
                        use_container_width=True
                    )
                else:
                    st.info("No trades executed yet. Adjust thresholds or wait for signals.")
            
            with col2:
                # Trading statistics
                st.markdown("#### 📊 Trading Statistics")
                
                total_trades = len(df_history[df_history['Action'].isin(['BUY', 'SELL'])])
                buy_trades = len(df_history[df_history['Action'] == 'BUY'])
                sell_trades = len(df_history[df_history['Action'] == 'SELL'])
                
                st.metric("Total Trades", total_trades)
                st.metric("Buy Orders", buy_trades)
                st.metric("Sell Orders", sell_trades)
                
                if total_trades > 0:
                    avg_trade_size = current_state['Shares'] / total_trades if total_trades > 0 else 0
                    st.metric("Avg Trade Size", f"{avg_trade_size:.1f} shares")
        
        # Portfolio allocation chart
        if current_state['Shares'] > 0 or current_state['Cash'] > 0:
            st.markdown("---")
            st.markdown("#### 🥧 Portfolio Allocation")
            
            cash_value = current_state['Cash']
            shares_value = current_state['Shares'] * current_state['Price']
            
            fig_pie = px.pie(
                values=[cash_value, shares_value],
                names=['Cash', 'TCS Shares'],
                title="Current Portfolio Allocation",
                color_discrete_map={'Cash': '#ff7f0e', 'TCS Shares': '#1f77b4'}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
    
    else:
        st.info("🎯 Start the simulation to view trading activity!")

with tab4:  # Reports Tab
    st.markdown("### 📋 Performance Reports")
    
    if current_state:
        history = get_trading_history(200)
        
        if history and len(history) > 10:
            df_history = pd.DataFrame(history)
            metrics = calculate_performance_metrics(history)
            
            # Comprehensive performance report
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Performance Summary")
                
                report_data = {
                    "Metric": [
                        "Initial Capital", "Current Portfolio", "Total Return", 
                        "Total Trades", "Win Rate", "Volatility", 
                        "Sharpe Ratio", "Maximum Drawdown"
                    ],
                    "Value": [
                        f"₹{TRADING_CONFIG['initial_cash']:,}",
                        f"₹{current_state['Portfolio']:,.2f}",
                        f"{metrics['total_return']:+.2f}%",
                        f"{metrics['total_trades']}",
                        f"{metrics['win_rate']:.1f}%",
                        f"{metrics['volatility']:.2f}%",
                        f"{metrics['sharpe_ratio']:.2f}",
                        f"{metrics['max_drawdown']:.2f}%"
                    ]
                }
                
                report_df = pd.DataFrame(report_data)
                st.dataframe(report_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### 📈 Cumulative Returns")
                
                returns = df_history['Portfolio'].pct_change().fillna(0)
                cumulative_returns = (1 + returns).cumprod() - 1
                
                fig_returns = go.Figure()
                fig_returns.add_trace(go.Scatter(
                    x=df_history['Date'], 
                    y=cumulative_returns * 100,
                    mode='lines',
                    name='Cumulative Returns',
                    line=dict(color='blue', width=2)
                ))
                
                fig_returns.update_layout(
                    title="Cumulative Returns (%)",
                    xaxis_title="Date",
                    yaxis_title="Returns (%)",
                    height=400
                )
                
                st.plotly_chart(fig_returns, use_container_width=True)
            
            # Export functionality
            st.markdown("---")
            st.markdown("#### 💾 Export Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export Trading History", use_container_width=True):
                    csv = df_history.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"trading_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📈 Export Performance Report", use_container_width=True):
                    report_csv = pd.DataFrame(report_data).to_csv(index=False)
                    st.download_button(
                        label="Download Report",
                        data=report_csv,
                        file_name=f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if st.button("🔄 Generate New Report", use_container_width=True):
                    st.rerun()
        
        else:
            st.info("📋 Collecting data for comprehensive reports... Please wait for more trading activity.")
    
    else:
        st.info("📋 Start the simulation to generate performance reports!")

# ======================
# Auto-refresh mechanism
# ======================
if auto_refresh and st.session_state.simulation_started:
    current_time = time.time()
    if current_time - st.session_state.last_update >= refresh_interval:
        st.session_state.last_update = current_time
        time.sleep(0.1)  # Small delay to prevent CPU overload
        st.rerun()

# ======================
# Enhanced Footer
# ======================
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ System Information")
st.sidebar.info(f"""
**🤖 Model**: Liquid Neural Network  
**📈 Asset**: TCS Stock (2020-2025)  
**⚡ Mode**: Real-time Simulation  
**🔄 Last Update**: {datetime.now().strftime('%H:%M:%S')}  
**💾 Data Points**: {len(get_trading_history(1000)) if get_trading_history(1000) else 0}
""")

# Add some helpful tips
with st.sidebar.expander("💡 Pro Tips"):
    st.markdown("""
    - **Higher speed** = faster results but more CPU usage
    - **Lower thresholds** = more frequent trading
    - **Enable performance stats** to monitor system health
    - **Use Analytics tab** for detailed technical analysis
    - **Export data** from Reports tab for further analysis
    """)

# Add disclaimer
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ **Disclaimer**: This is a simulation for educational purposes only. Not financial advice.")
