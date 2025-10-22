#!/usr/bin/env python3
"""
Test script to verify the trading simulation system is working correctly
"""

import time
from backend import (
    start_trading, stop_trading, get_trading_state, 
    get_trading_history, get_trading_status, reset_trading
)
from performance_monitor import (
    start_performance_monitoring, get_performance_stats
)

def test_trading_system():
    """Test the complete trading system"""
    print("🧪 Testing Real-Time TCS Trading Simulation System")
    print("=" * 50)
    
    # Test 1: Start performance monitoring
    print("1️⃣ Starting performance monitoring...")
    start_performance_monitoring()
    time.sleep(1)
    
    # Test 2: Start trading simulation
    print("2️⃣ Starting trading simulation...")
    success = start_trading(speed=5)  # 5x speed for testing
    if success:
        print("✅ Trading simulation started successfully")
    else:
        print("❌ Failed to start trading simulation")
        return
    
    # Test 3: Let it run for a few seconds
    print("3️⃣ Running simulation for 10 seconds...")
    for i in range(10):
        time.sleep(1)
        status = get_trading_status()
        current_state = get_trading_state()
        
        if status and current_state:
            print(f"   Step {status['current_step']}/{status['total_steps']} "
                  f"({status['progress']:.1f}%) - "
                  f"Price: ₹{current_state['Price']} - "
                  f"Action: {current_state['Action']} - "
                  f"Portfolio: ₹{current_state['Portfolio']:,.2f}")
        else:
            print(f"   Waiting for data... ({i+1}/10)")
    
    # Test 4: Check performance stats
    print("4️⃣ Checking performance statistics...")
    perf_stats = get_performance_stats()
    if perf_stats:
        print(f"   CPU Usage: {perf_stats['cpu_usage']:.1f}%")
        print(f"   Memory Usage: {perf_stats['memory_usage']:.1f}%")
        print(f"   Trading Speed: {perf_stats['trading_speed']:.2f} steps/sec")
    else:
        print("   Performance stats not available yet")
    
    # Test 5: Get trading history
    print("5️⃣ Checking trading history...")
    history = get_trading_history(10)
    if history:
        print(f"   Retrieved {len(history)} trading records")
        trades = [h for h in history if h['Action'] in ['BUY', 'SELL']]
        print(f"   Found {len(trades)} actual trades")
    else:
        print("   No trading history available")
    
    # Test 6: Stop simulation
    print("6️⃣ Stopping simulation...")
    stop_trading()
    time.sleep(1)
    
    final_status = get_trading_status()
    if final_status and not final_status['running']:
        print("✅ Simulation stopped successfully")
    else:
        print("⚠️ Simulation may still be running")
    
    print("\n🎉 System test completed!")
    print("\n📊 Test Results Summary:")
    print("=" * 30)
    
    if current_state:
        initial_cash = 100000
        final_portfolio = current_state['Portfolio']
        profit_loss = final_portfolio - initial_cash
        
        print(f"Initial Cash: ₹{initial_cash:,.2f}")
        print(f"Final Portfolio: ₹{final_portfolio:,.2f}")
        print(f"Profit/Loss: ₹{profit_loss:,.2f} ({(profit_loss/initial_cash)*100:.2f}%)")
        print(f"Total Steps Processed: {status['current_step'] if status else 'Unknown'}")
    
    print("\n💡 Next Steps:")
    print("1. Run 'python -m streamlit run frontend.py --server.port 8503' to start the web interface")
    print("2. Replace the demo model with your trained LNN model")
    print("3. Adjust configuration in config.py as needed")
    print("4. Monitor performance and adjust speed/refresh rates")

if __name__ == "__main__":
    try:
        test_trading_system()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        stop_trading()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        stop_trading()