"""
Performance monitoring for the TCS Trading Simulation
Monitors CPU usage, memory consumption, and trading performance
"""

import psutil
import time
import threading
from datetime import datetime
from backend import get_trading_status, get_trading_state

class PerformanceMonitor:
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'trading_steps_per_second': [],
            'timestamps': []
        }
        self.last_step = 0
        self.last_time = time.time()
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
        print("📊 Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                memory_percent = memory_info.percent
                
                # Trading metrics
                status = get_trading_status()
                current_time = time.time()
                
                if status and status['running']:
                    current_step = status['current_step']
                    time_diff = current_time - self.last_time
                    
                    if time_diff > 0:
                        steps_per_second = (current_step - self.last_step) / time_diff
                        self.stats['trading_steps_per_second'].append(steps_per_second)
                    
                    self.last_step = current_step
                
                self.last_time = current_time
                
                # Store metrics
                self.stats['cpu_usage'].append(cpu_percent)
                self.stats['memory_usage'].append(memory_percent)
                self.stats['timestamps'].append(datetime.now())
                
                # Limit history to prevent memory bloat
                max_history = 300  # 5 minutes at 1-second intervals
                for key in self.stats:
                    if len(self.stats[key]) > max_history:
                        self.stats[key] = self.stats[key][-max_history:]
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1)
    
    def get_current_stats(self):
        """Get current performance statistics"""
        if not self.stats['cpu_usage']:
            return None
        
        return {
            'cpu_usage': self.stats['cpu_usage'][-1] if self.stats['cpu_usage'] else 0,
            'memory_usage': self.stats['memory_usage'][-1] if self.stats['memory_usage'] else 0,
            'avg_cpu': sum(self.stats['cpu_usage'][-10:]) / min(10, len(self.stats['cpu_usage'])),
            'avg_memory': sum(self.stats['memory_usage'][-10:]) / min(10, len(self.stats['memory_usage'])),
            'trading_speed': self.stats['trading_steps_per_second'][-1] if self.stats['trading_steps_per_second'] else 0
        }
    
    def get_performance_report(self):
        """Generate performance report"""
        if not self.stats['cpu_usage']:
            return "No performance data available"
        
        stats = self.get_current_stats()
        
        report = f"""
📊 Performance Report
====================
Current CPU Usage: {stats['cpu_usage']:.1f}%
Current Memory Usage: {stats['memory_usage']:.1f}%
Average CPU (last 10s): {stats['avg_cpu']:.1f}%
Average Memory (last 10s): {stats['avg_memory']:.1f}%
Trading Speed: {stats['trading_speed']:.2f} steps/second

💡 Recommendations:
"""
        
        if stats['avg_cpu'] > 80:
            report += "- High CPU usage detected. Consider reducing simulation speed.\n"
        elif stats['avg_cpu'] < 20:
            report += "- Low CPU usage. You can increase simulation speed for faster results.\n"
        
        if stats['avg_memory'] > 80:
            report += "- High memory usage. Consider reducing history length.\n"
        
        if stats['trading_speed'] < 1:
            report += "- Slow trading simulation. Check system resources.\n"
        
        return report

# Global monitor instance
performance_monitor = PerformanceMonitor()

def start_performance_monitoring():
    """Start performance monitoring"""
    performance_monitor.start_monitoring()

def stop_performance_monitoring():
    """Stop performance monitoring"""
    performance_monitor.stop_monitoring()

def get_performance_stats():
    """Get current performance stats"""
    return performance_monitor.get_current_stats()

def get_performance_report():
    """Get performance report"""
    return performance_monitor.get_performance_report()

if __name__ == "__main__":
    # Test performance monitoring
    start_performance_monitoring()
    
    try:
        while True:
            time.sleep(5)
            print(get_performance_report())
    except KeyboardInterrupt:
        stop_performance_monitoring()