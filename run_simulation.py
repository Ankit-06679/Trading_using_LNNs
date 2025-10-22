#!/usr/bin/env python3
"""
Real-Time TCS Trading Simulation Launcher
Optimized for CPU efficiency and real-time performance
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'torch', 'pandas', 'numpy', 'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    return True

def main():
    """Launch the trading simulation"""
    print("🚀 Starting Real-Time TCS Trading Simulation...")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check if required files exist
    required_files = ['backend.py', 'frontend.py', 'lnn_final_model.pth', 'TCS_2020_present.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        sys.exit(1)
    
    print("✅ All requirements satisfied")
    print("🌐 Launching Streamlit app...")
    print("📊 Access the simulation at: http://localhost:8508")
    print("⚡ CPU-optimized for real-time performance")
    print("\n" + "="*50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'frontend.py',
            '--server.port', '8508',
            '--server.headless', 'false',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")
    except Exception as e:
        print(f"❌ Error launching simulation: {e}")

if __name__ == "__main__":
    main()