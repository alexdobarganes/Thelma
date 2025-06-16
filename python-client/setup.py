#!/usr/bin/env python3
"""
Setup Script for NinjaTrader WebSocket Client
===========================================

Automates the setup process for the WebSocket client.
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = ['logs', 'data', 'reports']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}/")

def install_dependencies():
    """Install Python dependencies"""
    print("🔧 Installing Python dependencies...")
    
    try:
        # Install basic dependencies
        basic_deps = [
            "websockets>=12.0",
            "asyncio",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "rich>=13.7.0"
        ]
        
        for dep in basic_deps:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ Installed: {dep}")
            
        # Try to install optional dependencies
        optional_deps = [
            "matplotlib>=3.7.0",
            "plotly>=5.17.0",
            "jupyter>=1.0.0"
        ]
        
        for dep in optional_deps:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                print(f"✅ Installed (optional): {dep}")
            except subprocess.CalledProcessError:
                print(f"⚠️  Optional dependency failed: {dep}")
                
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
        
    return True

def test_connection():
    """Test WebSocket connection"""
    print("\n🔍 Testing WebSocket connection...")
    print("Note: This requires NinjaTrader to be running with the WebSocket indicator")
    
    response = input("Do you want to test the connection now? (y/N): ").strip().lower()
    
    if response == 'y':
        print("🚀 Running connection test...")
        try:
            # Run simple test for 10 seconds
            subprocess.run([sys.executable, "simple_test.py"], timeout=10)
        except subprocess.TimeoutExpired:
            print("✅ Connection test completed (timed out after 10 seconds)")
        except subprocess.CalledProcessError:
            print("❌ Connection test failed")
        except KeyboardInterrupt:
            print("🛑 Connection test stopped by user")
    else:
        print("⏭️  Skipping connection test")

def show_usage_instructions():
    """Display usage instructions"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\n📋 Quick Start Guide:")
    print("\n1. Start NinjaTrader 8")
    print("2. Add the WebSocket indicator to any chart")
    print("3. Run the test client:")
    print("   python simple_test.py")
    print("\n📊 Advanced Usage:")
    print("   python websocket_client.py    # Full-featured client")
    print("\n📚 Documentation:")
    print("   See README.md for detailed instructions")
    print("\n🔧 Configuration:")
    print("   Edit connection settings in the Python files")
    print("   Default: ws://localhost:6789/")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("🚀 NinjaTrader WebSocket Client Setup")
    print("="*50)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        return False
    
    # Test connection (optional)
    test_connection()
    
    # Show usage instructions
    show_usage_instructions()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Setup completed successfully!")
        else:
            print("\n❌ Setup failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1) 