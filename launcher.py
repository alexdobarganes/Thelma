#!/usr/bin/env python3
"""
NinjaTrader 8 ML Strategy - System Launcher & Health Check
==========================================================

Production-ready launcher for the complete trading system.
Provides health checks, system validation, and manual entry points.

Author: Trading System Team
Version: 1.2.0
Date: 2025-06-17
"""

import os
import sys
import json
import pickle
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class SystemLauncher:
    """Complete system launcher with health checks and entry points."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.health_status = {}
        self.startup_time = datetime.now()
        
    def print_header(self):
        """Print system header with branding."""
        print(f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║              NinjaTrader 8 ML Strategy Launcher             ║
║                     Production System v1.2.0                ║
╚══════════════════════════════════════════════════════════════╝
{Colors.END}
{Colors.WHITE}🚀 Automated ES Futures Trading with Machine Learning
📅 System Check: {self.startup_time.strftime('%Y-%m-%d %H:%M:%S')}
🏗️ Project: {self.project_root.name}
{Colors.END}
        """)

    def check_project_structure(self) -> Tuple[bool, str]:
        """Verify optimized project structure."""
        required_dirs = [
            'config', 'docs', 'scripts', 'src', 'data', 
            'models', 'python-client', 'NT8', 'memory-bank'
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            if not (self.project_root / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            return False, f"Missing directories: {', '.join(missing_dirs)}"
        
        # Check semantic versioning structure
        production_path = self.project_root / 'models' / 'production'
        if not production_path.exists():
            return False, "Missing models/production directory"
            
        current_link = production_path / 'current'
        if not current_link.exists():
            return False, "Missing models/production/current symlink"
            
        return True, "✅ Project structure optimized and complete"

    def check_data_integrity(self) -> Tuple[bool, str]:
        """Verify data consolidation and integrity."""
        # Check raw data
        raw_data = self.project_root / 'data' / 'raw' / 'es_1m' / 'market_data.csv'
        if not raw_data.exists():
            return False, "❌ Missing raw ES data: data/raw/es_1m/market_data.csv"
        
        # Check processed features
        processed_data = self.project_root / 'data' / 'processed' / 'es_features_enhanced.csv'
        if not processed_data.exists():
            return False, "❌ Missing processed features: data/processed/es_features_enhanced.csv"
        
        # Verify data sizes
        raw_size_mb = raw_data.stat().st_size / (1024 * 1024)
        processed_size_mb = processed_data.stat().st_size / (1024 * 1024)
        
        if raw_size_mb < 30:  # Should be ~34MB for 2-year dataset
            return False, f"❌ Raw data too small: {raw_size_mb:.1f}MB (expected ~34MB)"
            
        if processed_size_mb < 500:  # Should be ~721MB for enhanced features
            return False, f"❌ Processed data too small: {processed_size_mb:.1f}MB (expected ~721MB)"
        
        return True, f"✅ Data integrity verified (Raw: {raw_size_mb:.1f}MB, Processed: {processed_size_mb:.1f}MB)"

    def check_production_model(self) -> Tuple[bool, str]:
        """Verify production model and metadata."""
        try:
            # Check current symlink
            current_path = self.project_root / 'models' / 'production' / 'current'
            if not current_path.exists():
                return False, "❌ Missing production/current symlink"
            
            # Check model file
            model_file = current_path / 'model.pkl'
            if not model_file.exists():
                return False, "❌ Missing model.pkl in current version"
            
            # Check metadata
            metadata_file = current_path / 'metadata.json'
            if not metadata_file.exists():
                return False, "❌ Missing metadata.json in current version"
            
            # Load and validate metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            required_fields = ['version', 'performance_metrics', 'deployment_status']
            missing_fields = [field for field in required_fields if field not in metadata]
            if missing_fields:
                return False, f"❌ Missing metadata fields: {', '.join(missing_fields)}"
            
            # Check performance metrics
            perf = metadata['performance_metrics']
            f1_score = perf.get('f1_score', 0)
            sharpe_ratio = perf.get('sharpe_ratio', 0)
            
            if f1_score < 0.44:
                return False, f"❌ F1 score below target: {f1_score:.4f} < 0.44"
            
            if sharpe_ratio < 1.20:
                return False, f"❌ Sharpe ratio below target: {sharpe_ratio:.4f} < 1.20"
            
            model_size_kb = model_file.stat().st_size / 1024
            
            return True, f"✅ Production model v{metadata['version']} (F1: {f1_score:.4f}, Sharpe: {sharpe_ratio:.2f}, {model_size_kb:.0f}KB)"
            
        except Exception as e:
            return False, f"❌ Model validation error: {str(e)}"

    def check_dependencies(self) -> Tuple[bool, str]:
        """Check Python dependencies."""
        try:
            config_file = self.project_root / 'config' / 'requirements.txt'
            if not config_file.exists():
                return False, "❌ Missing config/requirements.txt"
            
            # Check critical imports
            critical_modules = {
                'pandas': 'Data processing',
                'numpy': 'Numerical computing',
                'torch': 'PyTorch for GPU acceleration',
                'sklearn': 'Machine learning utilities'
            }
            
            missing_modules = []
            for module, description in critical_modules.items():
                try:
                    importlib.import_module(module)
                except ImportError:
                    missing_modules.append(f"{module} ({description})")
            
            if missing_modules:
                return False, f"❌ Missing modules: {', '.join(missing_modules)}"
            
            return True, "✅ All critical dependencies available"
            
        except Exception as e:
            return False, f"❌ Dependency check error: {str(e)}"

    def run_health_check(self):
        """Run complete system health check."""
        print(f"{Colors.YELLOW}{Colors.BOLD}🔍 SYSTEM HEALTH CHECK{Colors.END}")
        print("=" * 60)
        
        checks = [
            ("Project Structure", self.check_project_structure),
            ("Data Integrity", self.check_data_integrity),
            ("Production Model", self.check_production_model),
            ("Dependencies", self.check_dependencies),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                passed, message = check_func()
                status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
                print(f"{check_name:<25} {status} - {message}")
                self.health_status[check_name] = {'passed': passed, 'message': message}
                if not passed:
                    all_passed = False
            except Exception as e:
                print(f"{check_name:<25} {Colors.RED}❌ ERROR{Colors.END} - {str(e)}")
                self.health_status[check_name] = {'passed': False, 'message': str(e)}
                all_passed = False
        
        print("=" * 60)
        if all_passed:
            print(f"{Colors.GREEN}{Colors.BOLD}🎉 SYSTEM STATUS: ALL CHECKS PASSED{Colors.END}")
            print(f"{Colors.GREEN}✅ System ready for Week 3 platform integration{Colors.END}")
        else:
            print(f"{Colors.RED}{Colors.BOLD}⚠️  SYSTEM STATUS: ISSUES DETECTED{Colors.END}")
            print(f"{Colors.RED}❌ Please resolve issues before proceeding{Colors.END}")
        
        return all_passed

    def show_main_menu(self):
        """Display main system menu."""
        print(f"\n{Colors.BLUE}{Colors.BOLD}📋 SYSTEM OPERATIONS MENU{Colors.END}")
        print("=" * 60)
        
        menu_options = [
            ("1", "🔍 Run Health Check", "Complete system validation"),
            ("2", "🧠 Model Operations", "Train, validate, and manage models"),
            ("3", "📊 Data Operations", "Feature engineering and data processing"),
            ("4", "🌐 WebSocket Testing", "Test NT8 connectivity and data streaming"),
            ("5", "📈 Performance Dashboard", "View model metrics and comparisons"),
            ("6", "🚀 Deploy to NT8", "Update NinjaTrader indicators"),
            ("7", "📖 Documentation", "View project documentation"),
            ("8", "🧹 Maintenance", "Cleanup and optimization tools"),
            ("0", "❌ Exit", "Close launcher")
        ]
        
        for option, title, description in menu_options:
            print(f"{Colors.WHITE}{option}.{Colors.END} {Colors.CYAN}{title:<25}{Colors.END} - {description}")
        
        print("=" * 60)

    def main_loop(self):
        """Main interactive loop."""
        while True:
            self.show_main_menu()
            choice = input(f"\n{Colors.YELLOW}Enter your choice: {Colors.END}")
            
            if choice == '0':
                print(f"\n{Colors.GREEN}👋 Goodbye! System launcher closing...{Colors.END}")
                break
            elif choice == '1':
                self.run_health_check()
            elif choice == '2':
                print(f"\n{Colors.PURPLE}🧠 Model operations available in src/models/{Colors.END}")
            elif choice == '3':
                print(f"\n{Colors.GREEN}📊 Data operations available in src/data/{Colors.END}")
            elif choice == '4':
                print(f"\n{Colors.CYAN}🌐 WebSocket client: python-client/websocket_client.py{Colors.END}")
            elif choice == '5':
                print(f"\n{Colors.BLUE}📈 Dashboard: src/models/metrics_dashboard.py{Colors.END}")
            elif choice == '6':
                print(f"\n{Colors.YELLOW}🚀 Deploy script: scripts/deploy/update_indicator.sh{Colors.END}")
            elif choice == '7':
                print(f"\n{Colors.BLUE}📖 Documentation: docs/{Colors.END}")
            elif choice == '8':
                print(f"\n{Colors.YELLOW}🧹 Run: git clean -fd && find . -name '*.pyc' -delete{Colors.END}")
            else:
                print(f"{Colors.RED}❌ Invalid choice. Please try again.{Colors.END}")
            
            if choice != '0':
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='NinjaTrader 8 ML Strategy System Launcher')
    parser.add_argument('--health-check', action='store_true', help='Run health check only')
    
    args = parser.parse_args()
    
    launcher = SystemLauncher()
    launcher.print_header()
    
    if args.health_check:
        launcher.run_health_check()
    else:
        # Run initial health check
        health_ok = launcher.run_health_check()
        
        if health_ok:
            launcher.main_loop()
        else:
            print(f"\n{Colors.RED}⚠️ System health check failed. Please resolve issues before using the launcher.{Colors.END}")
            print(f"{Colors.YELLOW}💡 Run 'python launcher.py --health-check' for detailed diagnostics.{Colors.END}")

if __name__ == "__main__":
    main() 