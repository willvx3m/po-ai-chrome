#!/usr/bin/env python3
"""
Complete Example Usage of LSTM Price Prediction System

This script demonstrates the full workflow:
1. Training models
2. Making predictions
3. Running trading bot (demo mode)
4. Monitoring performance

Usage:
    python example_usage.py --mode train
    python example_usage.py --mode predict
    python example_usage.py --mode trade
    python example_usage.py --mode monitor
    python example_usage.py --mode all
"""

import subprocess
import sys
import time
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WorkflowManager:
    """Manages the complete AI trading workflow."""
    
    def __init__(self):
        self.scripts = {
            'train': 'run-cpu.py',
            'predict': 'prediction.py',
            'trade': 'trading_bot.py',
            'monitor': 'monitor.py'
        }
        
    def run_command(self, command, check=True):
        """Run a shell command and log output."""
        logging.info(f"Running: {command}")
        try:
            result = subprocess.run(command, shell=True, check=check, 
                                  capture_output=True, text=True)
            if result.stdout:
                logging.info(f"Output: {result.stdout.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed: {e}")
            if e.stdout:
                logging.error(f"Stdout: {e.stdout}")
            if e.stderr:
                logging.error(f"Stderr: {e.stderr}")
            raise
    
    def check_dependencies(self):
        """Check if all required files and dependencies exist."""
        logging.info("Checking dependencies...")
        
        # Check if data file exists
        if not Path('eurusd.json').exists():
            logging.error("eurusd.json not found! Please ensure you have the data file.")
            return False
        
        # Check if scripts exist
        for script_name, script_file in self.scripts.items():
            if not Path(script_file).exists():
                logging.error(f"{script_file} not found!")
                return False
        
        # Check if virtual environment is active (optional)
        if 'venv' not in sys.executable and 'conda' not in sys.executable:
            logging.warning("No virtual environment detected. Consider using venv_tf_cpu.")
        
        logging.info("All dependencies checked successfully")
        return True
    
    def train_models(self, quick_mode=False):
        """Train the LSTM models."""
        logging.info("=== TRAINING MODELS ===")
        
        if quick_mode:
            # Quick training for testing (1 hour)
            command = "python run-cpu.py --horizons 1 --batch_size 2048 --lookback 120"
        else:
            # Full training (balanced performance)
            command = "python run-cpu.py --horizons 1 3 5 10 --batch_size 1536 --lookback 300"
        
        self.run_command(command)
        logging.info("Model training completed!")
    
    def make_predictions(self):
        """Make predictions using trained models."""
        logging.info("=== MAKING PREDICTIONS ===")
        
        command = "python prediction.py --output predictions.json"
        self.run_command(command)
        
        logging.info("Predictions completed! Check predictions.json for results.")
    
    def run_trading_demo(self, max_cycles=5):
        """Run trading bot in demo mode."""
        logging.info("=== RUNNING TRADING BOT (DEMO) ===")
        
        # Create default config if not exists
        if not Path('trading_config.json').exists():
            self.run_command("python trading_bot.py --create_config")
        
        # Run bot with limited cycles for demo
        command = f"python trading_bot.py --max_iterations {max_cycles}"
        self.run_command(command)
        
        logging.info("Trading demo completed! Check trading_results.json for results.")
    
    def monitor_performance(self):
        """Monitor model performance."""
        logging.info("=== MONITORING PERFORMANCE ===")
        
        command = "python monitor.py --output monitoring_report.json"
        self.run_command(command)
        
        logging.info("Performance monitoring completed! Check monitoring_report.json for results.")
    
    def run_complete_workflow(self, quick_mode=False):
        """Run the complete workflow from training to monitoring."""
        logging.info("🚀 Starting Complete AI Trading Workflow")
        logging.info("=" * 60)
        
        try:
            # Step 1: Train models
            self.train_models(quick_mode)
            time.sleep(2)
            
            # Step 2: Make predictions
            self.make_predictions()
            time.sleep(2)
            
            # Step 3: Run trading demo
            self.run_trading_demo(max_cycles=3)
            time.sleep(2)
            
            # Step 4: Monitor performance
            self.monitor_performance()
            
            logging.info("✅ Complete workflow finished successfully!")
            self.print_results_summary()
            
        except Exception as e:
            logging.error(f"Workflow failed: {e}")
            raise
    
    def print_results_summary(self):
        """Print a summary of all results."""
        print("\n" + "=" * 60)
        print("🎯 WORKFLOW RESULTS SUMMARY")
        print("=" * 60)
        
        # Check for output files
        results_files = [
            ('predictions.json', 'Prediction Results'),
            ('trading_results.json', 'Trading Results'),
            ('monitoring_report.json', 'Performance Report'),
            ('trading_bot.log', 'Trading Log'),
            ('performance_history.json', 'Performance History')
        ]
        
        print("📁 Generated Files:")
        for filename, description in results_files:
            if Path(filename).exists():
                print(f"  ✅ {filename:<25} - {description}")
            else:
                print(f"  ❌ {filename:<25} - {description} (not found)")
        
        # Check for model files
        model_files = [
            'model_horizon_1min.h5',
            'model_horizon_3min.h5', 
            'model_horizon_5min.h5',
            'model_horizon_10min.h5',
            'scaler_common.pkl'
        ]
        
        print("\n🤖 Model Files:")
        for filename in model_files:
            if Path(filename).exists():
                size_mb = Path(filename).stat().st_size / (1024*1024)
                print(f"  ✅ {filename:<25} ({size_mb:.1f} MB)")
            else:
                print(f"  ❌ {filename:<25} (not found)")
        
        print("\n📊 Quick Analysis:")
        
        # Show prediction summary if available
        try:
            import json
            if Path('predictions.json').exists():
                with open('predictions.json', 'r') as f:
                    predictions = json.load(f)
                
                up_count = sum(1 for p in predictions.values() if p['direction'] == 'UP')
                total_count = len(predictions)
                
                print(f"  🔮 Predictions: {up_count}/{total_count} models predict UP")
                
                avg_confidence = sum(p['confidence'] for p in predictions.values()) / total_count
                print(f"  🎯 Average Confidence: {avg_confidence:.1%}")
        
        except Exception as e:
            logging.debug(f"Could not analyze predictions: {e}")
        
        # Show trading summary if available
        try:
            if Path('trading_results.json').exists():
                with open('trading_results.json', 'r') as f:
                    trading_results = json.load(f)
                
                final_balance = trading_results.get('final_balance', 10000)
                total_positions = trading_results.get('total_positions', 0)
                pnl = final_balance - 10000
                
                print(f"  💰 Trading Demo: ${pnl:+.2f} P&L on {total_positions} positions")
        
        except Exception as e:
            logging.debug(f"Could not analyze trading results: {e}")
        
        print("\n🚀 Next Steps:")
        print("  1. Review prediction accuracy in monitoring_report.json")
        print("  2. Analyze trading performance in trading_results.json")
        print("  3. Adjust trading_config.json for your risk preferences")
        print("  4. Run 'python trading_bot.py' for live trading (with real broker integration)")
        print("  5. Set up 'python monitor.py --continuous' for ongoing monitoring")
        
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Complete LSTM Trading System Workflow')
    parser.add_argument('--mode', choices=['train', 'predict', 'trade', 'monitor', 'all'], 
                       default='all', help='Workflow mode to run')
    parser.add_argument('--quick', action='store_true', 
                       help='Use quick mode for training (1 hour vs 4-6 hours)')
    parser.add_argument('--skip_deps', action='store_true',
                       help='Skip dependency checking')
    
    args = parser.parse_args()
    
    workflow = WorkflowManager()
    
    try:
        # Check dependencies
        if not args.skip_deps and not workflow.check_dependencies():
            print("\n❌ Dependency check failed!")
            print("Please ensure:")
            print("  1. eurusd.json data file exists")
            print("  2. All Python scripts are in the current directory")
            print("  3. TensorFlow environment is activated (optional)")
            return 1
        
        # Run selected workflow
        if args.mode == 'train':
            workflow.train_models(args.quick)
        elif args.mode == 'predict':
            workflow.make_predictions()
        elif args.mode == 'trade':
            workflow.run_trading_demo()
        elif args.mode == 'monitor':
            workflow.monitor_performance()
        elif args.mode == 'all':
            workflow.run_complete_workflow(args.quick)
        
        print("\n✅ Workflow completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Workflow interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 