import json
import pandas as pd
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from prediction import PricePredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelMonitor:
    """Monitor trained model performance and data drift."""
    
    def __init__(self, model_dir='./'):
        self.model_dir = model_dir
        self.predictor = PricePredictor(model_dir)
        self.performance_history = []
        
    def load_performance_history(self, history_file='performance_history.json'):
        """Load historical performance data."""
        try:
            with open(history_file, 'r') as f:
                self.performance_history = json.load(f)
            logging.info(f"Loaded {len(self.performance_history)} performance records")
        except FileNotFoundError:
            logging.info("No performance history found, starting fresh")
            self.performance_history = []
    
    def save_performance_history(self, history_file='performance_history.json'):
        """Save performance history to file."""
        with open(history_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2, default=str)
        logging.info(f"Saved performance history with {len(self.performance_history)} records")
    
    def prepare_test_data(self, data_path, test_period_days=7, lookback=300):
        """Prepare recent data for testing model performance."""
        try:
            # Load data
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            df['datetime_point'] = pd.to_datetime(df['datetime_point'])
            df = df.sort_values('datetime_point').reset_index(drop=True)
            
            # Get recent data for testing
            end_date = df['datetime_point'].max()
            start_date = end_date - timedelta(days=test_period_days)
            
            test_df = df[df['datetime_point'] >= start_date].copy()
            
            logging.info(f"Prepared test data: {len(test_df)} points from {start_date} to {end_date}")
            
            return test_df
            
        except Exception as e:
            logging.error(f"Failed to prepare test data: {e}")
            raise
    
    def create_test_sequences(self, df, horizons=[1, 3, 5, 10], lookback=300):
        """Create test sequences and targets from recent data."""
        try:
            # Add technical indicators
            df = self.predictor.add_technical_indicators(df)
            
            # Handle NaN values
            df[self.predictor.features] = df[self.predictor.features].ffill()
            df = df.dropna(subset=self.predictor.features)
            
            # Scale data
            scaled_data = np.clip(self.predictor.scaler.transform(df[self.predictor.features]), -1e5, 1e5)
            
            # Create sequences and targets
            max_h = max(horizons)
            num_sequences = len(df) - lookback - max_h + 1
            
            if num_sequences <= 0:
                raise ValueError(f"Insufficient data for sequences: need {lookback + max_h} points")
            
            sequences = np.zeros((num_sequences, lookback, len(self.predictor.features)), dtype=np.float32)
            targets = {h: [] for h in horizons}
            
            for i in range(num_sequences):
                sequences[i] = scaled_data[i:i + lookback, :]
            
            for h in horizons:
                for i in range(num_sequences):
                    future_price = df['close'].iloc[i + lookback - 1 + h]
                    current_price = df['close'].iloc[i + lookback - 1]
                    targets[h].append(1 if future_price > current_price else 0)
            
            return sequences, targets
            
        except Exception as e:
            logging.error(f"Failed to create test sequences: {e}")
            raise
    
    def evaluate_model_performance(self, data_path, horizons=[1, 3, 5, 10], lookback=300):
        """Evaluate model performance on recent data."""
        try:
            # Load models
            self.predictor.load_models(horizons)
            
            # Prepare test data
            test_df = self.prepare_test_data(data_path)
            test_sequences, test_targets = self.create_test_sequences(test_df, horizons, lookback)
            
            performance_results = {}
            
            for horizon in horizons:
                if horizon not in self.predictor.models:
                    logging.warning(f"Model for horizon {horizon}min not available")
                    continue
                
                model = self.predictor.models[horizon]
                y_true = np.array(test_targets[horizon])
                
                # Get predictions
                predictions = model.predict(test_sequences, verbose=0)
                y_pred = (predictions > 0.5).astype(int).flatten()
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # Calculate directional accuracy (more relevant for trading)
                directional_accuracy = np.mean(y_true == y_pred)
                
                # Calculate confidence distribution
                confidence_scores = np.maximum(predictions.flatten(), 1 - predictions.flatten())
                avg_confidence = np.mean(confidence_scores)
                high_conf_accuracy = accuracy_score(
                    y_true[confidence_scores > 0.7],
                    y_pred[confidence_scores > 0.7]
                ) if np.sum(confidence_scores > 0.7) > 0 else 0
                
                performance_results[horizon] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'directional_accuracy': directional_accuracy,
                    'avg_confidence': avg_confidence,
                    'high_conf_accuracy': high_conf_accuracy,
                    'num_predictions': len(y_true),
                    'positive_rate': np.mean(y_true),
                    'prediction_rate': np.mean(y_pred)
                }
                
                logging.info(f"Horizon {horizon}min: Accuracy={accuracy:.3f}, "
                           f"Precision={precision:.3f}, Confidence={avg_confidence:.3f}")
            
            return performance_results
            
        except Exception as e:
            logging.error(f"Performance evaluation failed: {e}")
            raise
    
    def detect_performance_degradation(self, current_performance, threshold=0.05):
        """Detect if model performance has degraded compared to historical performance."""
        alerts = []
        
        if not self.performance_history:
            logging.info("No historical performance data for comparison")
            return alerts
        
        # Get recent average performance
        recent_records = self.performance_history[-5:]  # Last 5 evaluations
        
        for horizon in current_performance:
            current_acc = current_performance[horizon]['directional_accuracy']
            
            # Calculate historical average
            historical_accs = [
                record['performance'][str(horizon)]['directional_accuracy']
                for record in recent_records
                if str(horizon) in record['performance']
            ]
            
            if historical_accs:
                historical_avg = np.mean(historical_accs)
                performance_drop = historical_avg - current_acc
                
                if performance_drop > threshold:
                    alert = {
                        'horizon': horizon,
                        'current_accuracy': current_acc,
                        'historical_average': historical_avg,
                        'performance_drop': performance_drop,
                        'severity': 'HIGH' if performance_drop > 0.1 else 'MEDIUM'
                    }
                    alerts.append(alert)
                    
                    logging.warning(f"Performance degradation detected for {horizon}min model: "
                                  f"Current={current_acc:.3f}, Historical={historical_avg:.3f}, "
                                  f"Drop={performance_drop:.3f}")
        
        return alerts
    
    def analyze_prediction_patterns(self, data_path, horizons=[1, 3, 5, 10]):
        """Analyze recent prediction patterns for insights."""
        try:
            # Get recent predictions
            predictions = self.predictor.predict_from_file(data_path)
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'consensus_direction': None,
                'confidence_levels': {},
                'horizon_agreement': {},
                'market_bias': None
            }
            
            # Analyze consensus
            up_predictions = sum(1 for p in predictions.values() if p['direction'] == 'UP')
            total_predictions = len(predictions)
            
            analysis['consensus_direction'] = 'BULLISH' if up_predictions > total_predictions / 2 else 'BEARISH'
            analysis['consensus_strength'] = abs(up_predictions - total_predictions / 2) / (total_predictions / 2)
            
            # Analyze confidence levels
            for horizon, pred in predictions.items():
                analysis['confidence_levels'][horizon] = pred['confidence']
            
            # Check for market bias (all models agreeing)
            if up_predictions == total_predictions:
                analysis['market_bias'] = 'STRONG_BULLISH'
            elif up_predictions == 0:
                analysis['market_bias'] = 'STRONG_BEARISH'
            else:
                analysis['market_bias'] = 'MIXED'
            
            return analysis
            
        except Exception as e:
            logging.error(f"Pattern analysis failed: {e}")
            return None
    
    def generate_performance_report(self, performance_results, alerts, pattern_analysis):
        """Generate a comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance': performance_results,
            'alerts': alerts,
            'pattern_analysis': pattern_analysis,
            'recommendations': []
        }
        
        # Generate recommendations
        if alerts:
            if any(alert['severity'] == 'HIGH' for alert in alerts):
                report['recommendations'].append("IMMEDIATE: Consider retraining models with HIGH severity alerts")
            if len(alerts) >= 2:
                report['recommendations'].append("WARNING: Multiple models showing degradation")
        
        # Check overall performance
        avg_accuracy = np.mean([perf['directional_accuracy'] for perf in performance_results.values()])
        if avg_accuracy < 0.52:
            report['recommendations'].append("CRITICAL: Overall accuracy below random chance")
        elif avg_accuracy < 0.55:
            report['recommendations'].append("CAUTION: Low overall accuracy")
        
        return report
    
    def create_performance_plots(self, output_dir='./plots/'):
        """Create visualization plots for model performance."""
        try:
            Path(output_dir).mkdir(exist_ok=True)
            
            if not self.performance_history:
                logging.warning("No performance history available for plotting")
                return
            
            # Extract performance data
            dates = [datetime.fromisoformat(record['timestamp']) for record in self.performance_history]
            horizons = list(self.performance_history[0]['performance'].keys())
            
            # Performance over time plot
            plt.figure(figsize=(12, 8))
            
            for horizon in horizons:
                accuracies = [
                    record['performance'][horizon]['directional_accuracy']
                    for record in self.performance_history
                    if horizon in record['performance']
                ]
                
                if accuracies and len(accuracies) == len(dates):
                    plt.plot(dates, accuracies, label=f'{horizon}min', marker='o')
            
            plt.title('Model Performance Over Time')
            plt.xlabel('Date')
            plt.ylabel('Directional Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/performance_over_time.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Latest performance comparison
            if self.performance_history:
                latest = self.performance_history[-1]['performance']
                
                plt.figure(figsize=(10, 6))
                horizons_list = list(latest.keys())
                accuracies = [latest[h]['directional_accuracy'] for h in horizons_list]
                confidences = [latest[h]['avg_confidence'] for h in horizons_list]
                
                x = np.arange(len(horizons_list))
                width = 0.35
                
                plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
                plt.bar(x + width/2, confidences, width, label='Avg Confidence', alpha=0.8)
                
                plt.xlabel('Horizon (minutes)')
                plt.ylabel('Score')
                plt.title('Latest Model Performance by Horizon')
                plt.xticks(x, [f'{h}min' for h in horizons_list])
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/latest_performance.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            logging.info(f"Performance plots saved to {output_dir}")
            
        except Exception as e:
            logging.error(f"Failed to create plots: {e}")
    
    def run_monitoring_cycle(self, data_path, output_file='monitoring_report.json'):
        """Run a complete monitoring cycle."""
        try:
            logging.info("Starting monitoring cycle...")
            
            # Load performance history
            self.load_performance_history()
            
            # Evaluate current performance
            performance_results = self.evaluate_model_performance(data_path)
            
            # Detect degradation
            alerts = self.detect_performance_degradation(performance_results)
            
            # Analyze patterns
            pattern_analysis = self.analyze_prediction_patterns(data_path)
            
            # Generate report
            report = self.generate_performance_report(performance_results, alerts, pattern_analysis)
            
            # Save report
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Update performance history
            self.performance_history.append(report)
            self.save_performance_history()
            
            # Create plots
            self.create_performance_plots()
            
            # Log summary
            logging.info(f"Monitoring cycle complete:")
            logging.info(f"  - Models evaluated: {len(performance_results)}")
            logging.info(f"  - Alerts generated: {len(alerts)}")
            logging.info(f"  - Report saved to: {output_file}")
            
            if alerts:
                logging.warning(f"Performance alerts detected:")
                for alert in alerts:
                    logging.warning(f"  - {alert['horizon']}min: {alert['severity']} "
                                  f"(drop: {alert['performance_drop']:.3f})")
            
            return report
            
        except Exception as e:
            logging.error(f"Monitoring cycle failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Model Performance Monitoring and Analysis')
    parser.add_argument('--model_dir', default='./', help='Directory containing trained models')
    parser.add_argument('--data_path', default='./eurusd.json', help='Path to data file')
    parser.add_argument('--output', default='monitoring_report.json', help='Output report file')
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 3, 5, 10], help='Horizons to monitor')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=3600, help='Monitoring interval in seconds (for continuous mode)')
    
    args = parser.parse_args()
    
    try:
        monitor = ModelMonitor(args.model_dir)
        
        if args.continuous:
            import time
            logging.info(f"Starting continuous monitoring every {args.interval} seconds...")
            
            while True:
                try:
                    monitor.run_monitoring_cycle(args.data_path, args.output)
                    time.sleep(args.interval)
                except KeyboardInterrupt:
                    logging.info("Monitoring stopped by user")
                    break
                except Exception as e:
                    logging.error(f"Monitoring cycle error: {e}")
                    time.sleep(60)  # Wait before retrying
        else:
            # Single monitoring run
            report = monitor.run_monitoring_cycle(args.data_path, args.output)
            
            # Print summary
            print("\n" + "="*60)
            print("MODEL MONITORING REPORT")
            print("="*60)
            
            print(f"Timestamp: {report['timestamp']}")
            print(f"Models Evaluated: {len(report['performance'])}")
            
            print("\nPerformance Summary:")
            for horizon, perf in report['performance'].items():
                print(f"  {horizon}min: Accuracy={perf['directional_accuracy']:.3f}, "
                      f"Confidence={perf['avg_confidence']:.3f}")
            
            if report['alerts']:
                print(f"\nAlerts ({len(report['alerts'])}):")
                for alert in report['alerts']:
                    print(f"  - {alert['horizon']}min: {alert['severity']} "
                          f"(performance drop: {alert['performance_drop']:.3f})")
            else:
                print("\nNo performance alerts")
            
            if report['recommendations']:
                print(f"\nRecommendations:")
                for rec in report['recommendations']:
                    print(f"  - {rec}")
            
            print("="*60)
        
    except Exception as e:
        logging.error(f"Monitoring failed: {e}")
        raise

if __name__ == "__main__":
    main() 