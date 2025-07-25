"""
Bulk backtest runner that tests all combinations of constants.
"""

from datetime import datetime
import pandas as pd
from run_ma import run_single_backtest

# Constants to test
SHORT_PERIOD_OPTIONS = [5, 10, 20, 30, 60]
LONG_PERIOD_OPTIONS = [30, 60, 90, 150, 240, 300]
STRATEGY_DURATION_OPTIONS = [3, 5, 8, 10]
MAX_POSITIONS_OPTIONS = [3, 4, 5, 6]
TREND_PERSIST_PERIOD_OPTIONS = [0, 5, 10, 20, 30, 60]
# SHORT_PERIOD_OPTIONS = [10]
# LONG_PERIOD_OPTIONS = [60]
# STRATEGY_DURATION_OPTIONS = [8]
# MAX_POSITIONS_OPTIONS = [4]
# TREND_PERSIST_PERIOD_OPTIONS = [10]

def run_bulk_backtest():
    """
    Run bulk backtest with all combinations of constants.
    """
    print("Starting bulk backtest...")
    print(f"Testing combinations:")
    print(f"- SHORT_PERIOD: {SHORT_PERIOD_OPTIONS}")
    print(f"- LONG_PERIOD: {LONG_PERIOD_OPTIONS}")
    print(f"- STRATEGY_DURATION: {STRATEGY_DURATION_OPTIONS}")
    print(f"- MAX_POSITIONS: {MAX_POSITIONS_OPTIONS}")
    print(f"- TREND_PERSIST_PERIOD: {TREND_PERSIST_PERIOD_OPTIONS}")
    print(f"Total combinations: {len(SHORT_PERIOD_OPTIONS) * len(LONG_PERIOD_OPTIONS) * len(STRATEGY_DURATION_OPTIONS) * len(MAX_POSITIONS_OPTIONS) * len(TREND_PERSIST_PERIOD_OPTIONS)}")
    
    all_results = []
    total_combinations = len(SHORT_PERIOD_OPTIONS) * len(LONG_PERIOD_OPTIONS) * len(STRATEGY_DURATION_OPTIONS) * len(MAX_POSITIONS_OPTIONS) * len(TREND_PERSIST_PERIOD_OPTIONS)
    current_combination = 0
    
    for short_period in SHORT_PERIOD_OPTIONS:
        for long_period in LONG_PERIOD_OPTIONS:
            if long_period < short_period:
                continue
            for trend_persist_period in TREND_PERSIST_PERIOD_OPTIONS:
                if trend_persist_period > long_period:
                    continue
                for strategy_duration in STRATEGY_DURATION_OPTIONS:
                    for max_positions in MAX_POSITIONS_OPTIONS:
                        if max_positions > strategy_duration:
                            continue
                        current_combination += 1
                        print(f"Running combination {current_combination}/{total_combinations}: "
                            f"SHORT_PERIOD={short_period}, "
                            f"LONG_PERIOD={long_period}, "
                            f"TREND_PERSIST_PERIOD={trend_persist_period}, "
                            f"DURATION={strategy_duration}, "
                            f"MAX_POSITIONS={max_positions}")
                        
                        result = run_single_backtest(short_period, long_period, strategy_duration, max_positions, trend_persist_period)
                        if result:
                            result['results'] = None
                            all_results.append(result)
                            print(f"  -> Trades: {result['total_trades']}, "
                                f"Profit: {result['total_profit']:.2f}, "
                                f"Amount: {result['total_amount']:.2f}, "
                                f"ROI: {result['roi']:.2f}")
                        else:
                            print(f"  -> Failed")
    # Create summary DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by total profit
    df_sorted = df.sort_values('roi', ascending=False)
    
    print("\n" + "="*80)
    print("BULK BACKTEST RESULTS")
    print("="*80)
    
    print("\nTop 10 Best Performing Combinations:")
    print(df_sorted[['short_period', 'long_period', 'trend_persist_period', 'strategy_duration', 'max_positions', 
                     'total_trades', 'total_profit', 'total_amount', 'avg_profit', 'roi']].head(10))
    
    print("\nSummary Statistics:")
    print(f"Total combinations tested: {len(df)}")
    print(f"Combinations with trades: {len(df[df['total_trades'] > 0])}")
    print(f"Best profit: {df['total_profit'].max():.2f}")
    print(f"Worst profit: {df['total_profit'].min():.2f}")
    print(f"Average profit: {df['total_profit'].mean():.2f}")
    print(f"Best ROI: {df['roi'].max():.2f}")
    print(f"Worst ROI: {df['roi'].min():.2f}")
    print(f"Average ROI: {df['roi'].mean():.2f}")
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"ma_bulk_backtest_results_{timestamp}.csv"
    df_sorted.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")
    
    return df_sorted

if __name__ == "__main__":
    run_bulk_backtest() 