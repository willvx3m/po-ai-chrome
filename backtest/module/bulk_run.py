"""
Bulk backtest runner that tests all combinations of constants.
"""

from datetime import datetime
import pandas as pd
from run import run_single_backtest

# Constants to test
MIN_CANDLES_FOR_TREND_OPTIONS = [30, 45, 60, 90, 120]
STRATEGY_DURATION_OPTIONS = [3, 5, 8, 10]
MAX_POSITIONS_OPTIONS = [3, 4, 5, 6]

def run_bulk_backtest():
    """
    Run bulk backtest with all combinations of constants.
    """
    print("Starting bulk backtest...")
    print(f"Testing combinations:")
    print(f"- MIN_CANDLES_FOR_TREND: {MIN_CANDLES_FOR_TREND_OPTIONS}")
    print(f"- STRATEGY_DURATION: {STRATEGY_DURATION_OPTIONS}")
    print(f"- MAX_POSITIONS: {MAX_POSITIONS_OPTIONS}")
    print(f"Total combinations: {len(MIN_CANDLES_FOR_TREND_OPTIONS) * len(STRATEGY_DURATION_OPTIONS) * len(MAX_POSITIONS_OPTIONS)}")
    print("="*80)
    
    all_results = []
    total_combinations = len(MIN_CANDLES_FOR_TREND_OPTIONS) * len(STRATEGY_DURATION_OPTIONS) * len(MAX_POSITIONS_OPTIONS)
    current_combination = 0
    
    for min_candles_for_trend in MIN_CANDLES_FOR_TREND_OPTIONS:
        for strategy_duration in STRATEGY_DURATION_OPTIONS:
            for max_positions in MAX_POSITIONS_OPTIONS:
                if max_positions > strategy_duration:
                    continue
                current_combination += 1
                print(f"Running combination {current_combination}/{total_combinations}: "
                      f"MIN_CANDLES={min_candles_for_trend}, "
                      f"DURATION={strategy_duration}, "
                      f"MAX_POSITIONS={max_positions}")
                
                result = run_single_backtest(min_candles_for_trend, strategy_duration, max_positions)
                if result:
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
    print(df_sorted[['min_candles_for_trend', 'strategy_duration', 'max_positions', 
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
    csv_filename = f"bulk_backtest_results_{timestamp}.csv"
    df_sorted.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")
    
    return df_sorted

if __name__ == "__main__":
    run_bulk_backtest() 