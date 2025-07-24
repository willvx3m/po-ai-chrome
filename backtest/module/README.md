"""
This is the backtest run module.
Steps:
0. define constants: json path
0. define functions: is_trend_line_broken, run_strategy, is_strategy_finished, calculate_trend_lines
1. read json file which contains an array of candle data (chronologically sorted, candle format: {datetime_point: "%Y-%m-%d %H:%M", open: number, high: number, low: number, close: number})
2. set iterator for time point. start from index 0.
3. define run status: CALCULATING_TREND || CHECKING_TREND_BREAK || RUNNING_STRATEGY
4. loop through the json file
    4.1 get current candle
    4.2 if the status is CALCULATING_TREND, calculate trend lines. if the trend lines are calculated, set the status to CHECKING_TREND_BREAK. if the trend lines are not calculated, continue to the next candle
    4.3 if the status is CHECKING_TREND_BREAK, check if the trend line is broken with current candle. if the trend line is broken, set the status to RUNNING_STRATEGY and start running the strategy. if the trend line is not broken, continue to the next candle.
    4.4 if the status is RUNNING_STRATEGY, continue running the strategy by calling run_strategy function. When the strategy is finished, grab the result and set the status to CALCULATING_TREND.
    4.5 if the iterator is the last candle, ignore any actions and finish the loop.
4. print the result
"""


"""
This is the strategy module.
Define the following functions:

1. define strategy variables:
    - STRATEGY_DURATION: duration of the strategy
    - MAX_POSITIONS: maximum number of positions
2. define function run_strategy
    parameters:
        - direction: "call" or "put"
        - positions: list of positions
        - start_index: index of the start of the strategy
        - current_index: index of the current candle
        - current_candle: current candle
    return:
        - positions: list of positions
        - is_finished: boolean indicating if the strategy is finished
        - total_profit: total profit of the strategy if the strategy is finished
    logic:
        - if the strategy is not started, create a new position at current candle - direction, amount: 1
        - if the strategy is finished, when current_index is equal to start_index + STRATEGY_DURATION, finish the strategy and return the result
        - if the number of positions is greater than MAX_POSITIONS, wait until the strategy is finished
        - if current candle violates the direction onwards the previous position, create a new position at current candle - direction, amount: twice the previous position amount
"""