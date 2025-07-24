# Strategy variables
STRATEGY_DURATION = 5  # Duration of the strategy in candles
MAX_POSITIONS = 3  # Maximum number of positions

def run_strategy(direction, positions, start_index, current_index, current_candle):
    """
    Run the trading strategy from start to finish.
    
    Args:
        direction: "call" or "put" - the initial direction of the strategy
        positions: list of positions - current positions in the strategy
        start_index: index of the start of the strategy
        current_index: index of the current candle
        current_candle: current candle data
    
    Returns:
        tuple: (positions, is_finished, total_profit)
    """
    # Check if strategy duration has been reached
    if current_index >= start_index + STRATEGY_DURATION:
        # Strategy finished, calculate total profit
        total_profit, total_amount = calculate_total_profit(positions, current_candle)
        return positions, True, total_profit, total_amount
    
    # If no positions exist, determine direction and create the first position
    if not positions:
        if direction is None:
            print("No direction provided, cannot create a new position")
            return positions, False, 0, 0
        
        new_position = create_position(direction, 1, current_candle, current_index)
        positions.append(new_position)
        return positions, False, 0, 0
    
    # Check if we've reached maximum positions
    if len(positions) >= MAX_POSITIONS:
        # Wait until strategy duration is reached
        return positions, False, 0, 0
    
    # Check if current candle violates the direction of the previous position
    previous_position = positions[-1]
    if violates_direction(previous_position, current_candle):
        # Create a new position with twice the amount, same direction
        new_amount = previous_position["amount"] * 2
        new_direction = previous_position["direction"]
        new_position = create_position(new_direction, new_amount, current_candle, current_index)
        positions.append(new_position)
    
    return positions, False, 0, 0

def create_position(direction, amount, candle, candle_index):
    """
    Create a new position.
    
    Args:
        direction: "call" or "put"
        amount: position amount
        candle: candle data
        candle_index: index of the candle
    
    Returns:
        dict: position data
    """
    return {
        "direction": direction,
        "amount": amount,
        "entry_price": (candle["low"] + candle["high"]) / 2,
        "entry_candle_index": candle_index,
        "entry_time": candle["datetime_point"],
    }

def violates_direction(position, current_candle):
    """
    Check if the current candle violates the direction of the position.
    
    Args:
        position: position data
        current_candle: current candle data
    
    Returns:
        bool: True if direction is violated, False otherwise
    """
    current_price = (current_candle["low"] + current_candle["high"]) / 2
    entry_price = position["entry_price"]
    
    if position["direction"] == "call":
        # For call position, violation occurs when price goes down
        return current_price < entry_price
    elif position["direction"] == "put":
        # For put position, violation occurs when price goes up
        return current_price > entry_price
    
    return False

def calculate_total_profit(positions, final_candle):
    """
    Calculate the total profit of all positions.
    
    Args:
        positions: list of positions
        final_candle: final candle data
    
    Returns:
        float: total profit
    """
    total_profit = 0
    total_amount = 0
    for position in positions:
        # Calculate profit for open positions
        current_price = (final_candle["high"] + final_candle["low"]) / 2
        entry_price = position["entry_price"]
        
        if position["direction"] == "call":
            if entry_price < current_price:
                # Profit for call position
                total_profit += 0.9 * position["amount"]
            elif entry_price > current_price:
                # Loss for call position
                total_profit -= position["amount"]
        elif position["direction"] == "put":
            if entry_price > current_price:
                # Profit for put position
                total_profit += 0.9 * position["amount"]
            elif entry_price < current_price:
                # Loss for put position
                total_profit -= position["amount"]
        total_amount += position["amount"]
    
    return total_profit, total_amount
    """
    Get a summary of the strategy execution.
    
    Args:
        positions: list of positions
        final_candle: final candle data
    
    Returns:
        dict: strategy summary
    """
    total_profit = calculate_total_profit(positions, final_candle)
    
    return {
        "total_positions": len(positions),
        "total_profit": total_profit,
        "positions": positions,
        "final_candle": final_candle
    }