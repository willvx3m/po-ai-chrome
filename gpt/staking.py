def martingale_stake(prev_trade_result, prev_stake, base_stake, multiplier=2):
    """
    Martingale logic:
    - If last trade lost → multiply stake by `multiplier`
    - If last trade won → reset to `base_stake`
    """
    if prev_trade_result is None:  # First trade
        return base_stake
    elif prev_trade_result:  # Win → reset
        return base_stake
    else:  # Loss → increase
        return prev_stake * multiplier
