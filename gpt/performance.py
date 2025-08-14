def performance_summary(trades_df):
    total_trades = len(trades_df)
    wins = trades_df["win"].sum()
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    net_profit = trades_df["profit"].sum()
    max_drawdown = (trades_df["balance"].cummax() - trades_df["balance"]).max()

    return {
        "Total Trades": total_trades,
        "Wins": wins,
        "Win Rate (%)": round(win_rate, 2),
        "Net Profit": round(net_profit, 2),
        "Max Drawdown": round(max_drawdown, 2)
    }
