# MARTINGALE: Self-explanatory

## Steps

1. Set N: position amount, P: starting price (arbitrary), T: position duration, I: run interval, MAX_PO: max positions
2. Create BUY
4. Set interval run: every I seconds
5. Repeat the following until position count reaches MAX_PO or amount reaches MAX_AMOUNT: 
   - If price falls below all positions, create with
    - Direction: BUY
    - Amount: prevAmount * 2