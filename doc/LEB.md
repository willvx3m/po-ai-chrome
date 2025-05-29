# Game: Expanding Pair-Positions (leb)

## Steps

1. Set N: position amount, P: starting price (arbitrary), T: position duration, I: run interval, MAX_PO: max positions
2. Create a BUY and a SELL position: (N, P, T)
3. Set interval run: every I seconds
4. Run:
3.1. Create a position
  - Direction towards P - BUY if price is below P, SELL otherwise
  - Amount: N/2
3.2 Repeat the following until position count reaches MAX_PO: 
    Wait until we can create a pairing position of the previous position
    - Direction: reverse
    - Amount: N

## Math

### Min outcome in several scenarios:

**Assuming MAX_PO = 6**

1. 0.5S (1B 1S) || (1B 1S) 0.5B : -0.58N
2. 0.5S (1B 1S) 1B || 1S (1B 1S) 0.5B : -0.62N
3. 1S (1B 1S) 0.5B 1B || 1S 0.5S (1B 1S) 1B : -0.66N
4. 1S 1S (1B 1S) 0.5B 1B || 1S 0.5S (1B 1S) 1B 1B : -0.7N

**Key Notes**

- Only 1 out of possible `3 ~ MAX_PO` ranges, it will have the negative outcome.
- Other ranges would have positive whose abs is bigger than the above loss. (Sure?)
- The possibility of the price falling into the negative range is 50% (pretty generous ;))
- So the compounding result > 0