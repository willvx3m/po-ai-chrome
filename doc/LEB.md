# LEB: Expanding Pair-Positions

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
- So the compounding result > 0 ?? Need full calculation with probability

P3: 20%, 0.58N -> 0.2
P4: 25%, 0.62N -> (1-0.2)*0.25 = 0.2
P5: 30%, 0.66N -> (1-0.2-0.2)*0.3 = 0.18
P6: 35%, 0.7N -> (1-0.2-0.2-0.18)*0.35 = 0.147
OTHER: 1-0.2-0.2-0.18-0.147=0.273

Loss: 0.2*0.58+0.2*0.62+0.18*0.66+0.147*0.7=0.4617N
Profit: 0.273N

N: 2
Session: 50
Loss: 46.17
Profit: 27.3


## Prospect

1. Deposit: 100$

2. Outcome
- Bonus: 100$
- Trading Profit: X

3. Expense (optional)
- KYC: 150$
- VPS, IP: 50$

4. Time Estimate
- A: 10, Avg Position: 5 (3-6) => 50
- Required volume: 100 * 100 = 10000
- Session: 10000 / 50 = 200 ~ 20hrs

5. Outcome: 100 + X - [200]

## Service and Cooperation

1. Sell one-on-one service

2. Co-operate with PO: increase their user base

## Tactic Updates

Speculate price movement before creating a next position. If it stays in "safe" zone, don't rush it.

## Promo Codes

 SUMMER2025: 100%, May 30th - Jun 15th


## Issue

Reload after canvas goes broke

## Report


```
[test4-box]

Found 406 regular positions
=P3=> Count: 76 ( 18.72 % )
=P3=> Failed: 71 ( 93.42 % ) ( 17.49 % )
=P3=> Total amount: 380
=P3=> Total profit: -78.64

=P4=> Count: 138 ( 33.99 % )
=P4=> Failed: 93 ( 67.39 % ) ( 22.91 % )
=P4=> Total amount: 966
=P4=> Total profit: -35.06

=P5=> Count: 98 ( 24.14 % )
=P5=> Failed: 65 ( 66.33 % ) ( 16.01 % )
=P5=> Total amount: 882
=P5=> Total profit: -37.38

=P6=> Count: 93 ( 22.91 % )
=P6=> Failed: 48 ( 51.61 % ) ( 11.82 % )
=P6=> Total amount: 1023
=P6=> Total profit: 22.15

=> Failed: 278 ( 68.47 % )
=> Total amount: 3255
=> Total profit: -129.09
```

```
[test-moz]

Found 200 regular positions
=P3=> Count: 36 ( 18 % )
=P3=> Failed: 35 ( 97.22 % ) ( 17.5 % )
=P3=> Total amount: 180
=P3=> Total profit: -39.92

=P4=> Count: 47 ( 23.5 % )
=P4=> Failed: 42 ( 89.36 % ) ( 21 % )
=P4=> Total amount: 329
=P4=> Total profit: -39.19

=P5=> Count: 40 ( 20 % )
=P5=> Failed: 34 ( 85 % ) ( 17 % )
=P5=> Total amount: 360
=P5=> Total profit: -30.76

=P6=> Count: 77 ( 38.5 % )
=P6=> Failed: 21 ( 27.27 % ) ( 10.5 % )
=P6=> Total amount: 847
=P6=> Total profit: 110.56

=> Failed: 132 ( 66 % )
=> Total amount: 1716
=> Total profit: 0.69
```