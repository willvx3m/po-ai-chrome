## Test Suites

1. May 27th 12:00 - 13:11

MS > 70 -> SELL, MS < 30 -> BUY
StartAmount: 2, Duration 2m (Then likely 1)

2. May 27th 13:30 - 22:50
   
MS > 70 -> BUY, MS < 30 -> SELL
StartAmount: 2, Duration: 3m / 2m (Then likely 1)

3. May 27th 22:55 - 
   
MS > 70 -> BUY, MS < 30 -> SELL
StartAmount: 2, Duration: 10m (Then likely 1)

4. May 29th 13:00 -  

LEB - Pairing position strategy\
Position Amount: 2/2/1/2/2/2, Duration: 5m

5. Jun 2nd 8:14 - 
   
Account: William\
LEB - Pairing position strategy\
Position Amount: 1/1/1/2/3/4, Duration: 5m\
Purpose: maximize profit from latter positions

6. Jun 2nd 9:40 - 
   
Account: Rapig\
Martingale\
Position Amount: 1/2/4/8/16, Duration: 2m, 1m\

Issues:
[DONE] - max cap: 8/16 (after reviewing the volume statistics)
- review profit per cap
[DONE] - flip BUY/SELL on 5/10 failure OR ??

Conduct TEST for 5 hrs (VOLUME reaches 10k = 200 * 50)


6.1 Jun 2nd 12:45PM - 

Resumed test with upgrades
- max cap
- flipping BUY/SELL


6.2 Jun 2nd 11:30 PM -

24hrs TEST

Issues:
- Change pair after straight failures (5/10)
 
7. Jun 4nd 04:18 AM - (Moz)

24hrs TEST - Martingale

START: 44463

Always BUY
Amount limit up to 16

Issues:
- Up-trending it seems to be winning, BUT no solution for a down trend. If DOWN persists, helpless loss

8. Jun 4nd 11:50 AM - (Moz)

24hrs TEST - Leb
START: 44231.19

9. Jun 5th 12:05 PM - (Moz)

Amount: 1, Duration: 4
Strategy: Tri
START: 44106.33

10. Jun 5th 12:05 PM - (Box)

Amount: 1, Duration: 2
Strategy: Tri
START: 49484.85

RESULT: Not enough win Rate, desired 57+%, but below 50%

11. Jun 6th 7:45 AM - (Moz)

STRATEGY: BOLK
Amount: 1/2/2 OR 1/2, Duration: 10m
START: 44068.56

12.  Jun 6th 7:45 AM - (Box)

STRATEGY: BOLK
Amount: 1/2/2 OR 1/2, Duration: 10m
START: 49433.20

13. Jun 6th 13:15 - (Moz)

STRATEGY: DBA
Amount: 4/5, Duration: 10m
START: 44057.16

14.  Jun 6th 13:15 - (Box)

STRATEGY: XOCY
Amount: 2/1/2 OR 2/1, Duration: 5m
START: 49404.40

15.   Jun 7th 13:30 - (Box)

STRATEGY: XOCY 2.0
Amount: 1/2/2 OR 1/2, Duration: 5m
START: 49373.59


16.   Jun 8th 10:40 - (Moz)

STRATEGY: XOCY 2.1 (Slack notification ON)
Amount: 1/2/2 OR 1/2, Duration: 5m
2nd Position: no spike restriction
START: 43976.76

17.    Jun 8th 13:30 - (Box)

STRATEGY: MAMA (Failure)
Amount: 1/4 OR 1, Duration: 4m
START: 

18.     Jun 9th 06:45 - (BOX)
1/2/{2}
STRATEGY: BOLK 2.0
Amount: 1/2/{2*}, Duration: 10m, Max Position: 10
*Position not creating in the last 2m
START: 49295.28

19.      Jun 11th 06:54 - (BOX)
1/2/4/8/16
STRATEGY: Martingale 3.0
Amount: 1/2/4/8/16, Duration: 10m, Max Position: 5
New position after breaking max price difference from last position
START: 49200.15

20.      Jun 12th 08:45 - Jun 14th 12:50 (BOX)
1/2/4/8/16
STRATEGY: Male (LEB + Martingale 3.0)
Amount: 1/2/4/8/16 (both sides), Duration: 10m, Max Position: 9
New position after breaking max price difference from last position
START: 49177.66

21.       Jun 12th 14:00 - (Moz)
1/2/4/8/16
STRATEGY: MAMA (SMA + Martingale 3.0)
Amount: 1/2/4/8/16, Direction: dynamic based on SMA, Duration: 10m, Max Position: 5
New position after breaking max price difference from last position
START: 43902.63

  -> Default Amount -> 5 since Jun 15th 15:25
  -> [ERROR] Blocking max amount (at 8$) until Jun 16th 06:26 -> increase to 80
  -> Increase settings.maxPositionAmount -> 80

  -> [CRITICAL] MAMA was running with max 8$ cap, since DF->5$, it used 80$ limit which caused more damage
   ===> FINAL Formula: Max Position: 4, Max position amount: DF * 8

[BUG] ISSUE: PO closed for some reason (Moz: Around Jun 14th 10:00 AM)
    It kept restarting, but couldn't properly start going.

[WATCH-SPOT] Constant decline between 07:00 ~ 10:00 (00:00 ~ 3:00 CST)

22. Jun 17th 07:00 - (Box)
STRATEGY: MAMA 2.0 (SMA + Martingale 3.0) + Progressive Default Amount

Default Amount: 1/2/4/8 -> Progressive: 1/2/3/4/5/7/9
Multipler: 1 (1/2/3/5)

Duration: 10m, Max Position: 4
START: 48644.66 (=BASE Amount)

23. Jun 19th 11:15 - (Moz)
STRATEGY: MAMA 2.2 (SMA + Martingale 3.0) + Progressive Default Amount
Non-OTC pair with minPayout: 80%

Duration: 10m, Max Position: 4
START: 43593.68 (=BASE Amount)


24. Jun 24th 09:55 - (VPS Blade)
STRATEGY: MAMA 2.2 - no progressive amount, with maxSpike
Include OTC Pair, minPayout: 80%

Duration: 10m, Max Position: 5
SMA Sample count: 6 (Recommended by backtesting, but still using maxSpike and <45s rule)

[IMPORTANT] FIRST VPS RUNNING
[RESULT] FAILED - window doesn't keep repainting; websocket connection closed

START: 49971.20


25. Jul 7th 13:50 - (PC)
STRATEGY: Martin Corpus
User: Box (EUR/USD OTC)

START: 43000.26
END - 42784 ;(
  
FIRST 50mins - incorrect match due to missing EMA51
 -> Update best match
 -> Start after recording 50m