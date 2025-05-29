# Auto positioning strategy

## Traits
- Create one or multiple pair of positions which calls the opposite directions.
- All positions are set to close at the same time.
- We aim to form a valid price range at which the closing price could be. The inner the price closes at, the more profit would come.

## Issues
- Possible failure to form any valid range. Price could keep sitting on the opposite side of the starting position, so we have make sure the price movement would exactly go the direction of the starting position.
- Setting up more pair of positions. Further positions can be built within currently active "range", and timing and monitoring could be crucial to set them up. Moreover, we need to make sure about the possibility of completing the pair positions.
- Percentage of having closing price within "ideal" range with a profit

## Strategy 1. Key-level based approach
This aparoach is used when we can spot an obvious key-level for the last period. Period varies depending on how long you'd like to take for the initial position.

### Steps to create positions

1. Identify key-level and draw the line
2. Set the position length in which
   - one cycle of flucation could be completed
   - key-level line can be included
3. Create a position towards the key-level with as big margin as possible. Ideally, it can be created when the current price is deviated as far as possible. But be sure it struggles to break out in the opposite direction of key-level.
4. Check the price movement and create a position towards the key-level with opposite direction as to the first one. This should be created before the mid-length of the first position. We need some tactics to select the best high/low point in order to create a maximum position range between 2 positions we created.
5. Wait until the positions close. Hopefully, the price keeps fluctuating around key-level and ends up inside the range.


## Strategy 2. Trend-line based approach
This approach is used when there is a clear uptrend.

### Steps to create positions

1. Draw the trend line. The up/down trend should be clear at least for the next cycle for which your position will be running.
2. Create a starting position at the base line. This should be near as possible as the trend line, even better beyond it with extra margin.
3. Check the price movement and create a position with opposite direction. Maximize the range as possible by picking up the furtherest point. Ideally, this can be picked before mid-length of the first position. If there isn't a ideal/possible pick, try until the final expiration to pick any position to form a valid range.
4. Wait until the positions close.

## Strategy 3. Local support/resistance based
[TO EDIT]

## Strategy 4. Min-risk based

1. Identify upcoming price range using AI models
2. Create a starting position with a likely CALL from above report
3. Every 10 seconds (set interval), do the following:
 - Get current price
 - Select X (CALL/PUT), Y (amount) to minimize the loss in every possible case (this can be restricted to nearest N areas)
 - Create a position with the figured X/Y. Skip it unless there is no proper X/Y pair found

## Dev task
[DONE] - Save/Restore settings
[DONE] - Open Trades->Opened tab
[DONE] - Check Has Active Trades Going On
[DONE] - Switch to currency pair with higher payout
[DONE] - Get current positions (with current price)
[DONE] - Create position
[DONE] - Figure out X/Y pair (use brute force)
[DONE] - Create starting position via Market Sentiment
[DONE]- Bug fix on switching currency pair (Not working on 2nd attempt) :: Not perfect, but working
[IGNORE] - Bug fix: Uncaught Error: Extension context invalidated.
- Automatic browser reload after X time (including re-start the run function)
- Create starting position via RSI
- BUG: changing pair would break the session, only change it when there is no active position
- CRUCIAL: Pick the correct direction & duration that you can gurantee 100% pairing

## Estimated outcome

### Case 1: 2 positions, Price inside the range

You get the profit from both positions.\
Initial: 200, Outcome: 192+192=384, Profit: +184\
Percentage: 10%

### Case 2: 2 positions, Price outside the range

You lose a small as the difference between profit/loss from both positions.\
Initial: 200, Outcome: 192, Loss: -8\
Percentage: 80%

### Case 3: 1 position
You lose the initial amount of position.\
Initial: 100, Outcome: 0, Loss: -100\
Percentage: 10%

### Overall outcome
With 10 shots:\
184 * 1 + (-8) * 8 + (-100) * 1 = 20


## Going-Live

1. Check Demo/Live Environment
2. Make sure to set AMOUNT LIMIT


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