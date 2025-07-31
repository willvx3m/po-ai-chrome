CORPUS 2.0

# MAKE CORPUS

1. Collect following metrics on every time point
 - trend15m, trend1h, trend6h
 - S/R for 15m/1h/6h
 - RSI (15m)
 - candles follow-ups (up to 10m)

2. Write snapshot using following encoding
 - decide encoding formula


# USE CORPUS

1. Take on live(testing) timepoint
2. Find top 3 best matching historical timepoints
 - define distance method to measure similarity
3. Predict virtual follow-up candles by weigh-summing follow-ups of historical data.
 - possible use of RAG
 - addition of media
4. Write 3 variation virtual follow-ups with
 - trend continuation
 - trend reversal
 - rapid fluctuation
5. Simulate strategies with virtual candles
6. Select the strategy with max profit (min risk)

## RUN STRATEGY
