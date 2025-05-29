# Transition Manager (maco)

## Terms
N: max positions\
T: position duration\
A: starting position amount\
X: starting price (random choice)
i: count of positions, between 0 and N \
a: position amount, between 1 and 100\
d: position direction, BUY or SELL
x: open price of position
PS(i): collection of all position sets of i positions\
PS(i, j): jth item in PS(i)\
R(i, j): set of ranges of PS(i, j)\

## Transition

### How to PS(i) -> PS(i+1)
PS(i+1) = PS(i+1, 0) + PS(i+1, 1) + ...

### How to PS(i, j) -> PS(i+1, j)
PS(i+1, j) = { p(k) }\
p(k) = (a, x, d) where\
-> a: 1-100\
-> x: random price in every r in R(i, j) (choose a middle price)\
-> d: BUY or SELL\

In here, k is between `0` ~ `100 * 2 * length(R(i, j)) - 1`

### Init data
PS(0) = Empty
PS(1) = { (A, X, 'BUY') }


### Metrics of a position set
MIN(PS(i, j)) = min({ outcome(x, PS(i, j)) }) where x in R(i, j)
MAX(PS(i, j)) = max({ outcome(x, PS(i, j)) }) where x in R(i, j)
AVG(PS(i, j)) = avg({ outcome(x, PS(i, j)) }) where x in R(i, j)

## Target
Figure out N, A, and consequent (a, x, d) for every scenario where\
- final metrics of all positions sets go positive

## Assumption
Following vars are selected to ensure dead start where you can't create a pairing 2nd positon.\
- X open price, naturally current price\
- D direction, refer to RSI or Market sentiments\
- T: position duration
