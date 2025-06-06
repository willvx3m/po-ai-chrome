# BOLK: Conditional 3 positions

## Steps

1. Create starting positions (duration: 10m)
   1. Create BUY - amount: 1
   2. Create SELL - amount: 2
2. For the first 3m, update max up/down deivation from starting positions
3. For the next 6m, try creating a BUY in the favorable position (below half)
   1. For the first 3m, IF price falls more than max THEN create a BUY position - amount: 2
   2. For the remaining 3m, IF no position yet THEN create a BUY position as long as possible - amount: 2
4. For the remaining 1m, try creating either a BUY or a SELL position
   1. IF the price is in below half, create a BUY position - amount: 2
   2. IF the price is in above half, create a SELL position - amount: 1

## Math

### Per Area

Win (Upper)	47.96%	
Perfect (Mid)	25.34%	
Fail (Below)	52.04%	
			
### Amount with BUY position (created in below half)
START UP	1		
START DOWN	2		
MID UP	2		
			
Profit Win	0.7		
Profit Perfect	2.6		
Profit Loss	-1.2		
			
Estimate	0.192760181		

### Amount with SELL position (created in above half)

START UP	2		
START DOWN	1		
MID UP	1		
			
Profit Win	1.7		
Profit Perfect	-0.2		
Profit Loss	-2.1		
			
Estimate	-0.758823529		

### Favoring Percentage	

Favor Percent	0.8		
Non-Favor   	0.2		
			
### Profit	
  1 Session: 0.002443439		
	1 hr: 0.012217195 (5 sessions)
  12 hrs: 0.146606335
			
### Volume
  1 Session: 4.8		
	1 hr: 24 (5 sessions)
  12 hrs: 288