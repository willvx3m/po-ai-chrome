// HOW TO MAKE THIS STRATEGY FILE
// COPY EXISTING STRATEGY FILE FROM CONTENT AND REMOVE ANY ACTIVE CODE
// MODIFY createStartingPosition to accept price and payout
// ADD endsAt to the return value, use Date.now() + newPositionDuration * 60 * 1000
// MODIFY calculateNextPosition to use endsAt
// THEN EXPORT THEM

const DEFAULT_SETTINGS = {
    enabled: false,
    defaultAmount: 1,
    defaultDuration: 120, // * 6
    maxPositionLimit: 3,
    maxPositionAmount: 100,
    interval: 10000,
    defaultDirection: 'BUY',
    maxPriceDifference: 0,
    previousRestart: (new Date()) * 1,
}

function createStartingPosition(settings, price, payout, timestamp) {
    const newPositionAmount = settings.defaultAmount;
    const newPositionDuration = settings.defaultDuration;
    const endsAt = new Date(timestamp + newPositionDuration * 60 * 1000);

    settings.maxPriceDifference = 0;
    settings.hitSouth = 0;
    settings.hitNorth = 0;

    return [
        {
            amount: newPositionAmount,
            direction: 'BUY',
            profit: payout,
            openPrice: price,
            startsAt: new Date(timestamp),
            endsAt: endsAt,
        },
        {
            amount: newPositionAmount,
            direction: 'SELL',
            profit: payout,
            openPrice: price,
            startsAt: new Date(timestamp),
            endsAt: endsAt,
        }
    ];
}

function calculateNextPosition(ps, price, newProfit, settings, timestamp) {
    const positions = ps;
    const secondsLeft = Math.abs(positions[0].endsAt - timestamp) / 1000;

    const priceDifference = Math.abs(price - positions[0].openPrice);
    const priceInNorth = price > positions[0].openPrice;

    var newPositionAmount;
    var newPositionDirection;

    if (priceInNorth) {
        settings.hitNorth++;
    } else {
        settings.hitSouth++;
    }

    if (secondsLeft > settings.defaultDuration * 60 * 9 / 10) {
        if (priceDifference > settings.maxPriceDifference) {
            // console.log('[cNP] MAX UPDATE:', priceDifference, settings.maxPriceDifference);
            settings.maxPriceDifference = priceDifference;
        }
        return null;
    } else if (positions.length === 2) { // && secondsLeft > settings.defaultDuration * 60 * 1 / 10) {
        if (priceInNorth) {
            // if(settings.hitSouth > settings.hitNorth){
            if (priceDifference > settings.maxPriceDifference) {
                newPositionAmount = settings.defaultAmount * 15;
                newPositionDirection = 'SELL';
            }
            // }
        } else {
            // if(settings.hitNorth > settings.hitSouth){
            if (priceDifference > settings.maxPriceDifference) {
                newPositionAmount = settings.defaultAmount * 15;
                newPositionDirection = 'BUY';
            }
            // }
        }
    } else if (positions.length === 3 && priceDifference > settings.maxPriceDifference) {
        const totalBuyAmount = positions.reduce((acc, position) => position.direction === 'BUY' ? acc + position.amount : acc, 0);
        const totalSellAmount = positions.reduce((acc, position) => position.direction === 'SELL' ? acc + position.amount : acc, 0);

        if (totalBuyAmount > totalSellAmount && priceInNorth) {
            newPositionAmount = settings.defaultAmount * 10;
            newPositionDirection = 'SELL';
        } else if (totalSellAmount > totalBuyAmount && !priceInNorth) {
            newPositionAmount = settings.defaultAmount * 10;
            newPositionDirection = 'BUY';
        }
    }

    if (newPositionAmount && newPositionDirection) {
        return [{
            amount: newPositionAmount,
            direction: newPositionDirection,
            profit: newProfit,
            openPrice: price,
            startsAt: new Date(timestamp),
            endsAt: positions[0].endsAt,
        }];
    }

    return null;
}

export { createStartingPosition, calculateNextPosition, DEFAULT_SETTINGS };