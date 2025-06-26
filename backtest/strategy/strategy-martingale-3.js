// HOW TO MAKE THIS STRATEGY FILE
// COPY EXISTING STRATEGY FILE FROM CONTENT AND REMOVE ANY ACTIVE CODE
// MODIFY createStartingPosition to accept price and payout
// ADD endsAt to the return value, use Date.now() + newPositionDuration * 60 * 1000
// MODIFY calculateNextPosition to use endsAt
// THEN EXPORT THEM

const DEFAULT_SETTINGS = {
    enabled: false,
    defaultAmount: 1,
    defaultDuration: 60, // * 6
    maxPositionLimit: 5,
    maxPositionAmount: 16,
    interval: 10000,
    defaultDirection: 'BUY',
    maxPriceDifference: 0,
    previousRestart: (new Date()) * 1,
}

function createStartingPosition(settings, price, payout, timestamp) {
    const newPositionAmount = settings.defaultAmount;
    const newPositionDuration = settings.defaultDuration;
    const newPositionDirection = settings.defaultDirection;
    const endsAt = new Date(timestamp + newPositionDuration * 60 * 1000);

    settings.maxPriceDifference = 0;

    return [
        {
            amount: newPositionAmount,
            direction: newPositionDirection,
            profit: payout,
            openPrice: price,
            startsAt: new Date(timestamp),
            endsAt: endsAt,
        }
    ];
}

function calculateNextPosition(ps, price, newProfit, settings, timestamp) {
    const positions = ps;
    const lastPosition = positions[positions.length - 1];
    const secondsLeft = Math.abs(positions[0].endsAt - timestamp) / 1000;
    const priceDifference = Math.abs(price - lastPosition.openPrice);
    
    if (positions.length <= 1 && secondsLeft > settings.defaultDuration * 60 * 8 / 10) {
        if (priceDifference > settings.maxPriceDifference) {
            settings.maxPriceDifference = priceDifference;
        }
        return null;
    }

    const needNewPosition = priceDifference > settings.maxPriceDifference && (lastPosition.direction === 'BUY' ? lastPosition.openPrice > price : lastPosition.openPrice < price);
    if(!needNewPosition){
        return null;
    }

    const newPositionAmount = lastPosition.amount * 2;
    if (newPositionAmount > settings.maxPositionAmount) {
        return null;
    }

    return [{
        amount: newPositionAmount,
        direction: settings.defaultDirection,
        profit: newProfit,
        openPrice: price,
        startsAt: new Date(timestamp),
        endsAt: positions[0].endsAt,
    }];
}

export { createStartingPosition, calculateNextPosition, DEFAULT_SETTINGS };