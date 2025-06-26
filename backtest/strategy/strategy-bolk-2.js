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
    maxPositionLimit: 10,
    maxPositionAmount: 2,
    interval: 10000,
    defaultDirection: 'BUY',
    maxPriceDifference: 0,
    previousRestart: (new Date()) * 1,
}

function createStartingPosition(settings, price, payout, timestamp) {
    const buyPositionAmount = settings.defaultAmount;
    const sellPositionAmount = settings.defaultAmount;
    const newPositionDuration = settings.defaultDuration;
    const endsAt = new Date(timestamp + newPositionDuration * 60 * 1000);

    settings.maxPriceDifference = 0;

    return [
        {
            amount: buyPositionAmount,
            direction: 'BUY',
            profit: payout,
            openPrice: price,
            startsAt: new Date(timestamp),
            endsAt: endsAt,
        },
        {
            amount: sellPositionAmount,
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
    const favorDirection = settings.defaultDirection;
    const priceInFavor = favorDirection === 'BUY' ? price < positions[0].openPrice : price > positions[0].openPrice;

    var newPositionAmount;
    var newPositionDirection;

    if (positions.length < 2) {
        // TODO create any pairing position
        return null;
    } else if (secondsLeft > settings.defaultDuration * 60 * 8 / 10) {
        if (priceDifference > settings.maxPriceDifference) {
            // console.log('[cNP] MAX UPDATE:', priceDifference, settings.maxPriceDifference);
            settings.maxPriceDifference = priceDifference;
        }
        return null;
    } else if (secondsLeft > settings.defaultDuration * 60 * 2 / 10) {
        if (!priceInFavor) {
            // console.log('[cNP] Price not in favor, waiting for break out');
            return null;
        }

        // console.log('[cNP] CHECK BREAK:', priceDifference, settings.maxPriceDifference);
        if (settings.maxPriceDifference > priceDifference) {
            return null;
        }

        newPositionDirection = favorDirection;
        newPositionAmount = settings.defaultAmount;
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