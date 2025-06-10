// HOW TO MAKE THIS STRATEGY FILE
// COPY EXISTING STRATEGY FILE FROM CONTENT AND REMOVE ANY ACTIVE CODE
// MODIFY createStartingPosition to accept price and payout
// ADD endsAt to the return value, use Date.now() + newPositionDuration * 60 * 1000
// MODIFY calculateNextPosition to use endsAt
// THEN EXPORT THEM

const DEFAULT_SETTINGS = {
    enabled: false,
    defaultAmount: 1,
    defaultDuration: 100, // * 6
    maxPositionLimit: 3,
    maxPositionAmount: 1,
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
    const firstPosition = positions[0];
    const secondsLeft = Math.abs(positions[0].endsAt - timestamp) / 1000;

    const priceDifference = Math.abs(firstPosition.openPrice - price);
    if (secondsLeft > settings.defaultDuration * 60 * 3 / 4) {
        if (settings.maxPriceDifference < priceDifference) {
            // console.log('[cNP] MAX UPDATE:', priceDifference, settings.maxPriceDifference);
            settings.maxPriceDifference = priceDifference;
        }
        return null;
    } else if (secondsLeft > settings.defaultDuration * 60 / 2) {
        // console.log('[cNP] CHECK BREAK:', priceDifference, settings.maxPriceDifference);
        if (settings.maxPriceDifference && settings.maxPriceDifference > priceDifference) {
            return null;
        }
    }

    const needBuy = firstPosition.direction === 'SELL' && firstPosition.openPrice > price;
    const needSell = firstPosition.direction === 'BUY' && firstPosition.openPrice < price;

    if (needBuy) {
        return [{
            amount: settings.defaultAmount,
            direction: 'BUY',
            profit: newProfit,
            openPrice: price,
            startsAt: new Date(timestamp),
            endsAt: positions[0].endsAt,
        }];
    } else if (needSell) {
        return [{
            amount: settings.defaultAmount,
            direction: 'SELL',
            profit: newProfit,
            openPrice: price,
            startsAt: new Date(timestamp),
            endsAt: positions[0].endsAt,
        }];
    }

    return null;
}

export { createStartingPosition, calculateNextPosition, DEFAULT_SETTINGS };