const DEFAULT_SETTINGS = {
    enabled: false,
    name: 'MAMA 3.0',
    defaultAmount: 1,
    defaultDuration: 5, // * 6
    maxPositionLimit: 3,
    maxPositionAmount: 16,
    interval: 10000,
    defaultDirection: 'BUY',
    maxPriceDifference: 0,
    previousRestart: (new Date()) * 1,
    smaSampleCount: 60, // Default: 60
    priceBook: [],
}

function getSMA(prices, smaSampleCount) {
    // if (!prices || prices.length < smaSampleCount) {
    //     return 0;
    // }

    const sum = prices.reduce((sum, price) => sum + price, 0);
    return prices.length > 0 ? sum / prices.length : 0;
}

function createStartingPosition(settings, price, payout, timestamp) {
    const priceBook = settings.priceBook;
    const sma = getSMA(priceBook, settings.smaSampleCount);
    const lastPrice = priceBook.length > 0 ? priceBook[priceBook.length - 1] : 0;
    // settings.defaultDirection = Math.random() > 0.5 ? 'BUY' : 'SELL';
    // settings.defaultDirection = 'BUY';
    settings.defaultDirection = sma < lastPrice ? 'BUY' : 'SELL';
    // settings.defaultDirection = sma > lastPrice ? 'BUY' : 'SELL';

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
    const needNewPosition = lastPosition.direction === 'BUY'
        ? lastPosition.openPrice > price
        : lastPosition.openPrice < price;

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