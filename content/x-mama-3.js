// SMA-based direction + Martingale
// No max spike check
const DEFAULT_SETTINGS = {
    userName: 'Unknown',
    name: 'MAMA 3.0',
    enabled: false,
    defaultAmount: 1,
    multiplier: 1,
    defaultDuration: 1,
    maxPositionLimit: 4,
    maxPositionAmount: 8,
    interval: 10000,
    defaultDirection: 'BUY',
    previousRestart: (new Date()) * 1,
    smaSampleCount: 6, // Default: 60
    minPayout: 80, // considered non-OTC pairs
    includeOTC: true, // include OTC pairs
}

function getSMA(prices, smaSampleCount) {
    if (!prices || prices.length < smaSampleCount) {
        return 0;
    }

    const sum = prices.reduce((sum, price) => sum + price, 0);
    return sum / prices.length;
}

function createStartingPosition(settings) {
    if (!settings.priceBook.book || settings.priceBook.book.length < settings.smaSampleCount) {
        console.log('[cSP] Price book not ready, skipping');
        return;
    }

    const sma = getSMA(settings.priceBook.book, settings.smaSampleCount);
    const lastPrice = settings.priceBook.book[settings.priceBook.book.length - 1];
    settings.defaultDirection = sma < lastPrice ? 'BUY' : 'SELL';
    console.log('[cSP] SMA:', sma, 'Last price:', lastPrice, 'Default direction:', settings.defaultDirection);

    const newPositionAmount = settings.defaultAmount;
    const newPositionDuration = settings.defaultDuration;
    const newPositionDirection = settings.defaultDirection;

    settings.maxPriceDifference = 0;
    saveSettings(settings);

    setEndTime(newPositionDuration, () => {
        // Check end time
        // This resolves the failure to create 2 starting positions
        var endTime = getEndTime();
        if (endTime) {
            const now = new Date();
            endTime = `${now.getFullYear()}-${now.getMonth() + 1}-${now.getDate()} ${endTime}`;
        }
        const newPositionSeconds = Math.abs(new Date(endTime) - new Date()) / 1000;
        if (!newPositionSeconds || newPositionSeconds < settings.defaultDuration * 60 - 30 || newPositionSeconds > settings.defaultDuration * 60 + 60) {
            console.log(`[cSP] Duration (${newPositionSeconds}s) is too short/long, `, 'EndTime:', endTime);
            // console.log('Restarting ...');
            // window.location.reload();
            return;
        }

        console.log('[cSP] Position duration set', newPositionDuration);
        createPosition(newPositionAmount, newPositionDirection, () => {
            console.log('[cSP] Position created', newPositionAmount, newPositionDirection);
        });
    });
}

function calculateNextPosition(ps, price, newProfit, settings) {
    const positions = ps.sort((a, b) => a.amount - b.amount);
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

    return {
        amount: newPositionAmount,
        direction: settings.defaultDirection,
        profit: newProfit
    }
}