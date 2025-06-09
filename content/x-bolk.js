// BOLK: Conditional 3-position strategy
// Position amount: 1/2/2 OR 1/2/1
const DEFAULT_SETTINGS = {
    enabled: false,
    defaultAmount: 1,
    defaultDuration: 10,
    maxPositionLimit: 3,
    maxPositionAmount: 2,
    interval: 10000,
    defaultDirection: 'BUY',
    maxPriceDifference: 0,
    previousRestart: (new Date()) * 1,
}

function createStartingPosition(settings) {
    // TODO: Use RSI to create dynamic starting positions (change favor direction: settings.defaultDirection)
    const buyPositionAmount = settings.defaultAmount;
    const sellPositionAmount = settings.defaultAmount * 2;
    const newPositionDuration = settings.defaultDuration;

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
            console.log('Restarting ...');
            window.location.reload();
            return;
        }

        console.log('[cSP] Position duration set', newPositionDuration);

        const favorDirection = settings.defaultDirection;
        const oppositeDirection = favorDirection === 'BUY' ? 'SELL' : 'BUY';
        createPosition(buyPositionAmount, favorDirection, () => {
            console.log('[cSP] Position created', buyPositionAmount, favorDirection);
            
            createPosition(sellPositionAmount, oppositeDirection, () => {
                console.log('[cSP] Position created', sellPositionAmount, oppositeDirection);
            });
        });
    });
}

function calculateNextPosition(ps, price, newProfit, settings) {
    const positions = ps;
    const [minutes, seconds] = positions[0]['timeLeft'].split(':').map(Number);
    const secondsLeft = minutes * 60 + seconds;

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
            console.log('[cNP] MAX UPDATE:', priceDifference, settings.maxPriceDifference);
            settings.maxPriceDifference = priceDifference;
            saveSettings(settings);
        }
        return null;
    } else if (secondsLeft > settings.defaultDuration * 60 * 2 / 10) {
        if (!priceInFavor) {
            console.log('[cNP] Price not in favor, waiting for break out');
            return null;
        }

        console.log('[cNP] CHECK BREAK:', priceDifference, settings.maxPriceDifference);
        if (settings.maxPriceDifference > priceDifference) {
            return null;
        }

        newPositionDirection = favorDirection;
        newPositionAmount = settings.defaultAmount * 2;
    }

    if (newPositionAmount && newPositionDirection) {
        return {
            amount: newPositionAmount,
            direction: newPositionDirection,
            profit: newProfit,
            openPrice: price,
        }
    }

    return null;
}