// DBA: Upgraded pair-position strategy
// Position amount: N+D/N OR N/N+D
const DEFAULT_SETTINGS = {
    enabled: false,
    defaultAmount: 4,
    defaultDuration: 10,
    maxPositionLimit: 2,
    maxPositionAmount: 5,
    interval: 10000,
    defaultDirection: 'BUY',
    maxPriceDifference: 0,
    previousRestart: (new Date()) * 1,
}

function createStartingPosition(settings) {
    const newPositionAmount = settings.defaultAmount;
    const newPositionDuration = settings.defaultDuration;
    var newPositionDirection;

    const marketSentiment = getMarketSentiment();
    if (parseInt(marketSentiment) > 70) {
        console.log(`[cSP] MSentiment ${marketSentiment} is > 70, creating a BUY position`);
        newPositionDirection = 'BUY';
    } else if (parseInt(marketSentiment) < 30) {
        console.log(`[cSP] MSentiment ${marketSentiment} is < 30, creating a SELL position`);
        newPositionDirection = 'SELL';
    } else {
        console.log(`[cSP] MSentiment is ${marketSentiment}, not creating a position`);
        return;
    }

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
        if (!newPositionSeconds || newPositionSeconds < settings.defaultDuration * 60 - 30 || newPositionSeconds > settings.defaultDuration * 60 + 120) {
            console.log(`[cSP] Duration (${newPositionSeconds}s) is too short/long, `, 'EndTime:', endTime);
            console.log('Restarting ...');
            window.location.reload();
            return;
        }

        console.log('[cSP] Position duration set', newPositionDuration);
        createPosition(newPositionAmount, newPositionDirection, () => {
            console.log('[cSP] Position created', newPositionAmount, newPositionDirection);
        });
    });
}

function calculateNextPosition(ps, price, newProfit, settings) {
    const positions = ps;
    const firstPosition = positions[0];
    const [minutes, seconds] = firstPosition['timeLeft'].split(':').map(Number);
    const secondsLeft = minutes * 60 + seconds;

    const priceDifference = Math.abs(firstPosition.openPrice - price);
    // First quarter, monitor and update max difference
    // Next quarter, create position as soon as it breaks max
    // After half duration, if no 3rd position yet => just create it
    if (secondsLeft > settings.defaultDuration * 60 * 3 / 4) {
        if (settings.maxPriceDifference < priceDifference) {
            console.log('[cNP] MAX UPDATE:', priceDifference, settings.maxPriceDifference);
            settings.maxPriceDifference = priceDifference;
            saveSettings(settings);
        }
        return null;
    } else if (secondsLeft > settings.defaultDuration * 60 / 2) {
        console.log('[cNP] CHECK BREAK:', priceDifference, settings.maxPriceDifference);
        if (settings.maxPriceDifference && settings.maxPriceDifference > priceDifference) {
            return null;
        }
    }

    const needBuy = firstPosition.direction === 'SELL' && firstPosition.openPrice > price;
    const needSell = firstPosition.direction === 'BUY' && firstPosition.openPrice < price;

    if (needBuy) {
        return {
            amount: settings.defaultAmount + 1,
            direction: 'BUY',
            profit: newProfit
        }
    } else if (needSell) {
        return {
            amount: settings.defaultAmount + 1,
            direction: 'SELL',
            profit: newProfit
        }
    }

    return null;
}