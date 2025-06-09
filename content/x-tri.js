// 3-position strategy
// Position amount: 1/1/1
const DEFAULT_SETTINGS = {
    name: 'TRI',
    enabled: false,
    defaultAmount: 1,
    defaultDuration: 4,
    maxPositionLimit: 3,
    maxPositionAmount: 1,
    interval: 10000,
    defaultDirection: 'BUY',
    maxPriceDifference: 0,
    previousRestart: (new Date()) * 1,
}

function createStartingPosition(settings) {
    const newPositionAmount = settings.defaultAmount;
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
        setPositionAmount(newPositionAmount, () => {
            createPosition(0, 'BUY', () => {
                console.log('[cSP] Position created', newPositionAmount, 'BUY');
            });
            createPosition(0, 'SELL', () => {
                console.log('[cSP] Position created', newPositionAmount, 'SELL');
            });
        });
    });
}

function calculateNextPosition(ps, price, newProfit, settings) {
    const positions = ps;
    const [minutes, seconds] = positions[0]['timeLeft'].split(':').map(Number);
    const secondsLeft = minutes * 60 + seconds;

    const priceDifference = Math.abs(positions[0].openPrice - price);
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

    const needBuy = positions.every(position => position.openPrice > price);
    const needSell = positions.every(position => position.openPrice < price);

    if (needBuy) {
        return {
            amount: settings.defaultAmount,
            direction: 'BUY',
            profit: newProfit
        }
    } else if (needSell) {
        return {
            amount: settings.defaultAmount,
            direction: 'SELL',
            profit: newProfit
        }
    }

    return null;
}