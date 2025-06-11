// Martingale strategy
// Position amount: 1/2/4/8/16
const DEFAULT_SETTINGS = {
    name: 'Martingale 3.0',
    enabled: false,
    defaultAmount: 1,
    defaultDuration: 10,
    maxPositionLimit: 5,
    maxPositionAmount: 16,
    interval: 10000,
    defaultDirection: 'BUY',
    previousRestart: (new Date()) * 1,
}

function createStartingPosition(settings) {
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
    const [minutes, seconds] = positions[0]['timeLeft'].split(':').map(Number);
    const secondsLeft = minutes * 60 + seconds;
    const priceDifference = Math.abs(price - lastPosition.openPrice);
    if (positions.length <= 1 && secondsLeft > settings.defaultDuration * 60 * 8 / 10) {
        if (priceDifference > settings.maxPriceDifference) {
            console.log('[cNP] MAX UPDATE:', priceDifference, settings.maxPriceDifference);
            settings.maxPriceDifference = priceDifference;
            saveSettings(settings);
        }
        return null;
    }

    console.log('[cNP] CHECK BREAK:', priceDifference, settings.maxPriceDifference);
    const needNewPosition = priceDifference > settings.maxPriceDifference && (lastPosition.direction === 'BUY' ? lastPosition.openPrice > price : lastPosition.openPrice < price);
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