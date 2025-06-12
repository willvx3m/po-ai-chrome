// Leb + Martingale 3.0
// Extending martingale on both sides
const DEFAULT_SETTINGS = {
    userName: 'Unknown',
    name: 'Male',
    enabled: false,
    defaultAmount: 1,
    defaultDuration: 10,
    maxPositionLimit: 9,
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
    const positions = ps;
    const buyPositions = ps.filter(p => p.direction === 'BUY').sort((a, b) => a.amount - b.amount);
    const sellPositions = ps.filter(p => p.direction === 'SELL').sort((a, b) => a.amount - b.amount);
    const firstPosition = buyPositions[0];
    const isOverStartingPrice = firstPosition.openPrice < price;
    const lastPosition = isOverStartingPrice ? (sellPositions.length > 0 ? sellPositions[sellPositions.length - 1] : firstPosition) : buyPositions[buyPositions.length - 1];
    const [minutes, seconds] = firstPosition['timeLeft'].split(':').map(Number);
    const secondsLeft = minutes * 60 + seconds;
    const priceDifference = Math.abs(price - lastPosition.openPrice);
    if (positions.length <= 1 && secondsLeft > settings.defaultDuration * 60 * 8 / 10) {
        if (priceDifference > settings.maxPriceDifference) {
            console.log('[cNP] MAX UPDATE:', priceDifference, settings.maxPriceDifference);
            settings.maxPriceDifference = priceDifference;
            saveSettings(settings);
        }
        return null;
    } else if (!settings.maxPriceDifference) {
        settings.maxPriceDifference = priceDifference;
        saveSettings(settings);
        return null;
    }

    if (secondsLeft < 45) {
        return null;
    }
    console.log('[cNP] CHECK BREAK:', priceDifference, settings.maxPriceDifference);
    const needNewPosition = priceDifference > settings.maxPriceDifference && (isOverStartingPrice ? lastPosition.openPrice < price : lastPosition.openPrice > price);
    if (!needNewPosition) {
        return null;
    }

    const newPositionAmount = lastPosition.amount * 2;
    const newPositionDirection = isOverStartingPrice ? 'SELL' : 'BUY';
    if (newPositionAmount > settings.maxPositionAmount) {
        return null;
    }

    return {
        amount: newPositionAmount,
        direction: newPositionDirection,
        profit: newProfit
    }
}