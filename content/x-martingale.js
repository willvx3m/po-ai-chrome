// Martingale strategy
// Position amount: 1/2/4/8/16
const DEFAULT_SETTINGS = {
    enabled: false,
    defaultAmount: 1,
    defaultDuration: 1,
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

    setEndTime(newPositionDuration, () => {
        // Check end time
        // This resolves the failure to create 2 starting positions
        var endTime = getEndTime();
        if (endTime) {
            const now = new Date();
            endTime = `${now.getFullYear()}-${now.getMonth() + 1}-${now.getDate()} ${endTime}`;
        }
        const newPositionSeconds = Math.abs(new Date(endTime) - new Date()) / 1000;
        if (!newPositionSeconds || newPositionSeconds < settings.defaultDuration * 60 - 20 || newPositionSeconds > settings.defaultDuration * 60 + 60) {
            console.log(`[cSP] Duration (${newPositionSeconds}s) is too short/long, `, 'EndTime:', endTime);
            console.log('Restarting ...');
            window.location.reload();
            return;
        }

        console.log('[cSP] Position duration set', newPositionDuration);
        setPositionAmount(newPositionAmount, () => {
            createPosition(newPositionAmount, newPositionDirection, () => {
                console.log('[cSP] Position created', newPositionAmount, newPositionDirection);
            });
        });
    });
}

function calculateNextPosition(ps, price, newProfit, settings) {
    const positions = ps.sort((a, b) => a.amount - b.amount);
    const needNewPosition = settings.defaultDirection === 'BUY' ? positions.every(position => position.openPrice > price) : positions.every(position => position.openPrice < price);
    if (needNewPosition) {
        const newPositionAmount = positions[positions.length - 1].amount * 2;
        if (newPositionAmount > settings.maxPositionAmount) {
            return null;
        }

        return {
            amount: newPositionAmount,
            direction: settings.defaultDirection,
            profit: newProfit
        }
    }

    return null;
}