// Main change from v2.0
// Position amount: 1/2/4/8/16
// Position direction: always up
// Martingale strategy

function createStartingPosition(settings) {
    settings.defaultDuration = 2;

    const newPositionAmount = 1;
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
        if (!newPositionSeconds || newPositionSeconds < settings.defaultDuration * 60 - 10) {
            console.log(`[cSP] Duration (${newPositionSeconds}s) is too short, `, 'EndTime:', endTime);
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
    const needNewPosition = positions.every(position => position.openPrice > price);
    if (needNewPosition) {
        const newPositionAmount = positions[positions.length - 1].amount * 2;
        return {
            amount: newPositionAmount,
            direction: 'BUY',
            profit: newProfit
        }
    }

    return null;
}