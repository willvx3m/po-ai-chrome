// Main change from v2.0
// Martingale strategy
// Position amount: 1/2/4/8
// Position direction: First UP, after Full Failure (1/2/4/8) flip direction

function createStartingPosition(settings) {
    settings.defaultDuration = 1;

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

function checkAndSaveSettings(settings, currentBalance) {
    if (settings.savedBalance) {
        const profit = currentBalance - settings.savedBalance;
        const fullFailureSum = 1 + 2 + 4 + 8;
        if (profit < 0 && Math.abs(profit) >= fullFailureSum) {
            settings.defaultDirection = settings.defaultDirection === 'BUY' ? 'SELL' : 'BUY';
            console.log('[cAS] Full failure detected, flipping direction to', settings.defaultDirection);
        }
    }

    settings.savedBalance = currentBalance;
    saveSettings(settings);
}

function calculateNextPosition(ps, price, newProfit, settings) {
    const positions = ps.sort((a, b) => a.amount - b.amount);
    const needNewPosition = positions.every(position => position.openPrice > price);
    if (needNewPosition) {
        const newPositionAmount = positions[positions.length - 1].amount * 2;
        if (newPositionAmount > 8) {
            return null;
        }

        return {
            amount: newPositionAmount,
            direction: settings.defaultDuration,
            profit: newProfit
        }
    }

    return null;
}