function createStartingPosition(settings) {
    const newPositionAmount = settings.defaultAmount;
    const newPositionDuration = settings.defaultDuration;
    setEndTime(newPositionDuration, () => {
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
    const buyPositions = positions.filter(position => position.direction === 'BUY');
    const sellPositions = positions.filter(position => position.direction === 'SELL');
    const totalBuyAmount = buyPositions.reduce((acc, position) => acc + position.amount, 0);
    const totalSellAmount = sellPositions.reduce((acc, position) => acc + position.amount, 0);
    const cutPosition = positions.filter(position => position.amount < settings.defaultAmount)[0];

    var needBuy;
    var needSell;
    var shouldCutAmount;

    // Starting position is only 1, skip
    if (buyPositions.length <= 0 || sellPositions.length <= 0) {
        return null;
    }

    if (totalBuyAmount === totalSellAmount) {
        shouldCutAmount = true;

        needBuy = positions.every(position => position.openPrice > price);
        needSell = positions.every(position => position.openPrice < price);
    } else if (totalBuyAmount > totalSellAmount) {
        needSell = positions.every(position => position.openPrice < price);
    } else if (totalBuyAmount < totalSellAmount) {
        needBuy = positions.every(position => position.openPrice > price);
    }

    if (needBuy) {
        return {
            amount: shouldCutAmount ? 1 * (settings.defaultAmount / 2).toFixed(1) : settings.defaultAmount,
            direction: 'BUY',
            profit: newProfit
        }
    } else if (needSell) {
        return {
            amount: shouldCutAmount ? 1 * (settings.defaultAmount / 2).toFixed(1) : settings.defaultAmount,
            direction: 'SELL',
            profit: newProfit
        }
    }

    return null;
}