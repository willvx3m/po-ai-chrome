// Main change from v1.0
// Position amount: 1/1/1/2/3/4 -> high risk, high reward

function createStartingPosition(settings) {
    const newPositionAmount = 1;
    const newPositionDuration = settings.defaultDuration;
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

    var needBuy;
    var needSell;
    var positionAmount;

    if (positions.length === 2) {
        positionAmount = 1;
    } else if (positions.length === 3) {
        positionAmount = 2;
    } else if (positions.length === 4) {
        positionAmount = 3;
    } else if (positions.length === 5) {
        positionAmount = 4;
    }

    if (!positionAmount) {
        return null;
    }

    if (totalBuyAmount === totalSellAmount) {
        needBuy = positions.every(position => position.openPrice > price);
        needSell = positions.every(position => position.openPrice < price);
    } else if (totalBuyAmount > totalSellAmount) {
        needSell = positions.every(position => position.openPrice < price);
    } else if (totalBuyAmount < totalSellAmount) {
        needBuy = positions.every(position => position.openPrice > price);
    }

    if (needBuy) {
        return {
            amount: positionAmount,
            direction: 'BUY',
            profit: newProfit
        }
    } else if (needSell) {
        return {
            amount: positionAmount,
            direction: 'SELL',
            profit: newProfit
        }
    }

    return null;
}