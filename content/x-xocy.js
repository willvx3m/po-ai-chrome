// XOCY: Progressive 3-pair strategy
// XOCY 1.0: Position amount: 2/1/2
// XOCY 2.0: Position amount: 1/2/2, COVER position after one minute
const DEFAULT_SETTINGS = {
    name: 'XOCY 2.0',
    enabled: false,
    defaultAmount: 2,
    defaultDuration: 5,
    maxPositionLimit: 3,
    maxPositionAmount: 2,
    interval: 10000,
    defaultDirection: 'BUY',
    maxPriceSpike: 0,
    previousRestart: (new Date()) * 1,
}

function createStartingPosition(settings) {
    const newPositionAmount = settings.defaultAmount / 2;
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

    settings.maxPriceSpike = 0;
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
        createPosition(newPositionAmount, newPositionDirection, () => {
            console.log('[cSP] Position created', newPositionAmount, newPositionDirection);
        });
    });
}

function calculateNextPosition(ps, price, newProfit, settings) {
    const positions = ps.sort((a, b) => a.amount - b.amount);
    const firstPosition = positions[0];
    const secondPosition = positions[1];
    const [minutes, seconds] = firstPosition['timeLeft'].split(':').map(Number);
    const secondsLeft = minutes * 60 + seconds;

    // create 2nd position as soon as it stays in favor zone - amount: 1
    // measure max price spike for interval (10s)
    // if price stays above openPrice + maxSpike => do nothing
    // if price falls below openPrice - maxSpike, create 3rd position - amount: 2
    const priceDifference = secondPosition ? Math.abs(price - secondPosition.openPrice) : Math.abs(price - firstPosition.openPrice);
    const lastPrice = settings.lastPrice || firstPosition.openPrice;
    const priceSpike = Math.abs(price - lastPrice);
    if(settings.maxPriceSpike < priceSpike) {
        console.log('[cNP] NEW MAX SPIKE:', priceSpike);
        settings.maxPriceSpike = priceSpike;
    }
    settings.lastPrice = price;
    saveSettings(settings);

    var newPositionAmount;
    var newPositionDirection;
    if (positions.length < 2 && secondsLeft < settings.defaultDuration * 60 * 4 / 5){ // Spike difference?
        if(firstPosition.direction === 'BUY' && price > firstPosition.openPrice) {
            newPositionAmount = settings.defaultAmount;
            newPositionDirection = 'SELL';
        } else if (firstPosition.direction === 'SELL' && price < firstPosition.openPrice) {
            newPositionAmount = settings.defaultAmount;
            newPositionDirection = 'BUY';
        }
    } else if(secondsLeft < settings.defaultDuration * 60 * 2 / 3 && priceDifference < settings.maxPriceSpike * 3) {
        if(secondPosition.direction === 'BUY' && price < secondPosition.openPrice) {
            newPositionAmount = settings.defaultAmount;
            newPositionDirection = 'BUY';
        } else if (secondPosition.direction === 'SELL' && price > secondPosition.openPrice) {
            newPositionAmount = settings.defaultAmount;
            newPositionDirection = 'SELL';
        }
    }

    if (newPositionAmount && newPositionDirection) {
        return {
            amount: newPositionAmount,
            direction: newPositionDirection,
            profit: newProfit
        }
    }

    return null;
}