// XOCY: Progressive 3-pair strategy
// Position amount: 2/1/2
const DEFAULT_SETTINGS = {
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
    const positions = ps.sort((a, b) => b.amount - a.amount);
    const firstPosition = positions[0];
    const [minutes, seconds] = firstPosition['timeLeft'].split(':').map(Number);
    const secondsLeft = minutes * 60 + seconds;

    // create 2nd position as soon as it stays in favor zone - amount: 1
    // measure max price spike for interval (10s)
    // if price stays above openPrice + maxSpike => do nothing
    // if price falls below openPrice - maxSpike, create 3rd position - amount: 2
    const priceDifference = Math.abs(price - firstPosition.openPrice);
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
    if (positions.length < 2){
        if(firstPosition.direction === 'BUY' && price > firstPosition.openPrice) {
            newPositionAmount = settings.defaultAmount / 2;
            newPositionDirection = 'SELL';
        } else if (firstPosition.direction === 'SELL' && price < firstPosition.openPrice) {
            newPositionAmount = settings.defaultAmount / 2;
            newPositionDirection = 'BUY';
        }
    } else if(secondsLeft < settings.defaultDuration * 60 * 2 / 3 && priceDifference < settings.maxPriceSpike) {
        if(firstPosition.direction === 'BUY' && price > firstPosition.openPrice) {
            newPositionAmount = settings.defaultAmount;
            newPositionDirection = 'SELL';
        } else if (firstPosition.direction === 'SELL' && price < firstPosition.openPrice) {
            newPositionAmount = settings.defaultAmount;
            newPositionDirection = 'BUY';
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