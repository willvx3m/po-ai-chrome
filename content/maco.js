function calculateNextPosition(ps, price, newProfit) {
    const positions = JSON.parse(JSON.stringify(ps));
    const oldRanges = getRanges(positions, null);
    // console.log('Old positions:', positions);
    // console.log('Old ranges:', oldRanges);
    const newPosition = {
        openPrice: price,
        amount: 0,
        direction: '',
        profit: newProfit,
        minNetProfit: Math.min(...oldRanges.map(range => getNetProfit(positions, range)))
    };

    // console.log('Current minNetProfit:', newPosition.minNetProfit);

    const ranges = getRanges(positions, price);
    if (ranges === null) {
        return null;
    }

    // console.log('New ranges:', ranges);
    for (var direction of ['BUY', 'SELL']) {
        // TODO: range and tick should be adjusted depending on new position index
        // 1st: tick = 10, 10 - 100
        // 2nd: tick = 5, 5 - 100
        // 3rd: tick = 3, 3 - 100 etc
        for (var amount = 1; amount <= 100; amount ++) {
            const newTestPosition = {
                openPrice: price,
                amount,
                profit: newProfit,
                direction
            };

            positions.push(newTestPosition);
            const netProfitsRanges = ranges.map(range => getNetProfit(positions, range));
            const newMinNetProfit = Math.min(...netProfitsRanges);
            const profitableRanges = netProfitsRanges.filter(netProfit => netProfit > 0);
            const hasBetterNetProfit = newPosition.minNetProfit < newMinNetProfit && profitableRanges.length > ranges.length / 2;
            // console.log('New Test position:', newTestPosition, ' - New minNetProfit:', newMinNetProfit, ' - Profitable ranges:', profitableRanges.length, ' out of', ranges.length);
            if (hasBetterNetProfit) {
                newPosition.amount = amount;
                newPosition.direction = direction;
                newPosition.minNetProfit = newMinNetProfit;
            }

            positions.pop();
        }
    }

    delete positions;
    if (newPosition.amount === 0) {
        return null;
    }

    return newPosition;
}

function getRanges(positions, price) {
    const tempPrices = positions.map(position => position.openPrice);
    if (price) {
        if (tempPrices.includes(price)) {
            return null;
        }
        tempPrices.push(price);
    }
    tempPrices.sort((a, b) => a - b);
    const prices = [...new Set(tempPrices)];
    delete tempPrices;

    const ranges = [];
    ranges.push({ rangeStart: -1, rangeEnd: prices[0] });
    for (var i = 0; i < prices.length - 1; i++) {
        ranges.push({ rangeStart: prices[i], rangeEnd: prices[i + 1] });
    }
    ranges.push({ rangeStart: prices[prices.length - 1], rangeEnd: 1000000 });

    return ranges;
}

function getNetProfit(positions, { rangeStart, rangeEnd }) {
    const netProfit = positions.reduce((acc, position) => {
        if (position.direction === 'BUY' && position.openPrice <= rangeStart) {
            return acc + position.amount * position.profit / 100;
        }
        if (position.direction === 'SELL' && position.openPrice >= rangeEnd) {
            return acc + position.amount * position.profit / 100;
        }

        return acc - position.amount;
    }, 0);

    // console.log('Getting netProfit for range:', { rangeStart, rangeEnd }, ' - Positions:', positions.length, ' - NetProfit:', netProfit);
    return netProfit;
}

function main() {
    const positions = [
        { openPrice: 50, amount: 3, direction: 'BUY', profit: 92 },
        // { openPrice: 150, amount: 100, direction: 'SELL', profit: 92 },
        // { openPrice: 70, amount: 100, direction: 'BUY', profit: 92 },
        // { openPrice: 120, amount: 100, direction: 'SELL', profit: 92 }
    ];


    const priceTargets = [100, 80];
    // for(var i = 0; i < 10; i ++){
    //     priceTargets.push(Math.round(Math.random() * 200));
    // }

    priceTargets.forEach((price) => {
        const nextPosition = calculateNextPosition(positions, price, 92);
        console.log(`Next Position at price:`, price, ' -> ', nextPosition);

        if (nextPosition) {
            positions.push(nextPosition);
        }
    });
}

main();