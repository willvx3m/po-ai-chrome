
const INFINITE_TOP = 10000000;
const INFINITE_BOTTOM = 0;
const MAX_AMOUNT = 10; // TODO 100
const MAX_POSITIONS = 4; // TODO 10
const arrayOfD = ['BUY', 'SELL'];
const arrayOfA = Array.from({ length: MAX_AMOUNT }, (_, i) => i + 1);

function createStartingPosition(settings) {
    // TODO: we should read RSI - RSI > 70 -> create a BUY, RSI < 30 -> create a SELL
    const marketSentiment = getMarketSentiment();
    if (parseInt(marketSentiment) > 70) {
        console.log(`[cSP] MSentiment ${marketSentiment} is > 70, creating a BUY position`);
        const newPositionAmount = settings.defaultAmount;
        const newPositionDuration = settings.defaultDuration;
        setEndTime(newPositionDuration, () => {
            console.log('[cSP] Position duration set', newPositionDuration);
            createPosition(newPositionAmount, 'BUY', () => {
                console.log('[cSP] Position created', newPositionAmount, 'BUY');
            });
        });
    } else if (parseInt(marketSentiment) < 30) {
        console.log(`[cSP] MSentiment ${marketSentiment} is < 30, creating a SELL position`);
        const newPositionAmount = settings.defaultAmount;
        const newPositionDuration = settings.defaultDuration;
        setEndTime(newPositionDuration, () => {
            console.log('[cSP] Position duration set', newPositionDuration);
            createPosition(newPositionAmount, 'SELL', () => {
                console.log('[cSP] Position created', newPositionAmount, 'SELL');
            });
        });
    } else {
        console.log(`[cSP] MSentiment is ${marketSentiment}, not creating a position`);
    }
}

function calculateNextPosition(ps, price, newProfit, settings) {
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
        for (var amount = 1; amount <= 100; amount++) {
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
    ranges.push({ rangeStart: INFINITE_BOTTOM, rangeEnd: prices[0] });
    for (var i = 0; i < prices.length - 1; i++) {
        ranges.push({ rangeStart: prices[i], rangeEnd: prices[i + 1] });
    }
    ranges.push({ rangeStart: prices[prices.length - 1], rangeEnd: INFINITE_TOP });

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

/* Tests */
function simpleTest() {
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

function bruteTest() {
    for (var N = 2; N <= MAX_POSITIONS; N++) {
        for (var A = 1; A <= MAX_AMOUNT; A++) {
            const arrayPS = [];
            arrayPS.push([]);
            arrayPS.push([
                { positions: [createAPosition(A, 'BUY', Math.round(INFINITE_TOP / 2))] }
            ]);

            for (var i = 2; i <= N; i++) {
                const newPS = [];
                arrayPS[i - 1].forEach((positionSet, index) => {
                    const nextPositionSets = getNextPositionSets(positionSet.positions, index);
                    newPS.push(...nextPositionSets);
                    delete nextPositionSets;
                });
                arrayPS.push(newPS);
            }

            console.log(' ================================ ');
            console.log('N:', N, ' - A:', A);

            const finalPS = arrayPS[N];
            finalPS.forEach((positionSet, index) => {
                const { minNetProfit, positiveRangePercentage } = getMetricPositionSet(positionSet);
                positionSet.minNetProfit = minNetProfit;
                positionSet.positiveRangePercentage = positiveRangePercentage;

                if (minNetProfit > -2 && positiveRangePercentage > 50) {
                    console.log('N:', N, ' A:', A, ' - PS #', index, ' - MNP:', minNetProfit);
                    console.log('Found a position set with positive minNetProfit:', positionSet);

                    // var prevN = N - 1;
                    // var parentPositionSetIndex = positionSet.parentPositionSetIndex;
                    // while(prevN >= 1){
                    //     const prevPS = arrayPS[prevN];
                    //     const parentPositionSet = positionSet.parentPositionSetIndex && prevPS[parentPositionSetIndex];
                    //     if(parentPositionSet){
                    //         const { minNetProfit, positiveRangePercentage } = getMetricPositionSet(parentPositionSet);
                    //         parentPositionSet.minNetProfit = minNetProfit;
                    //         parentPositionSet.positiveRangePercentage = positiveRangePercentage;

                    //         console.log('N:', prevN, ' - PS #', ' - MNP:', minNetProfit);
                    //         parentPositionSetIndex = parentPositionSet.parentPositionSetIndex;
                    //         prevN --;
                    //     } else {
                    //         break;
                    //     }
                    // }
                }

                delete ranges;
                delete rangeNetProfits;
            });

            delete arrayPS;
        }
    }
}

function getNextPositionSets(positions, parentPositionSetIndex) {
    const ranges = getRanges(positions, null);
    const arrayOfX = ranges.map(range => Math.round((range.rangeStart + range.rangeEnd) / 2));

    const nextPositionSets = [];
    arrayOfX.forEach(x => {
        arrayOfD.forEach(d => {
            arrayOfA.forEach(a => {
                nextPositionSets.push({
                    positions: [
                        ...positions,
                        createAPosition(a, d, x)
                    ],
                    parentPositionSetIndex
                });
            });
        });
    });

    delete ranges;
    delete arrayOfX;

    return nextPositionSets;
}

function getMetricPositionSet(positionSet) {
    const ranges = getRanges(positionSet.positions, null);
    const rangeNetProfits = ranges.map(range => getNetProfit(positionSet.positions, range));
    const positiveRangePercentage = Math.round(rangeNetProfits.filter(netProfit => netProfit > 0).length * 100 / rangeNetProfits.length);
    const minNetProfit = Math.min(...rangeNetProfits);
    return { minNetProfit, positiveRangePercentage };
}

function createAPosition(amount, direction, price) {
    return {
        openPrice: price,
        amount,
        direction,
        profit: 92
    };
}

// simpleTest();
// bruteTest();