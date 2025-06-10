const PROFIT = 92;
const STARTING_PRICE = 333;
const PRICE_UNIT = 1;
const MAX_PRICE_STEP = 3; // Set as 5 or 10
const N_PRICE_MOVEMENTS = 3; // Number of price movement
const MAX_POSITION_AMOUNT = 3; // Set as 10 or 20
const POSITION_DIRECTIONS = ['BUY', 'SELL'];

// move the price up to MAX_PRICE_STEP, direction could be up or down
// if it reaches N_PRICE_MOVEMENTS
function movePrice(currentPrice) {
    const direction = Math.random() > 0.5 ? 1 : -1;
    return currentPrice + direction * Math.floor(Math.random() * MAX_PRICE_STEP);
}

function formatPriceStep(price){
    const step = Math.round((price - STARTING_PRICE) / PRICE_UNIT);
    return step > 0 ? `+${step}` : step;
}

function formatPosition(position) {
    const direction = position.direction === 'BUY' ? 'B' : (position.direction === 'SELL' ? 'S' : '');
    return `${position.amount}${direction}@${formatPriceStep(position.price)}`;
}

function formatPositions(positions) {
    return positions.map(formatPosition).join('|');
}

function evaluatePosition(positions, price) {
    return positions.reduce((acc, position) => {
        if (position.direction === 'BUY') {
            if (price > position.price) {
                return acc + position.amount * PROFIT / 100;
            } else if (price === position.price) {
                return acc;
            } else {
                return acc - position.amount;
            }
        } else if (position.direction === 'SELL') {
            if (price < position.price) {
                return acc + position.amount * PROFIT / 100;
            } else if (price === position.price) {
                return acc;
            } else {
                return acc - position.amount;
            }
        } else {
            return acc;
        }
    }, 0);
}

function run(currentPrice, positions, nPriceMovements) {
    const positionLength = positions.length;
    const positionString = formatPositions(positions);

    if (nPriceMovements >= N_PRICE_MOVEMENTS) {
        const profit = evaluatePosition(positions, currentPrice);
        const message = `Price: ${formatPriceStep(currentPrice)} :: ${positionString} = ${profit.toFixed(2)}`;
        if (profit > 0) {
            console.warn(message);
            // const startingPrice = positions[0].price;
            // const endingPrice = currentPrice;

            // if (!truePaths[startingPrice]) {
            //     truePaths[startingPrice] = [];
            // }
            // if (!truePaths[startingPrice][endingPrice]) {
            //     truePaths[startingPrice][endingPrice] = [];
            // }
            // truePaths[startingPrice][endingPrice].push(message);

            if (!routeTruePaths[positionLength][positionString]) {
                // routeTruePaths[positionLength][positionString] = [];
                routeTruePaths[positionLength][positionString] = 0;
            }
            // routeTruePaths[positionLength][positionString].push(message);
            routeTruePaths[positionLength][positionString]++;
        }
        return profit > 0 ? 1 : 0;
    }

    var countTruePaths = 0;

    for (var a = positionLength > 0 ? 0 : 1; a <= MAX_POSITION_AMOUNT; a++) {
        for (const d of a > 0 ? POSITION_DIRECTIONS : ['']) {
            positions.push({
                amount: a,
                direction: d,
                price: currentPrice,
            });

            for (var nextPrice = currentPrice - MAX_PRICE_STEP; nextPrice <= currentPrice + MAX_PRICE_STEP; nextPrice++) {
                if (nextPrice === currentPrice) {
                    continue;
                }
                countTruePaths += run(nextPrice, positions, nPriceMovements + 1);
            }
            positions.pop();
        }
    }

    if (countTruePaths > 0) {
        routeTruePaths[positionLength][positionString] = (routeTruePaths[positionLength][positionString] || 0) + countTruePaths;
    }

    return countTruePaths;
}

var routeTruePaths = Array.from({ length: N_PRICE_MOVEMENTS + 1 }, () => ({}));
var truePaths = []; // truePath[STARTING_PRICE][ENDING_PRICE] = [...path]
var currentPrice = STARTING_PRICE;
const positions = [];
run(STARTING_PRICE, positions, 0);

// console.info(`Found ${truePaths.length} true paths`);
// console.info(truePaths);

// Object.keys(truePaths).forEach(startingPrice => {
//     Object.keys(truePaths[startingPrice]).forEach(endingPrice => {
//         const paths = truePaths[startingPrice][endingPrice];
//         console.info(`=== ${startingPrice} -> ${endingPrice} === ${paths.length} paths`);
//         console.info(paths.join('\n'));
//     });
// });

routeTruePaths.forEach((routeTruePath, positionLength) => {
    console.info(`=== P${positionLength} : ${Object.keys(routeTruePath).length} ===`);
    Object.keys(routeTruePath).forEach((positionString) => {
        // if(positionLength > 1 && !positionString.startsWith('[1B @333]')){
        //     return;
        // }
        // if(positionLength > 2 && !positionString.startsWith('[1B @333] [1B @334]')){
        //     return;
        // }

        if (positionLength < N_PRICE_MOVEMENTS) {
            console.info(`${positionString} - ${routeTruePath[positionString]}`);
        } else {
            console.info(`${positionString} - ${routeTruePath[positionString]}`);
            // console.info(`${positionString} - ${routeTruePath[positionString].length}`);
        }
    });
    console.info('--------------------------------');
});

const fs = require('fs');
const fileName = `routeTruePaths_${N_PRICE_MOVEMENTS}P_${MAX_POSITION_AMOUNT}A_${MAX_PRICE_STEP}S.json`;
fs.writeFileSync(fileName, JSON.stringify(routeTruePaths));