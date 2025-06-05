const fs = require('fs').promises;
const path = require('path');
// const fileName = 'test8-leb-moz.csv';
// const fileName = 'test4-box.csv';
const fileName = 'test-moz.csv';

/**
 * Reads a CSV file and converts it to an array of objects.
 * @param {string} filePath - The path to the CSV file.
 * @returns {Promise<Array<Object>>} - A promise that resolves to an array of objects representing the CSV data.
 */
async function readCSVToArray(filePath) {
    try {
        const data = await fs.readFile(filePath, 'utf8');
        const lines = data.split('\n');
        const headers = lines[0].split(',');
        const result = lines.slice(1).map(line => {
            const values = line.split(',');
            return headers.reduce((object, header, index) => {
                object[header.trim()] = values[index].trim();
                return object;
            }, {});
        });
        return result;
    } catch (err) {
        throw err;
    }
}

async function readPositionGroups() {
    const data = await readCSVToArray(path.join(__dirname, fileName));
    try {
        data.sort((a, b) => a['Close time'].toLowerCase().localeCompare(b['Close time'].toLowerCase()));
        data.sort((a, b) => a['Open time'].toLowerCase().localeCompare(b['Open time'].toLowerCase()));
        data.push([]); // Filling end

        const positionSets = [];
        var prevCloseTime;
        var positionSet = [];

        for (const row of data) {
            const closeTime = row['Close time'];
            if (prevCloseTime && closeTime !== prevCloseTime) {
                positionSets.push(positionSet);
                positionSet = [];
            }

            positionSet.push(row);
            prevCloseTime = closeTime;
        }

        return positionSets;
    } catch (err) {
        throw err;
    }
}

function identicalOpenTime(row1, row2) {
    return Math.abs(Date.parse(row1['Open time']) - Date.parse(row2['Open time'])) <= 1000;
}

async function main() {
    try {
        const positionSets = await readPositionGroups();
        console.log('File:', fileName, ' - Found', positionSets.length, 'position sets');

        var regularPositions = [];
        var oddPositions = [];
        var irregularPositions = [];

        for (const positionSet of positionSets) {
            const startingPositions = positionSet.filter(row => identicalOpenTime(row, positionSet[0]));
            if (startingPositions.length == 2) {
                regularPositions.push(positionSet);
            } else if (startingPositions.length > 2) {
                oddPositions.push(positionSet);
            } else {
                irregularPositions.push(positionSet);
            }
        }

        console.log();
        console.log('Found', oddPositions.length, 'odd positions');
        var oddTotalAmount = 0;
        var oddTotalProfit = 0;
        for (const position of oddPositions) {
            oddTotalAmount += position.reduce((acc, row) => acc + parseFloat(row['Trade amount']), 0);
            oddTotalProfit += position.reduce((acc, row) => acc + parseFloat(row['Profit']), 0);
        }
        console.log('=> Odd total amount:', oddTotalAmount.toFixed(2) * 1);
        console.log('=> Odd total profit:', oddTotalProfit.toFixed(2) * 1);

        console.log();
        console.log('Found', irregularPositions.length, 'irregular positions');
        var irregularTotalAmount = 0;
        var irregularTotalProfit = 0;
        for (const position of irregularPositions) {
            irregularTotalAmount += position.reduce((acc, row) => acc + parseFloat(row['Trade amount']), 0);
            irregularTotalProfit += position.reduce((acc, row) => acc + parseFloat(row['Profit']), 0);
            // console.log(position.map(row => row['Expiration']));
        }
        console.log('=> Irregular total amount:', irregularTotalAmount.toFixed(2) * 1);
        console.log('=> Irregular total profit:', irregularTotalProfit.toFixed(2) * 1);

        console.log();
        console.log('Found', regularPositions.length, 'regular positions');

        var regularTotalAmount = regularPositions.reduce((acc, position) => acc + position.reduce((acc, row) => acc + parseFloat(row['Trade amount']), 0), 0);
        var regularTotalProfit = regularPositions.reduce((acc, position) => acc + position.reduce((acc, row) => acc + parseFloat(row['Profit']), 0), 0);
        var countFailed = regularPositions.reduce((acc, position) => acc + (position.reduce((acc, row) => acc + parseFloat(row['Profit']), 0) < 0 ? 1 : 0), 0);

        for (var N = 3; N <= 6; N++) {
            var countP = 0;
            var countFailedP = 0;
            var regularTotalAmountP = 0;
            var regularTotalProfitP = 0;

            for (const position of regularPositions) {
                const positionAmount = position.reduce((acc, row) => acc + parseFloat(row['Trade amount']), 0);
                const positionProfit = position.reduce((acc, row) => acc + parseFloat(row['Profit']), 0);
                if (position.length === N) {
                    countP++;
                    regularTotalAmountP += positionAmount;
                    regularTotalProfitP += positionProfit;

                    if (positionProfit < 0) {
                        countFailedP++;
                    }
                }
            }

            console.log(`=P${N}=>`, `Count:`, countP, '(', (countP / regularPositions.length * 100).toFixed(2) * 1, '% )');
            console.log(`=P${N}=>`, `Failed:`, countFailedP, '(', (countFailedP / countP * 100).toFixed(2) * 1, '% )', '(', (countFailedP / regularPositions.length * 100).toFixed(2) * 1, '% )');
            console.log(`=P${N}=>`, `Total amount:`, regularTotalAmountP.toFixed(2) * 1);
            console.log(`=P${N}=>`, `Total profit:`, regularTotalProfitP.toFixed(2) * 1);
            console.log();
        }

        console.log('=> Failed:', countFailed, '(', (countFailed / regularPositions.length * 100).toFixed(2) * 1, '% )');
        console.log('=> Total amount:', regularTotalAmount.toFixed(2) * 1);
        console.log('=> Total profit:', regularTotalProfit.toFixed(2) * 1);

        analyzeLeb(regularPositions, 3);
        analyzeLeb(regularPositions, 4);
        analyzeLeb(regularPositions, 5);
        analyzeLeb(regularPositions, 6);
    } catch (err) {
        console.error(err);
    }
}

function analyzeLeb(regularPositions, stopIndex) {
    var successCount = 0;
    var rushCount = 0;

    for (const positions of regularPositions) {
        positions.sort((a, b) => a['Open time'].localeCompare(b['Open time']));
        if (positions.length < stopIndex) {
            continue;
        }

        const stopPosition = positions[stopIndex - 1];
        const stopPrice = stopPosition['Open price'] * 1;
        const stopDirection = stopPosition['Direction'];
        const closePrice = positions[0]['Close price'] * 1;
        const startPrice = positions[0]['Open price'] * 1;

        if (stopDirection === 'call' && closePrice > stopPrice) {
            successCount++;
        } else if (stopDirection === 'put' && closePrice < stopPrice) {
            successCount++;
        }

        if (startPrice < closePrice && stopPrice > closePrice) {
            rushCount++;
        }
    }

    console.log(`=> LEB Success (Stop: ${stopIndex}): ${successCount} (${successCount / regularPositions.length * 100}%)`);
    console.log(`=> LEB Rush (Stop: ${stopIndex}): ${rushCount} (${rushCount / regularPositions.length * 100}%)`);

    return successCount;
}

main();