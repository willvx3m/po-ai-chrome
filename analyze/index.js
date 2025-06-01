const fs = require('fs').promises;
const path = require('path');
const fileName = 'test4-box.csv';
// const fileName = 'test-moz.csv';

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
    try  {
        const positionSets = await readPositionGroups();
        console.log('File:', fileName, ' - Found', positionSets.length, 'position sets');

        var regularPositions = [];
        var oddPositions = [];
        var irregularPositions = [];

        for (const positionSet of positionSets) {
            const startingPositions = positionSet.filter(row => identicalOpenTime(row, positionSet[0]));
            if(startingPositions.length == 2) {
                regularPositions.push(positionSet);
            } else if(startingPositions.length > 2) {
                oddPositions.push(positionSet);
            } else {
                irregularPositions.push(positionSet);
            }
        }

        console.log('Found', regularPositions.length, 'regular positions');
        var regularTotalAmount = 0;
        var regularTotalProfit = 0;
        for (const position of regularPositions) {
            regularTotalAmount += position.reduce((acc, row) => acc + parseFloat(row['Trade amount']), 0);
            regularTotalProfit += position.reduce((acc, row) => acc + parseFloat(row['Profit']), 0);
        }

        console.log('=> Regular total amount:', regularTotalAmount.toFixed(2) * 1);
        console.log('=> Regular total profit:', regularTotalProfit.toFixed(2) * 1);

        console.log('Found', oddPositions.length, 'odd positions');
        var oddTotalAmount = 0;
        var oddTotalProfit = 0;
        for (const position of oddPositions) {
            oddTotalAmount += position.reduce((acc, row) => acc + parseFloat(row['Trade amount']), 0);
            oddTotalProfit += position.reduce((acc, row) => acc + parseFloat(row['Profit']), 0);
        }
        console.log('=> Odd total amount:', oddTotalAmount.toFixed(2) * 1);
        console.log('=> Odd total profit:', oddTotalProfit.toFixed(2) * 1);

        console.log('Found', irregularPositions.length, 'irregular positions');
        var irregularTotalAmount = 0;
        var irregularTotalProfit = 0;
        for (const position of irregularPositions) {
            irregularTotalAmount += position.reduce((acc, row) => acc + parseFloat(row['Trade amount']), 0);
            irregularTotalProfit += position.reduce((acc, row) => acc + parseFloat(row['Profit']), 0);
            console.log(position.map(row => row['Expiration']));
        }
        console.log('=> Irregular total amount:', irregularTotalAmount.toFixed(2) * 1);
        console.log('=> Irregular total profit:', irregularTotalProfit.toFixed(2) * 1);
    } catch (err) {
        console.error(err);
    }
}

main();