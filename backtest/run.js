const fs = require('fs');
const { parse } = require('csv-parse');
// const { createStartingPosition, calculateNextPosition, DEFAULT_SETTINGS } = require('./strategy');
// const { createStartingPosition, calculateNextPosition, DEFAULT_SETTINGS } = require('./strategy-bolk-2');
const { createStartingPosition, calculateNextPosition, DEFAULT_SETTINGS } = require('./strategy-martingale-3');
// const { createStartingPosition, calculateNextPosition, DEFAULT_SETTINGS } = require('./strategy-mama');

const period = '1m'; // 1w, 2w, 1m
// const symbol = 'EUR/USDT';
// const symbol = 'BTC/USDT';
// const symbol = 'aud/cad';
// const symbol = 'aud/cny';
// const symbol = 'gecko-eurc/usdc';
const symbol = 'gecko-chf/usdc';
const csvFile = `${symbol.replace('/', '_')}_ohlcv_${period}.csv`;
const fixedProfit = 80;

// Function to read CSV and extract close prices
function getPrices(csvFile, symbol, callback) {
    /**
     * Extract close prices from the CSV file for the specified symbol.
     * @param {string} csvFile - Path to the OHLCV CSV file.
     * @param {string} symbol - Asset symbol (e.g., 'BTC/USDT' or 'EUR/USDT').
     * @param {function} callback - Callback function to return the close prices array.
     * @returns {void} Calls callback with array of close prices.
     */
    const prices = [];

    fs.createReadStream(csvFile)
        .pipe(parse({ columns: true, trim: true }))
        .on('data', (row) => {
            if (symbol === 'EUR/USDT' || symbol === 'BTC/USDT') {
                if (row.symbol === symbol) {
                    prices.push({
                        price: parseFloat(row.close),
                        timestamp: row.timestamp
                    });
                }
            } else if (symbol.startsWith('gecko-')) {
                prices.push({
                    price: parseFloat(row.Close),
                    timestamp: row.Timestamp
                });
            } else {
                prices.push({
                    price: parseFloat(row['Open price']),
                    timestamp: row['Open time']
                });

            }
            // console.log(row['Open time'], row['Open price']);
            // console.log(row);
        })
        .on('end', () => {
            callback(prices);
        })
        .on('error', (err) => {
            console.error(`Error reading CSV: ${err.message}`);
            callback([]);
        });
}

// Function to evaluate positions
function evaluate(price, positions) {
    /**
     * Calculate the profit/loss of all positions at the given price.
     * @param {number} price - Current close price to evaluate positions.
     * @param {Array} positions - Array of positions, each as { price, amount, direction, payout }.
     * @returns {number} Total profit/loss for the positions.
     */
    let totalProfit = 0;
    for (const pos of positions) {
        const amount = pos.amount;
        const profit = pos.profit;
        const strike = pos.openPrice;
        const direction = pos.direction;

        // Check if position is in-the-money (ITM)
        if ((direction === 'BUY' && price > strike) || (direction === 'SELL' && price < strike)) {
            totalProfit += profit * amount / 100;
        } else {
            totalProfit -= amount; // Loss of investment
        }
    }
    return totalProfit;
}

// Function to run the backtest
function run(prices, symbol) {
    /**
     * Backtest the strategy by iterating through the price array and creating/evaluating positions.
     * @param {Array} prices - Array of close prices.
     * @param {string} symbol - Asset symbol (e.g., 'BTC/USDT' or 'EUR/USDT').
     * @returns {Object} Results including total profit, number of trades, win rate, and trade log.
     */
    const activePositions = [];
    const settings = DEFAULT_SETTINGS;
    settings.priceBook = [];
    const tradeLog = [];
    let totalProfit = 0;
    let totalTrades = 0;
    let winningTrades = 0;

    for (let i = 0; i < prices.length; i++) {
        const currentPrice = prices[i].price;
        const currentTime = new Date(prices[i].timestamp).getTime();

        if (activePositions.length > 0 && activePositions[0].endsAt <= currentTime) {
            const profit = evaluate(currentPrice, activePositions);
            totalProfit += profit;
            winningTrades += profit > 0 ? 1 : 0;
            tradeLog.push({
                endsAt: activePositions[0].endsAt,
                positions: [...activePositions],
                profit: profit,
                endPrice: currentPrice,
                endsAt: new Date(currentTime),
                maxPriceDifference: settings.maxPriceDifference
            });
            activePositions.length = 0;
            continue;
        }

        // Define new position
        var newPositions;
        if (activePositions.length === 0) {
            newPositions = createStartingPosition(settings, currentPrice, fixedProfit, currentTime);
        } else {
            newPositions = calculateNextPosition(activePositions, currentPrice, fixedProfit, settings, currentTime);
        }
        settings.priceBook.push(currentPrice);
        while (settings.priceBook.length > settings.smaSampleCount) {
            settings.priceBook.shift();
        }
        if (newPositions && activePositions.length < settings.maxPositionLimit) {
            newPositions.forEach(position => {
                // position.symbol = symbol; // Add symbol to position
                activePositions.push(position);
                totalTrades++;
            });
        }
    }

    return {
        totalProfit: totalProfit,
        totalTrades: totalTrades,
        tradeLog: tradeLog
    };
}

getPrices(csvFile, symbol, (prices) => {
    if (prices.length === 0) {
        console.error('No prices found for the specified symbol.');
        return;
    }
    if (symbol === 'aud/cad' || symbol === 'aud/cny') {
        prices = prices.reverse(); // only for aud/cny, aud/cad
    }

    // Run backtest
    const results = run(prices, symbol);

    // Print results
    let totalAmount = 0;
    let totalPlus = 0;
    let totalMinus = 0;
    let totalPlusCount = 0;
    let totalMinusCount = 0;

    results.tradeLog/*.slice(0, 5)*/.forEach((trade, index) => {
        const amount = trade.positions.reduce((acc, pos) => acc + pos.amount, 0);
        totalAmount += amount;
        if (trade.profit > 0) {
            totalPlus += trade.profit;
            totalPlusCount++;
        } else {
            totalMinus += trade.profit;
            totalMinusCount++;
        }

        console.log(
            `# ${index + 1}:`,
            `Amount:`, amount,
            `Profit:`, trade.profit.toFixed(2) * 1,
            // `Max Price Difference:`, trade.maxPriceDifference,
            `End Price:`, trade.endPrice,
            `Ends At:`, trade.endsAt,
            // `Positions:`, trade.positions
        );
    });

    console.log(`\nTotal Amount: ${totalAmount}`);
    console.log(`Total Profit: ${results.totalProfit.toFixed(2)}$`);
    console.log(`Total Sessions: ${results.tradeLog.length}`);
    console.log(`Total Trades: ${results.totalTrades}`);
    console.log(`Total Plus: ${totalPlus.toFixed(2)}$ (${totalPlusCount})`);
    console.log(`Total Minus: ${totalMinus.toFixed(2)}$ (${totalMinusCount})`);
    console.log(`Win Rate: ${(totalPlusCount / (totalPlusCount + totalMinusCount) * 100).toFixed(2)}%`);


    // Generate chart data
    // const chartData = {
    //     symbol: symbol,
    //     prices: prices.map(d => d.price),
    //     timestamps: prices.map(d => new Date(d.timestamp).toLocaleString('en-US', {
    //         timeZone: 'UTC',
    //         month: 'short',
    //         day: 'numeric',
    //         hour: '2-digit',
    //         minute: '2-digit',
    //         hour12: false
    //     }))
    // };

    // Save chart data to JSON file
    // fs.writeFileSync('chart_data.json', JSON.stringify(chartData, null, 2));
    // console.log('Chart data saved to chart_data.json');

});