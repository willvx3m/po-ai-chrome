const fs = require('fs');
const { parse } = require('csv-parse');
// const { createStartingPosition, calculateNextPosition, DEFAULT_SETTINGS } = require('./strategy');
// const { createStartingPosition, calculateNextPosition, DEFAULT_SETTINGS } = require('./strategy-bolk-2');
// const { createStartingPosition, calculateNextPosition, DEFAULT_SETTINGS } = require('./strategy-martingale-3');
// const { createStartingPosition, calculateNextPosition, DEFAULT_SETTINGS } = require('./strategy-mama');
const { createStartingPosition, calculateNextPosition, DEFAULT_SETTINGS } = require('./strategy-mama-3');

const period = '1d'; // 1w, 2w, 1m
// const symbol = 'EUR/USDT';
// const symbol = 'BTC/USDT';
// const symbol = 'aud/cad';
// const symbol = 'aud/cny';
// const symbol = 'gecko-eurc/usdc';
// const symbol = 'gecko-chf/usdc';
// const symbol = 'gecko-btc/usdt';
const symbol = 'po-aud_chf_otc';
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
            } else if (symbol.startsWith('po-')) {
                prices.push({
                    price: parseFloat(row.price),
                    timestamp: row.timestamp,
                    // payout: 92, //parseFloat(row.payout)
                    payout: parseFloat(row.payout)
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

// Function to simulate the backtest
function simulate(prices, symbol, settings) {
    /**
     * Backtest the strategy by iterating through the price array and creating/evaluating positions.
     * @param {Array} prices - Array of close prices.
     * @param {string} symbol - Asset symbol (e.g., 'BTC/USDT' or 'EUR/USDT').
     * @returns {Object} Results including total profit, number of trades, win rate, and trade log.
     */
    const activePositions = [];
    settings.priceBook = [];
    const tradeLog = [];
    let totalProfit = 0;
    let totalTrades = 0;
    let winningTrades = 0;

    for (let i = 0; i < prices.length; i++) {
        const currentPrice = prices[i].price;
        const currentPayout = prices[i].payout || fixedProfit;
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
            newPositions = createStartingPosition(settings, currentPrice, currentPayout, currentTime);
        } else {
            newPositions = calculateNextPosition(activePositions, currentPrice, currentPayout, settings, currentTime);
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

function main(prices, settings) {
    if (prices.length === 0) {
        console.error('No prices found for the specified symbol.');
        return;
    }
    if (symbol === 'aud/cad' || symbol === 'aud/cny') {
        prices = prices.reverse(); // only for aud/cny, aud/cad
    }

    console.log(prices[0]);

    // Run backtest
    const results = simulate(prices, symbol, settings);

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

    for (var p = 1; p <= 10; p++) {
        const pResults = results.tradeLog.filter(trade => trade.positions.length === p);
        if (pResults.length === 0) {
            continue;
        }
        const totalProfit = pResults.reduce((acc, trade) => acc + trade.profit, 0);
        const totalPlus = pResults.reduce((acc, trade) => acc + (trade.profit > 0 ? trade.profit : 0), 0);
        const totalMinus = pResults.reduce((acc, trade) => acc + (trade.profit < 0 ? trade.profit : 0), 0);
        const totalPlusCount = pResults.reduce((acc, trade) => acc + (trade.profit > 0 ? 1 : 0), 0);
        const totalMinusCount = pResults.reduce((acc, trade) => acc + (trade.profit < 0 ? 1 : 0), 0);
        console.log(`\nPositions: ${p}`, totalProfit > 0 ? '[WIN]' : '[LOSS]',
            `\nTotal Sessions: ${pResults.length}`,
            `\nTotal Trades: ${pResults.reduce((acc, trade) => acc + trade.positions.length, 0)}`,
            `\nTotal Profit: ${totalProfit.toFixed(2)}$`,
            `\nTotal Plus: ${totalPlus.toFixed(2)}$ (${totalPlusCount})`,
            `\nTotal Minus: ${totalMinus.toFixed(2)}$ (${totalMinusCount})`,
            `\nWin Rate: ${(totalPlusCount / (totalPlusCount + totalMinusCount) * 100).toFixed(2)}%`
        );
    }

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

    return {
        totalAmount: totalAmount,
        totalProfit: results.totalProfit.toFixed(2) * 1,
        totalTrades: results.totalTrades,
        tradeLog: results.tradeLog,
        winRate: (totalPlusCount / (totalPlusCount + totalMinusCount) * 100).toFixed(2)
    }
}

getPrices(csvFile, symbol, (prices) => {
    const settings = DEFAULT_SETTINGS;
    for (var _defaultAmount = 1; _defaultAmount <= 1; _defaultAmount++) {
        for (var _defaultDuration = 1; _defaultDuration <= 10; _defaultDuration++) {
            for (var _maxPositionLimit = 1; _maxPositionLimit <= 6; _maxPositionLimit++) {
                for (var _smaSampleCount = 6; _smaSampleCount <= 60; _smaSampleCount += 6) {
                    for (var _maxPositionAmount = 100; _maxPositionAmount <= 100; _maxPositionAmount++) {
                        for (var _interval = 10000; _interval <= 10000; _interval += 10000) {
                            settings.defaultAmount = _defaultAmount;
                            settings.defaultDuration = _defaultDuration;
                            if (!symbol.startsWith('po-')) {
                                settings.defaultDuration = settings.defaultDuration * 6;
                            }
                            settings.maxPositionLimit = _maxPositionLimit;
                            settings.maxPositionAmount = _maxPositionAmount;
                            settings.interval = _interval;
                            settings.smaSampleCount = _smaSampleCount;
                            const result = main(prices, settings);
                            if (result.totalProfit > 0) {
                                console.log(`[GEM] DA:${_defaultAmount} DD:${_defaultDuration} MP:${_maxPositionLimit} MA:${_maxPositionAmount} IN:${_interval} SM:${_smaSampleCount} TP:${result.totalProfit} TA:${result.totalAmount} TR:${result.totalTrades} WR:${result.winRate}`);

                                fs.appendFileSync('gem.csv', `\n${symbol},${_defaultAmount},${_defaultDuration},${_maxPositionLimit},${_maxPositionAmount},${_interval},${_smaSampleCount},${result.totalProfit},${result.totalAmount},${result.totalTrades},${result.winRate}`);
                            }
                        }
                    }
                }
            }
        }
    }
});