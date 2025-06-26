const fs = require('fs');
const { parse } = require('csv-parse');
const { createStartingPosition, calculateNextPosition, DEFAULT_SETTINGS } = require(
    // './strategy/strategy'
    // './strategy/strategy-bolk-2'
    // './strategy/strategy-martingale-3'
    // './strategy/strategy-mama'
    './strategy/strategy-mama-3'
);

const SYMBOLS = [
    // 'gecko-eurc/usdc',
    // 'gecko-chf/usdc',
    // 'gecko-btc/usdt',
    'po-aud_chf_otc',
    'po-aed_cny_otc',
    'po-eur_usd_otc',
    'po-aud_cad_otc'
];
const FIXED_PROFIT = 80;

// Function to read CSV and extract close prices
function getPrices(symbol, callback) {
    const csvFile = `ohlcv/${symbol.replace('/', '_')}.csv`;
    console.log(symbol, csvFile);
    /**
     * Extract close prices from the CSV file.
     * @param {function} callback - Callback function to return the close prices array.
     * @returns {void} Calls callback with array of close prices.
     */
    const prices = [];

    fs.createReadStream(csvFile)
        .pipe(parse({ columns: true, trim: true }))
        .on('data', (row) => {
            if (symbol.includes('EUR/USDT') || symbol.includes('BTC/USDT')) {
                prices.push({
                    price: parseFloat(row.close),
                    timestamp: row.timestamp
                });
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
function simulate(prices, settings) {
    /**
     * Backtest the strategy by iterating through the price array and creating/evaluating positions.
     * @param {Array} prices - Array of close prices.
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
        const currentPayout = prices[i].payout || FIXED_PROFIT;
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

function main(symbol, prices, settings, includeAnalysisPerPosition = false) {
    if (prices.length === 0) {
        console.error('No prices record');
        return;
    }
    if (symbol.includes('aud/cad') || symbol.includes('aud/cny')) {
        prices = prices.reverse(); // only for aud/cny, aud/cad
    }

    // Run backtest
    const results = simulate(prices, settings);

    // Print results
    let totalAmount = 0;
    let totalPlus = 0;
    let totalMinus = 0;
    let totalPlusCount = 0;
    let totalMinusCount = 0;
    let maxPlus = 0;
    let maxMinus = 0;
    let balanceDelta = 0;
    const balanceTrack = [];

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

        balanceDelta += trade.profit;
        if (balanceDelta > maxPlus) {
            maxPlus = balanceDelta;
        }
        if (balanceDelta < maxMinus) {
            maxMinus = balanceDelta;
        }

        balanceTrack.push(balanceDelta);

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

    if (includeAnalysisPerPosition) {
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
    }

    console.log(`\nTotal Amount: ${totalAmount}`);
    console.log(`Total Profit: ${results.totalProfit.toFixed(2)}$`);
    console.log(`Total Sessions: ${results.tradeLog.length}`);
    console.log(`Total Trades: ${results.totalTrades}`);
    console.log(`Total Plus: ${totalPlus.toFixed(2)}$ (${totalPlusCount})`);
    console.log(`Total Minus: ${totalMinus.toFixed(2)}$ (${totalMinusCount})`);
    console.log(`Max Plus: ${maxPlus.toFixed(2)}$`);
    console.log(`Max Minus: ${maxMinus.toFixed(2)}$`);
    console.log(`Win Rate: ${(totalPlusCount / (totalPlusCount + totalMinusCount) * 100).toFixed(2)}%`);

    return {
        totalAmount: totalAmount,
        totalProfit: results.totalProfit.toFixed(2) * 1,
        totalTrades: results.totalTrades,
        tradeLog: results.tradeLog,
        winRate: (totalPlusCount / (totalPlusCount + totalMinusCount) * 100).toFixed(2),
        maxPlus: maxPlus.toFixed(2),
        maxMinus: maxMinus.toFixed(2),
        balanceTrack: balanceTrack,
        prices
    }
}

function savePriceTrack(filePath, prices) {
    // timestamps: prices.map(d => new Date(d.timestamp).toLocaleString('en-US', {
    //     timeZone: 'UTC',
    //     month: 'short',
    //     day: 'numeric',
    //     hour: '2-digit',
    //     minute: '2-digit',
    //     hour12: false
    // }))
    fs.writeFileSync(
        filePath,
        prices.map(p => p.price).join('\n')
    );
}
SYMBOLS.forEach(symbol => getPrices(symbol, (prices) => {
    const settings = DEFAULT_SETTINGS;
    const includeAnalysisPerPosition = false;
    const isSavingGEM = false;
    const isSavingBalanceTrack = true;
    const isSavingPriceTrack = false;

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
                            const result = main(symbol, prices, settings, includeAnalysisPerPosition);
                            if (result.totalProfit > 0) {
                                console.log(`[GEM] DA:${_defaultAmount} DD:${_defaultDuration} MP:${_maxPositionLimit} MA:${_maxPositionAmount} IN:${_interval} SM:${_smaSampleCount} TP:${result.totalProfit} TA:${result.totalAmount} TR:${result.totalTrades} WR:${result.winRate} MP:${result.maxPlus} MM:${result.maxMinus}`);

                                if (isSavingGEM) {
                                    fs.appendFileSync(
                                        'gem.csv',
                                        `\n${symbol},${_defaultAmount},${_defaultDuration},${_maxPositionLimit},${_maxPositionAmount},${_interval},${_smaSampleCount},${result.totalProfit},${result.totalAmount},${result.totalTrades},${result.winRate},${result.maxPlus},${result.maxMinus}`
                                    );
                                }

                                if (isSavingBalanceTrack) {
                                    fs.appendFileSync(
                                        `balance_track/${symbol.replace('/', '_')}_DD_${settings.defaultDuration}_MP_${settings.maxPositionLimit}_MA_${settings.maxPositionAmount}_SMA_${settings.smaSampleCount}.csv`,
                                        result.balanceTrack.join('\n')
                                    );
                                }

                                if (isSavingPriceTrack) {
                                    savePriceTrack(
                                        `price_track/${symbol.replace('/', '_')}.csv`,
                                        result.prices
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}));