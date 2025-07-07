const fs = require('fs');
const { parse } = require('csv-parse');

/**
 * Extract close prices from the CSV file.
 * @param {function} callback - Callback function to return the close prices array.
 * @returns {void} Calls callback with array of close prices.
 */
function getPrices(symbol, csvFile, callback) {
    // const csvFile = `ohlcv/${symbol.replace('/', '_')}.csv`;
    console.log(symbol, csvFile);
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

/**
 * Calculate the profit/loss of all positions at the given price.
 * @param {number} price - Current close price to evaluate positions.
 * @param {Array} positions - Array of positions, each as { price, amount, direction, payout }.
 * @returns {number} Total profit/loss for the positions.
 */
function evaluate(price, positions) {
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

/**
 * Backtest the strategy by iterating through the price array and creating/evaluating positions.
 * @param {Array} prices - Array of close prices.
 * @returns {Object} Results including total profit, number of trades, win rate, and trade log.
 */
function simulate(prices, settings) {
    const { createStartingPosition, calculateNextPosition, FIXED_PROFIT } = settings;
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

/**
 * Backtest a single session
 * @param {Array} prices - Array of close prices.
 * @param {Object} settings - Strategy settings.
 * @param {number} startIndex - Index to start the backtest from.
 * @returns {Object} Results including positions, profit, end time, and end price.
 */
function simulateSingle(prices, settings, startIndex) {
    const { createStartingPosition, calculateNextPosition, FIXED_PROFIT } = settings;
    const activePositions = [];

    for (let i = startIndex; i < prices.length; i++) {
        const currentPrice = prices[i].price;
        const currentPayout = prices[i].payout || FIXED_PROFIT;
        const currentTime = new Date(prices[i].timestamp).getTime();

        if (activePositions.length > 0 && activePositions[0].endsAt <= currentTime) {
            const profit = evaluate(currentPrice, activePositions);
            return {
                positions: activePositions,
                profit: profit,
                endsAt: new Date(currentTime),
                endPrice: currentPrice,
            }
        }

        // Define new position
        var newPositions;
        if (activePositions.length === 0) {
            newPositions = createStartingPosition(settings, currentPrice, currentPayout, currentTime);
        } else {
            newPositions = calculateNextPosition(activePositions, currentPrice, currentPayout, settings, currentTime);
        }
        if (newPositions && activePositions.length < settings.maxPositionLimit) {
            newPositions.forEach(position => {
                activePositions.push(position);
            });
        }
    }

    return null;
}

/**
 * Calculate the Simple Moving Average (SMA) of the prices.
 * @param {Array} prices - Array of close prices.
 * @param {number} smaSampleCount - Number of samples to use for the SMA calculation.
 * @returns {number} The SMA value.
 */
function getSMA(prices, smaSampleCount) {
    if (prices.length === 0) return 0;
    if (prices.length < smaSampleCount) {
        return 0;
    }

    const aggPrices = prices.slice(-smaSampleCount);
    const sum = aggPrices.reduce((sum, price) => sum + price, 0);
    return aggPrices.length > 0 ? sum / aggPrices.length : 0;
}

/**
 * Calculate the Exponential Moving Average (EMA) of the prices.
 * @param {Array} prices - Array of close prices.
 * @param {number} period - Number of periods to use for the EMA calculation.
 * @returns {number} The EMA value.
 */
function getEMA(prices, period) {
    if (prices.length === 0) return 0;
    if (prices.length < period) {
        return 0;
    }

    const k = 2 / (period + 1); // Smoothing factor
    let ema = 0;

    // Calculate initial SMA for the first EMA value
    for (let i = 0; i < period; i++) {
        ema += prices[i];
    }
    ema /= period;

    // Calculate EMA up to the latest price
    for (let i = period; i < prices.length; i++) {
        ema = (prices[i] * k) + (ema * (1 - k));
    }

    return ema;
}

function getPriceCurveStatus(prices) {
    if (prices.length === 0) {
        return null;
    }

    // Calculate factors
    const ema9 = getEMA(prices, 9 * 6);
    const ema21 = getEMA(prices, 21 * 6);
    const sma50 = getSMA(prices, 50 * 6);
    const priceToEMA9 = ema9 > 0 ? Number((prices[prices.length - 1] / ema9).toFixed(4)) : 0;

    return {
        price: Number(prices[prices.length - 1].toFixed(4)),
        ema9: Number(ema9.toFixed(4)),
        ema21: Number(ema21.toFixed(4)),
        sma50: Number(sma50.toFixed(4)),
        priceToEMA9: priceToEMA9
    };
}

function findBestMatchingCurve(currentStatus, corpus) {
    if (!currentStatus.price || corpus.length === 0) {
        return null;
    }

    // Normalize values across corpus and current status for fair comparison
    const features = ['ema9', 'ema21', 'sma50', 'priceToEMA9'];

    // Find best match using Euclidean distance
    let bestMatch = null;
    let minDistance = Infinity;

    corpus.forEach((historicalStatus, index) => {
        // Normalize historical status
        const normalizedHistorical = features.reduce((acc, f, i) => {
            if (f !== 'priceToEMA9') {
                acc[f] = historicalStatus[f] * currentStatus.price / historicalStatus.price;
            }
            return acc;
        }, { price: currentStatus.price, priceToEMA9: historicalStatus.priceToEMA9 });

        // Calculate Euclidean distance
        const distance =
            Math.sqrt(
                features.reduce((sum, f) => {
                    return sum + Math.pow(currentStatus[f] - normalizedHistorical[f], 2);
                }, 0)
            );

        if (distance < minDistance) {
            minDistance = distance;
            bestMatch = { ...historicalStatus, corpusIndex: index, distance: Number(distance.toFixed(4)), settings: historicalStatus.settings, normalizedHistorical: normalizedHistorical };
        }
    });

    return bestMatch;
}

module.exports = {
    getPrices,
    evaluate,
    simulate,
    simulateSingle,
    getSMA,
    getEMA,
    getPriceCurveStatus,
    findBestMatchingCurve
}