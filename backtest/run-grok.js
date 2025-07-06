const fs = require('fs');

// Configuration object
const CONFIG = {
    emaShort1: 5,           // First short EMA period
    emaShort2: 10,          // Second short EMA period
    emaLong: 30,            // Long EMA period
    rsiPeriod: 14,          // RSI period
    rsiOverbought: 70,      // RSI threshold for Put
    rsiOversold: 30,        // RSI threshold for Call
    volatilityMin: 0.0002,  // Min volatility
    volatilityMax: 0.0015,  // Max volatility
    momentumCall: 55,       // Micro-momentum for Call (%)
    momentumPut: 45,        // Micro-momentum for Put (%)
    jumpThreshold: 0.001,   // Price jump threshold
    jumpWindow: 24,         // Price jump window (2 min)
    tradingStartHour: 13,   // Trading start hour (UTC, 8 AM EST)
    tradingEndHour: 17,     // Trading end hour (UTC, 12 PM EST)
    martingaleMax: 4,       // Max martingale levels
    baseStake: 1,         // Base stake ($)
    expirationPoints: 60    // Expiration (5 min = 30 points)
};

// Parse CSV
function parseCSV(filePath) {
    const data = fs.readFileSync(filePath, 'utf8').trim().split('\n');
    const headers = data[0].split(',');
    return data.slice(1).map(row => {
        const [timestamp, price, payout] = row.split(',');
        return { 
            timestamp: new Date(timestamp), 
            price: parseFloat(price), 
            payout: parseInt(payout) / 100 
        };
    });
}

// Calculate EMA
function calculateEMA(prices, period) {
    const k = 2 / (period + 1);
    let ema = prices[0];
    const emaArray = [ema];
    for (let i = 1; i < prices.length; i++) {
        ema = prices[i] * k + ema * (1 - k);
        emaArray.push(ema);
    }
    return emaArray;
}

// Calculate RSI
function calculateRSI(prices, period) {
    const rsi = [];
    const gains = [];
    const losses = [];
    for (let i = 1; i < prices.length; i++) {
        const diff = prices[i] - prices[i - 1];
        gains.push(diff > 0 ? diff : 0);
        losses.push(diff < 0 ? -diff : 0);
    }
    for (let i = 0; i < prices.length; i++) {
        if (i < period) {
            rsi.push(50);
            continue;
        }
        const avgGain = gains.slice(i - period, i).reduce((sum, g) => sum + g, 0) / period;
        const avgLoss = losses.slice(i - period, i).reduce((sum, l) => sum + l, 0) / period;
        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        rsi.push(100 - (100 / (1 + rs)));
    }
    return rsi;
}

// Calculate log returns
function calculateReturns(prices) {
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
        returns.push(Math.log(prices[i] / prices[i - 1]));
    }
    return returns;
}

// Calculate volatility
function calculateVolatility(returns, window) {
    const volatility = [];
    for (let i = 0; i < returns.length; i++) {
        if (i < window - 1) {
            volatility.push(0);
            continue;
        }
        const slice = returns.slice(i - window + 1, i + 1);
        const mean = slice.reduce((sum, r) => sum + r, 0) / window;
        const variance = slice.reduce((sum, r) => sum + (r - mean) ** 2, 0) / window;
        volatility.push(Math.sqrt(variance) * Math.sqrt(window));
    }
    return volatility;
}

// Calculate micro-momentum
function calculateMicroMomentum(returns, window) {
    const momentum = [];
    for (let i = 0; i < returns.length; i++) {
        if (i < window - 1) {
            momentum.push(0);
            continue;
        }
        const slice = returns.slice(i - window + 1, i + 1);
        const positive = slice.filter(r => r > 0).length;
        momentum.push((positive / window) * 100);
    }
    return momentum;
}

// Detect price jumps
function detectJumps(returns, window, threshold) {
    const jumps = [];
    for (let i = 0; i < returns.length; i++) {
        if (i < window - 1) {
            jumps.push(false);
            continue;
        }
        const slice = returns.slice(i - window + 1, i + 1);
        jumps.push(slice.some(r => Math.abs(r) > threshold));
    }
    return jumps;
}

// Log signal failure
function logSignalFailure(data, i, emaShort1, emaShort2, emaLong, rsi, volatility, microMomentum, reason) {
    const logEntry = `
Signal Failure at ${data[i].timestamp.toISOString()}:
  Price: ${data[i].price.toFixed(5)}
  Reason: ${reason}
  EMA(${CONFIG.emaShort1}): ${emaShort1[i].toFixed(5)}
  EMA(${CONFIG.emaShort2}): ${emaShort2[i].toFixed(5)}
  EMA(${CONFIG.emaLong}): ${emaLong[i].toFixed(5)}
  RSI(${CONFIG.rsiPeriod}): ${rsi[i].toFixed(2)}
  Volatility: ${volatility[i].toFixed(6)}
  Micro-Momentum: ${microMomentum[i].toFixed(2)}%
`;
    console.log(logEntry);
    fs.appendFileSync('trades_log.txt', logEntry);
}

// Log sequence
function logTrade(sequence, data, exitIdx, emaShort1, emaShort2, emaLong, rsi, volatility, microMomentum) {
    let logEntry = `\nSequence ${sequence.id}:\n`;
    let totalProfit = 0;
    for (const pos of sequence.positions) {
        totalProfit += pos.profit;
        logEntry += `  Position ${pos.level}:
    Entry Time: ${data[pos.entryIdx].timestamp.toISOString()}
    Entry Price: ${data[pos.entryIdx].price.toFixed(5)}
    Direction: ${pos.direction}
    Stake: $${pos.stake.toFixed(2)}
    Martingale Level: ${pos.level}
    EMA(${CONFIG.emaShort1}): ${emaShort1[pos.entryIdx].toFixed(5)}
    EMA(${CONFIG.emaShort2}): ${emaShort2[pos.entryIdx].toFixed(5)}
    EMA(${CONFIG.emaLong}): ${emaLong[pos.entryIdx].toFixed(5)}
    RSI(${CONFIG.rsiPeriod}): ${rsi[pos.entryIdx].toFixed(2)}
    Volatility: ${volatility[pos.entryIdx].toFixed(6)}
    Micro-Momentum: ${microMomentum[pos.entryIdx].toFixed(2)}%
    Exit Time: ${data[exitIdx].timestamp.toISOString()}
    Exit Price: ${data[exitIdx].price.toFixed(5)}
    Profit/Loss: $${pos.profit.toFixed(2)}
    Outcome: ${pos.profit > 0 ? 'Win' : 'Loss'}\n`;
    }
    logEntry += `  Sequence Total Profit/Loss: $${totalProfit.toFixed(2)}\n`;
    console.log(logEntry);
    fs.appendFileSync('trades_log.txt', logEntry);
}

// Backtest function
function backtest(filePath) {
    const data = parseCSV(filePath);
    const prices = data.map(d => d.price);
    const payouts = data.map(d => d.payout);
    const returns = calculateReturns(prices);
    const emaShort1 = calculateEMA(prices, CONFIG.emaShort1);
    const emaShort2 = calculateEMA(prices, CONFIG.emaShort2);
    const emaLong = calculateEMA(prices, CONFIG.emaLong);
    const rsi = calculateRSI(prices, CONFIG.rsiPeriod);
    const volatility = calculateVolatility(returns, 60);
    const microMomentum = calculateMicroMomentum(returns, 30);
    const jumps = detectJumps(returns, CONFIG.jumpWindow, CONFIG.jumpThreshold);

    let sequences = [];
    let balance = 0;
    let sequence = null;
    let sequenceId = 1;

    for (let i = Math.max(CONFIG.emaLong, CONFIG.rsiPeriod); i < data.length - CONFIG.expirationPoints; i++) {
        const hour = data[i].timestamp.getUTCHours();
        // if (hour < CONFIG.tradingStartHour || hour >= CONFIG.tradingEndHour) continue;

        const isBullish = (emaShort1[i] > emaLong[i] || emaShort2[i] > emaLong[i]) && rsi[i] < CONFIG.rsiOverbought;
        const isBearish = (emaShort1[i] < emaLong[i] || emaShort2[i] < emaLong[i]) && rsi[i] > CONFIG.rsiOversold;
        const validVolatility = volatility[i] >= CONFIG.volatilityMin && volatility[i] <= CONFIG.volatilityMax;
        const noJump = !jumps[i];

        let failureReason = '';
        if (!isBullish && !isBearish) failureReason = 'No EMA crossover or RSI conflict';
        else if (!validVolatility) failureReason = 'Volatility out of range';
        else if (isBullish && microMomentum[i] <= CONFIG.momentumCall) failureReason = 'Insufficient Call momentum';
        else if (isBearish && microMomentum[i] >= CONFIG.momentumPut) failureReason = 'Insufficient Put momentum';
        else if (jumps[i]) failureReason = 'Price jump detected';
        // if (failureReason) logSignalFailure(data, i, emaShort1, emaShort2, emaLong, rsi, volatility, microMomentum, failureReason);

        if (!sequence) {
            if (isBullish && validVolatility && microMomentum[i] > CONFIG.momentumCall && noJump) {
                sequence = {
                    id: sequenceId++,
                    direction: 'Call',
                    expirationIdx: i + CONFIG.expirationPoints,
                    positions: [{ level: 1, entryIdx: i, stake: CONFIG.baseStake, direction: 'Call' }]
                };
            } else if (isBearish && validVolatility && microMomentum[i] < CONFIG.momentumPut && noJump) {
                sequence = {
                    id: sequenceId++,
                    direction: 'Put',
                    expirationIdx: i + CONFIG.expirationPoints,
                    positions: [{ level: 1, entryIdx: i, stake: CONFIG.baseStake, direction: 'Put' }]
                };
            }
        } else if (i < sequence.expirationIdx) {
            const lastPos = sequence.positions[sequence.positions.length - 1];
            const lastPrice = prices[lastPos.entryIdx];
            const currentPrice = prices[i];
            const lastLost = (sequence.direction === 'Call' && currentPrice <= lastPrice) ||
                             (sequence.direction === 'Put' && currentPrice >= lastPrice);
            if (lastLost && sequence.positions.length < CONFIG.martingaleMax && validVolatility && noJump &&
                ((sequence.direction === 'Call' && isBullish && microMomentum[i] > CONFIG.momentumCall) ||
                 (sequence.direction === 'Put' && isBearish && microMomentum[i] < CONFIG.momentumPut))) {
                sequence.positions.push({
                    level: sequence.positions.length + 1,
                    entryIdx: i,
                    stake: lastPos.stake * 2,
                    direction: sequence.direction
                });
            }
        }

        if (sequence && i >= sequence.expirationIdx) {
            const exitPrice = prices[i];
            let sequenceProfit = 0;
            for (const pos of sequence.positions) {
                const isWin = (sequence.direction === 'Call' && exitPrice > pos.entryPrice) ||
                              (sequence.direction === 'Put' && exitPrice < pos.entryPrice);
                pos.profit = isWin ? pos.stake * payouts[i] : -pos.stake;
                pos.entryPrice = prices[pos.entryIdx];
                sequenceProfit += pos.profit;
            }
            balance += sequenceProfit;
            sequences.push(sequence);
            logTrade(sequence, data, i, emaShort1, emaShort2, emaLong, rsi, volatility, microMomentum);
            sequence = null;
        }
    }

    const winRate = sequences.filter(s => s.positions.some(p => p.profit > 0)).length / sequences.length || 0;
    const profitFactor = sequences.reduce((sum, s) => sum + s.positions.reduce((pSum, p) => pSum + (p.profit > 0 ? p.profit : 0), 0), 0) /
                         Math.abs(sequences.reduce((sum, s) => sum + s.positions.reduce((pSum, p) => pSum + (p.profit < 0 ? p.profit : 0), 0), 0)) || 0;
    const totalReturn = balance;
    const drawdowns = [];
    let peak = 0;
    let currentBalance = 0;
    for (let s of sequences) {
        const sequenceProfit = s.positions.reduce((sum, p) => sum + p.profit, 0);
        currentBalance += sequenceProfit;
        peak = Math.max(peak, currentBalance);
        drawdowns.push(peak - currentBalance);
    }
    const maxDrawdown = Math.max(...drawdowns) || 0;

    const summary = `
Summary:
  Config: ${JSON.stringify(CONFIG)}
  Win Rate: ${(winRate * 100).toFixed(2)}%
  Profit Factor: ${profitFactor.toFixed(2)}
  Total Return: $${totalReturn.toFixed(2)}
  Max Drawdown: $${maxDrawdown.toFixed(2)}
  Total Sequences: ${sequences.length}
`;
    console.log(summary);
    fs.appendFileSync('trades_log.txt', summary);
}

// Run backtest
fs.writeFileSync('trades_log.txt', '');
backtest('ohlcv/po-EUR-USD-OTC-INPUT.csv');