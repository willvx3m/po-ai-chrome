const fs = require('fs');
const { getPrices, simulate } = require('./utils');
const { DEFAULT_SETTINGS } = require(
    // './strategy/strategy'
    // './strategy/strategy-leb'
    // './strategy/strategy-bolk-2'
    // './strategy/strategy-martingale-3'
    // './strategy/strategy-mama'
    // './strategy/strategy-mama-3'
    './strategy/strategy-mama-4'
);

const SYMBOLS = [
    // 'gecko-eurc/usdc',
    // 'gecko-chf/usdc',
    // 'gecko-btc/usdt',
    
    // 'po-aud_chf_otc',
    // 'po-aed_cny_otc',
    // 'po-eur_usd_otc',
    // 'po-aud_cad_otc',
    // 'po-aud_usd_otc',
    // 'po-aaa',

    // 'po-aud_cad_otc_1',
    // 'po-aud_cad_otc_2',
    // 'po-aud_cad_otc_3',

    'po-AUD-CAD-OTC-INPUT',
    // 'po-AUD-USD-OTC-INPUT',
    // 'po-EUR-USD-OTC-INPUT',
];
const BACKTEST_IND = '0707-audcad'

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

        balanceTrack.push(balanceDelta.toFixed(2));

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
SYMBOLS.forEach(symbol => getPrices(symbol, `ohlcv/${symbol.replace('/', '_')}.csv`, (prices) => {
    const settings = DEFAULT_SETTINGS;
    const includeAnalysisPerPosition = false;
    const isSavingGEM = true;
    const isSavingBalanceTrack = true;
    const isSavingPriceTrack = false;

    // create balance_track_${BACKTEST_IND} folder if not exists
    if (isSavingBalanceTrack && !fs.existsSync(`balance_track_${BACKTEST_IND}`)) {
        fs.mkdirSync(`balance_track_${BACKTEST_IND}`, { recursive: true });
    }
    if (isSavingPriceTrack && !fs.existsSync(`price_track_${BACKTEST_IND}`)) {
        fs.mkdirSync(`price_track_${BACKTEST_IND}`, { recursive: true });
    }

    for (var _defaultAmount = 1; _defaultAmount <= 1; _defaultAmount++) {
        for (var _defaultDuration = 4; _defaultDuration <= 10; _defaultDuration++) {
            for (var _maxPositionLimit = 3; _maxPositionLimit <= 6; _maxPositionLimit++) {
                for (var _smaSampleCount = 6; _smaSampleCount <= 60; _smaSampleCount += 6) {
                    for (var _smaBaseSampleCount = _smaSampleCount * 2; _smaBaseSampleCount <= 180; _smaBaseSampleCount += 12) {
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
                                settings.smaBaseSampleCount = _smaBaseSampleCount;
                                const result = main(symbol, prices, settings, includeAnalysisPerPosition);
                                if (result.totalProfit > 0) {
                                    console.log(`[GEM] [${symbol}] DA:${_defaultAmount} DD:${_defaultDuration} MP:${_maxPositionLimit} MA:${_maxPositionAmount} IN:${_interval} SMA:${_smaSampleCount} SMAB:${_smaBaseSampleCount} TP:${result.totalProfit} TA:${result.totalAmount} TR:${result.totalTrades} WR:${result.winRate} MP:${result.maxPlus} MM:${result.maxMinus}`);

                                    if (isSavingGEM) {
                                        fs.appendFileSync(
                                            `gem-${BACKTEST_IND}.csv`,
                                            `\n${symbol},${_defaultAmount},${_defaultDuration},${_maxPositionLimit},${_maxPositionAmount},${_interval},${_smaSampleCount},${_smaBaseSampleCount},${result.totalProfit},${result.totalAmount},${result.totalTrades},${result.winRate},${result.maxPlus},${result.maxMinus}`
                                        );
                                    }

                                    if (isSavingBalanceTrack) {
                                        fs.writeFileSync(
                                            `balance_track_${BACKTEST_IND}/${symbol.replace('/', '_')}_DD_${settings.defaultDuration}_MP_${settings.maxPositionLimit}_MA_${settings.maxPositionAmount}_SMA_${settings.smaSampleCount}_SMAB_${settings.smaBaseSampleCount}.csv`,
                                            result.balanceTrack.join('\n\n')
                                        );
                                    }

                                    if (isSavingPriceTrack) {
                                        savePriceTrack(
                                            `price_track_${BACKTEST_IND}/${symbol.replace('/', '_')}.csv`,
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
    }
}));