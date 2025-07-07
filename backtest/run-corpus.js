// Generate Corpus
// [Input]
// 1. SYMBOL
// 2. SETTINGS[] (defaultDuration, maxPositionLimit, defaultDirection)
// 3. PRICES[] (price, timestamp)

// [Output]
// CORPUS

const fs = require('fs');
const { getPrices, simulateSingle, getPriceCurveStatus } = require('./utils');
const { DEFAULT_SETTINGS } = require(
    './strategy/strategy-martingale'
);

const SYMBOLS = [
    // 'po-AUD-CAD-OTC-INPUT',
    'po-AUD-USD-OTC-INPUT',
    // 'po-EUR-USD-OTC-INPUT',
];
const strategyName = DEFAULT_SETTINGS.name || 'DEFAULT';

const SETTINGS = [];
for (var _defaultDuration = 3; _defaultDuration <= 10; _defaultDuration++) {
    for (var _maxPositionLimit = 3; _maxPositionLimit <= 6; _maxPositionLimit++) {
        ['BUY', 'SELL'].forEach(_direction => {
            const settings = { ...DEFAULT_SETTINGS };
            settings.defaultDuration = _defaultDuration;
            settings.maxPositionLimit = _maxPositionLimit;
            settings.defaultDirection = _direction;
            SETTINGS.push(settings);
        });
    }
}

// console.log(SETTINGS.length);
// SETTINGS.forEach(settings => {
//     console.log(`${settings.defaultDuration} ${settings.maxPositionLimit} ${settings.defaultDirection}`);
// });

SYMBOLS.forEach(symbol => getPrices(symbol, `ohlcv/${symbol.replace('/', '_')}.csv`, (prices) => {
    const priceBook = [];
    const corpus = [];
    for (let i = 0; i < prices.length; i++) {
        const currentPrice = prices[i].price;
        priceBook.push(currentPrice);

        const screenshot = getPriceCurveStatus(priceBook);
        let maxProfit = 0;
        let maxProfitSettings = null;

        SETTINGS.forEach(settings => {
            const result = simulateSingle(prices, settings, i);
            if (result && result.profit > maxProfit) {
                maxProfit = result.profit;
                maxProfitSettings = settings;
            }
        });

        if (!maxProfit) {
            continue;
        }

        console.log(
            `#`, i, ` - `,
            `Price:`, screenshot.price,
            `EMA9:`, screenshot.ema9,
            `EMA21:`, screenshot.ema21,
            `SMA50:`, screenshot.sma50,
            `PriceToEMA9:`, screenshot.priceToEMA9, ` - `,
            `Profit:`, maxProfit.toFixed(2), `,`,
            `Settings:`, `${maxProfitSettings.defaultDuration}/${maxProfitSettings.maxPositionLimit}/${maxProfitSettings.defaultDirection}`
        );

        corpus.push({
            price: screenshot.price,
            ema9: screenshot.ema9,
            ema21: screenshot.ema21,
            sma50: screenshot.sma50,
            priceToEMA9: screenshot.priceToEMA9,
            settings: {
                defaultDuration: maxProfitSettings.defaultDuration,
                maxPositionLimit: maxProfitSettings.maxPositionLimit,
                defaultDirection: maxProfitSettings.defaultDirection
            }
        });
    }

    fs.writeFileSync(`corpus/${strategyName}-${symbol.replace('/', '_')}.json`, JSON.stringify(corpus, null, 2));
}));