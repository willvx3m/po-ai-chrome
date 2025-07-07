const fs = require('fs');
const { findBestMatchingCurve, getPriceCurveStatus } = require('./utils');

// Sample corpus of historical price curve statuses
// const corpus = JSON.parse(fs.readFileSync(`corpus/MARTINGALE-po-AUD-CAD-OTC-INPUT.json`, 'utf8'));
const corpus = JSON.parse(fs.readFileSync(`corpus/MARTINGALE-po-EUR-USD-OTC-INPUT.json`, 'utf8'));

// Sample priceBook
const priceBook = Array(50 * 6).fill().map((_, i) => 1.1500 + Math.random() * 0.01);
const currentStatus = getPriceCurveStatus(priceBook);
console.log('currentStatus', currentStatus);

const bestMatch = findBestMatchingCurve(currentStatus, corpus);
console.log(bestMatch);
// Output example:
// {
//   ema9: 100.12345678,
//   ema21: 99.87654321,
//   sma50: 98.34567890,
//   priceToEMA9: 1.00234567,
//   timestamp: "2025-07-01",
//   corpusIndex: 0,
//   distance: 0.01234567
// }