const express = require('express');
const fs = require('fs');
const path = require('path');
const { findBestMatchingCurve, getPriceCurveStatus } = require('../utils');

const app = express();
const PORT = process.env.PORT || 3000;

// CORS middleware to allow any request from any host
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
    
    // Handle preflight requests
    if (req.method === 'OPTIONS') {
        res.sendStatus(200);
    } else {
        next();
    }
});

// Middleware to parse JSON requests
app.use(express.json());

// CSV logging function
function logToCSV(data) {
    const logFile = path.join(__dirname, 'api-logs.csv');
    const timestamp = new Date().toISOString();
    
    // Check if file exists to write header
    const fileExists = fs.existsSync(logFile);
    
    const csvLine = [
        timestamp,
        data.symbol || 'unknown',
        data.price,
        data.current_ema9,
        data.current_ema21,
        data.current_sma50,
        data.current_priceToEMA9,
        data.match_corpus_index,
        data.distance,
        data.match_ema9,
        data.match_ema21,
        data.match_sma50,
        data.match_priceToEMA9,
        data.match_setting_duration,
        data.match_setting_position_limit,
        data.match_direction
    ].map(field => `"${field || ''}"`).join(',');
    
    const header = 'timestamp,symbol,price,current_ema9,current_ema21,current_sma50,current_priceToEMA9,match_corpus_index,distance,match_ema9,match_ema21,match_sma50,match_priceToEMA9,match_setting_duration,match_setting_position_limit,match_direction';
    
    if (!fileExists) {
        fs.writeFileSync(logFile, header + '\n');
    }
    
    fs.appendFileSync(logFile, csvLine + '\n');
    console.log(`Logged API call to ${logFile}`);
}

// Load corpus data
let corpusData = null;

function loadCorpus(corpusFile = 'MARTINGALE-po-EUR-USD-OTC-INPUT.json') {
    try {
        const corpusPath = path.join(__dirname, corpusFile);
        const data = fs.readFileSync(corpusPath, 'utf8');
        corpusData = JSON.parse(data);
        console.log(`Loaded corpus with ${corpusData.length} entries from ${corpusFile}`);
        return true;
    } catch (error) {
        console.error(`Error loading corpus from ${corpusFile}:`, error.message);
        return false;
    }
}

// Load default corpus on startup
loadCorpus();

// Endpoint to get best matching curve
app.post('/api/find-best-match', (req, res) => {
    try {
        const { priceBook, corpusFile } = req.body;

        // Validate input
        if (!priceBook || !Array.isArray(priceBook)) {
            return res.status(400).json({
                error: 'priceBook is required and must be an array'
            });
        }

        // Load corpus if different file is requested
        if (corpusFile && corpusFile !== 'MARTINGALE-po-EUR-USD-OTC-INPUT.json') {
            if (!loadCorpus(corpusFile)) {
                return res.status(400).json({
                    error: `Failed to load corpus file: ${corpusFile}`
                });
            }
        }

        // Ensure corpus is loaded
        if (!corpusData) {
            return res.status(500).json({
                error: 'Corpus data not loaded'
            });
        }

        // Get current price curve status
        const currentStatus = getPriceCurveStatus(priceBook);
        
        if (!currentStatus) {
            return res.status(400).json({
                error: 'Unable to calculate price curve status from provided priceBook'
            });
        }

        // Find best matching curve
        const bestMatch = findBestMatchingCurve(currentStatus, corpusData);

        if (!bestMatch) {
            return res.status(404).json({
                error: 'No matching curve found'
            });
        }

        // Log to CSV
        const logData = {
            symbol: req.body.symbol || 'unknown',
            price: currentStatus.price,
            current_ema9: currentStatus.ema9,
            current_ema21: currentStatus.ema21,
            current_sma50: currentStatus.sma50,
            current_priceToEMA9: currentStatus.priceToEMA9,
            match_corpus_index: bestMatch.corpusIndex,
            distance: bestMatch.distance || 0,
            match_ema9: bestMatch.ema9,
            match_ema21: bestMatch.ema21,
            match_sma50: bestMatch.sma50,
            match_priceToEMA9: bestMatch.priceToEMA9,
            match_setting_duration: bestMatch.settings?.defaultDuration || '',
            match_setting_position_limit: bestMatch.settings?.maxPositionLimit || '',
            match_direction: bestMatch.settings?.defaultDirection || ''
        };
        
        logToCSV(logData);

        // Return the best match with additional metadata
        res.json({
            success: true,
            currentStatus,
            bestMatch: {
                corpusIndex: bestMatch.corpusIndex,
                distance: bestMatch.distance,
                settings: bestMatch.settings,
                timestamp: bestMatch.timestamp,
                price: bestMatch.price,
                ema9: bestMatch.ema9,
                ema21: bestMatch.ema21,
                sma50: bestMatch.sma50,
                priceToEMA9: bestMatch.priceToEMA9
            }
        });

    } catch (error) {
        console.error('Error processing request:', error);
        res.status(500).json({
            error: 'Internal server error',
            message: error.message
        });
    }
});

// Endpoint to list available corpus files
app.get('/api/corpus-files', (req, res) => {
    try {
        const files = fs.readdirSync(__dirname)
            .filter(file => file.endsWith('.json'))
            .map(file => ({
                name: file,
                size: fs.statSync(path.join(__dirname, file)).size
            }));
        
        res.json({
            success: true,
            files
        });
    } catch (error) {
        res.status(500).json({
            error: 'Failed to list corpus files',
            message: error.message
        });
    }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        corpusLoaded: corpusData !== null,
        corpusSize: corpusData ? corpusData.length : 0
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Available endpoints:`);
    console.log(`  POST /api/find-best-match - Find best matching price curve`);
    console.log(`  GET  /api/corpus-files - List available corpus files`);
    console.log(`  GET  /api/health - Health check`);
});

module.exports = app;
