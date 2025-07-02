const fs = require('fs').promises;
const path = require('path');
const { parse } = require('csv-parse/sync');

// Default max correction percentage if not specified
const DEFAULT_MAX_CORRECTION = 10;

async function analyzeCSVFiles(folderPath, maxCorrectionPercent = DEFAULT_MAX_CORRECTION, outputFolder) {
    try {
        // Validate input folder
        if (!(await fs.stat(folderPath)).isDirectory()) {
            throw new Error('Input folder does not exist');
        }

        // Create output folder if specified
        if (outputFolder) {
            await fs.mkdir(outputFolder, { recursive: true });
        }

        // Read all files in folder
        const files = (await fs.readdir(folderPath)).filter(file => file.endsWith('.csv'));
        const results = [];

        for (const file of files) {
            const filePath = path.join(folderPath, file);
            try {
                // Read and parse CSV
                const content = await fs.readFile(filePath);
                const values = parse(content, {
                    skip_empty_lines: true,
                    cast: value => parseFloat(value)
                }).flat();

                // Analyze trend
                const analysis = analyzeTrend(values, maxCorrectionPercent);
                if (analysis.isRising) {
                    results.push({
                        file,
                        maxCorrection: analysis.maxCorrection,
                        growthDepth: analysis.growthDepth,
                        allowedCorrection: analysis.allowedCorrection
                    });
                }
            } catch (err) {
                console.error(`Error processing ${file}: ${err.message}`);
            }
        }

        // Display results
        console.log('\nFiles with consistent rising trend:');
        if (results.length === 0) {
            console.log('No files meet the rising trend criteria');
            return;
        }

        for (const result of results) {
            console.log(`\nFile: ${result.file}`);
            console.log(`Max Correction: ${result.maxCorrection.toFixed(2)}, Growth Depth: ${result.growthDepth.toFixed(2)}, Allowed Correction: ${result.allowedCorrection.toFixed(2)}`);

            // Copy CSV file and corresponding PNG (if exists) to output folder
            if (outputFolder) {
                const csvPath = path.join(folderPath, result.file);
                const outputCsvPath = path.join(outputFolder, result.file);
                await fs.copyFile(csvPath, outputCsvPath);

                // Check for and copy corresponding PNG file
                const pngFile = result.file.replace('.csv', '.png');
                const pngPath = path.join(folderPath, pngFile);
                const outputPngPath = path.join(outputFolder, pngFile);
                try {
                    await fs.access(pngPath); // Check if PNG exists
                    await fs.copyFile(pngPath, outputPngPath);
                    console.log(`Copied ${pngFile} to output folder`);
                } catch {
                    console.log(`Note: ${pngFile} not found, skipping`);
                }
            }
        }

        console.log('Total files:', files.length);
        console.log('Total files with consistent rising trend:', results.length);
    } catch (err) {
        console.error('Error:', err.message);
    }
}

function analyzeTrend(values, maxCorrectionPercent) {
    if (values.length < 2) {
        return { isRising: false };
    }

    // Calculate global max and min values
    const maxValue = Math.max(...values);
    const minValue = Math.min(...values);
    const growthDepth = maxValue - minValue;
    const allowedCorrection = growthDepth * maxCorrectionPercent / 100;

    let currentMax = values[0];
    let maxCorrection = 0;
    let isRising = true;

    for (let i = 1; i < values.length; i++) {
        if (values[i] < values[i - 1]) {
            const correction = currentMax - values[i];
            if (correction > allowedCorrection) {
                isRising = false;
                break;
            } else if (maxCorrection < correction) {
                maxCorrection = correction;
            }
        } else {
            // Update current max
            currentMax = Math.max(currentMax, values[i]);
        }
    }

    return {
        isRising,
        maxCorrection,
        growthDepth,
        allowedCorrection
    };
}

// Parse command line arguments
function parseArgs() {
    const args = process.argv.slice(2);
    if (args.length < 1) {
        console.error('Usage: node script.js <folderPath> [maxCorrectionPercent] [outputFolder]');
        process.exit(1);
    }

    return {
        folderPath: args[0],
        maxCorrectionPercent: parseFloat(args[1]) || DEFAULT_MAX_CORRECTION,
        outputFolder: args[2]
    };
}

// Main execution
async function main() {
    const { folderPath, maxCorrectionPercent, outputFolder } = parseArgs();
    await analyzeCSVFiles(folderPath, maxCorrectionPercent, outputFolder);
}

main().catch(err => console.error('Fatal error:', err.message));