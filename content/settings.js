function readSettings(callback) {
    chrome.storage?.local?.get(['settings'], (data) => {
        const settings = data.settings;
        callback(settings);
    });
}

function saveSettings(settings) {
    chrome.storage.local.set({ settings }, () => {
        console.log('[saveSettings] Settings updated:', settings);
    });
}

function checkAndSaveSettings(settings, currentBalance) {
    // Skip flipping direction
    // if (settings.savedBalance) {
    //     const profit = currentBalance - settings.savedBalance;
    //     const fullFailureSum = 1 + 2 + 4 + 8;
    //     if (profit < 0 && Math.abs(profit) >= fullFailureSum) {
    //         settings.defaultDirection = settings.defaultDirection === 'BUY' ? 'SELL' : 'BUY';
    //         console.log('[cAS] Full failure detected, flipping direction to', settings.defaultDirection);
    //     }
    // }

    settings.savedBalance = currentBalance;
    saveSettings(settings);
}

function restartRequired(settings) {
    const previousRestart = settings.previousRestart;
    const now = (new Date()) * 1;
    if (previousRestart) {
        const buffer = 60 * 60 * 1000; // 1 hour
        if (now - previousRestart * 1 > buffer) {
            settings.previousRestart = now;
            return true;
        }
        
        return false;
    } else {
        settings.previousRestart = now;
        return true;
    }
}