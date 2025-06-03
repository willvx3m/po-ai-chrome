function readSettings(callback) {
    chrome.storage?.local?.get(['settings'], (data) => {
        const settings = data.settings || {};
        callback(settings);
    });
}

function saveSettings(settings) {
    chrome.storage.local.set({ settings }, () => {
        console.log('[saveSettings] Settings updated:', settings);
    });
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