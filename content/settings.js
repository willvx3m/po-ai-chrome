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