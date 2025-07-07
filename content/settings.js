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

function insertPriceBook(settings, price, currentPair) {
    if (!settings.priceBook) {
        settings.priceBook = {
            pair: currentPair,
            book: [price]
        };
        return;
    }
    if (settings.priceBook.pair != currentPair) {
        settings.priceBook.pair = currentPair;
        settings.priceBook.book = [price];
    } else {
        settings.priceBook.book.push(price);

        const maxPriceBookLength = settings.maxPriceBookLength || 60;
        while (settings.priceBook?.book && settings.priceBook.book.length > maxPriceBookLength) {
            settings.priceBook.book.shift();
        }
    }
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