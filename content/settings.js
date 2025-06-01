// Draw settings
// {
//   "defaultAmount": 10, // 10 USD
//   "defaultDuration": 10, // 10 minutes
//   "maxPositionLimit": 10, // 10 positions
//   "interval": 10000, // 10 seconds
//   "defaultDirection": "BUY"
// }

function readSettings(callback){
    chrome.storage?.local?.get(['settings'], (data) => {
        const settings = data.settings || {};
        if(!settings.defaultAmount){
            settings.defaultAmount = 2;
        }
        if(!settings.defaultDuration){
            settings.defaultDuration = 5;
        }
        if(!settings.maxPositionLimit){
            settings.maxPositionLimit = 6; // 3 pairs for leb
        }
        if(!settings.interval){
            settings.interval = 10000;
        }
        if(!settings.defaultDirection){
            settings.defaultDirection = 'BUY';
        }

        callback(settings);
    });
}