chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  console.log('[Content.onMessage] request:', request);
  if (request.action === 'run') {
    run();
  } else if (request.action === 'setDuration') {
    setDuration(request.data);
  } else if (request.action === 'setAmount') {
    setAmount(request.data);
  }
});

window.addEventListener("load", () => {
  console.log("[PO] Window fully loaded, including all resources! Running in 10 seconds...");
  setTimeout(run, 10000);
});

function sendSlackMessage(message) {
  chrome.runtime.sendMessage({
    action: 'sendSlackNotification',
    message: message,
  }, (response) => {
    if (chrome.runtime.lastError) {
      console.error('Error sending slack message:', chrome.runtime.lastError.message);
      return;
    }
  });
}

// Main function
function run() {
  readSettings((settings) => {
    if (!settings) {
      console.log('[run] No settings found, using default settings');
      saveSettings(settings = DEFAULT_SETTINGS);
    }

    const enabled = settings.enabled;
    if (!enabled) {
      console.log('[run] Extension is disabled');
      return;
    }

    if (!settings.priceBook) {
      settings.priceBook = {
        pair: '',
        book: []
      };
    }

    const minPayout = settings.minPayout || 90;
    const payoutNumber = getCurrentPayout();
    const currentPair = getCurrentPair();
    if (currentPair != settings.priceBook.pair) {
      settings.priceBook.pair = currentPair;
      settings.priceBook.book = [];
    }

    const shouldChangePair = payoutNumber < minPayout || (settings.includeOTC === false && currentPair.includes('OTC'));
    if (!shouldChangePair && settings.priceBook.book.length < settings.smaSampleCount) {
      // Gather price
      console.log('[run] Gathering price for', currentPair, ' Book length:', settings.priceBook.book.length);
      if (openPendingTrades()) {
        const currentPrice = getCurrentPrice();
        if (currentPrice) {
          settings.priceBook.book.push(currentPrice);
          saveSettings(settings);
        }
      }
    } else if (openActiveTrades()) {
      const currentBalance = getCurrentBalance();
      const currentQTMode = getCurrentQTMode();
      console.log('[run] Balance:', currentBalance, 'Mode:', currentQTMode, 'Payout:', payoutNumber);

      if (!hasActivePosition()) {
        if (shouldChangePair) {
          console.log(`[run] Pair change required, CurrentPayout: (${payoutNumber}), MinPayout: (${minPayout}), includeOTC: (${settings.includeOTC})`);
          changeTopPairAndOpenActiveTrades(minPayout, settings.includeOTC);
          return;
        } else {
          console.log('[run] No active position, creating a new position');
          if (restartRequired(settings)) {
            console.log('[run] Restart required, reloading window');
            saveSettings(settings);
            window.location.reload();
            return;
          }

          settings.savedBalance = currentBalance;
          settings.savedQTMode = currentQTMode;
          settings.defaultAmount = getDefaultAmount(currentBalance, settings.multiplier, settings.baseAmount, settings.riskDepth);
          saveSettings(settings);
          createStartingPosition(settings);

          const data = {
            userName: settings.userName,
            pair: currentPair,
            payout: payoutNumber,
            qtMode: currentQTMode,
            balance: currentBalance,
          };

          if(settings.slackChannelID){
            setTimeout(() => sendSlackMessage(JSON.stringify(data)), 0);
          }

          if(settings.urlDataServer){
            setTimeout(() => sendDataToServer(settings.urlDataServer, data), 0);
          }
        }
      } else {
        getActivePositions((positions, price, amount, outcome, timeLeft) => {
          console.log('[run] Positions:', positions);
          console.log(`[run] Price: ${price}, Amount: ${amount} -> ${outcome}, ${timeLeft} left`);

          if (price === 0) {
            console.info('[run] Failed to get price, skipping');
            return;
          }

          settings.priceBook.book.push(price);
          settings.priceBook.book.shift();
          saveSettings(settings);

          if (positions.length < settings.maxPositionLimit) {
            const newPosition = calculateNextPosition(positions, price, payoutNumber, settings);
            if (newPosition) {
              createPosition(newPosition.amount, newPosition.direction, () => {
                console.log(`[run] => New position created: ${newPosition.amount} - ${newPosition.direction}`);
              });
            }
          }

          delete positions;
        });
      }
    }

    setTimeout(run, settings.interval);
  });
}

function setDuration(duration) {
  console.log('[setDuration] Duration:', duration);
  if (!duration || duration * 1 < 1) {
    return;
  }

  setEndTime(duration, () => {
    console.log('[setDuration] Duration set');
  });
}

function setAmount(amount) {
  console.log('[setAmount] Amount:', amount);
  if (!amount || amount * 1 <= 0) {
    return;
  }

  setPositionAmount(amount, () => {
    console.log('[setAmount] Amount set');
  });
}