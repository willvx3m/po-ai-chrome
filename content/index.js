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
      settings = DEFAULT_SETTINGS;
      saveSettings(settings);
    }

    const enabled = settings.enabled;
    if (!enabled) {
      console.log('[run] Extension is disabled');
      return;
    }

    const minPayout = settings.minPayout || 90;
    const payoutNumber = getCurrentPayout();
    const currentPair = getCurrentPair();
    const hasActivePosition = getHasActivePosition();

    const notPreferredPair = (settings.includeOTC === false && currentPair.includes('OTC')) || (settings.preferredPair && !currentPair.includes(settings.preferredPair));
    const payoutNotEnough = payoutNumber < minPayout;
    const shouldChangePair = notPreferredPair || (payoutNotEnough && !settings.preferredPair);
    const isRestartRequired = restartRequired(settings);

    if (!hasActivePosition && isRestartRequired) {
      console.log('[run] Restart required, reloading window');
      saveSettings(settings);
      window.location.reload();
      return;
    }

    if (!settings.urlQueryServer) {
      console.log('[run] Please set the Query server URL');
    }

    if (!settings.priceBook?.book) {
      settings.priceBook = {
        pair: currentPair,
        book: [],
      }
    }

    if (!hasActivePosition && !shouldChangePair && (settings.priceBook.book.length < settings.smaSampleCount || payoutNotEnough || !settings.urlQueryServer)) {
      console.log('[run] Gathering price for', currentPair, ' Book length:', settings.priceBook.book.length);
      if (openPendingTrades()) {
        const currentPrice = getCurrentPrice();
        if (currentPrice) {
          insertPriceBook(settings, currentPrice, currentPair);
          saveSettings(settings);
        }
      }
    } else if (!hasActivePosition && shouldChangePair) {
      console.log(`[run] Pair change required, CurrentPayout: (${payoutNumber}), MinPayout: (${minPayout}), includeOTC: (${settings.includeOTC}), PreferredPair: (${settings.preferredPair})`);
      changeTopPairAndOpenActiveTrades(minPayout, settings.includeOTC, settings.preferredPair);
      return;
    } else if (openActiveTrades()) {
      if (!hasActivePosition) {
        const currentBalance = getCurrentBalance();
        const currentQTMode = getCurrentQTMode();
        console.log('[run] Balance:', currentBalance, 'Mode:', currentQTMode, 'Payout:', payoutNumber);
        settings.savedBalance = currentBalance;
        settings.savedQTMode = currentQTMode;
        if (settings.baseAmount) {
          settings.defaultAmount = getDefaultAmount(currentBalance, settings.multiplier, settings.baseAmount, settings.riskDepth);
        }

        setTimeout(() => {

          fetch(settings.urlQueryServer, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              symbol: currentPair,
              priceBook: settings.priceBook.book
            })
          })
            .then(response => response.json())
            .then(data => {
              console.log('[run] Query server response:', data);
              if (!data.success) {
                console.log('[run] Invalid response from query server, skipping position creation');
                return;
              }

              const strategy = data.bestMatch?.settings || {};
              console.log('[run] Strategy:', strategy.defaultDuration, strategy.maxPositionLimit, strategy.defaultDirection);

              if (strategy.defaultDuration) {
                settings.defaultDuration = data.defaultDuration;
              }
              if (strategy.maxPositionLimit) {
                settings.maxPositionLimit = data.maxPositionLimit;
              }
              if (strategy.defaultDirection) {
                settings.defaultDirection = data.defaultDirection;
              }
              if (strategy.defaultAmount) {
                settings.defaultAmount = data.defaultAmount;
              }

              saveSettings(settings);
              createStartingPosition(settings);

              const messageData = {
                userName: settings.userName,
                pair: currentPair,
                payout: payoutNumber,
                qtMode: currentQTMode,
                balance: currentBalance,
              };

              if (settings.slackChannelID) {
                setTimeout(() => sendSlackMessage(JSON.stringify(messageData)), 0);
              }

              if (settings.urlDataServer) {
                setTimeout(() => sendDataToServer(settings.urlDataServer, messageData), 0);
              }
            })
            .catch(error => {
              console.error('[run] Error querying server:', error);
            });
        }, 0);
      } else {
        getActivePositions((positions, price, amount, outcome, timeLeft) => {
          console.log('[run] Positions:', positions);
          console.log(`[run] Price: ${price}, Amount: ${amount} -> ${outcome}, ${timeLeft} left`);

          if (price === 0) {
            console.info('[run] Failed to get price, skipping');
            return;
          }

          insertPriceBook(settings, price, currentPair);
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