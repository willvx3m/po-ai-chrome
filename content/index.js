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

    if (openActiveTrades()) {
      const currentBalance = getCurrentBalance();
      const currentQTMode = getCurrentQTMode();
      const payoutNumber = getCurrentPayout();
      console.log('[run] Balance:', currentBalance, 'Mode:', currentQTMode, 'Payout:', payoutNumber);

      if (!hasActivePosition()) {
        if (payoutNumber < 90) {
          console.log(`[run] Payout (${payoutNumber}) is less than 90, changing to top pair`);
          changeTopPairAndOpenActiveTrades();
        } else {
          console.log('[run] No active position, creating a new position');
          if (restartRequired(settings)) {
            console.log('[run] Restart required, reloading window');
            saveSettings(settings);
            window.location.reload();
            return;
          }

          checkAndSaveSettings(settings, currentBalance);
          createStartingPosition(settings);

          const message = `[${settings.userName} (${settings.name})] ${currentBalance}, ${currentQTMode}`;
          setTimeout(() => sendSlackMessage(message), 0);
        }
      } else {
        getActivePositions((positions, price, amount, outcome, timeLeft) => {
          console.log('[run] Positions:', positions);
          console.log(`[run] Price: ${price}, Amount: ${amount} -> ${outcome}, ${timeLeft} left`);

          if (price === 0) {
            console.warn('[run] Failed to get price, skipping');
            return;
          }

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