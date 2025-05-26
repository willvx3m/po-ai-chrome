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

// Main function
function run() {
  chrome.storage.local.get(['settings'], (data) => {
    console.log('[run] settings', data.settings);
    const settings = data.settings || {};
    const enabled = settings.enabled;
    if (!enabled) {
      console.log('[run] Extension is disabled');
      return;
    }

    const payoutNumber = getCurrentPayout();
    // console.log('[run] Payout is', payoutNumber);
    if (payoutNumber < 90) {
      console.log(`[run] Payout (${payoutNumber}) is less than 90, changing to top pair`);
      changeTopPairAndOpenActiveTrades();
    } else {
      if (openActiveTrades()) {
        if (!hasActivePosition()) {
          console.log('[run] No active position');
          // TODO: create position: RANDOM??
        } else {
          getActivePositions((positions, price, payout, outcome, timeLeft) => {
            console.log('[run] Current positions:', positions);
            console.log(`[run] Current price: ${price}, Total payout: ${payout}, Total outcome: ${outcome}, Time left: ${timeLeft}`);

            const newPosition = calculateNextPosition(positions, payout, outcome, timeLeft, price);
            const maxPositionLimit = settings.maxPositionLimit || 10;
            const newPositionRequired = newPosition && positions.length < maxPositionLimit;
            if (newPositionRequired) {
              console.log(`[run] New position amount: ${newPosition.amount}, direction: ${newPosition.direction}`);
              createPosition(newPosition.amount, newPosition.direction, () => {
                console.log('[run] Position created');
              });
            }

            delete positions;
          });

          // const marketSentiment = getMarketSentiment();
          // console.log(`[run] Market sentiment: ${marketSentiment}`);
        }
      }
    }

    const interval = settings.interval || 10000;
    setTimeout(run, interval);
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
  if (!amount || amount * 1 < 1) {
    return;
  }

  setPositionAmount(amount, () => {
    console.log('[setAmount] Amount set');
  });
}