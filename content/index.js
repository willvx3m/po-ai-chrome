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
  readSettings((settings) => {
    console.log('[run] settings', settings);
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
          console.log('[run] No active position, creating a new position');

          // TODO: we should read RSI - RSI > 70 -> create a BUY, RSI < 30 -> create a SELL
          const marketSentiment = getMarketSentiment();
          if (parseInt(marketSentiment) > 70) {
            console.log(`[run] MSentiment ${marketSentiment} is > 70, creating a BUY position`);
            const newPositionAmount = settings.defaultAmount;
            const newPositionDuration = settings.defaultDuration;
            setEndTime(newPositionDuration, () => {
              console.log('[run] Position duration set', newPositionDuration);
              createPosition(newPositionAmount, 'BUY', () => {
                console.log('[run] Position created', newPositionAmount, 'BUY');
              });
            });
          } else if (parseInt(marketSentiment) < 30) {
            console.log(`[run] MSentiment ${marketSentiment} is < 30, creating a SELL position`);
            const newPositionAmount = settings.defaultAmount;
            const newPositionDuration = settings.defaultDuration;
            setEndTime(newPositionDuration, () => {
              console.log('[run] Position duration set', newPositionDuration);
              createPosition(newPositionAmount, 'SELL', () => {
                console.log('[run] Position created', newPositionAmount, 'SELL');
              });
            });
          } else {
            console.log(`[run] MSentiment ${marketSentiment} is ${marketSentiment}, not creating a position`);
          }
        } else {
          getActivePositions((positions, price, amount, outcome, timeLeft) => {
            console.log('[run] Current positions:', positions);
            console.log(`[run] Current price: ${price}, Total amount: ${amount}, Total outcome: ${outcome}, Time left: ${timeLeft}`);

            const newPosition = calculateNextPosition(positions, price, payoutNumber);
            const maxPositionLimit = settings.maxPositionLimit;
            const newPositionRequired = newPosition && positions.length < maxPositionLimit;
            if (newPositionRequired) {
              console.log(`[run] New position amount: ${newPosition.amount}, direction: ${newPosition.direction}, minNetProfit: ${newPosition.minNetProfit}`);
              createPosition(newPosition.amount, newPosition.direction, () => {
                console.log('[run] Position created');
              });
            }

            delete positions;
          });
        }
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
  if (!amount || amount * 1 < 1) {
    return;
  }

  setPositionAmount(amount, () => {
    console.log('[setAmount] Amount set');
  });
}