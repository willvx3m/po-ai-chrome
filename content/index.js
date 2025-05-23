chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  console.log('[Content.onMessage] request:', request);
  if (request.action === 'run') {
    run();
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
    console.log('[run] Payout is', payoutNumber);
    if (payoutNumber < 90) {
      console.log('[run] Payout is less than 90, changing to top pair');
      changeTopPairAndOpenActiveTrades();
    } else {
      console.log('[run] Payout is greater than 90, do your thing...');
      // createPositionUsingAI();
    }

    const interval = settings.interval || 5000;
    setTimeout(run, interval);
  });
}