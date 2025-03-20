function clickButton() {
  window.hasRun = false;
  
  chrome.storage.local.get("enabled", (data) => {
    console.log('clickButton, enabled: ', data.enabled);

    if (data.enabled) {
      const payout = document.querySelector('div.value__val-start').innerText;
      console.log('payout', payout);
      const payoutNumber = parseFloat(payout);
      if(payoutNumber < 90) {
        console.log('Payout is', payoutNumber, 'Changing to top pair');
        const pairDropDown = document.querySelector('a.pair-number-wrap');
        console.log('pairDropDown', pairDropDown);
        pairDropDown.click();

        setTimeout(() => {
          const topPair = document.querySelector('ul.alist-currency li.alist__item:first-child a');
          console.log('topPair', topPair);
          topPair.click();
          
          setTimeout(() => {
            const openedTradesTab = document.querySelector('div.widget-slot__header ul li:first-child a');
            console.log('openedTradesTab', openedTradesTab);
            openedTradesTab.click();

            const assetsDropdown = document.querySelector('div.drop-down-modal div.assets-block');
            if(assetsDropdown) {
              assetsDropdown.remove();
            }
          }, 200);
        }, 200);
      } else {
        const hasActivePosition = document.querySelector('div.deals-list__item div.animated') !== null;
        const button = document.querySelector('a.ai-trading-btn');

        if (hasActivePosition) {
          console.log('Already has an active position, skipping');
        } else if (button) {
          button.click();
          console.log('Placed a new order');
        } else {
          console.log('No active position and no button found');
        }
      }

      if(!window.hasRun) {
        setTimeout(clickButton, 5000);
        window.hasRun = true; // block duplicate calls
      }
    }
  });
}

console.log("Content script is running");
if(!window.hasRun) {
  setTimeout(clickButton, 5000);
  window.hasRun = true;
}