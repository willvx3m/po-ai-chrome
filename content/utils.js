function getCurrentPayout() {
    const payout = document.querySelector('div.value__val-start');
    if (!payout) {
        console.warn('[getCurrentPayout] No payout found');
        return 0;
    }
    const payoutNumber = parseFloat(payout.innerText);
    return payoutNumber;
}

function changeTopPairAndOpenActiveTrades() {
    const pairDropDown = document.querySelector('a.pair-number-wrap');
    pairDropDown.click();

    setTimeout(() => {
        const topPair = document.querySelector('ul.alist-currency li.alist__item:first-child a');
        // console.log('[changePair] topPair', topPair);
        topPair.click();

        setTimeout(() => {
            const openedTradesTab = document.querySelector('div.widget-slot__header ul li:first-child a');
            // console.log('[changePair] openedTradesTab', openedTradesTab);
            openedTradesTab.click();

            const assetsDropdown = document.querySelector('div.drop-down-modal div.assets-block');
            if (assetsDropdown) {
                assetsDropdown.remove();
            }
        }, 200);
    }, 200);
}

function hasActivePosition() {
    const activePositions = document.querySelectorAll('div.deals-list__item div.animated');
    return activePositions.length;
}

function createPositionUsingAI() {
    const button = document.querySelector('a.ai-trading-btn');
    if (button) {
        button.click();
    }
}