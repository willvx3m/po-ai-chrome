// Getters
function getCurrentPayout() {
    const payout = document.querySelector('div.value__val-start');
    if (!payout) {
        console.warn('[getCurrentPayout] No payout found');
        return 0;
    }
    const payoutNumber = parseFloat(payout.innerText);
    return payoutNumber;
}

function getMarketSentiment() {
    return document.querySelector('div.market-watch-panel__content span.pb__number-start').innerText;
}

function hasActivePosition() {
    const activePositions = document.querySelectorAll('div.deals-list__item div.animated');
    return activePositions.length;
}

function getActivePositions(callback) {
    const positions = [];

    const activePositions = document.querySelectorAll('div.deals-list__item div.animated');
    activePositions.forEach(position => {
        positions.push({
            pair: position.querySelector('div.item-row:first-child > div > a').innerText,
            direction: position.querySelector('i.fa-arrow-up') ? 'BUY' : 'SELL',
            profit: parseInt(position.querySelector('div.item-row:first-child span.price-up').innerText),
            amount: 0,
            openPrice: 0,
            currentProfit: 0,
            currentPrice: 0,
            timeLeft: position.querySelector('div.item-row:first-child div:last-child').innerText,

            domElement: position,
            buttonTogglePosition: position.querySelector('div.item-row:first-child span.price-up'),
        });
    });

    if (activePositions.length === 0) {
        console.log('[getCurrentPositionsAndPrice] No active positions');
        return;
    }

    const getPositionDetail = (positions, index) => {
        if (index >= positions.length) {
            if (callback) {
                var price = positions[0].currentPrice;
                var amount = positions.reduce((acc, position) => acc + position.amount, 0);
                var outcome = positions.reduce((acc, position) => acc + position.outcome, 0);
                var timeLeft = positions[0].timeLeft;
                callback(positions, price, amount, outcome, timeLeft);
            }

            return;
        }

        const position = positions[index];
        const positionDom = position.domElement;
        const buttonTogglePosition = position.buttonTogglePosition;
        const isPositionShowingFullInfo = positionDom.classList.contains('open-full-info');

        if (!isPositionShowingFullInfo) {
            buttonTogglePosition.click();
        }

        setTimeout(() => {
            if (!positionDom.querySelector('div.price-info__prices div.price-info__prices-item:first-child')) {
                console.warn('[getPositionDetail] Position box not opened');
                return;
            }

            position.openPrice = 1 * positionDom.querySelector('div.price-info__prices div.price-info__prices-item:first-child').innerText.replace('Open price:\n', '').trim();
            position.currentPrice = 1 * positionDom.querySelector('div.price-info__prices div.price-info__prices-item:nth-child(2)').innerText.replace('Current price:\n', '').trim();
            position.amount = 1 * positionDom.querySelector('div.forecast > div:first-child > div:nth-child(2) span').innerText.replace('$', '').trim();
            position.currentProfit = 1 * positionDom.querySelector('div.forecast > div:first-child > div:nth-child(3) span').innerText.replace('$', '').replace('+', '').trim();
            position.outcome = position.currentProfit > 0 ? position.amount + position.currentProfit : 0;

            price = position.currentPrice;

            buttonTogglePosition.click();
            setTimeout(() => {
                getPositionDetail(positions, index + 1);
            }, 0);
        }, 0);
    }

    getPositionDetail(positions, 0);
}

function getEndTime() {
    const endTime = document.querySelector('div.block--expiration-inputs div.value__val');
    if (!endTime) {
        console.warn('[getEndTime] No end time found');
        return 0;
    }

    return endTime.innerText;
}

// Actions
function changeTopPairAndOpenActiveTrades() {
    const pairDropDown = document.querySelector('a.pair-number-wrap');
    pairDropDown.click();

    setTimeout(() => {
        const topPair = document.querySelector('ul.alist-currency li.alist__item:first-child a');
        // console.log('[changePair] topPair', topPair);
        topPair.click();

        setTimeout(() => {
            openActiveTrades();
            // Keep the modal open
            // const assetsDropdown = document.querySelector('div.drop-down-modal div.assets-block');
            // if (assetsDropdown) {
            //     assetsDropdown.remove();
            // }
        }, 500);
    }, 500);
}

function openActiveTrades() {
    const activeTab = document.querySelector('div.widget-slot__header ul li.active a');
    if (activeTab.innerText === 'Opened') {
        return true;
    }

    console.log('[openActiveTrades] Opening opened trades tab');

    const openedTradesTab = document.querySelector('div.widget-slot__header ul li:first-child a');
    openedTradesTab.click(); // TODO: check if this is correct

    return false;
}

function setEndTime(time, callback) {
    var timeLabel = document.querySelector('div.block--expiration-inputs div.block__title');
    if (!timeLabel) {
        console.warn('[setEndTime] No time label found');
        return;
    }

    if (timeLabel.innerText === 'Time') { // Switch from duration-based to timepoint-based
        const timeToggleButton = document.querySelector('div.block--expiration-inputs div.control__buttons');
        if (!timeToggleButton) {
            console.warn('[setEndTime] Failed to set time: toggle button not found');
            return;
        }

        timeToggleButton.click();
    }

    setTimeout(() => {
        const timeBlock = document.querySelector('div.block.block--expiration-inputs div.block__control.control div.value__val');
        timeBlock.click();

        setTimeout(() => {
            const btnMinusHour = document.querySelector('div.trading-panel-modal__in > div.rw:nth-child(1) a.btn-minus');
            const btnPlusMinute = document.querySelector('div.trading-panel-modal__in > div.rw:nth-child(2) a.btn-plus');

            const clickButtons = [
                btnMinusHour,
                btnMinusHour,
                ...Array(time * 1).fill(btnPlusMinute),
                timeBlock,
            ];

            const funcClickButton = (index) => {
                if (index < 0 || index >= clickButtons.length) {
                    if (callback) {
                        callback();
                    }
                    return;
                }

                const button = clickButtons[index];
                button.click();
                setTimeout(() => {
                    funcClickButton(index + 1);
                }, 100);
            }

            funcClickButton(0);
        }, 0);
    }, 0);
}

function setPositionAmount(amount, callback) {
    const amountBlock = document.querySelector('div.block.block--bet-amount div.block__control.control div.value__val');
    amountBlock.click();

    setTimeout(() => {
        const btnClear = document.querySelector('div.trading-panel-modal__in div.virtual-keyboard__keys div.virtual-keyboard__col:last-child');
        const btnDigits = document.querySelectorAll('div.trading-panel-modal__in div.virtual-keyboard__keys div.virtual-keyboard__col div.virtual-keyboard__input');
        const btnDigitsMap = {};
        btnDigits.forEach(btn => {
            const digit = btn.innerText;
            btnDigitsMap[digit] = btn;
        });

        const amountDigits = String(amount * 1).split('');

        const clickButtons = [
            btnClear,
            btnClear,
            btnClear,
            ...amountDigits.map(digit => btnDigitsMap[digit]),
            amountBlock,
        ];

        const funcClickButton = (index) => {
            if (index < 0 || index >= clickButtons.length) {
                if (callback) {
                    callback();
                }
                return;
            }

            const button = clickButtons[index];
            button.click();
            setTimeout(() => {
                funcClickButton(index + 1);
            }, 1);
        }

        funcClickButton(0);
    }, 0);
}

// Create position
function createPositionUsingAI() {
    const button = document.querySelector('a.ai-trading-btn');
    if (button) {
        button.click();
    }
}

function createPosition(amount, direction, callback) {
    if (amount === 0) {
        const btnBuy = document.querySelector('div.tour-action-buttons-container a.btn-call');
        const btnSell = document.querySelector('div.tour-action-buttons-container a.btn-put');

        if (direction === 'BUY') {
            btnBuy.click();
        } else {
            btnSell.click();
        }

        if (callback) {
            callback();
        }
        return;
    }

    setPositionAmount(amount, () => {
        setTimeout(() => {
            const btnBuy = document.querySelector('div.tour-action-buttons-container a.btn-call');
            const btnSell = document.querySelector('div.tour-action-buttons-container a.btn-put');

            if (direction === 'BUY') {
                btnBuy.click();
            } else {
                btnSell.click();
            }

            if (callback) {
                callback();
            }
        }, 0);
    });
}