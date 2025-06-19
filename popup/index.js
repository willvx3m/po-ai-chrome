var settings = {};

document.addEventListener('DOMContentLoaded', function () {
  chrome.storage.local.get(['settings'], function (result) {
    settings = result.settings || {};
    displaySettings(settings);
  });
});

document.getElementById('toggle').addEventListener('click', () => {
  console.log('[toggle] Current settings', settings);
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs.length > 0) {
      const activeTab = tabs[0];
      const url = new URL(activeTab.url);

      // Check if the domain is pocketoption.com
      if (url.hostname === 'pocketoption.com') {
        settings.enabled = !settings.enabled;
        chrome.storage.local.set({ settings }, () => {
          console.log('[toggle] New settings', settings);
          displaySettings(settings);
          chrome.tabs.sendMessage(activeTab.id, { action: 'run', data: 'Sample Data' });
        });
      } else {
        console.log("The active tab is not on pocketoption.com");
      }
    } else {
      console.log("No active tab found");
    }
  });
});

document.getElementById('buttonDuration').addEventListener('click', () => {
  const duration = document.getElementById('duration').value;
  console.log('[buttonDuration] Duration:', duration);
  dispatchMessageToPO('setDuration', duration);
});

document.getElementById('buttonAmount').addEventListener('click', () => {
  const amount = document.getElementById('amount').value;
  console.log('[buttonAmount] Amount:', amount);
  dispatchMessageToPO('setAmount', amount);
});

document.getElementById('buttonDefaultAmount').addEventListener('click', () => {
  const defaultAmount = document.getElementById('defaultAmount').value;
  console.log('[buttonDefaultAmount] Default Amount:', defaultAmount);
  settings.defaultAmount = defaultAmount * 1;
  chrome.storage.local.set({ settings }, () => {
    console.log('[buttonDefaultAmount] New settings', settings);
    displaySettings(settings);
  });
});

document.getElementById('buttonUserName').addEventListener('click', () => {
  const userName = document.getElementById('userName').value;
  console.log('[buttonUserName] User Name:', userName);
  settings.userName = userName;
  chrome.storage.local.set({ settings }, () => {
    console.log('[buttonUserName] New settings', settings);
    displaySettings(settings);
  });
});

document.getElementById('buttonDefaultDuration').addEventListener('click', () => {
  const defaultDuration = document.getElementById('defaultDuration').value;
  console.log('[buttonDefaultDuration] Default Duration:', defaultDuration);
  settings.defaultDuration = defaultDuration * 1;
  chrome.storage.local.set({ settings }, () => {
    console.log('[buttonDefaultDuration] New settings', settings);
    displaySettings(settings);
  });
});

document.getElementById('buttonMaxPositionLimit').addEventListener('click', () => {
  const maxPositionLimit = document.getElementById('maxPositionLimit').value;
  console.log('[buttonMaxPositionLimit] Max Position Limit:', maxPositionLimit);
  settings.maxPositionLimit = maxPositionLimit * 1;
  chrome.storage.local.set({ settings }, () => {
    console.log('[buttonMaxPositionLimit] New settings', settings);
    displaySettings(settings);
  });
});

document.getElementById('buttonMaxPositionAmount').addEventListener('click', () => {
  const maxPositionAmount = document.getElementById('maxPositionAmount').value;
  console.log('[buttonMaxPositionAmount] Max Position Amount:', maxPositionAmount);
  settings.maxPositionAmount = maxPositionAmount * 1;
  chrome.storage.local.set({ settings }, () => {
    console.log('[buttonMaxPositionAmount] New settings', settings);
    displaySettings(settings);
  });
});

document.getElementById('buttonInterval').addEventListener('click', () => {
  const interval = document.getElementById('interval').value;
  console.log('[buttonInterval] Interval:', interval);
  settings.interval = interval * 1;
  chrome.storage.local.set({ settings }, () => {
    console.log('[buttonInterval] New settings', settings);
    displaySettings(settings);
  });
});

document.getElementById('buttonDefaultDirection').addEventListener('click', () => {
  const defaultDirection = document.getElementById('defaultDirection').value;
  console.log('[buttonDefaultDirection] Default Direction:', defaultDirection);
  settings.defaultDirection = defaultDirection;
  chrome.storage.local.set({ settings }, () => {
    console.log('[buttonDefaultDirection] New settings', settings);
    displaySettings(settings);
  });
});

document.getElementById('testSlackMessage').addEventListener('click', () => {
  chrome.runtime.sendMessage({ action: 'sendSlackNotification', message: 'Hello Slack!' });
});

document.getElementById('buttonSlackChannelID').addEventListener('click', () => {
  const slackChannelID = document.getElementById('slackChannelID').value;
  console.log('[buttonSlackChannelID] Slack Channel ID:', slackChannelID);
  settings.slackChannelID = slackChannelID;
  chrome.storage.local.set({ settings }, () => {
    console.log('[buttonSlackChannelID] New settings', settings);
    displaySettings(settings);
  });
});

document.getElementById('buttonMultiplier').addEventListener('click', () => {
  const multiplier = document.getElementById('multiplier').value;
  console.log('[buttonMultiplier] Multiplier:', multiplier);
  settings.multiplier = multiplier * 1;
  chrome.storage.local.set({ settings }, () => {
    console.log('[buttonMultiplier] New settings', settings);
    displaySettings(settings);
  });
});

document.getElementById('buttonBaseAmount').addEventListener('click', () => {
  const baseAmount = document.getElementById('baseAmount').value;
  console.log('[buttonBaseAmount] Base Amount:', baseAmount);
  settings.baseAmount = baseAmount * 1;
  chrome.storage.local.set({ settings }, () => {
    console.log('[buttonBaseAmount] New settings', settings);
    displaySettings(settings);
  });
});

document.getElementById('buttonRiskDepth').addEventListener('click', () => {
  const riskDepth = document.getElementById('riskDepth').value;
  console.log('[buttonRiskDepth] Risk Depth:', riskDepth);
  settings.riskDepth = riskDepth * 1;
  chrome.storage.local.set({ settings }, () => {
    console.log('[buttonRiskDepth] New settings', settings);
    displaySettings(settings);
  });
});

document.getElementById('buttonMinPayout').addEventListener('click', () => {
  const minPayout = document.getElementById('minPayout').value;
  console.log('[buttonMinPayout] Min Payout:', minPayout);
  settings.minPayout = minPayout * 1;
  chrome.storage.local.set({ settings }, () => {
    console.log('[buttonMinPayout] New settings', settings);
    displaySettings(settings);
  });
});

document.getElementById('buttonIncludeOTC').addEventListener('click', () => {
  const includeOTC = document.getElementById('includeOTC').checked;
  console.log('[buttonIncludeOTC] Include OTC:', includeOTC);
  settings.includeOTC = includeOTC;
  chrome.storage.local.set({ settings }, () => {
    console.log('[buttonIncludeOTC] New settings', settings);
    displaySettings(settings);
  });
});

function dispatchMessageToPO(action, data) {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs.length > 0) {
      const activeTab = tabs[0];
      const url = new URL(activeTab.url);

      // Check if the domain is pocketoption.com
      if (url.hostname === 'pocketoption.com') {
        chrome.tabs.sendMessage(activeTab.id, { action, data });
      } else {
        console.log("The active tab is not on pocketoption.com");
      }
    } else {
      console.log("No active tab found");
    }
  });
}

function displaySettings(settings) {
  const enabled = settings?.enabled;
  const statusElement = document.getElementById("settingDisplay");
  statusElement.innerHTML = `Enabled: ${enabled ? "Yes" : "No"}<br>
    User Name: ${settings.userName || "N/A"}<br>
    Strategy Name: ${settings.name || "N/A"}<br>
    Default Amount: ${settings.defaultAmount || "N/A"}<br>
    Multiplier: ${settings.multiplier || "N/A"}<br>
    Base Amount: ${settings.baseAmount || "N/A"}<br>
    Risk Depth: ${settings.riskDepth || "N/A"}<br>
    Default Duration: ${settings.defaultDuration || "N/A"}<br>
    Max Position Limit: ${settings.maxPositionLimit || "N/A"}<br>
    Max Position Amount: ${settings.maxPositionAmount || "N/A"}<br>
    Interval: ${settings.interval || "N/A"}<br>
    Default Direction: ${settings.defaultDirection || "N/A"}<br>
    Slack Channel ID: ${settings.slackChannelID || "N/A"}<br>
    Min Payout: ${settings.minPayout || "90"}<br>
    Include OTC: ${settings.includeOTC === false ? "No" : "Yes"}<br>
  `;
}