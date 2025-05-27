var settings = {};

document.addEventListener('DOMContentLoaded', function() {
  chrome.storage.local.get(['settings'], function(result) {
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
    Default Amount: ${settings.defaultAmount || "N/A"}<br>
    Default Duration: ${settings.defaultDuration || "N/A"}<br>
    Max Position Limit: ${settings.maxPositionLimit || "N/A"}<br>
    Interval: ${settings.interval || "N/A"}<br>
    Default Direction: ${settings.defaultDirection || "N/A"}`;
}