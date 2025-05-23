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

function displaySettings(settings) {
  const enabled = settings?.enabled;
  const statusElement = document.getElementById("settingDisplay");
  statusElement.innerHTML = enabled ? "Enabled" : "Disabled";
}