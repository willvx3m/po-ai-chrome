document.getElementById('toggle').addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs.length > 0) {
      const activeTab = tabs[0];
      const url = new URL(activeTab.url);

      // Check if the domain is pocketoption.com
      if (url.hostname === 'pocketoption.com') {
        chrome.storage.local.get(["enabled", "scriptInjected"], (data) => {
          const newStatus = !data.enabled;
          chrome.storage.local.set({ enabled: newStatus }, () => {
            const statusElement = document.getElementById("toggle");
            statusElement.innerHTML = newStatus ? "Disable" : "Enable";

            if (newStatus) {
              chrome.scripting.executeScript({
                target: { tabId: activeTab.id },
                files: ['content.js']
              }, () => {
                chrome.storage.local.set({ scriptInjected: true });
              });
            } else {
              chrome.storage.local.set({ scriptInjected: false });
            }

            chrome.tabs.reload(activeTab.id);
          });
        });
      } else {
        console.log("The active tab is not on pocketoption.com");
      }
    } else {
      console.log("No active tab found");
    }
  });
});