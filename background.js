chrome.runtime.onInstalled.addListener(() => {
  console.log('Extension installed');
  chrome.storage.local.set({
    settings: {
      enabled: false,
      defaultAmount: 2,
      defaultDuration: 10,
      maxPositionLimit: 10,
      interval: 10000,
      defaultDirection: 'BUY'
    }
  });
});

// chrome.action.onClicked.addListener((tab) => {
//   console.log('Button clicked');
//   chrome.storage.local.get("enabled", (data) => {
//     const newStatus = !data.enabled;
//     chrome.storage.local.set({ enabled: newStatus });
//     console.log('Status updated to:', newStatus);
//   });
// });