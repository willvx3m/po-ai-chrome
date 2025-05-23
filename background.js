// chrome.runtime.onInstalled.addListener(() => {
//   console.log('Extension installed');
//   chrome.storage.local.set({ enabled: false });
// });

// chrome.action.onClicked.addListener((tab) => {
//   console.log('Button clicked');
//   chrome.storage.local.get("enabled", (data) => {
//     const newStatus = !data.enabled;
//     chrome.storage.local.set({ enabled: newStatus });
//     console.log('Status updated to:', newStatus);
//   });
// });