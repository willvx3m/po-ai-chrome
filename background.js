const SLACK_BOT_TOKEN = 'xoxb-523895185494-9041191622752-HM9RyRlScZgl0qeUYzVcyzAI';
let CHANNEL_ID = 'C090JCZ6TPU';

async function sendSlackNotification(message) {
  chrome.storage?.local?.get(['settings'], async (data) => {
    const settings = data.settings;
    const customChannelID = settings?.slackChannelID;
    if (customChannelID) {
      CHANNEL_ID = customChannelID;
    }

    try {
        const response = await fetch('https://slack.com/api/chat.postMessage', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${SLACK_BOT_TOKEN}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                channel: CHANNEL_ID,
                text: message
            })
        });
  
        const data = await response.json();
        if (data.ok) {
            console.log('Notification sent successfully:', data);
        } else {
            console.error('Error sending notification:', data.error);
        }
    } catch (error) {
        console.error('Fetch error:', error);
    }
  });
}

chrome.runtime.onInstalled.addListener(() => {
  console.log('Extension installed');
  sendSlackNotification('Hello onInstalled!');
  // chrome.storage.local.set({
  //   settings: DEFAULT_SETTINGS
  // });
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'sendSlackNotification') {
    sendSlackNotification(request.message);
    return true;
  }
});

