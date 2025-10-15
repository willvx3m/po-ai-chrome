let SLACK_BOT_TOKEN = ''; // Set via settings or environment variable
let CHANNEL_ID = '';

async function sendSlackNotification(message) {
  chrome.storage?.local?.get(['settings'], async (data) => {
    const settings = data.settings;
    const customChannelID = settings?.slackChannelID;
    const customBotToken = settings?.slackBotToken;
    if (customChannelID && customBotToken) {
      SLACK_BOT_TOKEN = customBotToken;
      CHANNEL_ID = customChannelID;
    } else {
      return;
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

async function testQueryServer(data) {
  console.log('[testQueryServer] Test Query Server:');
  chrome.storage?.local?.get(['settings'], async (data) => {
    const settings = data.settings;
    const urlQueryServer = settings?.urlQueryServer;
    if (!urlQueryServer) {
      console.log('[testQueryServer] No URL Query Server found');
      return;
    }

    const response = await fetch(urlQueryServer, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbol: 'TEST',
        priceBook: [1.15, 1.16, 1.17, 1.18, 1.19],
        settings: settings,
      }),
    });

    const responseData = await response.json();
    console.log('[testQueryServer] Response:', responseData);
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
  if (request.action === 'testQueryServer') {
    testQueryServer(request.data);
    return true;
  }
});

