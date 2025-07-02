const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');

// Character set for the 5-character string (0-9, a-z)
const characters = '0123456789abcdefghijklmnopqrstuvwxyz';
const charLength = characters.length; // 36
const combinationLength = 5; // XXXXX
const totalCombinations = Math.pow(charLength, combinationLength); // 36^5 = 60,466,176
const batchSize = 100; // Number of concurrent requests

// Files for caching and output
const outputFile = path.join(__dirname, 'live_domains.txt');
const indexFile = path.join(__dirname, 'last_index.txt');

// Function to convert an index to a 5-character string
function indexToString(index) {
  let str = '';
  for (let i = 0; i < combinationLength; i++) {
    const charIndex = Math.floor(index / Math.pow(charLength, combinationLength - 1 - i)) % charLength;
    str += characters[charIndex];
  }
  return str;
}

// Function to read the last index from file
async function getLastIndex() {
  try {
    const data = await fs.readFile(indexFile, 'utf8');
    const index = parseInt(data.trim(), 10);
    if (isNaN(index) || index < 0 || index >= totalCombinations) {
      return 0;
    }
    return index;
  } catch {
    return 0; // If file doesn't exist or is invalid, start from 0
  }
}

// Function to save the last index to file
async function saveLastIndex(index) {
  await fs.writeFile(indexFile, index.toString());
}

// Function to check if a domain is live
async function checkDomain(domain) {
  try {
    const response = await axios.get(`https://${domain}`, { timeout: 5000 });
    if (response.status === 200) {
      console.log(`Domain ${domain} is LIVE (Status: ${response.status})`);
      return { domain, isLive: true };
    } else {
      console.log(`Domain ${domain} returned status: ${response.status}`);
      return { domain, isLive: false };
    }
  } catch (error) {
    if (error.response && error.response.status === 404) {
    //   console.log(`Domain ${domain} is NOT LIVE (404 Not Found)`);
    } else {
    //   console.log(`Domain ${domain} failed: ${error.message}`);
    }
    return { domain, isLive: false };
  }
}

// Function to check a batch of domains concurrently
async function checkDomainBatch(startIndex, batchSize) {
  const promises = [];
  const domains = [];
  // Generate batch of domains
  for (let i = startIndex; i < startIndex + batchSize && i < totalCombinations; i++) {
    const randomString = indexToString(i);
    const domain = `web-production-${randomString}.up.railway.app`;
    domains.push(domain);
    promises.push(checkDomain(domain));
  }

  // Wait for all requests in the batch to complete
  const results = await Promise.all(promises);

  // Collect live domains
  const liveDomains = results.filter(result => result.isLive).map(result => result.domain);
  if (liveDomains.length > 0) {
    await fs.appendFile(outputFile, liveDomains.join('\n') + '\n');
  }

  return domains.length; // Return number of domains checked in this batch
}

// Function to check all possible domains in batches
async function checkAllDomains(delayMs = 1000) {
  // Read the last index to resume from
  let currentIndex = await getLastIndex();
  console.log(`Starting from index ${currentIndex} of ${totalCombinations} total combinations`);

  // Ensure output file exists
  try {
    await fs.access(outputFile);
  } catch {
    await fs.writeFile(outputFile, 'Live Domains:\n');
  }

  while (currentIndex < totalCombinations) {
    const domainsChecked = await checkDomainBatch(currentIndex, batchSize);
    currentIndex += domainsChecked;

    // Save the current index after each batch
    await saveLastIndex(currentIndex);

    // Log progress
    console.log(`Progress: Checked ${currentIndex} of ${totalCombinations} domains (${((currentIndex / totalCombinations) * 100).toFixed(2)}%)`);

    // Delay between batches
    if (currentIndex < totalCombinations) {
      await new Promise(resolve => setTimeout(resolve, delayMs));
    }
  }
}

// Run the script, resuming from the last index
checkAllDomains(1000)
  .then(() => {
    console.log('All domain checks completed. Live domains saved to live_domains.txt');
    // Optionally, clear the index file when done
    return fs.writeFile(indexFile, '0');
  })
  .catch(err => console.error('Error during domain checks:', err));