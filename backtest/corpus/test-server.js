const fetch = require('node-fetch');

// Test the server API
async function testServer() {
    const baseUrl = 'http://localhost:3000';
    
    try {
        // Test health endpoint
        console.log('Testing health endpoint...');
        const healthResponse = await fetch(`${baseUrl}/api/health`);
        const healthData = await healthResponse.json();
        console.log('Health check:', healthData);
        
        // Test corpus files endpoint
        console.log('\nTesting corpus files endpoint...');
        const filesResponse = await fetch(`${baseUrl}/api/corpus-files`);
        const filesData = await filesResponse.json();
        console.log('Available corpus files:', filesData);
        
        // Test find best match endpoint
        console.log('\nTesting find best match endpoint...');
        const priceBook = Array(50 * 6).fill().map((_, i) => 1.1500 + Math.random() * 0.01);
        
        fetch(`${baseUrl}/api/find-best-match`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: 'EUR-USD-OTC',
                priceBook: priceBook,
                corpusFile: 'MARTINGALE-po-EUR-USD-OTC-INPUT.json' // optional
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Best match result:', JSON.stringify(data, null, 2));
        })
        .catch(error => {
            console.error('Error testing server:', error.message);
        });
        
    } catch (error) {
        console.error('Error testing server:', error.message);
    }
}

// Run the test
testServer(); 