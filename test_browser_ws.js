// Test WebSocket connection from browser perspective
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8000/ws');

ws.on('open', function open() {
  console.log('WebSocket connected');
  
  const config = {
    uid: 'browser_test',
    language: 'en',
    task: 'transcribe',
    model: 'tiny',
    use_vad: true,
    multilingual: false
  };
  
  ws.send(JSON.stringify(config));
  console.log('Sent config:', config);
});

ws.on('message', function message(data) {
  console.log('Received:', data.toString());
});

ws.on('error', function error(err) {
  console.error('WebSocket error:', err.message);
});

ws.on('close', function close(code, reason) {
  console.log('WebSocket closed:', code, reason.toString());
});

setTimeout(() => {
  console.log('Closing connection');
  ws.close();
}, 5000);