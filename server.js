// Express server with WebSocket support for the hand tracking application
const express = require('express');
const http = require('http');
const path = require('path');
const { initWebSocketServer } = require('./src/websocket');

// Create Express app
const app = express();
const port = 3001;

// Create HTTP server
const server = http.createServer(app);

// Initialize WebSocket server
const wss = initWebSocketServer(server);

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, 'public')));

// Redirect root to index.html
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// API endpoint for getting available gestures
app.get('/api/gestures', (req, res) => {
  res.json({
    gestures: [
      { symbol: "👍", name: "Thumbs Up" },
      { symbol: "👎", name: "Thumbs Down" },
      { symbol: "✋", name: "Open Hand" },
      { symbol: "✊", name: "Closed Fist" }
    ]
  });
});

// Start the server
server.listen(port, () => {
  console.log(`Hand tracking server running at http://localhost:${port}`);
  console.log(`WebSocket server running at ws://localhost:${port}/ws`);
  console.log(`Open your browser and navigate to http://localhost:${port}`);
});
