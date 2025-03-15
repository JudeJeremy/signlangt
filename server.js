// Simple Express server for serving the hand tracking application
const express = require('express');
const path = require('path');
const app = express();
const port = 3001;

// Serve static files from the current directory
app.use(express.static(__dirname));

// Redirect root to index.html
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Start the server
app.listen(port, () => {
  console.log(`Hand tracking server running at http://localhost:${port}`);
  console.log(`Open your browser and navigate to http://localhost:${port}`);
});
