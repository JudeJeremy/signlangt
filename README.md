# Sign Language Tracking Project

This project uses MediaPipe for hand tracking and sign language recognition. The application has been restructured to separate front-end and back-end functionality.

## Project Structure

- `public/`: Front-end files
  - `index.html`: Main HTML file
  - `css/styles.css`: Stylesheet
  - `js/client.js`: Client-side JavaScript
- `src/`: Back-end files
  - `websocket.js`: WebSocket server implementation
  - `handTracking.js`: Hand tracking and gesture recognition logic
- `python/`: Python scripts for OpenCV-based hand tracking
- `server.js`: Express server with WebSocket support
- `vite.config.js`: Configuration for the Vite development server

## Setup

1. Install dependencies:
```bash
npm install
```

2. Run the application using one of the following methods:

### Method 1: Using Express Server (Recommended)
```bash
npm run express
```
or
```bash
npm start
```
Then open your browser and navigate to http://localhost:3001

### Method 2: Using Vite
```bash
npm run dev
```
Then open your browser and navigate to http://localhost:3001

## Features

- Real-time hand tracking using MediaPipe
- WebSocket communication between client and server
- Server-side gesture recognition
- Fallback to client-side processing when server is unavailable
- Support for multiple gestures:
  - 👍 Thumbs Up
  - 👎 Thumbs Down
  - ✋ Open Hand
  - ✊ Closed Fist

## Troubleshooting

If you encounter issues with the local server:

1. Make sure all dependencies are installed:
```bash
npm install
```

2. Check if port 3001 is already in use by another application. If so, you can change the port in both `vite.config.js` and `server.js`.

3. Try running the application using the alternative method (Vite or Express).

4. Check the browser console for any JavaScript errors.
