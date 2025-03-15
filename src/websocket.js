const WebSocket = require('ws');
const { recognizeGesture } = require('./handTracking');

// Initialize WebSocket server
function initWebSocketServer(server) {
    const wss = new WebSocket.Server({ server });
    
    console.log('WebSocket server initialized');
    
    wss.on('connection', (ws) => {
        console.log('Client connected to WebSocket');
        
        // Send a welcome message
        ws.send(JSON.stringify({
            type: 'info',
            message: 'Connected to sign language transcription server'
        }));
        
        // Handle messages from clients
        ws.on('message', (message) => {
            try {
                const data = JSON.parse(message);
                
                // Handle different message types
                if (data.type === 'landmarks') {
                    handleLandmarks(ws, data);
                }
            } catch (error) {
                console.error('Error processing message:', error);
                ws.send(JSON.stringify({
                    type: 'error',
                    message: 'Error processing message'
                }));
            }
        });
        
        // Handle client disconnection
        ws.on('close', () => {
            console.log('Client disconnected from WebSocket');
        });
    });
    
    return wss;
}

// Handle hand landmarks data
function handleLandmarks(ws, data) {
    // Extract landmarks from the message
    const { handIndex, landmarks } = data;
    
    // Process landmarks to recognize gestures
    const gesture = recognizeGesture(landmarks);
    
    // If a gesture was recognized, send it back to the client
    if (gesture) {
        ws.send(JSON.stringify({
            type: 'gesture',
            gesture: gesture
        }));
    }
}

module.exports = { initWebSocketServer };
