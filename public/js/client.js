// We'll initialize these variables after the DOM is loaded
let videoElement;
let canvasElement;
let canvasCtx;
let statusElement;
let transcriptionOutput;
let clearButton;

// MediaPipe objects
let hands;
let camera;

// Import MediaPipe drawing utilities
const mpHands = window.Hands;
const mpDrawingUtils = window.drawConnectors;
const mpDrawingStyles = window.drawLandmarks;

// For gesture recognition
let lastGestureTime = 0;
let gestureBuffer = [];

// WebSocket connection
let socket;

// Wait for DOM to load before accessing elements
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    videoElement = document.getElementById('videoElement');
    canvasElement = document.getElementById('canvasElement');
    canvasCtx = canvasElement.getContext('2d');
    statusElement = document.getElementById('statusElement');
    transcriptionOutput = document.getElementById('transcriptionOutput');
    clearButton = document.getElementById('clearButton');
    
    // Set canvas dimensions
    canvasElement.width = 640;
    canvasElement.height = 480;
    
    // Initialize MediaPipe Hands
    hands = new Hands({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${file}`;
        }
    });
    
    // Configure hands
    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7
    });
    
    // Set up result handling
    hands.onResults(onResults);
    
    // Initialize camera
    camera = new Camera(videoElement, {
        onFrame: async () => {
            try {
                await hands.send({image: videoElement});
            } catch (error) {
                console.error('Error in hand tracking:', error);
                statusElement.textContent = 'Error in hand tracking: ' + error.message;
            }
        },
        width: 640,
        height: 480
    });
    
    // Start camera with error handling
    camera.start()
        .then(() => {
            statusElement.textContent = 'Camera started, tracking hands...';
            console.log('Camera started successfully');
        })
        .catch(error => {
            statusElement.textContent = `Error starting camera: ${error.message}`;
            console.error('Error starting camera:', error);
        });
    
    // Clear button functionality
    clearButton.addEventListener('click', () => {
        gestureBuffer = [];
        transcriptionOutput.textContent = 'Sign language transcription will appear here...';
    });
    
    // Initialize WebSocket connection
    initWebSocket();
    
    // Add debugging info
    console.log('DOM loaded, initialization complete');
});

// Initialize WebSocket connection to the server
function initWebSocket() {
    // Create WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    socket = new WebSocket(wsUrl);
    
    // Connection opened
    socket.addEventListener('open', (event) => {
        console.log('WebSocket connection established');
        statusElement.textContent = 'Connected to server, tracking hands...';
    });
    
    // Listen for messages from the server
    socket.addEventListener('message', (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'gesture') {
            // Add to gesture buffer
            gestureBuffer.push(data.gesture);
            
            // Update transcription
            if (gestureBuffer.length > 5) {
                gestureBuffer.shift(); // Keep buffer size limited
            }
            transcriptionOutput.textContent = gestureBuffer.join(" ");
        }
    });
    
    // Connection closed
    socket.addEventListener('close', (event) => {
        console.log('WebSocket connection closed');
        statusElement.textContent = 'Connection to server lost. Reconnecting...';
        
        // Try to reconnect after a delay
        setTimeout(initWebSocket, 3000);
    });
    
    // Connection error
    socket.addEventListener('error', (error) => {
        console.error('WebSocket error:', error);
        statusElement.textContent = 'Error in server connection';
    });
}

// Process results from hand tracking
function onResults(results) {
    // Log results for debugging
    console.log('Got results:', results.multiHandLandmarks ? 
                `${results.multiHandLandmarks.length} hands detected` : 
                'No hands detected');
    
    // Clear canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw camera feed on canvas
    canvasCtx.drawImage(
        results.image, 0, 0, canvasElement.width, canvasElement.height
    );
    
    // Update status
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        statusElement.textContent = `Detected ${results.multiHandLandmarks.length} hand(s)`;
        
        // Process each detected hand
        results.multiHandLandmarks.forEach((landmarks, handIndex) => {
            // Draw hand landmarks with connections
            window.drawConnectors(
                canvasCtx, landmarks, mpHands.HAND_CONNECTIONS,
                {color: '#00FF00', lineWidth: 5}
            );
            window.drawLandmarks(
                canvasCtx, landmarks,
                {color: '#FF0000', lineWidth: 2, radius: 4}
            );
            
            // Extract and log thumb tip position (similar to Python code)
            const thumbTip = landmarks[4]; // Thumb tip is index 4
            console.log(`Hand ${handIndex} Thumb Tip Position: ${thumbTip.x.toFixed(3)}, ${thumbTip.y.toFixed(3)}`);
            
            // Send hand landmarks to server for processing
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({
                    type: 'landmarks',
                    handIndex: handIndex,
                    landmarks: landmarks
                }));
            } else {
                // Fallback to client-side gesture recognition if server connection is not available
                recognizeGesture(landmarks);
            }
        });
    } else {
        statusElement.textContent = 'No hands detected';
    }
}

// Simple gesture recognition function (fallback for when server is not available)
function recognizeGesture(landmarks) {
    // This is a very basic placeholder for gesture recognition
    // In a real implementation, you would have more sophisticated logic
    
    // Example: Detect if thumb is up (very simplified)
    const thumbTip = landmarks[4];
    const indexFingerTip = landmarks[8];
    
    // Simple check if thumb is higher than index finger
    if (thumbTip.y < indexFingerTip.y) {
        // Throttle gesture detection to avoid rapid changes
        const now = Date.now();
        if (now - lastGestureTime > 1000) { // 1 second cooldown
            lastGestureTime = now;
            
            // Add to gesture buffer
            gestureBuffer.push("👍");
            
            // Update transcription
            if (gestureBuffer.length > 5) {
                gestureBuffer.shift(); // Keep buffer size limited
            }
            transcriptionOutput.textContent = gestureBuffer.join(" ");
        }
    }
}
