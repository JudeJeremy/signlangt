// DOM Elements
let videoElement;
let canvasElement;
let canvasCtx;
let statusElement;
let transcriptionOutput;
let clearButton;
let confidenceElement;

// Socket connection
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
    confidenceElement = document.getElementById('confidenceElement');
    
    // Set canvas dimensions
    canvasElement.width = 640;
    canvasElement.height = 480;
    
    // Initialize WebSocket connection
    initWebSocket();
    
    // Clear button functionality
    clearButton.addEventListener('click', () => {
        transcriptionOutput.textContent = '';
        transcriptionOutput.dataset.content = '';
    });
});

// Initialize WebSocket connection to the Python backend
function initWebSocket() {
    socket = io('https://singlangt.onrender.com/');
    
    // Connection opened
    socket.on('connect', () => {
        console.log('Connected to Python backend');
        statusElement.textContent = 'Connected to hand tracking server';
        updateStatus('Connected', 'success');
        
        // Start hand tracking
        socket.emit('start_tracking');
    });
    
    // Handle incoming frames and tracking data
    socket.on('frame', (data) => {
        // Update video frame
        const img = new Image();
        img.onload = () => {
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            canvasCtx.drawImage(img, 0, 0, canvasElement.width, canvasElement.height);
            
            // Draw hand position guide
            drawHandGuide();
        };
        img.src = 'data:image/jpeg;base64,' + data.frame;
        
        // Update status and transcription if gesture detected
        if (data.gesture) {
            const { letter, confidence } = data.gesture;
            
            // Update confidence display
            updateConfidence(confidence);
            
            // Update status with detected letter
            updateStatus(`Detected: ${letter} (${confidence}% confident)`, 'success');
            
            // Update transcription
            const currentContent = transcriptionOutput.dataset.content || '';
            if (currentContent === '' || currentContent.slice(-1) !== letter) {
                transcriptionOutput.dataset.content = currentContent + letter;
                transcriptionOutput.textContent = transcriptionOutput.dataset.content;
            }
        } else {
            updateStatus('No gesture detected', 'info');
            updateConfidence(0);
        }
    });
    
    // Handle status messages
    socket.on('status', (data) => {
        updateStatus(data.message, 'info');
    });
    
    // Handle errors
    socket.on('error', (data) => {
        console.error('Server error:', data.message);
        updateStatus(data.message, 'error');
    });
    
    // Connection closed
    socket.on('disconnect', () => {
        console.log('Disconnected from Python backend');
        updateStatus('Disconnected from server. Reconnecting...', 'error');
        
        // Try to reconnect after a delay
        setTimeout(initWebSocket, 3000);
    });
}

// Update status message with color coding
function updateStatus(message, type) {
    statusElement.textContent = message;
    statusElement.className = 'status ' + type;
}

// Update confidence meter
function updateConfidence(confidence) {
    if (confidenceElement) {
        confidenceElement.style.width = confidence + '%';
        confidenceElement.style.backgroundColor = getConfidenceColor(confidence);
    }
}

// Get color for confidence level
function getConfidenceColor(confidence) {
    if (confidence >= 90) return '#4CAF50';  // Green
    if (confidence >= 70) return '#8BC34A';  // Light Green
    if (confidence >= 50) return '#FFC107';  // Amber
    if (confidence >= 30) return '#FF9800';  // Orange
    return '#F44336';  // Red
}

// Draw hand position guide
function drawHandGuide() {
    const ctx = canvasCtx;
    const width = canvasElement.width;
    const height = canvasElement.height;
    
    // Draw guide box
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    
    // Center box for hand placement
    const boxSize = Math.min(width, height) * 0.6;
    const x = (width - boxSize) / 2;
    const y = (height - boxSize) / 2;
    
    ctx.beginPath();
    ctx.rect(x, y, boxSize, boxSize);
    ctx.stroke();
    
    // Reset line dash
    ctx.setLineDash([]);
}
