// Game variables
let socket;
let gameCanvas;
let gameCtx;
let videoElement;
let canvasElement;
let canvasCtx;
let statusElement;
let confidenceElement;
let scoreElement;
let timerElement;
let levelElement;
let currentWordElement;
let detectedLettersElement;
let livesElement;
let startButton;
let restartButton;
let gameOverModal;
let finalScoreElement;
let wordsSpelledElement;
let finalLevelElement;
let playAgainButton;

// Game state
let gameState = {
    isRunning: false,
    score: 0,
    lives: 3,
    level: 1,
    timeRemaining: 60, // seconds
    currentWord: '',
    currentLetterIndex: 0,
    detectedLetters: '',
    wordsSpelled: [],
    lastDetectedLetter: '',
    lastDetectionTime: 0,
    timerInterval: null
};

// Constants
const LETTER_DETECTION_COOLDOWN = 800; // ms - increased to make detection more reliable
const SUPPORTED_LETTERS = ['A', 'B', 'I', 'K', 'W'];
const POINTS_PER_LETTER = 10;
const WORD_BONUS_MULTIPLIER = 1.5;
const LEVEL_WORD_COUNT = 5; // Words to spell before leveling up

// Word list (using A, B, I, K, W)
const WORD_LIST = [
    // Short words (2-3 letters)
    'AB', 'AW', 'BA', 'KA', 'WA', 'AI', 'BI', 'KI', 'WI', 'IA',
    'BAK', 'WAK', 'BAW', 'KAB', 'WAB', 'AIK', 'BIK', 'KIA', 'WIB', 'IKA',
    
    // Medium words (4 letters)
    'BAKA', 'WAKA', 'ABBA', 'KAWA', 'BAWK', 'BAKI', 'WIKI', 'KIWI', 'BIKA', 'AIKI',
    
    // Long words (5+ letters)
    'AIKAWA', 'KABIKI', 'WAKIBA', 'AWKWARD', 'KAWABATA', 'WAKAWAKA'
];

// Wait for DOM to load before accessing elements
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    videoElement = document.getElementById('videoElement');
    canvasElement = document.getElementById('canvasElement');
    canvasCtx = canvasElement.getContext('2d');
    statusElement = document.getElementById('statusElement');
    confidenceElement = document.getElementById('confidenceElement');
    scoreElement = document.getElementById('scoreElement');
    timerElement = document.getElementById('timerElement');
    levelElement = document.getElementById('levelElement');
    currentWordElement = document.getElementById('currentWordElement');
    detectedLettersElement = document.getElementById('detectedLettersElement');
    livesElement = document.getElementById('livesElement');
    startButton = document.getElementById('startButton');
    restartButton = document.getElementById('restartButton');
    gameOverModal = document.getElementById('gameOverModal');
    finalScoreElement = document.getElementById('finalScoreElement');
    wordsSpelledElement = document.getElementById('wordsSpelledElement');
    finalLevelElement = document.getElementById('finalLevelElement');
    playAgainButton = document.getElementById('playAgainButton');
    
    // Set up game canvas
    gameCanvas = document.getElementById('gameCanvas');
    gameCtx = gameCanvas.getContext('2d');
    gameCanvas.width = 800;
    gameCanvas.height = 400;
    
    // Set canvas dimensions for hand tracking
    canvasElement.width = 320;
    canvasElement.height = 240;
    
    // Initialize WebSocket connection
    initWebSocket();
    
    // Set up game controls
    startButton.addEventListener('click', startGame);
    restartButton.addEventListener('click', startGame);
    playAgainButton.addEventListener('click', function() {
        gameOverModal.style.display = 'none';
        startGame();
    });
    
    // Initial game setup
    resetGame();
    drawStartScreen();
    
    // Initialize timer display
    updateTimerDisplay();
    
    console.log("Game initialized");
});

// Initialize WebSocket connection to the Python backend
function initWebSocket() {
    socket = io('http://localhost:5000');
    
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
        };
        img.src = 'data:image/jpeg;base64,' + data.frame;
        
        // Update confidence display
        if (data.gesture) {
            const { letter, confidence } = data.gesture;
            updateConfidence(confidence);
            
            // Process detected letter for the game
            if (gameState.isRunning) {
                processDetectedLetter(letter);
            }
        } else {
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

// Process detected letter for the game
function processDetectedLetter(letter) {
    // Check if letter is supported
    if (!SUPPORTED_LETTERS.includes(letter)) {
        return;
    }
    
    // Apply cooldown to prevent rapid letter detection
    const now = Date.now();
    if (now - gameState.lastDetectionTime < LETTER_DETECTION_COOLDOWN) {
        return;
    }
    
    // Always process the letter after cooldown, regardless if it's the same as before
    gameState.lastDetectedLetter = letter;
    gameState.lastDetectionTime = now;
    
    // Add letter to detected letters display
    gameState.detectedLetters = letter;
    detectedLettersElement.textContent = letter;
    
    // Check if letter matches the expected letter in the current word
    checkLetter(letter);
    
    // Debug message
    console.log(`Detected letter: ${letter}`);
}

// Check if the detected letter matches the expected letter
function checkLetter(letter) {
    if (gameState.currentWord.length === 0) return;
    
    const expectedLetter = gameState.currentWord[gameState.currentLetterIndex];
    
    if (letter === expectedLetter) {
        // Correct letter
        gameState.currentLetterIndex++;
        
        // Update display
        updateWordDisplay();
        
        // Play success sound
        playSound('letter');
        
        // Check if word is complete
        if (gameState.currentLetterIndex >= gameState.currentWord.length) {
            // Word completed successfully
            wordCompleted();
        }
    } else {
        // Incorrect letter
        // Lose a life
        gameState.lives--;
        updateLivesDisplay();
        
        // Play error sound
        playSound('error');
        
        // Show error message
        showErrorMessage();
        
        // Check if game over
        if (gameState.lives <= 0) {
            endGame();
        } else {
            // Reset current word and get a new one
            setTimeout(() => {
                getNextWord();
            }, 1500);
        }
    }
}

// Update the word display to show progress
function updateWordDisplay() {
    let html = '';
    
    for (let i = 0; i < gameState.currentWord.length; i++) {
        let letterClass = '';
        if (i < gameState.currentLetterIndex) {
            letterClass = 'letter-correct';
        } else if (i === gameState.currentLetterIndex) {
            letterClass = 'letter-current';
        } else {
            letterClass = 'letter-pending';
        }
        
        html += `<span class="${letterClass}">${gameState.currentWord[i]}</span>`;
    }
    
    currentWordElement.innerHTML = html;
}

// Handle successful word completion
function wordCompleted() {
    // Calculate score
    const baseScore = gameState.currentWord.length * POINTS_PER_LETTER;
    const bonus = gameState.currentWord.length >= 4 ? WORD_BONUS_MULTIPLIER : 1;
    const scoreGain = Math.floor(baseScore * bonus);
    
    // Add score
    gameState.score += scoreGain;
    scoreElement.textContent = gameState.score;
    
    // Add to words spelled
    gameState.wordsSpelled.push(gameState.currentWord);
    
    // Show success message
    showSuccessMessage(scoreGain);
    
    // Play success sound
    playSound('word');
    
    // Check level progression
    if (gameState.wordsSpelled.length % LEVEL_WORD_COUNT === 0) {
        levelUp();
    }
    
    // Get next word after a delay
    setTimeout(() => {
        getNextWord();
    }, 1500);
}

// Get the next word to spell
function getNextWord() {
    // Filter words by level difficulty
    let wordPool;
    
    if (gameState.level === 1) {
        // Level 1: Short words (2-3 letters)
        wordPool = WORD_LIST.filter(word => word.length <= 3);
    } else if (gameState.level === 2) {
        // Level 2: Medium words (3-4 letters)
        wordPool = WORD_LIST.filter(word => word.length >= 3 && word.length <= 4);
    } else {
        // Level 3+: Any word, with preference for longer words
        wordPool = WORD_LIST;
        
        // Add some longer words multiple times to increase their probability
        const longWords = WORD_LIST.filter(word => word.length > 4);
        wordPool = wordPool.concat(longWords);
    }
    
    // Get a random word
    const randomIndex = Math.floor(Math.random() * wordPool.length);
    gameState.currentWord = wordPool[randomIndex];
    gameState.currentLetterIndex = 0;
    
    // Update display
    updateWordDisplay();
}

// Level up
function levelUp() {
    gameState.level++;
    levelElement.textContent = gameState.level;
    
    // Add time bonus for leveling up
    const timeBonus = 15;
    gameState.timeRemaining += timeBonus;
    updateTimerDisplay();
    
    // Show level up message
    showLevelUpMessage(timeBonus);
    
    // Play level up sound
    playSound('levelUp');
}

// Update lives display
function updateLivesDisplay() {
    if (livesElement) {
        let livesHtml = '';
        for (let i = 0; i < gameState.lives; i++) {
            livesHtml += '❤️ ';
        }
        for (let i = gameState.lives; i < 3; i++) {
            livesHtml += '❌ ';
        }
        livesElement.innerHTML = livesHtml;
    }
}

// Show success message
function showSuccessMessage(score) {
    const message = document.createElement('div');
    message.textContent = `Correct! +${score}`;
    message.className = 'game-message success';
    
    document.querySelector('.game-area').appendChild(message);
    
    // Animate and remove after a delay
    setTimeout(() => {
        message.classList.add('fade-out');
        setTimeout(() => {
            message.remove();
        }, 500);
    }, 1000);
}

// Show error message
function showErrorMessage() {
    const message = document.createElement('div');
    message.textContent = 'Incorrect! -1 Life';
    message.className = 'game-message error';
    
    document.querySelector('.game-area').appendChild(message);
    
    // Animate and remove after a delay
    setTimeout(() => {
        message.classList.add('fade-out');
        setTimeout(() => {
            message.remove();
        }, 500);
    }, 1000);
}

// Show level up message
function showLevelUpMessage(timeBonus) {
    const message = document.createElement('div');
    message.textContent = `LEVEL UP! ${gameState.level} (+${timeBonus}s)`;
    message.className = 'game-message level-up';
    
    document.querySelector('.game-area').appendChild(message);
    
    // Animate and remove after a delay
    setTimeout(() => {
        message.classList.add('fade-out');
        setTimeout(() => {
            message.remove();
        }, 500);
    }, 2000);
}

// Play sound effect
function playSound(type) {
    // Simple sound effect using Web Audio API
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        switch (type) {
            case 'letter':
                oscillator.type = 'sine';
                oscillator.frequency.setValueAtTime(440, audioContext.currentTime);
                gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
                oscillator.start();
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
                oscillator.stop(audioContext.currentTime + 0.2);
                break;
                
            case 'word':
                oscillator.type = 'sine';
                oscillator.frequency.setValueAtTime(523.25, audioContext.currentTime);
                gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
                oscillator.start();
                oscillator.frequency.setValueAtTime(659.25, audioContext.currentTime + 0.1);
                oscillator.frequency.setValueAtTime(783.99, audioContext.currentTime + 0.2);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
                oscillator.stop(audioContext.currentTime + 0.3);
                break;
                
            case 'error':
                oscillator.type = 'sawtooth';
                oscillator.frequency.setValueAtTime(110, audioContext.currentTime);
                gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
                oscillator.start();
                oscillator.frequency.setValueAtTime(100, audioContext.currentTime + 0.1);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
                oscillator.stop(audioContext.currentTime + 0.2);
                break;
                
            case 'levelUp':
                oscillator.type = 'square';
                oscillator.frequency.setValueAtTime(440, audioContext.currentTime);
                gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
                oscillator.start();
                oscillator.frequency.setValueAtTime(554.37, audioContext.currentTime + 0.1);
                oscillator.frequency.setValueAtTime(659.25, audioContext.currentTime + 0.2);
                oscillator.frequency.setValueAtTime(880, audioContext.currentTime + 0.3);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.4);
                oscillator.stop(audioContext.currentTime + 0.4);
                break;
        }
    } catch (e) {
        console.error('Error playing sound:', e);
    }
}

// Start the game
function startGame() {
    resetGame();
    gameState.isRunning = true;
    
    // Hide start button, show restart button
    startButton.style.display = 'none';
    restartButton.style.display = 'inline-block';
    
    // Get first word
    getNextWord();
    
    // Start game loop
    requestAnimationFrame(gameLoop);
    
    // Start timer
    startTimer();
    
    console.log("Game started");
}

// Start the timer
function startTimer() {
    // Clear any existing timer
    if (gameState.timerInterval) {
        clearInterval(gameState.timerInterval);
    }
    
    // Set up timer interval
    gameState.timerInterval = setInterval(() => {
        // Decrease time
        gameState.timeRemaining--;
        
        // Update timer display
        updateTimerDisplay();
        
        // Check if time is up
        if (gameState.timeRemaining <= 0) {
            clearInterval(gameState.timerInterval);
            endGame();
        }
    }, 1000);
    
    console.log("Timer started");
}

// Update timer display
function updateTimerDisplay() {
    if (timerElement) {
        // Format time as MM:SS
        const minutes = Math.floor(gameState.timeRemaining / 60);
        const seconds = gameState.timeRemaining % 60;
        const timeString = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        
        // Update display
        timerElement.textContent = timeString;
        
        // Update color based on time remaining
        if (gameState.timeRemaining <= 10) {
            timerElement.style.color = '#F44336'; // Red
        } else if (gameState.timeRemaining <= 30) {
            timerElement.style.color = '#FFC107'; // Amber
        } else {
            timerElement.style.color = '#4CAF50'; // Green
        }
        
        console.log(`Timer updated: ${timeString}`);
    } else {
        console.error("Timer element not found");
    }
}

// Reset game state
function resetGame() {
    gameState.isRunning = false;
    gameState.score = 0;
    gameState.lives = 3;
    gameState.level = 1;
    gameState.timeRemaining = 60;
    gameState.currentWord = '';
    gameState.currentLetterIndex = 0;
    gameState.detectedLetters = '';
    gameState.wordsSpelled = [];
    gameState.lastDetectedLetter = '';
    gameState.lastDetectionTime = 0;
    
    // Clear any existing timer
    if (gameState.timerInterval) {
        clearInterval(gameState.timerInterval);
        gameState.timerInterval = null;
    }
    
    // Update UI
    scoreElement.textContent = gameState.score;
    levelElement.textContent = gameState.level;
    currentWordElement.innerHTML = '';
    detectedLettersElement.textContent = '';
    updateLivesDisplay();
    updateTimerDisplay();
}

// End the game
function endGame() {
    gameState.isRunning = false;
    
    // Clear any existing timer
    if (gameState.timerInterval) {
        clearInterval(gameState.timerInterval);
        gameState.timerInterval = null;
    }
    
    // Update game over modal with statistics
    finalScoreElement.textContent = gameState.score;
    
    // Update words spelled count
    if (wordsSpelledElement) {
        wordsSpelledElement.textContent = gameState.wordsSpelled.length;
    }
    
    // Update final level
    if (finalLevelElement) {
        finalLevelElement.textContent = gameState.level;
    }
    
    // Show game over modal
    gameOverModal.style.display = 'flex';
    
    console.log("Game ended");
}

// Game loop
function gameLoop() {
    if (!gameState.isRunning) return;
    
    // Clear canvas
    gameCtx.clearRect(0, 0, gameCanvas.width, gameCanvas.height);
    
    // Draw background
    drawBackground();
    
    // Draw game board
    drawGameBoard();
    
    // Continue game loop
    requestAnimationFrame(gameLoop);
}

// Draw background
function drawBackground() {
    // Chess.com inspired background
    gameCtx.fillStyle = '#E8EDF9';  // Light blue-gray background
    gameCtx.fillRect(0, 0, gameCanvas.width, gameCanvas.height);
    
    // Add subtle pattern
    gameCtx.strokeStyle = '#D6DBEA';
    gameCtx.lineWidth = 1;
    
    // Draw grid pattern
    const gridSize = 20;
    for (let x = 0; x < gameCanvas.width; x += gridSize) {
        gameCtx.beginPath();
        gameCtx.moveTo(x, 0);
        gameCtx.lineTo(x, gameCanvas.height);
        gameCtx.stroke();
    }
    
    for (let y = 0; y < gameCanvas.height; y += gridSize) {
        gameCtx.beginPath();
        gameCtx.moveTo(0, y);
        gameCtx.lineTo(gameCanvas.width, y);
        gameCtx.stroke();
    }
}

// Draw game board
function drawGameBoard() {
    // Draw board background
    gameCtx.fillStyle = '#F5F5F5';
    gameCtx.strokeStyle = '#333';
    gameCtx.lineWidth = 2;
    
    // Draw rounded rectangle for board
    const margin = 20;
    const radius = 10;
    const boardX = margin;
    const boardY = margin;
    const boardWidth = gameCanvas.width - (margin * 2);
    const boardHeight = gameCanvas.height - (margin * 2);
    
    gameCtx.beginPath();
    gameCtx.moveTo(boardX + radius, boardY);
    gameCtx.lineTo(boardX + boardWidth - radius, boardY);
    gameCtx.quadraticCurveTo(boardX + boardWidth, boardY, boardX + boardWidth, boardY + radius);
    gameCtx.lineTo(boardX + boardWidth, boardY + boardHeight - radius);
    gameCtx.quadraticCurveTo(boardX + boardWidth, boardY + boardHeight, boardX + boardWidth - radius, boardY + boardHeight);
    gameCtx.lineTo(boardX + radius, boardY + boardHeight);
    gameCtx.quadraticCurveTo(boardX, boardY + boardHeight, boardX, boardY + boardHeight - radius);
    gameCtx.lineTo(boardX, boardY + radius);
    gameCtx.quadraticCurveTo(boardX, boardY, boardX + radius, boardY);
    gameCtx.closePath();
    
    gameCtx.fill();
    gameCtx.stroke();
    
    // Draw grid lines
    gameCtx.strokeStyle = '#DDD';
    gameCtx.lineWidth = 1;
    
    // Horizontal lines
    for (let y = boardY + 50; y < boardY + boardHeight; y += 50) {
        gameCtx.beginPath();
        gameCtx.moveTo(boardX, y);
        gameCtx.lineTo(boardX + boardWidth, y);
        gameCtx.stroke();
    }
    
    // Vertical lines
    for (let x = boardX + 50; x < boardX + boardWidth; x += 50) {
        gameCtx.beginPath();
        gameCtx.moveTo(x, boardY);
        gameCtx.lineTo(x, boardY + boardHeight);
        gameCtx.stroke();
    }
    
    // Draw current word if game is running
    if (gameState.isRunning && gameState.currentWord) {
        // Draw word title
        gameCtx.fillStyle = '#333';
        gameCtx.font = 'bold 24px "Roboto", sans-serif';
        gameCtx.textAlign = 'center';
        gameCtx.textBaseline = 'middle';
        gameCtx.fillText('Spell this word:', gameCanvas.width / 2, boardY + 50);
        
        // Draw the word
        gameCtx.font = 'bold 48px "Roboto", sans-serif';
        gameCtx.fillText(gameState.currentWord, gameCanvas.width / 2, boardY + 120);
        
        // Draw progress
        gameCtx.font = '24px "Roboto", sans-serif';
        gameCtx.fillText(`Progress: ${gameState.currentLetterIndex}/${gameState.currentWord.length}`, gameCanvas.width / 2, boardY + 180);
        
        // Draw detected letter
        if (gameState.detectedLetters) {
            gameCtx.fillStyle = '#4D8BBD';
            gameCtx.font = 'bold 36px "Roboto", sans-serif';
            gameCtx.fillText(`Detected: ${gameState.detectedLetters}`, gameCanvas.width / 2, boardY + 240);
        }
    }
}

// Draw start screen
function drawStartScreen() {
    gameCtx.clearRect(0, 0, gameCanvas.width, gameCanvas.height);
    
    // Draw chess.com inspired background
    drawBackground();
    
    // Draw title box
    const titleBoxWidth = 600;
    const titleBoxHeight = 300;
    const titleBoxX = (gameCanvas.width - titleBoxWidth) / 2;
    const titleBoxY = (gameCanvas.height - titleBoxHeight) / 2;
    
    // Draw box with shadow
    gameCtx.shadowColor = 'rgba(0, 0, 0, 0.2)';
    gameCtx.shadowBlur = 10;
    gameCtx.shadowOffsetX = 5;
    gameCtx.shadowOffsetY = 5;
    
    gameCtx.fillStyle = 'white';
    gameCtx.beginPath();
    gameCtx.roundRect(titleBoxX, titleBoxY, titleBoxWidth, titleBoxHeight, 10);
    gameCtx.fill();
    
    // Reset shadow
    gameCtx.shadowColor = 'transparent';
    gameCtx.shadowBlur = 0;
    gameCtx.shadowOffsetX = 0;
    gameCtx.shadowOffsetY = 0;
    
    // Draw border
    gameCtx.strokeStyle = '#DDD';
    gameCtx.lineWidth = 2;
    gameCtx.stroke();
    
    // Draw title
    gameCtx.fillStyle = '#333';
    gameCtx.font = 'bold 48px "Roboto", sans-serif';
    gameCtx.textAlign = 'center';
    gameCtx.fillText('Sign Language Spelling', gameCanvas.width / 2, titleBoxY + 70);
    
    // Draw subtitle
    gameCtx.font = '24px "Roboto", sans-serif';
    gameCtx.fillText('Spell words using ASL signs', gameCanvas.width / 2, titleBoxY + 120);
    
    // Draw instructions
    gameCtx.font = '18px "Roboto", sans-serif';
    gameCtx.fillText('Press the Start Game button to begin', gameCanvas.width / 2, titleBoxY + 170);
    gameCtx.fillText('Supported letters: A, B, I, K, W', gameCanvas.width / 2, titleBoxY + 200);
    
    // Draw example letters
    const letterSize = 40;
    const spacing = 10;
    const startX = (gameCanvas.width - ((letterSize + spacing) * 5 - spacing)) / 2;
    const startY = titleBoxY + 230;
    
    for (let i = 0; i < SUPPORTED_LETTERS.length; i++) {
        const letter = SUPPORTED_LETTERS[i];
        const x = startX + i * (letterSize + spacing);
        const y = startY;
        
        // Draw letter background
        gameCtx.fillStyle = '#E8EDF9';
        gameCtx.strokeStyle = '#333';
        gameCtx.lineWidth = 1;
        
        gameCtx.beginPath();
        gameCtx.roundRect(x, y, letterSize, letterSize, 5);
        gameCtx.fill();
        gameCtx.stroke();
        
        // Draw letter
        gameCtx.fillStyle = '#333';
        gameCtx.font = 'bold 20px "Roboto", sans-serif';
        gameCtx.textAlign = 'center';
        gameCtx.textBaseline = 'middle';
        gameCtx.fillText(letter, x + letterSize / 2, y + letterSize / 2);
    }
}
