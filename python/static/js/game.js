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
let healthElement;
let levelElement;
let currentWordElement;
let detectedLettersElement;
let startButton;
let restartButton;
let gameOverModal;
let finalScoreElement;
let playAgainButton;

// Game state
let gameState = {
    isRunning: false,
    score: 0,
    health: 100,
    level: 1,
    monsters: [],
    detectedLetters: '',
    currentMonsterIndex: 0,
    lastDetectedLetter: '',
    lastDetectionTime: 0
};

// Constants
const MONSTER_SPEED_BASE = 1;
const MONSTER_SPEED_INCREMENT = 0.2;
const PLAYER_POSITION_X = 50;
const PLAYER_SIZE = 60;
const MONSTER_SIZE = 80;
const HEALTH_DECREASE_RATE = 10;
const LETTER_DETECTION_COOLDOWN = 1000; // ms
const SUPPORTED_LETTERS = ['A', 'B', 'K', 'W'];

// Word list (using only A, B, K, W)
const WORD_LIST = [
    'AB', 'AW', 'BA', 'KA', 'WA',
    'BAK', 'WAK', 'BAW', 'KAB', 'WAB',
    'BAKA', 'WAKA', 'ABBA', 'KAWA', 'BAWK',
    'AWKWARD', 'KAWABATA', 'WAKAWAKA'
];

// Monster class
class Monster {
    constructor(word, level) {
        this.word = word;
        this.x = gameCanvas.width;
        this.y = 100 + Math.random() * (gameCanvas.height - 200);
        this.speed = MONSTER_SPEED_BASE + (level - 1) * MONSTER_SPEED_INCREMENT;
        this.color = this.getRandomColor();
        this.currentLetterIndex = 0;
        this.isDefeated = false;
        this.isAttacking = false;
        this.attackTimer = 0;
    }

    update() {
        if (!this.isDefeated) {
            this.x -= this.speed;

            // Check if monster reached the player
            if (this.x <= PLAYER_POSITION_X + PLAYER_SIZE / 2 && !this.isAttacking) {
                this.isAttacking = true;
                this.attackTimer = 0;
            }

            // Handle attack
            if (this.isAttacking) {
                this.attackTimer++;
                if (this.attackTimer >= 60) { // Attack every 60 frames (about 1 second)
                    gameState.health -= HEALTH_DECREASE_RATE;
                    healthElement.style.width = gameState.health + '%';
                    
                    // Update health bar color
                    if (gameState.health <= 30) {
                        healthElement.style.backgroundColor = '#F44336'; // Red
                    } else if (gameState.health <= 60) {
                        healthElement.style.backgroundColor = '#FFC107'; // Amber
                    }
                    
                    this.attackTimer = 0;
                    
                    // Check if game over
                    if (gameState.health <= 0) {
                        endGame();
                    }
                }
            }
        } else {
            // Move defeated monster away
            this.y -= 2;
            this.x += 1;
            
            // Remove monster if it's off screen
            if (this.y < -MONSTER_SIZE || this.x > gameCanvas.width + MONSTER_SIZE) {
                const index = gameState.monsters.indexOf(this);
                if (index !== -1) {
                    gameState.monsters.splice(index, 1);
                }
            }
        }
    }

    draw() {
        gameCtx.save();
        
        // Draw monster body
        gameCtx.fillStyle = this.color;
        gameCtx.beginPath();
        gameCtx.arc(this.x, this.y, MONSTER_SIZE / 2, 0, Math.PI * 2);
        gameCtx.fill();
        
        // Draw monster eyes
        gameCtx.fillStyle = 'white';
        gameCtx.beginPath();
        gameCtx.arc(this.x - 15, this.y - 10, 10, 0, Math.PI * 2);
        gameCtx.arc(this.x + 15, this.y - 10, 10, 0, Math.PI * 2);
        gameCtx.fill();
        
        // Draw monster pupils
        gameCtx.fillStyle = 'black';
        gameCtx.beginPath();
        gameCtx.arc(this.x - 15, this.y - 10, 5, 0, Math.PI * 2);
        gameCtx.arc(this.x + 15, this.y - 10, 5, 0, Math.PI * 2);
        gameCtx.fill();
        
        // Draw monster mouth
        if (this.isDefeated) {
            // X eyes for defeated monster
            gameCtx.strokeStyle = 'black';
            gameCtx.lineWidth = 3;
            gameCtx.beginPath();
            gameCtx.moveTo(this.x - 20, this.y - 15);
            gameCtx.lineTo(this.x - 10, this.y - 5);
            gameCtx.moveTo(this.x - 10, this.y - 15);
            gameCtx.lineTo(this.x - 20, this.y - 5);
            gameCtx.moveTo(this.x + 10, this.y - 15);
            gameCtx.lineTo(this.x + 20, this.y - 5);
            gameCtx.moveTo(this.x + 20, this.y - 15);
            gameCtx.lineTo(this.x + 10, this.y - 5);
            gameCtx.stroke();
            
            // Sad mouth
            gameCtx.beginPath();
            gameCtx.arc(this.x, this.y + 10, 15, Math.PI, Math.PI * 2, true);
            gameCtx.stroke();
        } else if (this.isAttacking) {
            // Angry attacking mouth
            gameCtx.fillStyle = 'black';
            gameCtx.beginPath();
            gameCtx.arc(this.x, this.y + 10, 20, 0, Math.PI);
            gameCtx.fill();
            
            // Teeth
            gameCtx.fillStyle = 'white';
            gameCtx.beginPath();
            gameCtx.moveTo(this.x - 15, this.y + 10);
            gameCtx.lineTo(this.x - 10, this.y + 20);
            gameCtx.lineTo(this.x - 5, this.y + 10);
            gameCtx.fill();
            
            gameCtx.beginPath();
            gameCtx.moveTo(this.x + 5, this.y + 10);
            gameCtx.lineTo(this.x + 10, this.y + 20);
            gameCtx.lineTo(this.x + 15, this.y + 10);
            gameCtx.fill();
        } else {
            // Normal mouth
            gameCtx.fillStyle = 'black';
            gameCtx.beginPath();
            gameCtx.arc(this.x, this.y + 10, 15, 0, Math.PI);
            gameCtx.fill();
        }
        
        // Draw word above monster
        gameCtx.fillStyle = 'black';
        gameCtx.font = '20px Arial';
        gameCtx.textAlign = 'center';
        
        // Draw each letter with appropriate color
        for (let i = 0; i < this.word.length; i++) {
            let letterColor;
            if (i < this.currentLetterIndex) {
                letterColor = '#4CAF50'; // Green for correct letters
            } else if (i === this.currentLetterIndex) {
                letterColor = '#2196F3'; // Blue for current letter
            } else {
                letterColor = '#333'; // Dark gray for pending letters
            }
            
            gameCtx.fillStyle = letterColor;
            gameCtx.fillText(this.word[i], this.x - ((this.word.length - 1) * 10) + (i * 20), this.y - 40);
        }
        
        gameCtx.restore();
    }

    checkLetter(letter) {
        if (this.currentLetterIndex < this.word.length && 
            letter === this.word[this.currentLetterIndex]) {
            this.currentLetterIndex++;
            
            // Check if monster is defeated
            if (this.currentLetterIndex >= this.word.length) {
                this.isDefeated = true;
                this.isAttacking = false;
                
                // Add score based on word length and level
                const scoreGain = this.word.length * 10 * gameState.level;
                gameState.score += scoreGain;
                scoreElement.textContent = gameState.score;
                
                // Show score popup
                showScorePopup(this.x, this.y, scoreGain);
                
                return true;
            }
            return true;
        }
        return false;
    }

    getRandomColor() {
        const colors = [
            '#FF6B6B', // Red
            '#4D96FF', // Blue
            '#6BCB77', // Green
            '#FFD93D', // Yellow
            '#9B72AA'  // Purple
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }
}

// Wait for DOM to load before accessing elements
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    videoElement = document.getElementById('videoElement');
    canvasElement = document.getElementById('canvasElement');
    canvasCtx = canvasElement.getContext('2d');
    statusElement = document.getElementById('statusElement');
    confidenceElement = document.getElementById('confidenceElement');
    scoreElement = document.getElementById('scoreElement');
    healthElement = document.getElementById('healthElement');
    levelElement = document.getElementById('levelElement');
    currentWordElement = document.getElementById('currentWordElement');
    detectedLettersElement = document.getElementById('detectedLettersElement');
    startButton = document.getElementById('startButton');
    restartButton = document.getElementById('restartButton');
    gameOverModal = document.getElementById('gameOverModal');
    finalScoreElement = document.getElementById('finalScoreElement');
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
    
    // Check if this is a new letter
    if (letter !== gameState.lastDetectedLetter) {
        gameState.lastDetectedLetter = letter;
        gameState.lastDetectionTime = now;
        
        // Add letter to detected letters
        gameState.detectedLetters += letter;
        detectedLettersElement.textContent = gameState.detectedLetters;
        
        // Check if letter matches current monster's next letter
        if (gameState.monsters.length > 0) {
            const currentMonster = gameState.monsters[gameState.currentMonsterIndex];
            if (currentMonster.checkLetter(letter)) {
                // Update current word display
                updateCurrentWordDisplay(currentMonster);
                
                // Check if monster is defeated
                if (currentMonster.isDefeated) {
                    // Move to next monster if available
                    if (gameState.currentMonsterIndex < gameState.monsters.length - 1) {
                        gameState.currentMonsterIndex++;
                        updateCurrentWordDisplay(gameState.monsters[gameState.currentMonsterIndex]);
                    }
                }
            }
        }
    }
}

// Update current word display
function updateCurrentWordDisplay(monster) {
    if (!monster) {
        currentWordElement.innerHTML = '';
        return;
    }
    
    let html = '';
    for (let i = 0; i < monster.word.length; i++) {
        let letterClass = '';
        if (i < monster.currentLetterIndex) {
            letterClass = 'letter-correct';
        } else if (i === monster.currentLetterIndex) {
            letterClass = 'letter-current';
        } else {
            letterClass = 'letter-pending';
        }
        
        html += `<span class="${letterClass}">${monster.word[i]}</span>`;
    }
    
    currentWordElement.innerHTML = html;
}

// Start the game
function startGame() {
    resetGame();
    gameState.isRunning = true;
    
    // Hide start button, show restart button
    startButton.style.display = 'none';
    restartButton.style.display = 'inline-block';
    
    // Start game loop
    requestAnimationFrame(gameLoop);
    
    // Start spawning monsters
    spawnMonster();
}

// Reset game state
function resetGame() {
    gameState.isRunning = false;
    gameState.score = 0;
    gameState.health = 100;
    gameState.level = 1;
    gameState.monsters = [];
    gameState.detectedLetters = '';
    gameState.currentMonsterIndex = 0;
    gameState.lastDetectedLetter = '';
    gameState.lastDetectionTime = 0;
    
    // Update UI
    scoreElement.textContent = gameState.score;
    healthElement.style.width = gameState.health + '%';
    healthElement.style.backgroundColor = '#4CAF50';
    levelElement.textContent = gameState.level;
    currentWordElement.innerHTML = '';
    detectedLettersElement.textContent = '';
}

// End the game
function endGame() {
    gameState.isRunning = false;
    
    // Show game over modal
    finalScoreElement.textContent = gameState.score;
    gameOverModal.style.display = 'flex';
}

// Game loop
function gameLoop() {
    if (!gameState.isRunning) return;
    
    // Clear canvas
    gameCtx.clearRect(0, 0, gameCanvas.width, gameCanvas.height);
    
    // Draw background
    drawBackground();
    
    // Draw player
    drawPlayer();
    
    // Update and draw monsters
    updateMonsters();
    
    // Check level progression
    checkLevelProgression();
    
    // Continue game loop
    requestAnimationFrame(gameLoop);
}

// Draw background
function drawBackground() {
    // Draw sky
    const gradient = gameCtx.createLinearGradient(0, 0, 0, gameCanvas.height);
    gradient.addColorStop(0, '#87CEEB');
    gradient.addColorStop(1, '#E0F7FA');
    gameCtx.fillStyle = gradient;
    gameCtx.fillRect(0, 0, gameCanvas.width, gameCanvas.height);
    
    // Draw ground
    gameCtx.fillStyle = '#8BC34A';
    gameCtx.fillRect(0, gameCanvas.height - 50, gameCanvas.width, 50);
}

// Draw player
function drawPlayer() {
    const x = PLAYER_POSITION_X;
    const y = gameCanvas.height - 80;
    
    gameCtx.save();
    
    // Draw player body
    gameCtx.fillStyle = '#2196F3';
    gameCtx.beginPath();
    gameCtx.arc(x, y, PLAYER_SIZE / 2, 0, Math.PI * 2);
    gameCtx.fill();
    
    // Draw player eyes
    gameCtx.fillStyle = 'white';
    gameCtx.beginPath();
    gameCtx.arc(x - 10, y - 10, 8, 0, Math.PI * 2);
    gameCtx.arc(x + 10, y - 10, 8, 0, Math.PI * 2);
    gameCtx.fill();
    
    // Draw player pupils
    gameCtx.fillStyle = 'black';
    gameCtx.beginPath();
    gameCtx.arc(x - 8, y - 10, 4, 0, Math.PI * 2);
    gameCtx.arc(x + 12, y - 10, 4, 0, Math.PI * 2);
    gameCtx.fill();
    
    // Draw player mouth
    gameCtx.beginPath();
    gameCtx.arc(x, y + 5, 15, 0, Math.PI);
    gameCtx.stroke();
    
    gameCtx.restore();
}

// Update and draw monsters
function updateMonsters() {
    for (let i = 0; i < gameState.monsters.length; i++) {
        gameState.monsters[i].update();
        gameState.monsters[i].draw();
    }
}

// Spawn a new monster
function spawnMonster() {
    if (!gameState.isRunning) return;
    
    // Get a random word based on level
    const word = getRandomWord(gameState.level);
    
    // Create a new monster
    const monster = new Monster(word, gameState.level);
    gameState.monsters.push(monster);
    
    // If this is the first monster, set it as current
    if (gameState.monsters.length === 1) {
        gameState.currentMonsterIndex = 0;
        updateCurrentWordDisplay(monster);
    }
    
    // Schedule next monster spawn
    const spawnDelay = Math.max(5000 - (gameState.level * 500), 2000);
    setTimeout(spawnMonster, spawnDelay);
}

// Get a random word based on level
function getRandomWord(level) {
    let wordPool;
    
    if (level <= 2) {
        // Level 1-2: Short words (2-3 letters)
        wordPool = WORD_LIST.filter(word => word.length <= 3);
    } else if (level <= 5) {
        // Level 3-5: Medium words (3-4 letters)
        wordPool = WORD_LIST.filter(word => word.length >= 3 && word.length <= 4);
    } else {
        // Level 6+: Any word, with preference for longer words
        wordPool = WORD_LIST;
        
        // Add some longer words multiple times to increase their probability
        const longWords = WORD_LIST.filter(word => word.length > 4);
        wordPool = wordPool.concat(longWords);
    }
    
    return wordPool[Math.floor(Math.random() * wordPool.length)];
}

// Check level progression
function checkLevelProgression() {
    // Level up every 200 points
    const newLevel = Math.floor(gameState.score / 200) + 1;
    
    if (newLevel > gameState.level) {
        gameState.level = newLevel;
        levelElement.textContent = gameState.level;
        
        // Show level up message
        showLevelUpMessage();
    }
}

// Show level up message
function showLevelUpMessage() {
    const message = document.createElement('div');
    message.textContent = 'LEVEL UP!';
    message.style.position = 'absolute';
    message.style.top = '50%';
    message.style.left = '50%';
    message.style.transform = 'translate(-50%, -50%)';
    message.style.fontSize = '48px';
    message.style.fontWeight = 'bold';
    message.style.color = '#FFD700';
    message.style.textShadow = '2px 2px 4px rgba(0, 0, 0, 0.5)';
    message.style.zIndex = '100';
    
    document.querySelector('.game-area').appendChild(message);
    
    // Animate the message
    let opacity = 1;
    let size = 48;
    
    const animate = () => {
        opacity -= 0.02;
        size += 1;
        
        message.style.opacity = opacity;
        message.style.fontSize = size + 'px';
        
        if (opacity > 0) {
            requestAnimationFrame(animate);
        } else {
            message.remove();
        }
    };
    
    requestAnimationFrame(animate);
}

// Show score popup
function showScorePopup(x, y, score) {
    const popup = document.createElement('div');
    popup.textContent = '+' + score;
    popup.style.position = 'absolute';
    popup.style.top = (y + gameCanvas.offsetTop) + 'px';
    popup.style.left = (x + gameCanvas.offsetLeft) + 'px';
    popup.style.fontSize = '24px';
    popup.style.fontWeight = 'bold';
    popup.style.color = '#4CAF50';
    popup.style.zIndex = '100';
    
    document.querySelector('.game-area').appendChild(popup);
    
    // Animate the popup
    let opacity = 1;
    let posY = y;
    
    const animate = () => {
        opacity -= 0.02;
        posY -= 2;
        
        popup.style.opacity = opacity;
        popup.style.top = (posY + gameCanvas.offsetTop) + 'px';
        
        if (opacity > 0) {
            requestAnimationFrame(animate);
        } else {
            popup.remove();
        }
    };
    
    requestAnimationFrame(animate);
}

// Draw start screen
function drawStartScreen() {
    gameCtx.clearRect(0, 0, gameCanvas.width, gameCanvas.height);
    
    // Draw background
    drawBackground();
    
    // Draw title
    gameCtx.fillStyle = '#333';
    gameCtx.font = 'bold 48px Arial';
    gameCtx.textAlign = 'center';
    gameCtx.fillText('Sign Slayer', gameCanvas.width / 2, 100);
    
    // Draw subtitle
    gameCtx.font = '24px Arial';
    gameCtx.fillText('Defeat monsters with ASL signs!', gameCanvas.width / 2, 150);
    
    // Draw instructions
    gameCtx.font = '18px Arial';
    gameCtx.fillText('Press the Start Game button to begin', gameCanvas.width / 2, 200);
    gameCtx.fillText('Supported letters: A, B, K, W', gameCanvas.width / 2, 230);
    
    // Draw player and monster example
    drawPlayer();
    
    // Example monster
    gameCtx.fillStyle = '#FF6B6B';
    gameCtx.beginPath();
    gameCtx.arc(gameCanvas.width - 100, gameCanvas.height - 80, MONSTER_SIZE / 2, 0, Math.PI * 2);
    gameCtx.fill();
    
    // Monster eyes
    gameCtx.fillStyle = 'white';
    gameCtx.beginPath();
    gameCtx.arc(gameCanvas.width - 115, gameCanvas.height - 90, 10, 0, Math.PI * 2);
    gameCtx.arc(gameCanvas.width - 85, gameCanvas.height - 90, 10, 0, Math.PI * 2);
    gameCtx.fill();
    
    // Monster pupils
    gameCtx.fillStyle = 'black';
    gameCtx.beginPath();
    gameCtx.arc(gameCanvas.width - 115, gameCanvas.height - 90, 5, 0, Math.PI * 2);
    gameCtx.arc(gameCanvas.width - 85, gameCanvas.height - 90, 5, 0, Math.PI * 2);
    gameCtx.fill();
    
    // Monster mouth
    gameCtx.fillStyle = 'black';
    gameCtx.beginPath();
    gameCtx.arc(gameCanvas.width - 100, gameCanvas.height - 70, 15, 0, Math.PI);
    gameCtx.fill();
    
    // Example word
    gameCtx.fillStyle = 'black';
    gameCtx.font = '20px Arial';
    gameCtx.textAlign = 'center';
    gameCtx.fillText('BAK', gameCanvas.width - 100, gameCanvas.height - 120);
}
