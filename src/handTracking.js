// Gesture recognition thresholds and parameters
const GESTURE_COOLDOWN = 1000; // milliseconds
let lastGestureTime = 0;

/**
 * Recognize gestures from hand landmarks
 * @param {Array} landmarks - Array of hand landmarks from MediaPipe
 * @returns {string|null} - Recognized gesture or null if no gesture recognized
 */
function recognizeGesture(landmarks) {
    // Check if enough time has passed since the last gesture
    const now = Date.now();
    if (now - lastGestureTime < GESTURE_COOLDOWN) {
        return null;
    }
    
    // Recognize thumbs up gesture
    if (isThumbsUp(landmarks)) {
        lastGestureTime = now;
        return "👍";
    }
    
    // Recognize thumbs down gesture
    if (isThumbsDown(landmarks)) {
        lastGestureTime = now;
        return "👎";
    }
    
    // Recognize open hand gesture
    if (isOpenHand(landmarks)) {
        lastGestureTime = now;
        return "✋";
    }
    
    // Recognize closed fist gesture
    if (isClosedFist(landmarks)) {
        lastGestureTime = now;
        return "✊";
    }
    
    // No gesture recognized
    return null;
}

/**
 * Check if the hand is making a thumbs up gesture
 * @param {Array} landmarks - Array of hand landmarks from MediaPipe
 * @returns {boolean} - True if thumbs up gesture detected
 */
function isThumbsUp(landmarks) {
    // Thumb tip is higher than index finger tip
    const thumbTip = landmarks[4];
    const indexFingerTip = landmarks[8];
    
    return thumbTip.y < indexFingerTip.y;
}

/**
 * Check if the hand is making a thumbs down gesture
 * @param {Array} landmarks - Array of hand landmarks from MediaPipe
 * @returns {boolean} - True if thumbs down gesture detected
 */
function isThumbsDown(landmarks) {
    // Thumb tip is lower than index finger tip
    const thumbTip = landmarks[4];
    const indexFingerTip = landmarks[8];
    
    return thumbTip.y > indexFingerTip.y && thumbTip.y > landmarks[0].y;
}

/**
 * Check if the hand is making an open hand gesture
 * @param {Array} landmarks - Array of hand landmarks from MediaPipe
 * @returns {boolean} - True if open hand gesture detected
 */
function isOpenHand(landmarks) {
    // All finger tips should be higher than their middle joints
    const fingerTips = [8, 12, 16, 20]; // Index, middle, ring, pinky tips
    const fingerMiddle = [6, 10, 14, 18]; // Corresponding middle joints
    
    // Check if all fingers are extended
    const allFingersExtended = fingerTips.every((tip, i) => 
        landmarks[tip].y < landmarks[fingerMiddle[i]].y
    );
    
    return allFingersExtended;
}

/**
 * Check if the hand is making a closed fist gesture
 * @param {Array} landmarks - Array of hand landmarks from MediaPipe
 * @returns {boolean} - True if closed fist gesture detected
 */
function isClosedFist(landmarks) {
    // All finger tips should be lower than their middle joints
    const fingerTips = [8, 12, 16, 20]; // Index, middle, ring, pinky tips
    const fingerMiddle = [6, 10, 14, 18]; // Corresponding middle joints
    
    // Check if all fingers are curled
    const allFingersCurled = fingerTips.every((tip, i) => 
        landmarks[tip].y > landmarks[fingerMiddle[i]].y
    );
    
    return allFingersCurled;
}

module.exports = {
    recognizeGesture
};
