import os
import shutil
import json

def update_signlanguage_model_js():
    """
    Update the signLanguageModel.js file to work with the trained model
    """
    # Path to the signLanguageModel.js file
    js_file = '../src/signLanguageModel.js'
    backup_file = '../src/signLanguageModel_backup.js'
    
    # Create a backup if it doesn't exist
    if not os.path.exists(backup_file) and os.path.exists(js_file):
        shutil.copy(js_file, backup_file)
        print(f"Created backup of {js_file} as {backup_file}")
    
    # Read the original file
    with open(js_file, 'r') as f:
        content = f.read()
    
    # Update the ASL_ALPHABET mapping to include all 26 letters
    asl_alphabet_update = """
// Define the ASL alphabet mapping
const ASL_ALPHABET = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 
    25: 'Z', 26: 'SPACE', 27: 'DELETE', 28: 'NOTHING'
};

// Model loading state
let model = null;
let isModelLoading = false;
let modelConfidence = 0.0;
"""
    
    # Replace the ASL_ALPHABET section
    start_marker = "// Define the ASL alphabet mapping"
    end_marker = "// Model loading state"
    
    start_pos = content.find(start_marker)
    end_pos = content.find(end_marker)
    
    if start_pos != -1 and end_pos != -1:
        # Replace the section
        content = content[:start_pos] + asl_alphabet_update + content[content.find('\n', end_pos) + 1:]
    else:
        print("Warning: Could not find the ASL_ALPHABET section to update.")
    
    # Update the model loading function to use the trained model
    load_model_update = """
/**
 * Load the pre-trained model
 * @returns {Promise<tf.LayersModel>} The loaded model
 */
async function loadModel() {
    if (model) {
        return model; // Return cached model if already loaded
    }

    if (isModelLoading) {
        // If model is currently loading, wait until it's done
        while (isModelLoading) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        return model;
    }

    try {
        isModelLoading = true;
        console.log('Loading sign language model...');
        
        // Try to load the custom trained model
        try {
            model = await tf.loadLayersModel('/models/tfjs_model/model.json');
            console.log('Loaded custom trained ASL model');
            modelConfidence = 0.9; // Higher confidence for the custom model
        } catch (modelError) {
            console.warn('Could not load custom model:', modelError);
            
            // Fall back to the placeholder model
            console.log('Falling back to placeholder model');
            model = await createPlaceholderModel();
            modelConfidence = 0.7; // Lower confidence for the placeholder model
        }
        
        console.log('Sign language model loaded successfully');
        return model;
    } catch (error) {
        console.error('Error loading sign language model:', error);
        throw error;
    } finally {
        isModelLoading = false;
    }
}
"""
    
    # Replace the loadModel function
    start_marker = "/**\n * Load the pre-trained model"
    end_marker = "}"
    
    start_pos = content.find(start_marker)
    if start_pos != -1:
        # Find the end of the function
        function_start = content.find("async function loadModel()", start_pos)
        if function_start != -1:
            # Find the closing brace of the function
            brace_count = 0
            end_pos = function_start
            
            for i in range(function_start, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            if end_pos > function_start:
                # Replace the function
                content = content[:start_pos] + load_model_update + content[end_pos:]
            else:
                print("Warning: Could not find the end of the loadModel function.")
        else:
            print("Warning: Could not find the loadModel function declaration.")
    else:
        print("Warning: Could not find the loadModel function to update.")
    
    # Write the updated content back to the file
    with open(js_file, 'w') as f:
        f.write(content)
    
    print(f"Updated {js_file} to work with the trained model.")
    return True

def create_model_metadata():
    """
    Create a metadata.json file for the model
    """
    # Create models directory if it doesn't exist
    models_dir = '../public/models/tfjs_model'
    os.makedirs(models_dir, exist_ok=True)
    
    # Create metadata.json
    metadata = {
        "name": "ASL Alphabet Recognition Model",
        "description": "Model trained on ASL alphabet gestures",
        "version": "1.0.0",
        "input_shape": [224, 224, 3],
        "output_shape": [26],  # 26 letters (A-Z)
        "classes": [chr(65 + i) for i in range(26)],  # A-Z
        "preprocessing": {
            "resize": [224, 224],
            "normalize": "0-1"
        }
    }
    
    # Write metadata to file
    metadata_path = os.path.join(models_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created model metadata at {metadata_path}")
    return True

def update_client_js():
    """
    Update the client.js file to handle model-based recognition
    """
    # Path to the client.js file
    js_file = '../public/js/client.js'
    backup_file = '../public/js/client_backup.js'
    
    # Create a backup if it doesn't exist
    if not os.path.exists(backup_file) and os.path.exists(js_file):
        shutil.copy(js_file, backup_file)
        print(f"Created backup of {js_file} as {backup_file}")
    
    # Read the original file
    with open(js_file, 'r') as f:
        content = f.read()
    
    # Update the socket.on('frame') handler to handle model-based recognition
    frame_handler_update = """
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
            updateConfidence(confidence * 100);  // Convert to percentage
            
            // Update status with detected letter
            updateStatus(`Detected: ${letter} (${(confidence * 100).toFixed(0)}% confident)`, 'success');
            
            // Update transcription
            const currentContent = transcriptionOutput.dataset.content || '';
            
            // Only add the letter if it's different from the last one
            // or if it's been a while since we added a letter (to allow repeating letters)
            const lastLetter = currentContent.slice(-1);
            const now = Date.now();
            const lastLetterTime = parseInt(transcriptionOutput.dataset.lastLetterTime || '0');
            
            if (currentContent === '' || 
                letter !== lastLetter || 
                (now - lastLetterTime > 1000)) {  // 1 second cooldown for repeating letters
                
                transcriptionOutput.dataset.content = currentContent + letter;
                transcriptionOutput.textContent = transcriptionOutput.dataset.content;
                transcriptionOutput.dataset.lastLetterTime = now.toString();
                
                // Scroll to the bottom of the transcription
                transcriptionOutput.scrollTop = transcriptionOutput.scrollHeight;
            }
        } else {
            updateStatus('No gesture detected', 'info');
            updateConfidence(0);
        }
    });
"""
    
    # Replace the socket.on('frame') handler
    start_marker = "    // Handle incoming frames and tracking data"
    end_marker = "    });"
    
    start_pos = content.find(start_marker)
    if start_pos != -1:
        # Find the end of the handler
        handler_end = content.find(end_marker, start_pos)
        if handler_end != -1:
            # Find the next line after the handler
            next_line = content.find('\n', handler_end) + 1
            
            # Replace the handler
            content = content[:start_pos] + frame_handler_update + content[next_line:]
        else:
            print("Warning: Could not find the end of the frame handler.")
    else:
        print("Warning: Could not find the frame handler to update.")
    
    # Write the updated content back to the file
    with open(js_file, 'w') as f:
        f.write(content)
    
    print(f"Updated {js_file} to handle model-based recognition.")
    return True

if __name__ == "__main__":
    print("Updating frontend to work with the trained model...")
    
    # Update signLanguageModel.js
    if update_signlanguage_model_js():
        print("Successfully updated signLanguageModel.js")
    else:
        print("Failed to update signLanguageModel.js")
    
    # Create model metadata
    if create_model_metadata():
        print("Successfully created model metadata")
    else:
        print("Failed to create model metadata")
    
    # Update client.js
    if update_client_js():
        print("Successfully updated client.js")
    else:
        print("Failed to update client.js")
    
    print("\nFrontend update complete!")
    print("You can now run the application to use the trained model for ASL recognition.")
