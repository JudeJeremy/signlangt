import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

def generate_synthetic_data(num_samples=1000):
    """
    Generate synthetic data for ASL recognition.
    This is a placeholder for real data collection.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        X: Features (9 Euclidean distances)
        y: Labels (ASL letters)
    """
    # Define the ASL letters
    asl_letters = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z'
    ]
    
    # Generate synthetic data
    X = []
    y = []
    
    for _ in range(num_samples):
        # Generate a random ASL letter
        letter_idx = random.randint(0, len(asl_letters) - 1)
        letter = asl_letters[letter_idx]
        
        # Generate 9 random distances with some pattern based on the letter
        distances = np.random.rand(9) * 0.5  # Base randomness
        
        # Add some pattern based on the letter index
        for i in range(9):
            # Add a pattern: each letter has a unique signature in the distances
            distances[i] += 0.1 * ((letter_idx + i) % 26) / 26
        
        X.append(distances)
        y.append(letter_idx)
    
    return np.array(X), np.array(y)

def train_model():
    """
    Train a simple neural network for ASL recognition.
    
    Returns:
        model: Trained TensorFlow model
    """
    # Generate synthetic data
    X, y = generate_synthetic_data(num_samples=5000)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create a simple neural network
    model = keras.Sequential([
        keras.layers.Input(shape=(9,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(26, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    return model, scaler

def convert_to_tflite(model, output_path):
    """
    Convert a TensorFlow model to TensorFlow Lite format.
    
    Args:
        model: TensorFlow model
        output_path: Path to save the TensorFlow Lite model
    """
    # Convert the model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Train the model
    model, scaler = train_model()
    
    # Convert to TensorFlow Lite
    tflite_path = os.path.join(models_dir, "asl_model.tflite")
    convert_to_tflite(model, tflite_path)
    
    print("Done!")
