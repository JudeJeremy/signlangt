import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import random

def create_dataset_directories(base_dir='dataset'):
    """
    Create directories for the organized dataset
    
    Args:
        base_dir: Base directory for the organized dataset
    """
    # Create main directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create train, validation, and test directories
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create class directories (A-Z)
        for i in range(26):
            class_dir = os.path.join(split_dir, chr(65 + i))  # ASCII 65 = 'A'
            os.makedirs(class_dir, exist_ok=True)

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess an image for the model
    
    Args:
        image_path: Path to the image
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image
    """
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    return img

def organize_dataset(gestures_dir='gestures', output_dir='dataset', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Organize the dataset into train, validation, and test sets
    
    Args:
        gestures_dir: Directory containing the gesture folders
        output_dir: Output directory for the organized dataset
        train_ratio: Ratio of images for training
        val_ratio: Ratio of images for validation
        test_ratio: Ratio of images for testing
    """
    # Create dataset directories
    create_dataset_directories(output_dir)
    
    # Process each letter folder (0-25 for A-Z)
    for i in range(26):
        folder_name = str(i)
        folder_path = os.path.join(gestures_dir, folder_name)
        letter = chr(65 + i)  # ASCII 65 = 'A'
        
        print(f"Processing folder {folder_name} (Letter {letter})...")
        
        # Get all image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Shuffle the files to ensure random distribution
        random.shuffle(image_files)
        
        # Split the files into train, validation, and test sets
        n_files = len(image_files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Copy files to their respective directories
        for files, split in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
            target_dir = os.path.join(output_dir, split, letter)
            
            for file in files:
                src_path = os.path.join(folder_path, file)
                dst_path = os.path.join(target_dir, file)
                
                # Preprocess and save the image
                img = preprocess_image(src_path)
                if img is not None:
                    cv2.imwrite(dst_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        print(f"  - {len(train_files)} training images")
        print(f"  - {len(val_files)} validation images")
        print(f"  - {len(test_files)} test images")

def create_data_generators(dataset_dir='dataset', batch_size=32, img_size=(224, 224)):
    """
    Create data generators for training, validation, and testing
    
    Args:
        dataset_dir: Directory containing the organized dataset
        batch_size: Batch size for training
        img_size: Image size for the model
        
    Returns:
        train_generator, val_generator, test_generator
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and testing
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(dataset_dir, 'val'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_dir, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def visualize_samples(dataset_dir='dataset', num_samples=5):
    """
    Visualize sample images from each class
    
    Args:
        dataset_dir: Directory containing the organized dataset
        num_samples: Number of samples to visualize per class
    """
    # Get all class directories
    train_dir = os.path.join(dataset_dir, 'train')
    class_dirs = sorted(os.listdir(train_dir))
    
    # Create a figure
    plt.figure(figsize=(15, 15))
    
    # For each class, display some samples
    for i, class_dir in enumerate(class_dirs):
        class_path = os.path.join(train_dir, class_dir)
        images = os.listdir(class_path)[:num_samples]
        
        for j, image in enumerate(images):
            img_path = os.path.join(class_path, image)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(len(class_dirs), num_samples, i * num_samples + j + 1)
            plt.imshow(img)
            plt.title(f"Class: {class_dir}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_dir, 'samples.png'))
    plt.close()

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare ASL dataset')
    parser.add_argument('--gestures-dir', type=str, default='../gestures', help='Directory containing gesture folders')
    parser.add_argument('--output-dir', type=str, default='../dataset', help='Output directory for the organized dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Ratio of images for training')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Ratio of images for validation')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Ratio of images for testing')
    parser.add_argument('--no-visualize', action='store_true', help='Skip dataset visualization')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
    # Organize the dataset
    organize_dataset(
        gestures_dir=args.gestures_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Visualize samples if not skipped
    if not args.no_visualize:
        visualize_samples(dataset_dir=args.output_dir, num_samples=3)
    
    print("Dataset preparation complete!")
