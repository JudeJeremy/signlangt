import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import datetime
import pandas as pd
from data_preparation import create_data_generators

def create_model(input_shape=(224, 224, 3), num_classes=26):
    """
    Create a transfer learning model based on ResNet50V2
    
    Args:
        input_shape: Input shape for the model
        num_classes: Number of classes (26 for A-Z)
        
    Returns:
        Compiled model
    """
    # Load pre-trained ResNet50V2 model without top layers
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_generator, val_generator, epochs=20, batch_size=32, model_dir='models'):
    """
    Train the model
    
    Args:
        model: Model to train
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Number of epochs to train
        batch_size: Batch size
        model_dir: Directory to save the model
        
    Returns:
        Training history
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    log_dir = os.path.join(model_dir, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard]
    )
    
    # Save the final model
    model.save(os.path.join(model_dir, 'final_model.h5'))
    
    return history

def unfreeze_and_finetune(model, train_generator, val_generator, epochs=10, batch_size=32, model_dir='models'):
    """
    Unfreeze some layers of the base model and fine-tune
    
    Args:
        model: Model to fine-tune
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Number of epochs to train
        batch_size: Batch size
        model_dir: Directory to save the model
        
    Returns:
        Training history
    """
    # Unfreeze some layers of the base model
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            # This is the base model
            # Unfreeze the last 30 layers
            for i, base_layer in enumerate(layer.layers):
                if i >= len(layer.layers) - 30:
                    base_layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'best_finetuned_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    log_dir = os.path.join(model_dir, 'logs', 'finetune_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Fine-tune the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard]
    )
    
    # Save the final fine-tuned model
    model.save(os.path.join(model_dir, 'final_finetuned_model.h5'))
    
    return history

def evaluate_model(model, test_generator, model_dir='models'):
    """
    Evaluate the model on the test set
    
    Args:
        model: Model to evaluate
        test_generator: Test data generator
        model_dir: Directory to save evaluation results
    """
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Generate classification report
    report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(model_dir, 'classification_report.csv'))
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Save test accuracy to a file
    with open(os.path.join(model_dir, 'test_accuracy.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")

def plot_training_history(history, model_dir='models'):
    """
    Plot training history
    
    Args:
        history: Training history
        model_dir: Directory to save plots
    """
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    plt.close()

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ASL recognition model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for initial training')
    parser.add_argument('--fine-tune-epochs', type=int, default=10, help='Number of epochs for fine-tuning')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=224, help='Image size for training')
    parser.add_argument('--dataset-dir', type=str, default='../dataset', help='Directory containing the dataset')
    parser.add_argument('--model-dir', type=str, default='../models', help='Directory to save the model')
    parser.add_argument('--no-finetune', action='store_true', help='Skip fine-tuning step')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Parameters
    img_size = (args.img_size, args.img_size)
    batch_size = args.batch_size
    initial_epochs = args.epochs
    fine_tune_epochs = args.fine_tune_epochs
    dataset_dir = args.dataset_dir
    model_dir = args.model_dir
    
    # Create data generators
    train_generator, val_generator, test_generator = create_data_generators(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        img_size=img_size
    )
    
    # Create model
    model = create_model(input_shape=(*img_size, 3), num_classes=26)
    
    # Print model summary
    model.summary()
    
    # Train model
    print(f"\nTraining model (initial phase) for {initial_epochs} epochs with batch size {batch_size}...")
    history = train_model(
        model=model,
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=initial_epochs,
        batch_size=batch_size,
        model_dir=model_dir
    )
    
    # Plot training history
    plot_training_history(history, model_dir=model_dir)
    
    # Fine-tune model if not skipped
    if not args.no_finetune:
        print(f"\nFine-tuning model for {fine_tune_epochs} epochs...")
        fine_tune_history = unfreeze_and_finetune(
            model=model,
            train_generator=train_generator,
            val_generator=val_generator,
            epochs=fine_tune_epochs,
            batch_size=batch_size,
            model_dir=model_dir
        )
        
        # Plot fine-tuning history
        plot_training_history(fine_tune_history, model_dir=os.path.join(model_dir, 'finetune'))
    else:
        print("\nSkipping fine-tuning step as requested.")
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, test_generator, model_dir=model_dir)
    
    print("\nModel training and evaluation complete!")
