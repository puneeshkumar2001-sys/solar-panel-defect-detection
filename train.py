"""
Training Script for Solar Panel Defect Detection Model
Run this script to train the model on your own dataset
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime

from model import SolarDefectDetector


def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")


def train_model(train_dir, val_dir, epochs=50, batch_size=32,
                model_save_path='models/solar_defect_model.h5'):
    """
    Train the solar panel defect detection model

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_save_path: Path to save the trained model
    """

    print("=" * 60)
    print("Solar Panel Defect Detection - Model Training")
    print("=" * 60)

    # Create output directories
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Initialize model
    print("\n[1/6] Initializing model...")
    detector = SolarDefectDetector()
    print(f"Model architecture created with {detector.model.count_params():,} parameters")

    # Data augmentation for training
    print("\n[2/6] Setting up data augmentation...")
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.15,
        fill_mode='nearest'
    )

    # No augmentation for validation
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    # Load data
    print("\n[3/6] Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=detector.img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    print("\n[4/6] Loading validation data...")
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=detector.img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Number of classes: {train_generator.num_classes}")
    print(f"Class indices: {train_generator.class_indices}")

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [
        # Save best model
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),

        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),

        # TensorBoard logging
        TensorBoard(
            log_dir=f'logs/fit_{timestamp}',
            histogram_freq=1
        )
    ]

    # Train model
    print("\n[5/6] Training model...")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {train_generator.samples // batch_size}")
    print("-" * 60)

    history = detector.model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    print(f"\n[6/6] Saving final model to {model_save_path}...")
    detector.save_model(model_save_path)

    # Plot training history
    print("\nGenerating training visualizations...")
    plot_training_history(history, 'results/training_history.png')

    # Evaluate on validation set
    print("\nEvaluating model on validation set...")
    val_generator.reset()
    predictions = detector.model.predict(val_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes

    # Classification report
    class_names = list(val_generator.class_indices.keys())
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, 'results/confusion_matrix.png')

    # Final metrics
    val_loss, val_accuracy = detector.model.evaluate(val_generator, verbose=0)
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"\nModel saved to: {model_save_path}")
    print(f"Training history: results/training_history.png")
    print(f"Confusion matrix: results/confusion_matrix.png")
    print(f"TensorBoard logs: logs/fit_{timestamp}")
    print("\nTo view TensorBoard logs, run:")
    print(f"  tensorboard --logdir logs/fit_{timestamp}")
    print("=" * 60)

    return history, detector


def main():
    parser = argparse.ArgumentParser(description='Train Solar Panel Defect Detection Model')

    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to validation data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--output', type=str, default='models/solar_defect_model.h5',
                        help='Path to save trained model (default: models/solar_defect_model.h5)')

    args = parser.parse_args()

    # Validate directories exist
    if not os.path.exists(args.train_dir):
        print(f"Error: Training directory not found: {args.train_dir}")
        return

    if not os.path.exists(args.val_dir):
        print(f"Error: Validation directory not found: {args.val_dir}")
        return

    # Train model
    train_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.output
    )


if __name__ == '__main__':
    main()
