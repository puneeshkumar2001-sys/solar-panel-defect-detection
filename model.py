import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
from datetime import datetime
import json
import os


class SolarDefectDetector:
    """
    Solar Panel Defect Detection Model
    Detects: Micro-cracks, Scratches, Discoloration, Soldering Issues
    """

    def __init__(self, model_path=None):
        self.img_size = (224, 224)
        self.defect_types = ['Micro-Crack', 'Scratch', 'Discoloration', 'Soldering Issue']

        if model_path and os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
        else:
            self.model = self._build_model()

    def _build_model(self):
        """Build CNN model for defect classification"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(224, 224, 3)),

            # Convolutional blocks
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),

            # Output layer (5 classes: 4 defects + 1 no defect)
            layers.Dense(5, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if isinstance(image, str):
            image = cv2.imread(image)

        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, self.img_size)

        # Normalize
        image = image.astype('float32') / 255.0

        return image

    def detect_defects(self, image):
        """
        Detect defects in solar panel image
        Returns: dict with defect type, confidence, and status
        """
        # Preprocess
        processed_img = self.preprocess_image(image)

        # Add batch dimension
        img_array = np.expand_dims(processed_img, axis=0)

        # Predict
        predictions = self.model.predict(img_array, verbose=0)

        # Get results
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])

        # Class 0 = No Defect, Classes 1-4 = Defects
        if class_idx == 0:
            status = 'PASS'
            defect_type = 'No Defect'
        else:
            status = 'FAIL'
            defect_type = self.defect_types[class_idx - 1]

        result = {
            'status': status,
            'defect_type': defect_type,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'all_predictions': {
                'No Defect': float(predictions[0][0]),
                'Micro-Crack': float(predictions[0][1]),
                'Scratch': float(predictions[0][2]),
                'Discoloration': float(predictions[0][3]),
                'Soldering Issue': float(predictions[0][4])
            }
        }

        return result

    def annotate_image(self, image, result):
        """Annotate image with detection results"""
        if isinstance(image, str):
            image = cv2.imread(image)

        annotated = image.copy()
        h, w = annotated.shape[:2]

        # Color based on status
        color = (0, 255, 0) if result['status'] == 'PASS' else (0, 0, 255)

        # Add border
        cv2.rectangle(annotated, (10, 10), (w - 10, h - 10), color, 5)

        # Add text background
        text = f"{result['status']}: {result['defect_type']}"
        confidence_text = f"Confidence: {result['confidence'] * 100:.1f}%"

        # Calculate text size for background
        font = cv2.FONT_HERSHEY_BOLD
        text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
        conf_size = cv2.getTextSize(confidence_text, font, 0.8, 2)[0]

        # Draw text backgrounds
        cv2.rectangle(annotated, (20, 20), (30 + text_size[0], 70 + text_size[1]), (0, 0, 0), -1)
        cv2.rectangle(annotated, (20, 80 + text_size[1]), (30 + conf_size[0], 110 + text_size[1] + conf_size[1]),
                      (0, 0, 0), -1)

        # Draw text
        cv2.putText(annotated, text, (25, 50 + text_size[1]), font, 1.5, color, 3)
        cv2.putText(annotated, confidence_text, (25, 100 + text_size[1] + conf_size[1]), font, 0.8, (255, 255, 255), 2)

        # If defect detected, add additional markers
        if result['status'] == 'FAIL':
            # Add corner markers to highlight defect area (simulated)
            margin = 100
            cv2.circle(annotated, (w // 2, h // 2), 50, (0, 0, 255), 3)
            cv2.putText(annotated, "DEFECT AREA", (w // 2 - 80, h // 2 - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return annotated

    def save_model(self, path='models/solar_defect_model.h5'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")

    def train(self, train_data, train_labels, validation_data=None, epochs=50, batch_size=32):
        """
        Train the model
        train_data: numpy array of images
        train_labels: numpy array of one-hot encoded labels
        """
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]

        history = self.model.fit(
            train_data,
            train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history


class InspectionLogger:
    """Log all inspections for records and analytics"""

    def __init__(self, log_file='data/inspection_log.json'):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Initialize log file if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                json.dump([], f)

    def log_inspection(self, result, panel_id=None):
        """Log an inspection result"""
        # Load existing logs
        with open(self.log_file, 'r') as f:
            logs = json.load(f)

        # Add new log entry
        log_entry = {
            'panel_id': panel_id or f"PANEL_{len(logs) + 1:06d}",
            'timestamp': result['timestamp'],
            'status': result['status'],
            'defect_type': result['defect_type'],
            'confidence': result['confidence']
        }

        logs.append(log_entry)

        # Save updated logs
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)

        return log_entry

    def get_statistics(self):
        """Get inspection statistics"""
        with open(self.log_file, 'r') as f:
            logs = json.load(f)

        if not logs:
            return {
                'total_inspections': 0,
                'pass_count': 0,
                'fail_count': 0,
                'yield_rate': 0,
                'defect_breakdown': {}
            }

        total = len(logs)
        pass_count = sum(1 for log in logs if log['status'] == 'PASS')
        fail_count = total - pass_count

        # Defect breakdown
        defect_counts = {}
        for log in logs:
            if log['status'] == 'FAIL':
                defect_type = log['defect_type']
                defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1

        return {
            'total_inspections': total,
            'pass_count': pass_count,
            'fail_count': fail_count,
            'yield_rate': (pass_count / total * 100) if total > 0 else 0,
            'defect_breakdown': defect_counts
        }
