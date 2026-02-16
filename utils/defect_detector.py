"""
Core defect detection logic using OpenCV
"""

import cv2
import numpy as np
import streamlit as st

class DefectDetector:
    """
    Solar panel defect detection using computer vision
    """
    
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.defect_types = ['crack', 'scratch', 'discoloration', 'soldering_issue']
        
    def detect_defects(self, image):
        """
        Detect defects in solar panel image
        
        Args:
            image: numpy array of the panel image
            
        Returns:
            annotated_image: image with defects highlighted
            defects: list of defect locations and types
            defect_count: number of defects found
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding to find anomalies
        thresh = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours (potential defects)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area (remove noise)
        min_area = image.shape[0] * image.shape[1] * 0.001  # 0.1% of image area
        max_area = image.shape[0] * image.shape[1] * 0.1   # 10% of image area
        
        defects = []
        annotated_image = image.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area < area < max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Classify defect type based on shape characteristics
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                # Simple classification logic
                if circularity < 0.3:
                    defect_type = 'crack'
                elif w/h > 3 or h/w > 3:
                    defect_type = 'scratch'
                elif area > (image.shape[0] * image.shape[1] * 0.05):
                    defect_type = 'discoloration'
                else:
                    defect_type = 'soldering_issue'
                
                # Draw defect highlight
                cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(
                    annotated_image, 
                    defect_type, 
                    (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 0, 0), 
                    2
                )
                
                defects.append({
                    'type': defect_type,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area
                })
        
        # Apply threshold-based filtering
        defect_count = len(defects)
        if defect_count > 0 and self.threshold < 0.5:
            # Higher sensitivity in demo mode
            pass
        elif defect_count > 0 and self.threshold >= 0.5:
            # Lower sensitivity - only keep larger defects
            defects = [d for d in defects if d['area'] > min_area * 2]
            defect_count = len(defects)
        
        return annotated_image, defects, defect_count
    
    def simulate_defect(self, image):
        """
        Simulate a defect for testing purposes
        """
        result = image.copy()
        h, w = result.shape[:2]
        
        # Add random scratch
        cv2.line(
            result,
            (np.random.randint(0, w), np.random.randint(0, h)),
            (np.random.randint(0, w), np.random.randint(0, h)),
            (0, 0, 0),
            2
        )
        
        return result
