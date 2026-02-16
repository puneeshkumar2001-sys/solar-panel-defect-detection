import cv2
import numpy as np

class DefectDetector:
    def __init__(self, threshold=0.3):
        self.threshold = threshold
    
    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = image.shape[0] * image.shape[1] * 0.001
        defects = []
        result = image.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                defects.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area})
                cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        return result, defects, len(defects)
