import numpy as np
import cv2
import random

class CameraSimulator:
    def __init__(self):
        self.frame = 0
    
    def get_frame(self):
        # Create blank panel
        panel = np.ones((400, 400, 3), dtype=np.uint8) * 200
        
        # Add grid
        for i in range(0, 400, 100):
            cv2.line(panel, (i, 0), (i, 400), (150, 150, 150), 2)
            cv2.line(panel, (0, i), (400, i), (150, 150, 150), 2)
        
        # Randomly add defect
        if random.random() < 0.2:
            x, y = random.randint(50, 300), random.randint(50, 300)
            cv2.line(panel, (x, y), (x+50, y+50), (0, 0, 0), 3)
        
        self.frame += 1
        return panel
