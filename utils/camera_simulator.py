"""
Camera simulator for factory integration demo
Simulates GigE camera feed via RTSP stream
"""

import cv2
import numpy as np
import random

class CameraSimulator:
    """
    Simulates a factory camera feed
    In production, this would connect to real GigE cameras via RTSP
    """
    
    def __init__(self):
        self.frame_count = 0
        self.good_panels = 0
        self.defective_panels = 0
        
        # Create sample panel templates
        self.create_templates()
    
    def create_templates(self):
        """Create synthetic panel images for simulation"""
        self.templates = []
        
        # Good panel template (clean)
        good_panel = np.ones((400, 400, 3), dtype=np.uint8) * 200
        # Add grid lines (simulating solar cells)
        for i in range(0, 400, 100):
            cv2.line(good_panel, (i, 0), (i, 400), (150, 150, 150), 2)
            cv2.line(good_panel, (0, i), (400, i), (150, 150, 150), 2)
        self.templates.append(('good', good_panel))
        
        # Defective panel 1 (with cracks)
        defective1 = good_panel.copy()
        for _ in range(3):
            x1, y1 = random.randint(50, 350), random.randint(50, 350)
            x2, y2 = x1 + random.randint(20, 50), y1 + random.randint(20, 50)
            cv2.line(defective1, (x1, y1), (x2, y2), (0, 0, 0), 3)
        self.templates.append(('crack', defective1))
        
        # Defective panel 2 (with scratches)
        defective2 = good_panel.copy()
        for _ in range(2):
            x1, y1 = random.randint(30, 370), random.randint(30, 370)
            x2, y2 = x1 + random.randint(100, 200), y1
            cv2.line(defective2, (x1, y1), (x2, y2), (100, 100, 100), 2)
        self.templates.append(('scratch', defective2))
        
        # Defective panel 3 (discoloration)
        defective3 = good_panel.copy()
        x, y = random.randint(100, 250), random.randint(100, 250)
        cv2.rectangle(defective3, (x, y), (x+100, y+100), (100, 100, 150), -1)
        self.templates.append(('discoloration', defective3))
    
    def stream_frames(self):
        """
        Generator that yields frames continuously
        In production: cap = cv2.VideoCapture("rtsp://camera-ip:554/stream")
        """
        while True:
            # Simulate 80% good, 20% defective panels
            if random.random() < 0.8:
                frame = self.templates[0][1].copy()  # Good panel
                self.good_panels += 1
            else:
                # Random defective template
                idx = random.randint(1, len(self.templates)-1)
                frame = self.templates[idx][1].copy()
                self.defective_panels += 1
            
            # Add some random noise (real camera has noise)
            noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
            frame = cv2.add(frame, noise)
            
            self.frame_count += 1
            yield frame
    
    def get_stats(self):
        """Get camera statistics"""
        return {
            'frames_processed': self.frame_count,
            'good_panels': self.good_panels,
            'defective_panels': self.defective_panels,
            'yield_rate': (self.good_panels / max(1, self.frame_count)) * 100
        }
