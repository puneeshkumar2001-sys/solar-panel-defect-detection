import numpy as np
import cv2
import random
from datetime import datetime, timedelta


def generate_sample_solar_panel(defect_type='none', size=(800, 600)):
    """
    Generate synthetic solar panel image for demo purposes
    defect_type: 'none', 'crack', 'scratch', 'discoloration', 'soldering'
    """
    h, w = size

    # Create base solar panel (grid of cells)
    panel = np.ones((h, w, 3), dtype=np.uint8) * 30  # Dark blue background

    # Draw solar cells (6x10 grid)
    cell_w = w // 6
    cell_h = h // 10
    margin = 5

    for i in range(10):
        for j in range(6):
            x1 = j * cell_w + margin
            y1 = i * cell_h + margin
            x2 = (j + 1) * cell_w - margin
            y2 = (i + 1) * cell_h - margin

            # Slight color variation for realistic look
            color_var = random.randint(-10, 10)
            cell_color = (40 + color_var, 60 + color_var, 100 + color_var)
            cv2.rectangle(panel, (x1, y1), (x2, y2), cell_color, -1)

            # Draw cell borders (bus bars)
            cv2.rectangle(panel, (x1, y1), (x2, y2), (180, 180, 180), 1)

            # Draw fingers (thin lines on cells)
            for k in range(3):
                line_x = x1 + (x2 - x1) * (k + 1) // 4
                cv2.line(panel, (line_x, y1), (line_x, y2), (200, 200, 200), 1)

    # Add defects based on type
    if defect_type == 'crack':
        # Add micro-cracks
        for _ in range(random.randint(2, 4)):
            start_x = random.randint(w // 4, 3 * w // 4)
            start_y = random.randint(h // 4, 3 * h // 4)
            end_x = start_x + random.randint(-100, 100)
            end_y = start_y + random.randint(50, 150)

            # Draw crack with slight randomness
            points = []
            steps = 20
            for i in range(steps + 1):
                t = i / steps
                x = int(start_x + (end_x - start_x) * t + random.randint(-5, 5))
                y = int(start_y + (end_y - start_y) * t + random.randint(-5, 5))
                points.append([x, y])

            points = np.array(points)
            cv2.polylines(panel, [points], False, (0, 0, 0), 2)

    elif defect_type == 'scratch':
        # Add scratches
        for _ in range(random.randint(3, 6)):
            start_x = random.randint(0, w)
            start_y = random.randint(0, h)
            end_x = start_x + random.randint(-200, 200)
            end_y = start_y + random.randint(-50, 50)
            cv2.line(panel, (start_x, start_y), (end_x, end_y), (100, 100, 100), 2)

    elif defect_type == 'discoloration':
        # Add discolored patches
        for _ in range(random.randint(2, 4)):
            center_x = random.randint(w // 4, 3 * w // 4)
            center_y = random.randint(h // 4, 3 * h // 4)
            radius = random.randint(40, 80)

            # Create discoloration overlay
            overlay = panel.copy()
            color = (random.randint(80, 120), random.randint(80, 100), random.randint(60, 80))
            cv2.circle(overlay, (center_x, center_y), radius, color, -1)
            cv2.addWeighted(overlay, 0.6, panel, 0.4, 0, panel)

    elif defect_type == 'soldering':
        # Add soldering issues (broken or misaligned bus bars)
        for i in range(random.randint(3, 6)):
            cell_i = random.randint(0, 9)
            cell_j = random.randint(0, 5)
            x1 = cell_j * cell_w + margin
            y1 = cell_i * cell_h + margin
            x2 = (cell_j + 1) * cell_w - margin
            y2 = (cell_i + 1) * cell_h - margin

            # Draw broken solder joint
            joint_x = (x1 + x2) // 2
            joint_y = (y1 + y2) // 2
            cv2.circle(panel, (joint_x, joint_y), 8, (0, 0, 200), -1)
            cv2.circle(panel, (joint_x, joint_y), 12, (0, 0, 150), 2)

    # Add slight noise for realism
    noise = np.random.normal(0, 5, panel.shape).astype(np.uint8)
    panel = cv2.add(panel, noise)

    return panel


def create_sample_dataset(num_samples_per_class=20):
    """Create sample dataset for demo purposes"""
    defect_types = ['none', 'crack', 'scratch', 'discoloration', 'soldering']

    images = []
    labels = []

    for defect_type in defect_types:
        for _ in range(num_samples_per_class):
            img = generate_sample_solar_panel(defect_type)
            images.append(img)
            labels.append(defect_types.index(defect_type))

    return np.array(images), np.array(labels)


def generate_demo_logs(num_logs=100):
    """Generate demo inspection logs for analytics"""
    logs = []

    defect_types = ['No Defect', 'Micro-Crack', 'Scratch', 'Discoloration', 'Soldering Issue']
    statuses = ['PASS', 'FAIL']

    # Generate logs over past 7 days
    start_date = datetime.now() - timedelta(days=7)

    for i in range(num_logs):
        # 85% pass rate (realistic for good production)
        status = random.choices(statuses, weights=[85, 15])[0]

        if status == 'PASS':
            defect_type = 'No Defect'
            confidence = random.uniform(0.90, 0.99)
        else:
            defect_type = random.choice(defect_types[1:])  # Pick a defect
            confidence = random.uniform(0.75, 0.95)

        # Random timestamp within past 7 days
        timestamp = start_date + timedelta(
            days=random.randint(0, 7),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        log = {
            'panel_id': f"PANEL_{i + 1:06d}",
            'timestamp': timestamp.isoformat(),
            'status': status,
            'defect_type': defect_type,
            'confidence': confidence
        }

        logs.append(log)

    return logs


def save_demo_images(output_dir='demo_images'):
    """Save demo solar panel images"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    defect_types = {
        'good_panel': 'none',
        'cracked_panel': 'crack',
        'scratched_panel': 'scratch',
        'discolored_panel': 'discoloration',
        'soldering_issue': 'soldering'
    }

    saved_paths = {}

    for name, defect_type in defect_types.items():
        img = generate_sample_solar_panel(defect_type)
        path = os.path.join(output_dir, f'{name}.jpg')
        cv2.imwrite(path, img)
        saved_paths[name] = path

    return saved_paths
