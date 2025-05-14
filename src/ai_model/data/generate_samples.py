import numpy as np
import cv2
import os
from tqdm import tqdm
import random
from ai_model.config import DATA_DIR

def create_retinal_background(size=(380, 380)):
    """Create a synthetic retinal background with blood vessels."""
    background = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    background[:, :] = [random.randint(180, 220), random.randint(180, 220), random.randint(150, 180)]
    
    # optic disc (bright circular area)
    center = (random.randint(size[0]//4, 3*size[0]//4), 
             random.randint(size[1]//4, 3*size[1]//4))
    radius = random.randint(20, 30)
    cv2.circle(background, center, radius, (255, 255, 255), -1)
    
    # blood vessels (more realistic pattern)
    for _ in range(30):
        start_point = (
            center[0] + random.randint(-radius, radius),
            center[1] + random.randint(-radius, radius)
        )
        end_point = (
            random.randint(0, size[0]),
            random.randint(0, size[1])
        )
        thickness = random.randint(1, 3)
        # Dark red color for vessels
        color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
        cv2.line(background, start_point, end_point, color, thickness)
    
    return background

def add_lesions(image, severity):
    """Add synthetic lesions based on severity level (0-4)."""
    if severity == 0:
        return image  # No lesions for normal retina
    
    if severity >= 1:
        for _ in range(random.randint(5, 15)):
            x = random.randint(0, image.shape[1]-1)
            y = random.randint(0, image.shape[0]-1)
            radius = random.randint(1, 3)
            cv2.circle(image, (x, y), radius, (0, 0, 255), -1)
    
    # hemorrhages (larger red spots with irregular shapes)
    if severity >= 2:
        for _ in range(random.randint(3, 8)):
            x = random.randint(0, image.shape[1]-1)
            y = random.randint(0, image.shape[0]-1)
            radius = random.randint(3, 8)
            points = []
            for i in range(8):
                angle = i * np.pi / 4
                r = radius * random.uniform(0.8, 1.2)
                points.append((
                    int(x + r * np.cos(angle)),
                    int(y + r * np.sin(angle))
                ))
            points = np.array(points, np.int32)
            cv2.fillPoly(image, [points], (0, 0, 255))
    
    # exudates (yellow-white spots with soft edges)
    if severity >= 3:
        for _ in range(random.randint(5, 10)):
            x = random.randint(0, image.shape[1]-1)
            y = random.randint(0, image.shape[0]-1)
            radius = random.randint(2, 6)
            for r in range(radius, 0, -1):
                alpha = r / radius
                color = (int(255 * alpha), int(255 * alpha), int(200 * alpha))
                cv2.circle(image, (x, y), r, color, -1)
    
    # Add severe lesions (large areas of damage)
    if severity >= 4:
        for _ in range(random.randint(2, 5)):
            x = random.randint(0, image.shape[1]-1)
            y = random.randint(0, image.shape[0]-1)
            radius = random.randint(10, 20)
            # Create irregular dark area
            points = []
            for i in range(12):
                angle = i * np.pi / 6
                r = radius * random.uniform(0.7, 1.3)
                points.append((
                    int(x + r * np.cos(angle)),
                    int(y + r * np.sin(angle))
                ))
            points = np.array(points, np.int32)
            cv2.fillPoly(image, [points], (0, 0, 0))
    
    return image

def add_noise(image, severity):
    """Add realistic noise to the image."""
    noise = np.random.normal(0, 5 + severity * 2, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    #  slight blur for more realism
    if severity > 0:
        image = cv2.GaussianBlur(image, (3, 3), 0.5)
    
    # add vignette effect (iam not an ai xD)
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/4)
    kernel_y = cv2.getGaussianKernel(rows, rows/4)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    image = image * mask[:, :, np.newaxis]
    
    return image

def generate_sample(severity):
    """Generate a single synthetic retinal image."""
    image = create_retinal_background()
    
    image = add_lesions(image, severity)
    image = add_noise(image, severity)
    
    return image

def generate_dataset(output_dir=DATA_DIR, samples_per_class=200):
    """Generate a complete synthetic dataset."""
    for i in range(5):
        os.makedirs(os.path.join(output_dir, f'class_{i}'), exist_ok=True)
    
    # Generate samples for each class
    for severity in range(5):
        print(f"Generating samples for class {severity}...")
        for i in tqdm(range(samples_per_class)):
            image = generate_sample(severity)
            output_path = os.path.join(output_dir, f'class_{severity}', f'sample_{i:04d}.jpg')
            cv2.imwrite(output_path, image)

if __name__ == "__main__":
    generate_dataset(samples_per_class=200)
    print(f"Generated synthetic dataset in {DATA_DIR}") 