import numpy as np
import cv2
import os
from pathlib import Path

# Step 1: Create a red dummy image (224x224 pixels)
dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
dummy_img[:] = [0, 0, 255]  # Red color in BGR format

# Step 2: Build absolute path to the output file
project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / "output"
save_path = output_dir / "test_image.jpg"

# Step 3: Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Step 4: Try saving the image
success = cv2.imwrite(str(save_path), dummy_img)

# Step 5: Confirm result
if success and save_path.exists():
    print(f"✅ Dummy image saved successfully at:\n{save_path}")
else:
    print("❌ Dummy image save FAILED.")
