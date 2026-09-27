import cv2
import os

targets = ["1001_GPSMamba_Local_Enhance_data2_x1.png", "1014_GPSMamba_Local_Enhance_data2_x1.png", "1028_GPSMamba_Local_Enhance_data2_x1.png"]
path = "datasets2/GPSMamba"

for name in targets:
    f = os.path.join(path, name)
    if os.path.exists(f):
        img = cv2.imread(f)
        print(f"{name}: {img.shape}")
    else:
        print(f"{name}: NOT FOUND")
