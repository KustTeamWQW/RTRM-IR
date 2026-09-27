import cv2
import os
import glob

path = "datasets2/GPSMamba"
pattern = "????_GPSMamba_Local_Enhance_data2_x1.png"
files = sorted(glob.glob(os.path.join(path, pattern)))

processed = []
failed = 0
examples = []

for f in files:
    name = os.path.basename(f)
    try:
        num = int(name[:4])
        if 1001 <= num <= 1028:
            img = cv2.imread(f)
            if img is None:
                failed += 1
                continue
            
            old_shape = img.shape
            # resize to (width=640, height=512)
            resized = cv2.resize(img, (640, 512))
            cv2.imwrite(f, resized)
            
            processed.append(f)
            if len(examples) < 3:
                examples.append(f"{name}: {old_shape} -> {resized.shape}")
    except Exception as e:
        failed += 1

print(f"Processed: {len(processed)}")
print(f"Failed: {failed}")
for ex in examples:
    print(ex)
