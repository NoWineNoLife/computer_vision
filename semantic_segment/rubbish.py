import cv2
import random
from glob import glob

path = "/home/hozumi/datasets/mallampti/images/*.png"

train_str = ""
val_str = ""
for png_file in glob(path):
    img_file = str(png_file).replace(".png", ".jpg")
    oneline = img_file + ", " + png_file + "\n"
    if random.randint(0, 5) == 0:
        val_str = val_str + oneline

    else:
        train_str = train_str + oneline

with open("/home/hozumi/datasets/mallampti/train.txt", 'w') as f:
    f.write(train_str)
f.close()

with open("/home/hozumi/datasets/mallampti/val.txt", 'w') as f:
    f.write(val_str)
f.close()

