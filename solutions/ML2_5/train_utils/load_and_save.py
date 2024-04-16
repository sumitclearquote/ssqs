#This code loads an image and saves it back to disk: Helps in solving corrupt data error.
from PIL import Image, ImageFile
import os
from tqdm import tqdm
import sys
sys.path.append("..")

from utils.data_utils import read_img, get_size
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000000


dtype = "val"

data_dir = f"datasets/wheelrim-pad-cover_yolo_dataset/{dtype}/images"


exts = ["jpg", "jpeg", "png"]

all_images = os.listdir(data_dir)

print("Total images before loading and saving: ", len(all_images))
print("Current folder size: ", get_size(data_dir))

failed_imgs = 0
for imgname in tqdm(all_images):
    if any([imgname.lower().endswith(i) for i in exts]):
        imgpath = f"{data_dir}/{imgname}"
        img = read_img(imgpath, img_type='pil')
        if img is not None:
            img.save(imgpath)
        else:
            failed_imgs += 1
            
            
print("Folder size after loading and saving: ", get_size(data_dir))            
print("Total images after loading and saving: ", len(os.listdir(data_dir)))
print("\n Total Failed images: ", failed_imgs)
print("\nDone")

