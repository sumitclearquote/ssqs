'''This script visualizes the license plates annotations from yolo labels for every image. .'''
import sys
sys.path.append("..")
import os
from tqdm import tqdm
from utils.data_utils import *

def yolo_to_corners(x_center, y_center, width, height, image_width, image_height):
    # Reverse the normalization
    xmin = int((x_center - width / 2) * image_width)
    ymin = int((y_center - height / 2) * image_height)
    xmax = int((x_center + width / 2) * image_width)
    ymax = int((y_center + height / 2) * image_height)

    return [xmin, ymin, xmax, ymax]


def visualize_and_save(images_path, labels_path,dest_dir, class_list, limit=10):
    c = 0
    for imgname in tqdm(os.listdir(f"{images_path}")):
        imgpath = f"{images_path}/{imgname}"
        imgname_without_extension = imgname.split(".")[0]
        label_path = f"{labels_path}/{imgname_without_extension}.txt"
        
        annot_file = read_textfile(label_path) #Returns a list of annotations
        
        img = read_img(imgpath)
        h, w, _ = img.shape
        for annot in annot_file:
            class_idx = int(annot.split(" ")[0])
            identity = class_list[class_idx]
            annot_list = annot.split(" ")[1:]
            annot_list = [float(i) for i in annot_list]
            bbox = yolo_to_corners(*annot_list, w, h)
            img = draw_box(img, bbox, color= (0,255,0))
            img = write_text(img, identity, bbox[:2])
        
        
        Image.fromarray(img).save(f"{dest_dir}/{imgname}")
        c +=1
        if c >= limit:
            return "Done"








if __name__ == '__main__':
    #Change the below
    dtype = "val"
    limit = 300 #No of images to visualize
    dataset_path = "datasets/wheelrim-pad-cover_yolo_dataset"
    dest_dir = "datasets/viz_annots/wheelrim_yolo_viz"
    class_list = ['fender_cover', 'lifting_pads', 'wheelrim']  # the list of labels that was used to create yolo dataset
    
    os.makedirs(dest_dir, exist_ok=True)



    images_path = f"{dataset_path}/{dtype}/images"
    labels_path = f"{dataset_path}/{dtype}/labels"
    

    print(visualize_and_save(images_path, labels_path, dest_dir, class_list, limit=limit))
    