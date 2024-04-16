'''Script to Convert VGG Format annotations to Yolo format

## Convert VGG format to YOLOv8
* Each image will have corresponding label txt file with the same name.
* each label text file will be of format: class_idx box_center_x box_center_y bbox_width bbox_height. 
* FOr the coordinates, each value is divided by the corresponding height and width of the image. x axis values are divided by width. y-axis values are divided by height.
* class_idx is got by sorting the panel_count dictionary by key in ascending order and using their respective indexes in the list.

'''


import numpy as np
import os
from tqdm import tqdm
from collections import Counter
import shutil
import sys
sys.path.append("..")

from utils.data_utils import read_img, read_json
from utils.annotation_utils import *

def fix_out_of_bound_bbox(bbox, img_h, img_w):
    x1,y1,x2,y2 = bbox
    x1 = max(x1, 1)
    y1 = max(y1, 1)
    x2 = min(x2, img_w-1) 
    y2 = min(y2, img_h-1)
    return x1,y1,x2,y2

def convert_yolo_dict_to_dataset(yolo_annots, data_dir, dest_dir, dataset_type):
    for folder_path, annot_list in tqdm(yolo_annots.items()):
        imgpath = f"{data_dir}/{dataset_type}/{folder_path}"
        imgname_without_extension = folder_path.split("/")[-1].split(".")[0]
        
        txtfile = ""
        for annot in annot_list:
            txtfile +=  annot + "\n"
        
        txtfile = txtfile.strip("\n")
        
        shutil.copy(imgpath, f"{dest_dir}/{dataset_type}/images")
        with open(f"{dest_dir}/{dataset_type}/labels/{imgname_without_extension}.txt", "w") as f:
            f.write(txtfile)
            

def vgg_to_yolo(vgg_annotation, relevant_classes, class_list, dataset_type = None, data_dir = "new_van_data_resized", annot_type = 'poly'):
    yolo_annotations = {}
    ids = []
    for imgname, annotation_info in tqdm(vgg_annotation.items()):
        folderpath = annotation_info['filename']
        img_path = f"{data_dir}/{dataset_type}/{folderpath}"
        
        
        img = read_img(img_path)
        
        if img is None:continue
        
        img_h, img_w, _ = img.shape
        
        
        regions = annotation_info['regions']

        for region in regions:
            if annot_type == 'rect':
                bbox, identity = get_bbox_from_rect_annot(region)
            
            else:#poly
                coords, identity = get_coords_from_annot(region)
                if len(coords) < 3 or identity not in relevant_classes:
                    continue
                
                bbox = get_bbox_from_polycoords(coords)
                
            ids.append(identity)
            class_idx = class_list.index(identity)
            
            x1, y1, x2, y2 = fix_out_of_bound_bbox(bbox, img_h, img_w)
            
            

            # Normalize coordinates
            normalized_x = (x1 + x2) / (2 * img_w)
            normalized_y = (y1 + y2) / (2 * img_h)
            normalized_width = (x2 - x1) / img_w
            normalized_height = (y2 - y1) / img_h
            

            # YOLO annotation format: class x_center y_center width height
            yolo_annotation = f"{class_idx} {normalized_x:.6f} {normalized_y:.6f} {normalized_width:.6f} {normalized_height:.6f}"
            
            if folderpath not in yolo_annotations:
                yolo_annotations[folderpath] = [yolo_annotation]
            else:
                yolo_annotations[folderpath].append(yolo_annotation)
                
        # Add imgname with empty annotation:
        if folderpath not in yolo_annotations:
            yolo_annotations[folderpath] = []


    return yolo_annotations, ids



dataset_type = "val"
data_dir = "datasets/Project_SSQS_Fixed_Camera"
dest_dir = "datasets/wheelrim-pad-cover_yolo_dataset"

annot_type = 'poly' #or poly

os.makedirs(f"{dest_dir}/{dataset_type}/images", exist_ok=True)
os.makedirs(f"{dest_dir}/{dataset_type}/labels", exist_ok=True)


relevant_classes = sorted(['fender_cover', 'lifting_pads', 'wheelrim']) #-> ['fender_cover', 'lifting_pads', 'wheelrim']

class_list = relevant_classes # This will be used to assign class_idx in the annotations. Change accordingly


vgg_annotation_file = read_json(f"{data_dir}/{dataset_type}/via_region_data.json")


# Create a dict conforming to yolo format: {foldername/imgname: [annot1, annot2]}
yolo_annots, ids = vgg_to_yolo(vgg_annotation_file, relevant_classes, class_list, dataset_type=dataset_type, data_dir=data_dir, annot_type=annot_type)

print("yolo dict created. Saving to directory ...")

# Create the yolo dataset using the yolo dict
convert_yolo_dict_to_dataset(yolo_annots, data_dir, dest_dir, dataset_type)

#Save the panel count for the entire yolo dataset
dump_json(dict(Counter(ids)), f"{dest_dir}/{dataset_type}/class_count.json", indent = None)

print(f"Yolo dataset stored at {dest_dir}/{dataset_type}")