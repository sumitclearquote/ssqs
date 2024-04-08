import json
import numpy as np
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
from numpyencoder import NumpyEncoder


def get_number(item):
    return int(item.split('_')[1].split('.')[0])

def read_json(json_path):
    json_file = json.load(open(json_path, "r"))
    return json_file

def dump_json(json_file, dest_path, indent = 4):
    try:
        with open(dest_path, "w") as f:
            json.dump(json_file, f, indent = indent, cls = NumpyEncoder)
    except Exception as e:
        print("Error occurred in dumping json: ", e)

        
def show(img, title):
    plt.figure(figsize = (12,8))
    plt.imshow(img)
    plt.title(title)
    plt.show()
        
        
def get_coords_from_annot(region, identity_name = 'identity'):
    '''
    Returns the identity (lamp name or damage name) and the polygon coordinates of a particular panel/damage
    
    Coordinates Return Format: Numpy Array -> [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    '''
    coords = list(zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y']))
    coords = np.array([[i,j] for (i,j) in coords])
    identity = region['region_attributes'][identity_name]
    
    return coords, identity

def get_bbox_from_rect_annot(region):
    '''
    Returns the identity (class name) and the bbox coords of a particular class
    
    bbox return format: List -> [x1,y1,x2,y2]
    '''
    x1 = region['shape_attributes']['x']
    y1 = region['shape_attributes']['y']
    x2 = region['shape_attributes']['width'] + x1
    y2 = region['shape_attributes']['height'] + y1
    bbox = [int(x1),int(y1),int(x2),int(y2)]
    
    identity = region['region_attributes']['identity']
    
    return bbox, identity
    

def get_bbox_from_polycoords(coords):
    '''
    Retrieves coordinates from polygon coords that makes up a bbox.
    
    bbox return format: List -> [x1,y1,x2,y2]
    '''
    coords = np.array(coords)
    x1 = min(coords[:, 0])
    x2 = max(coords[:, 0])
    y1 = min(coords[:, 1])
    y2 = max(coords[:, 1])
    
    bbox = [int(x1),int(y1),int(x2),int(y2)]
    return bbox
    

def read_img(img_path, img_type = "array"):
    '''
    Return a Numpy image or a PIL image based on the img_type in the argument.
    '''
    try:
        if img_type == "array":
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = Image.open(img_path).convert("RGB")
            
        return img
    except Exception as e:
        print("Exception while reading Image: ", e)
        print("Img Path: ", img_path)
        return None
    


def get_total_image_count_in_subfolders(dataset_path):
    '''
    Gets the total number of files in all the subfolders in a folder(dataset_path argument)
    '''
    total_count = 0
    for folder in os.listdir(dataset_path):
        if folder.endswith("Store") or folder.endswith("json") :continue
        total_count += len(os.listdir(f"{dataset_path}/{folder}"))
        
    return total_count


def draw_polygon(img, coords):
    '''
    Draws a polygon using the coords around an image and send the image
    '''
    
    cv2.polylines(img, [coords], True, (255,0,0), 2)
        
    return img

def draw_box(img, coords, color = None, thickness = 3):
    '''
    Draws a bbox using the bbox coords around an image and send the image
    '''
  
    color = (255,0,0) if color is None else color
    
    x1,y1,x2,y2 = coords
    
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
   
    return img
  
def write_text(img, text, pt):
    '''
    Writes a text on a specific part of the bbox in an image and sends it.
    '''

    pt = (pt[0]+10, pt[1]-3)
    cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    return img


def get_size(path):
    '''Returns the total size occupied by a file or folder on disk.
    Args:
    path: path to the file/folder
    '''
    total_size = 0
    
    if os.path.isfile(path):
        #return f"{os.path.getsize(path) / 1000 / 1000:.2f} MB"
        return f"{os.path.getsize(path) / 1000 / 1000:.2f}"
    else:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)

    return f"{total_size  / 1000 / 1000 / 1000:.2f} GB"
    #return f"{total_size  / 1000 / 1000 / 1000:.2f}"


def get_correct_imgname(imgname):
    '''Return the corrected imgname in case where there is something after "jpg" 
    
    '''
    if 'jpg' in imgname:
        imgname = imgname.split(".")[0] + '.jpg'
    elif 'png' in imgname:
        imgname = imgname.split(".")[0] + '.png'
    elif 'jpeg' in imgname:
        imgname = imgname.split(".")[0] + '.jpeg'
    elif 'JPG' in imgname:
        imgname = imgname.split(".")[0] + '.JPG'
    elif 'PNG' in imgname:
        imgname = imgname.split(".")[0] + '.PNG'
    elif 'JPEG' in imgname:
        imgname = imgname.split(".")[0] + '.JPEG'
        
    return imgname


def plot_subplots(img1, img2, xlabel="X", ylabel="Y", title1='image1', title2='image2',  figsize = (12,8)):
    fig, axs = plt.subplots(1, 2, figsize = figsize)

    axs[0].imshow(img1)
    axs[1].imshow(img2)
    axs[0].set_title(title1)
    axs[1].set_title(title2)
    axs[0].xlabel(xlabel)
    axs[1].xlabel(xlabel)
    axs[0].ylabel(ylabel)
    axs[1].ylabel(ylabel)
    plt.show()
    
    
    
    
def read_textfile(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        
    return lines