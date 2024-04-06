import random
import cv2
import sys
sys.path.append("utils")
from data_utils import *
all_lamps = ["leftheadlamp", "rightheadlamp", "lefttaillamp", "righttaillamp", "leftfoglamp", "rightfoglamp"]


def get_min_dim(imgname):
    last_digit = imgname.split(".")[0][-1]
    last_digit = int(last_digit) if last_digit.isnumeric() else last_digit
    min_dim = 500 if last_digit in [0,1] else 520 if last_digit in [2,3] else \
              540 if last_digit in [4,5] else 560 if last_digit in [6,7] else \
              580 if last_digit in [8,9] else 512
              
    return min_dim
    


def expand_bbox(bbox, img_h, img_w, percent = 0.1,  alpha = 0):
    '''
    Expands a bbox area by adding a alpha to each coordinate
    '''
    x1,y1,x2,y2 = bbox
    
    boxh = y2-y1
    boxw = x2-x1
    
    height_alpha = int(boxh * percent)
    width_alpha = int(boxw * percent)
    
    
    x1 = max(0, x1 - width_alpha)
    y1 = max(0, y1 - height_alpha)
    x2 = min(x2 + width_alpha, img_w)
    y2 = min(y2 + height_alpha, img_h)
    
    return [int(x1),int(y1),int(x2),int(y2)]


def damage_in_lamp(lamp_bbox,  damage_bbox):
    '''
    Returns True if a particular damage is inside a lamp area.
    
    '''
    if damage_bbox[0] >= lamp_bbox[0] and damage_bbox[1] >= lamp_bbox[1] and damage_bbox[2] <= lamp_bbox[2] and damage_bbox[3] <= lamp_bbox[3]:
        return True
    else:
        return False
    
    
    
def create_new_region(lamp_bbox, damage_bbox, identity):
    '''
    create region key for the annot dictionary by adding x_points and y_points.
    '''
    X1, Y1, X2, Y2 = lamp_bbox
    x1,y1,x2,y2 = damage_bbox
    
    # Scale down damage coords: adjusted to the cropped lamp image
    x1 = x1 - X1
    y1 = y1 - Y1
    x2 = x2 - X1
    y2 = y2 - Y1
    
    damage_x_points = [x1, x2, x2, x1]
    damage_y_points = [y1, y1, y2, y2]
    new_region = {"shape_attributes": {"name": "polygon", 
                                      "all_points_x":damage_x_points,
                                        "all_points_y": damage_y_points,
                                      },
                 "region_attributes": {"identity":identity}
                }
    
    return new_region



def create_new_annotation(lamps, damages, imgname, img_h, img_w, expansion_percent):
    '''
    Creates a Dictionary with annotation for all lamps (cropped images on disk) in a full-size standard image.
    '''
    new_annot = {}
    for lamp in lamps: #each lamp is a cropped image on disk
        new_annotation = {}
        new_regions = []
        for damage in damages:
            lamp_bbox = get_bbox_from_polycoords(lamp[1])
            
            #if lamp[0] in all_lamps[2:]: #if lamp is any of taillamps, foglamps.
            #    exp_percent = expansion_percent[0]
            #else:
            #    exp_percent = expansion_percent[1]
            exp_percent = expansion_percent
            
            #lamp_bbox = expand_bbox(lamp_bbox, img_h, img_w, percent = exp_percent)
            lamp_bbox = expand_to_fixed_dimension(*lamp_bbox, img_w, img_h, imgname)
            
            damage_bbox = get_bbox_from_polycoords(damage[1])

            
            if damage_in_lamp(lamp_bbox, damage_bbox):
                # create annotation
                new_regions.append(create_new_region(lamp_bbox, damage_bbox, damage[0]))
                
        new_img_name = f"{imgname.split('.')[0]}_{lamp[0]}.{imgname.split('.')[-1]}"
        #Below is an annotation for one cropped image.
        new_annotation.update({new_img_name: {"filename": new_img_name,
                                              "size": 0,
                                            "regions": new_regions,
                                            "file_attributes": {}
                                            }
                               })
        new_annot.update(new_annotation)
    
    
    return new_annot   



def expand_to_fixed_dimension(x1,y1,x2,y2,width,height, imgname)-> tuple:
    ''' Expands the bbox of the lamp to a certain size-min_dim, to cover more area.
    
    Args: 
    - (x1,y1,x2,y2): Coordinates of the lamp
    - width, height:  the width and height of the whole image.
    
    Returns: 
    - (x1,y1,x2,y2): Expanded bbox coordinages
    '''
    #min_dim = 512
    #Choose a min_dim based on the range the last digit in the image name is
    min_dim = get_min_dim(imgname)
    
    boxh = y2 - y1
    boxw = x2-x1
  
    height_delta = min_dim - boxh
    width_delta  = min_dim - boxw
    if height_delta  > 0:
            # bring height to min height
        margin = int(height_delta/2)
        #print("Height margin: ",margin)
        y1 -= margin
        y2 += (height_delta-margin)
    if width_delta > 0:
        margin = int(width_delta/2)
        #print("Width margin:", margin)
        x1 -= margin
        x2 += (width_delta - margin)
    if x1 < 0: # shift the box towards other side.
        x2 -= x1
        x1 = 0
    if y1 < 0 :
        y2 -= y1
        y1 = 0
    if x2  > width:
        x1 =  max(x1 -(x2-width),0)
        x2 = width - 1
    if y2 > height:
        y1 =max(y1 -(y2-height),0)
        y2 = height -1

    if x1 < 0: x1 = 0
    if y1 < 0 :y1 = 0

    return (int(x1),int(y1),int(x2),int(y2))


def get_resized_img(img, target_size):
    ''' Resizes the image by resizing the larger side, which is greater than target_size by 100 pixels, to a fixed target size and resizes the other side w.r.t to the aspect ratio of the original image
    Args:
    h: original image height
    w: original image width
    
    target_size: The size to which the larger side of the image is to be resized.
    '''
    h, w, _ = img.shape
    
    # Resize only if the larger side is greater than target_size + 100
    if max(w, h) <= target_size + 100:
        return img, None
    
    
    aspect_ratio = w/h
    

    if w >= h:
        new_w = target_size
        new_h = int(target_size / aspect_ratio)
    elif h > w:
        new_w = int(target_size * aspect_ratio)
        new_h = target_size
        
    resized_img = cv2.resize(img, (new_w, new_h))
    
    return resized_img, aspect_ratio
    
    
def get_resized_coords(xpoints, ypoints, h, w, new_h, new_w):
    ''' Calculate the new bbox coordinates for the resized image w.r.t to the width and height scale.
    
    '''
    width_scale = new_w / w
    height_scale = new_h / h
    
    xpoints =  [int(x * width_scale) for x in xpoints]
    ypoints  = [int(y * height_scale) for y in ypoints]
    
    return xpoints, ypoints


def yolo_to_corners(x_center, y_center, width, height, image_width, image_height):
    # Reverse the normalization
    xmin = int((x_center - width / 2) * image_width)
    ymin = int((y_center - height / 2) * image_height)
    xmax = int((x_center + width / 2) * image_width)
    ymax = int((y_center + height / 2) * image_height)

    return xmin, ymin, xmax, ymax