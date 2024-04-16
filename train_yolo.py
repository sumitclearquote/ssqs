from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version
import sys
from ultralytics import YOLO

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def __init__(self, p=1.0):
    """Initialize the transform object for YOLO bbox formatted params."""
    self.p = p
    self.transform = None
    prefix = colorstr("\n\nalbumentations: ")
    try:
        import albumentations as A

        check_version(A.__version__, "1.0.3", hard=True)  # version requirement

        # Transforms
        T = [
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.05),
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_lower=75, p=0.02),
        ]
        '''
        Add custom augmentation here
        T += [A.HorizontalFlip(p=0.2),
              A.VerticalFlip(p=0.15),
              ]
        '''
        self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
    except ImportError:  # package not installed, skip
        pass
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")


# Replace the constructor of the repo with the above one. Use this constructor to add whatever custom augmentations is needed. 
Albumentations.__init__ = __init__


#==============================================================================================================

#'''
#YOLOv8s --->============================================================================================================
print("Training yolov8n ...\n")

# A directory inside "yolo_runs" will be created with the below name
project_name = "wheelrim_cover_pads"

#name of the dataset folder
dataset_name = "wheelrim-pad-cover_yolo_dataset"
yolo_cfg = "wheelrim_data.yaml" #name of the yolo cfg yaml file inside dataset
train_version = "v1_n"

# Add other HPs here
model_file = "yolov8n.yaml"



#Load a Model
model = YOLO(model_file)

project_path = f'/home/paintfrmladmin01/datadrive/ssqs/yolo_runs/{project_name}'



config  ={  'data': f"/home/paintfrmladmin01/datadrive/ssqs/datasets/{dataset_name}/{yolo_cfg}", 
            'epochs': 30,
            'batch': 256,
            'imgsz':224,
            'device':device,
            'patience':5,
            'project':project_path,
            'name':train_version,
            'close_mosaic': 5,
            'mosaic':0.2,
            'fliplr':0.4
        }

# Train the Model -> yolov8s
results = model.train(**config)
#'''


'''
# YOLOv8m START ================================================================================================================
print("Training yolov8m ...\n")

# Add other HPs here
model_file = "yolov8m.yaml"

train_version = "v4_m"

#Load a Model
model = YOLO(model_file)

project_path = '/home/paintfrmladmin01/datadrive/LPBlur/ultralytics/my_runs/lpblur/'
#project_path = '/home/paintfrmladmin01/datadrive/LPBlur/runs'


config  ={  'data': "/home/paintfrmladmin01/datadrive/LPBlur/datasets/LP_yolo_dataset/lp_data.yaml", 
            'epochs': 80,
            'batch': 74,
            'imgsz':480,
            'device':device,
            'patience':10,
            'project':project_path,
            'name':train_version,
            'close_mosaic': 5,
            'mosaic':0.55
        }

# Train the Model -> yolov8m
results = model.train(**config)
'''