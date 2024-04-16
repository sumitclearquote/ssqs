import sys
import numpy as np
import cv2
import torch
import torchvision
import sys

sys.path.append("..")

from utils.data_utils import *
from modelLoader import allBodystylePanels

def detectObj(predictor, test_image, imgpath):
    ''' Object Detection model to detect LP plate.
    '''
    #test_image = cv2.imread(imgpath)
    #print('Image here',test_image,flush=True)
    try:
        outputs = predictor(test_image)

    except Exception as e:
        msg = 'Error in object detection'
        exc_type, exc_obj, exc_tb = sys.exc_info()
        json_output = {'imgpath': imgpath, 'error': str(e), 'lineNo':  str(exc_tb.tb_lineno)}
        print(json_output)

    out = outputs["instances"].to("cpu")
    all_fields = out.get_fields()
    p = {}
    p['class_ids'] = all_fields['pred_classes'].numpy()
    p['rois'] = all_fields['pred_boxes'].tensor.numpy()
    p['scores'] = all_fields['scores'].numpy()
    p['class_names'] = []
    
    for i in p['class_ids']:
        p['class_names'].append(allBodystylePanels[i])

    
    return p