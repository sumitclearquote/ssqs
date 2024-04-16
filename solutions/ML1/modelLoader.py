'''This Script contains functions to load panel detection models'''
import sys
import torch
import torchvision

#Detectron2________________________________________________________________________________________________________________
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects import point_rend

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# FOr allbodystyle panels model
PANEL_THRESH = 0.15

allBodystylePanels = ['alloywheel', 'bonnet', 'doorhandle', 'footstep', 'frontbumper', 'frontbumpercladding', 'frontbumperfiller', 
                'frontbumpergrille', 'frontws', 'fuelcap', 'headlightwasher', 'indicator', 'leftapillar', 'leftbootlamp', 'leftbpillar', 'leftbumperfiller', 
                'leftcabcorner', 'leftcpillar', 'leftdpillar', 'leftfender', 'leftfoglamp', 'leftfrontdoor', 'leftfrontdoorcladding', 'leftfrontdoorglass', 
                'leftfrontventglass', 'leftheadlamp', 'leftorvm', 'leftqpanel', 'leftquarterglass', 'leftquarterglass_pickup', 'leftreardoor', 'leftreardoorcladding', 
                'leftreardoorglass', 'leftrearfender', 'leftrearventglass', 'leftroofside', 'leftrunningboard', 'lefttaillamp', 'leftwa', 'licenseplate', 'logo', 
                'lowerbumpergrille', 'namebadge', 'rearbumper', 'rearbumper_pickup', 'rearbumpercladding', 'rearws', 'Reflector', 'rightapillar', 'rightbootlamp', 
                'rightbpillar', 'rightbumperfiller', 'rightcabcorner', 'rightcpillar', 'rightdpillar', 'rightfender', 'rightfoglamp', 'rightfrontdoor', 
                'rightfrontdoorcladding', 'rightfrontdoorglass', 'rightfrontventglass', 'rightheadlamp', 'rightorvm', 'rightqpanel', 'rightquarterglass', 
                'rightquarterglass_pickup', 'rightreardoor', 'rightreardoorcladding', 'rightreardoorglass', 'rightrearfender', 'rightrearventglass', 'rightroofside', 
                'rightrunningboard', 'righttaillamp', 'rightwa', 'Roof', 'roofrail', 'sensor', 'sunroof', 'tailgate', 'towbarcover', 'tyre', 'wheelcap', 'wheelrim', 'wiper',
                "cabinroof","doorglass_van","frontwall","hinge","leftbrakelamp",
                "leftcabcorner_van","leftcabcornercladding","leftcpillar_van","leftfrontwacladding","leftqpanel_van","leftqpanelcladding",
                "leftrearcover","leftsidefiller","leftsidefillercladding","leftsidewall","leftslidingdoor","leftslidingdoorcladding",
                "lefttailgate","lefttailgatecladding","leftvalence","leftvalencecladding","lowerrearfiller","rearbumper_van","rearroofside",
                "reartailbar","rightbrakelamp","rightcabcorner_van","rightcabcornercladding","rightcpillar_van","rightfrontwacladding",
                "rightqpanel_van","rightqpanelcladding","rightrearcover","rightsidefiller","rightsidefillercladding","rightsidewall","rightslidingdoor",
                "rightslidingdoorcladding","righttailgate","righttailgatecladding","rightvalence","rightvalencecladding"]

def load_point_rend_weights(detType, registered_name, classes, config_file_name, weights_path, bspi, threshold, device, extra):
    scratch_metadata = MetadataCatalog.get(registered_name).set(thing_classes=classes)
    
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)

    cfg.merge_from_file(config_file_name)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = bspi  # 1024 for panels 512 for damages
    
    # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = len(classes)

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4, 8, 16, 32, 64, 128, 256, 512, 1024]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 1.33, 1.5, 2.0]]
    
    cfg.MODEL.DEVICE = device

    if extra:
        cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION = 21
        cfg.MODEL.POINT_HEAD.FC_DIM = 512
    
    predictor = DefaultPredictor(cfg)
    # print('Using:', cfg.MODEL.DEVICE, flush=True)
    # print('Loading ', cfg.MODEL.WEIGHTS, ' for', detType, flush=True)
    return predictor


def loadAllBodystylePanelModel():
    panel_predictor = load_point_rend_weights('panels',
                                          'merged_panels_train',
                                          allBodystylePanels,
                                          "detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_101_FPN_3x_coco.yaml",
                                          "models/all_bodystyle_panels_02_02_2024_v_1.pth",
                                          1024,
                                          float(PANEL_THRESH),
                                          device,
                                          False)
    return panel_predictor