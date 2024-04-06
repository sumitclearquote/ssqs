import torch
from ultralytics import YOLO
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_lp_detection_model():
    checkpoint_path = "models/model_lp_yolov8m_detector_v_1.pt"
    lp_detector = YOLO(checkpoint_path).to(device)
    return lp_detector


def detect_licenseplate(model, test_image,conf_threshold = 0.60):
    ''' Detects License plate and replaces/appends it to the primary panel detections
    
    test_image -> np array image
    '''
    try:
        results = model(test_image, imgsz=480, iou=0.7,conf=conf_threshold, device=device, verbose = False)
    except Exception as e:
        print("Error in LP Detection")
        print("Error Message: ", e)
        
        
    result = results[0]
    
    #If no detections found. Return primary detection unchanged
    if len(result.boxes.cpu().numpy()) == 0:
        return None
    

    boxes = result.boxes.cpu().numpy()
    
    bboxes = boxes.xyxy # np.array([[x1,y2,x2,y2], [x1,y1,x2,y2]])
    scores = boxes.conf # np.array([conf1, conf2])
    
    largest_area = 0
    for i, (bbox, conf) in enumerate(zip(bboxes, scores)):
        x1,y1,x2,y2 = bbox
        current_width, current_height = x2-x1, y2-y1
        current_area = current_height * current_width

        # Store the largest bbox in 
        if current_area > largest_area:
            largest_bbox_idx = i
            largest_area = current_area
            
        
    bbox = bboxes[largest_bbox_idx]
    bbox = [int(i) for i in bbox]
    score = scores[largest_bbox_idx]
    
    return bbox  # (x1,y1,x2,y2)
        
        