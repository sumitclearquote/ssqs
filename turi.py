import turicreate as tc
import os
import random
from PIL import Image
from solutions.utils.data_utils import read_img, read_json, get_coords_from_annot, get_bbox_from_polycoords, draw_box, show
#from solutions.utils.annotation_utils import shrink_bbox




# Inference with wheelrim trained model
class_name = "fendercover" #-> ["wheelrim", "liftingpads", "fendercover"]
data_dir = f"solutions/sbrm_test_data/{class_name}_testing"
dest_dir = f"results/{class_name}"
os.makedirs(dest_dir, exist_ok=True)

if class_name == "wheelrim":
    model = tc.load_model(f"models/{class_name}.model")
else:
    model = tc.load_model(f"models/{class_name}.model")


test_image_set = os.listdir(data_dir)#[:8]
#random.shuffle(test_image_set)

print("Total images in test set: ", len(os.listdir(test_image_set)))

# Load test images
test_images = tc.SFrame({'image':[tc.Image(f'{data_dir}/{imgname}') for imgname in test_image_set if not imgname.endswith(("Store", "json")) and "cropped" not in imgname], 
                         'imgname': [imgname for imgname in test_image_set if not imgname.endswith(("Store", "json")) and "cropped" not in imgname]})



data_dir = f"{class_name}_testing"
#test_images = [tc.Image(f"{data_dir}/{imgname}") for imgname in os.listdir(data_dir) if not imgname.endswith(("Store", "json")) and "cropped" not in imgname]


test_images['predictions'] = model.predict(test_images)

print("Total predictions: ", len(test_images['predictions']))

# Draw prediction bounding boxes on the test images
test_images['annotated_predictions'] = tc.one_shot_object_detector.util.draw_bounding_boxes(test_images['image'],test_images['predictions']) 

#display image
for i in range(len(test_images['predictions'])):
    Image.fromarray(test_images['annotated_predictions'][i].pixel_data).save(f"{dest_dir}/{class_name}_frame{i}.png")