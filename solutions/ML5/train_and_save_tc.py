import turicreate as tc
import os
tc.config.set_num_gpus(1)

print("GPUs being used based on config: ", tc.config.get_num_gpus())

# Load the starter images
starter_images = tc.SFrame({'image':[tc.Image('wheelrim_testing/cropped_wheelrim.png')], 'label':['wheelrim']})


#Load the background images
images = [tc.Image(f"wheelrim_testing/{imgname}") for imgname in os.listdir("wheelrim_testing") if not imgname.endswith(("Store", "json"))]

bg_sarray = tc.SArray(images)


# Load test images
test_images = tc.SFrame({'image':[tc.Image('wheelrim_testing/frame_11910.jpg'), 
                                  tc.Image('wheelrim_testing/frame_16180.jpg'),
                                  tc.Image('wheelrim_testing/frame_9520.jpg')]})


# Create a model. This step will take a few hours on CPU and about an hour on GPU
model = tc.one_shot_object_detector.create(starter_images, 'label', backgrounds = bg_sarray)


# Save predictions on the test set
test_images['predictions'] = model.predict(test_images)

# Draw prediction bounding boxes on the test images
test_images['annotated_predictions'] = \
    tc.one_shot_object_detector.util.draw_bounding_boxes(test_images['image'],
        test_images['predictions']) 
    
    
print("Total Predictions for the test images: \n", len(test_images['predictions']))

print("\n\n")
print("Annotated Predictions for the test images: \n", test_images['annotated_predictions'])
print("\n\n")

print("Saving model to disk ..")
# Save the model for later use in TuriCreate
model.save('models/wheelrim_with_bg.model')
print("model saved to models/wheelrim.model")