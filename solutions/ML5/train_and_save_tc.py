import turicreate as tc
import os
tc.config.set_num_gpus(1)

print("GPUs being used based on config: ", tc.config.get_num_gpus())


def train_one_shot_detector(data_dir, test_images, class_name, starter_images_list, iteration = 1):
    # Load the starter images
    starter_images = tc.SFrame({'image':starter_images_list, 'label':[class_name]})


    #Load the background images
    images = [tc.Image(f"{data_dir}/{imgname}") for imgname in os.listdir(data_dir) if not imgname.endswith(("Store", "json"))]

    bg_sarray = tc.SArray(images)


    # Load test images
    test_images = tc.SFrame({'image':test_images})


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
    if iteration != 1:
        model.save(f'models/{class_name}_{iteration}.model')
        print(f"Model Trained and saved to models/{class_name}_{iteration}.model")
    else:
        model.save(f'models/{class_name}.model')
        print(f"Model Trained and saved to models/{class_name}.model")

if __name__ == '__main__':
    
    class_name = "liftingpads"
    
    
    data_dir = f"{class_name}_testing"
    test_images = [tc.Image(f"{data_dir}/{imgname}") for imgname in os.listdir(data_dir) if not imgname.endswith(("Store", "json")) or "cropped" not in imgname]
    
    total_starter_images = 2
    starter_images_list = [tc.Image(f"{data_dir}/cropped_{class_name}{i}.png") for i in range(total_starter_images)]
    
    iteration = 1
    
    train_one_shot_detector(data_dir, test_images, class_name, starter_images_list, iteration=1)