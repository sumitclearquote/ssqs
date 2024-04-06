'''This script downloads frames from a single video. And stores it in dest_dir
'''


import os
import cv2
from imutils.video import FileVideoStream
from imutils.video import FPS, count_frames
import numpy as np
import imutils
import time
import cv2
from tqdm import tqdm

def download_frames(video_path, interval, dest_dir, total_frames, lower_limit = 0, upper_limit = 0):
    # Reading video using imutils
    fvs = FileVideoStream(video_path).start()
    time.sleep(1.0)
    
    # start the FPS timer
    fps = FPS().start()
    
    # Initialize progress bar
    pbar = tqdm(total=total_frames)

    frame_count = 0
    frame_saved = 0
    while fvs.more():
        frame = fvs.read()
        
        frame_count += 1
        
        
        if interval == 0 and lower_limit < frame_count < upper_limit: #download all frames
            frame_saved += 1
            cv2.imwrite(f"{dest_dir}/frame_{frame_count}.jpg", frame) #image names will be frame_1.jpg, frame10.jpg etc
            
            
        # Save n frames in a second equally spaced between total frames in that second. 
        elif interval != 0 and frame_count % interval == 0 and frame_count > lower_limit and frame_saved < upper_limit:
            # Save the frame
            frame_saved += 1
            cv2.imwrite(f"{dest_dir}/frame_{frame_count}.jpg", frame) #image names will be frame_1.jpg, frame10.jpg etc


        fps.update()
        
        # Update progress bar
        pbar.update(1)
        
    fvs.stop()
    pbar.close()
    return frame_saved
        
    




if __name__ == '__main__':
    
    camera_angle = "front"
    
    data_dir = "video_data"
    recorded_date = "april2"
    
    
    video_dir = "April 02 15.11-15.36"
    
    camera_dir = [i for i in os.listdir(f"{data_dir}/{recorded_date}/{video_dir}") if camera_angle in i][0]
    video_file_name = [i for i in os.listdir(f"{data_dir}/{recorded_date}/{video_dir}/{camera_dir}") if i.endswith("mp4")][0]
    video_path = f"{data_dir}/{recorded_date}/{video_dir}/{camera_dir}/{video_file_name}"
    
    #video_path = ""
    
    # download frames stream wise or 2 frames per second
    download_all = False
    lower_limit = 10  #from what frame to start saving
    upper_limit = 1000 # uptill what frame to save
   
   
   
   
    dest_dir = f"image_data/{recorded_date}/{video_dir}"
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get video details
    # Get the frames per second (fps) of the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_to_save_per_second = 2 # total frames to save in one second out of all the frames in a second
    total_frames_in_video = count_frames(video_path)
    
    if download_all: # Download in a sequence
        interval = 0 
    else:
        # Divide the FPS by the num of frames that need to be extracted per second. This will divide the total frames in a second in equal parts and download the frames at these intervals.
        interval = int(round(fps / total_frames_to_save_per_second))
    
    print("Downloading Frames ...")
    total_downloaded = download_frames(video_path, interval, dest_dir, total_frames_in_video, lower_limit = lower_limit, upper_limit=upper_limit)
    print("Frames Downloaded: ", total_downloaded)