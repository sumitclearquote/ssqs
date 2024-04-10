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


def get_video_details(video_path):
    ''' Returns the FPS and the Total frames in a video
    '''
    
    # Get the frames per second (fps) of the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #total_frames_in_video = count_frames(video_path)
    print("FPS of the video: ", fps)
    #print("Total Frames in the video: ", total_frames_in_video)
    return fps#, total_frames_in_video




def download_all_frames(video_path, interval, dest_dir, video_file_name):
    # Reading video using imutils
    fvs = FileVideoStream(video_path).start()
    time.sleep(1.0)
    
    # start the FPS timer
    fps = FPS().start()
    

    frame_count = 0
    frame_saved = 0
    while fvs.more():
        frame = fvs.read()
        
        frame_count += 1
        
        
        if frame_count % interval == 0: 
            frame_saved += 1
            cv2.imwrite(f"{dest_dir}/{video_file_name}_frame_{frame_saved}.jpg", frame) #image names will be frame_1.jpg, frame10.jpg etc
            
        if frame_saved ==500:
            print("Frame 500")
        

        fps.update()
        

        
    fvs.stop()
    return frame_saved
        
    




if __name__ == '__main__':
    
    
    
    data_dir = "video_data"
    camera_angle = "fixedcam" #[headcam, fixedcam]
    recorded_date = ["april2", "april4"]
    
    already_extracted_videos = [] # add name of video files that have already been extracted. These videos will be skipped while looping and downloading frames.
    
    to_extract_videos = ["Camera_03_Maruti_RTC_Maruti_RTC_20240404150459_20240404151459_4716330.mp4"] # Add only those names of video files that needs to be extracted
    
    # COntrol Flags:
    total_frames_to_save_per_second = 2          # total frames to save in one second out of all the frames in a second
    compute_fps_and_total_frames_to_display = True
    download_frames = False # Keep this false to debug or test the script
    
    for rec_date in os.listdir(f"{data_dir}/{camera_angle}"):
        if rec_date.endswith("Store"): continue
        videos_path = f"{data_dir}/{camera_angle}/{rec_date}"
        for video_file in os.listdir(videos_path):                    
            if video_file.endswith("Store") or (video_file in already_extracted_videos and video_file not in to_extract_videos): continue
            
            
            
            # Create a custom name to add to image names
            video_name = "_".join(video_file.split(".")[0].split(" "))
            
            #path to the video
            video_path = f"{videos_path}/{video_file}"
            
            
            if compute_fps_and_total_frames_to_display:
                print("Getting video details")
                start = time.time()
                fps = get_video_details(video_path)
                end = time.time()
                total_time = end-start
               
            
            interval = int(round(fps / total_frames_to_save_per_second))
            
            
            
            
            print(video_name)
            print(video_path)
            print(interval)
            print("Total time taken to get video details: ", total_time)
            print("fps: ", fps)
            print("Total frames: ", total_frames)
            
            
            if download_frames:
                print(f"Downloading Frames for {video_file} ...")
                total_downloaded = download_all_frames(video_path, interval, dest_dir, video_name)
                print(f"Total Frames downloaded for {video_file}")
                print("=========================================================================")