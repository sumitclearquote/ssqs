'''This script has functions to download frames from fixed cameras.'''

import os
import cv2
from imutils.video import FileVideoStream
from imutils.video import FPS, count_frames
import numpy as np
import imutils
import time
import cv2
from tqdm import tqdm


def get_video_details(video_path, print_fps_and_total_frames = False):
    ''' Returns the FPS and the Total frames in a video
    '''
    
    # Get the frames per second (fps) of the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_in_video = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    if print_fps_and_total_frames:
        print("Getting video details")
        print("FPS of the video: ", fps)
        print("Total Frames in the video: ", total_frames_in_video)
    return fps, total_frames_in_video




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


def download_headcam_frames(data_dir, camera_type, recorded_date, already_extracted_videos, to_extract_videos, dest_dir):
    for rec_date in os.listdir(f"{data_dir}/{camera_type}"): #example: "april2"
        if rec_date.endswith("Store"): continue
        if rec_date not in recorded_date: continue
        videos_path = f"{data_dir}/{camera_type}/{rec_date}"
        
        for video_file in os.listdir(f"{videos_path}"): #A single mp4 video file.
            if video_file.endswith("Store"):continue
            if not video_file.lower().endswith("mp4"):continue
            
            if video_file in already_extracted_videos:continue
            
            if to_extract_videos != [] and video_file not in to_extract_videos:continue
            
            # Create a custom name to add to image names
            video_name = "_".join(video_file.split(".")[0].split(" "))
            
            #path to the video
            video_path = f"{videos_path}/{video_file}"
            
            
            
            if compute_fps_and_total_frames_to_display:
                fps, total_frames = get_video_details(video_path, print_fps_and_total_frames=print_fps_and_total_frames)
                
            interval = int(round(fps / total_frames_to_save_per_second))
            
            total_downloaded = "No Frames Downloaded"
            if download_frames:
                os.makedirs(dest_dir, exist_ok=True)  
                print(f"Downloading Frames for {video_file} ...")
                total_downloaded = download_all_frames(video_path, interval, dest_dir, video_name)
                print(f"Total Frames downloaded for {video_path}: ", total_downloaded)

        


def download_fixedcam_frames(data_dir, camera_type, camera_angles, recorded_date, already_extracted_videos, to_extract_videos, dest_dir):
    
    for rec_date in os.listdir(f"{data_dir}/{camera_type}"):
        if rec_date.endswith("Store"): continue
        if rec_date not in recorded_date: continue
        videos_path = f"{data_dir}/{camera_type}/{rec_date}"
        for video_dir in os.listdir(videos_path):
            if video_dir.endswith("Store"):continue
            if video_dir in already_extracted_videos:continue
            if to_extract_videos != [] and video_dir not in to_extract_videos:continue
            for video_file in os.listdir(f"{videos_path}/{video_dir}"): #A single mp4 video file.
                if video_file.endswith("Store"):continue
                if not video_file.lower().endswith("mp4"):continue
                if not any(cam_angle in video_file.lower() for cam_angle in camera_angles):continue
                
                # Create a custom name to add to image names
                video_name = "_".join(video_dir.split(".")[0].split(" "))
                
                #path to the video
                video_path = f"{videos_path}/{video_dir}/{video_file}"
                
                
                if compute_fps_and_total_frames_to_display:
                    fps, total_frames = get_video_details(video_path, print_fps_and_total_frames=print_fps_and_total_frames)
                    
                interval = int(round(fps / total_frames_to_save_per_second))
                    
                print(video_path)
                print("--------")
                
                total_downloaded = "No Frames Downloaded"
                if download_frames:
                    os.makedirs(dest_dir, exist_ok=True)  
                    print(f"Downloading Frames for {video_path} ...")
                    total_downloaded = download_all_frames(video_path, interval, dest_dir, video_name)
                    print(f"Total Frames downloaded for {video_path}: ", total_downloaded)
                    
                
                    
                    
    
if __name__ == '__main__':
    data_dir = "video_data"
    camera_type = "fixedcam" #[headcam, fixedcam] #Always add only one of these unless all frames need to be combined.
    recorded_date = ["april4"] # only extract frames from these dates. Add all dates to extract all frames from all videos.
    camera_angles = ["camera_03"]  #["camera_01", "camera_02", "camera_03"] # only for fixed cams. Only download these camera angles. #add all camera angles to extract all frames from all videos.
    
    # Add names of specific videos to not/ to extract.
    # For fixed cam, add the video_dir in the below lists: ex: "APRIL 04 15.05-15.15"
    # For headcam, add the actual video_file name: ex: go pro 16 34.MP4
    # keep the below lists empty to extract all videos.
    already_extracted_videos = [] # add name of video files that have already been extracted. These videos will be skipped while looping and downloading frames.
    
    to_extract_videos = ["APRIL 04 15.05-15.15"] # Add only those names of video files that needs to be extracted
    
    # Control Flags:
    total_frames_to_save_per_second = 2          # total frames to save in one second out of all the frames in a second
    compute_fps_and_total_frames_to_display = True
    download_frames = True # Keep this false to debug or test the script
    print_fps_and_total_frames = False #whether to print the fps and total frames calculated
    

    dest_dir =  f"image_data/april4/APRIL 04 15.05-15.15"   
             
    
    
    if camera_type == "fixedcam":
        download_fixedcam_frames(data_dir, camera_type, camera_angles, recorded_date, already_extracted_videos, to_extract_videos, dest_dir)
        
    elif camera_type == "headcam":
        download_headcam_frames(data_dir, camera_type, recorded_date, already_extracted_videos, to_extract_videos, dest_dir)
    
    
    