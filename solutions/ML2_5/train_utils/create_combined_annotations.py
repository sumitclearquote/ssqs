#This script Combines all annotation files and images from all indiviudal folders to the parent folder

import os
from tqdm import tqdm
import sys
sys.path.append("..")

import shutil
from utils.data_utils import dump_json, read_json

# Make this True if images also need to be copied to parent folder
combine_images = False

save_combined_json = True

replace_imgname_with_folderimgname = True

# Provide the parent directory name in CLI : datasets/Project_SSQS_Fixed_Camera/train
data_dir = sys.argv[1]

# Destination directory where combined json is stored: datasets/Project_SSQS_Fixed_Camera/train
dest_dir = sys.argv[2]

# Name of the combined json file: via_region_data
dest_name = sys.argv[3]

folders = os.listdir(data_dir)
combined_annot = {}
a = 0
for folder in tqdm(folders):
    if folder.endswith("Store") or folder.endswith("json"): continue
    folder_path = f"{data_dir}/{folder}"
    annot_file = read_json(f"{folder_path}/via_region_data.json")
    for imgname, annot in annot_file.items():
        #if imgname in ['LkaaaZAsrm_1709745087412.jpg', 'tOSWjlIfAv_1709745057253.jpg', '6Zf8XzEwBv_1709745152154.jpg', 'y7ZDIK5gkP_1709745183693.jpg', 'ceZTmTWLsi_1709745120978.jpg', 'rLxbEYZ8gA_1709745073936.jpg', 'NnkYXJLh8Z_1709745195251.jpg', 'N9QLA6CyjA_1709745249508.jpg', '9Zt2O2bLuC_1709745077871.jpg']:continue #annotations not present
        if imgname not in os.listdir(f"{data_dir}/{folder}"):
            print(f"{imgname} not in {folder}")
            continue
        
        if combine_images:
            source_path = f"{data_dir}/{folder}/{imgname}"
            dest_path = f"{dest_dir}/{imgname}"
            shutil.copy(source_path, dest_path)
        
        if replace_imgname_with_folderimgname:
            # replace the imgname with folder/imgname
            filename = annot['filename'].replace("https://cq-workflow.s3.amazonaws.com/", "")
            annot['filename'] = f"{folder}/{filename}"
        
        if imgname in combined_annot:
            a += 1

        
        combined_annot.update({imgname: annot})
        
        
print("Total Annotations: ", len(combined_annot))
print("Total Images which exists in more than one folders: ", a)

if save_combined_json:
    dump_json(combined_annot, f"{dest_dir}/{dest_name}.json", indent = None)
    
print(f"Combined Annotation file saved at {dest_dir}/{dest_name}")