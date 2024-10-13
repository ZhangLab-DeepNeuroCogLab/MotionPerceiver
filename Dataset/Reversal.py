import pandas as pd
import os
import decord
from decord import cpu
import numpy as np
import cv2
import os

input_csv_path = ""
input_folder = ""
output_folder = ""
        
data = pd.read_csv(input_csv_path,header=None,sep=",") 
fourcc = cv2.VideoWriter_fourcc(*'H264')

for index, row in data.iterrows():
    
    ################## read videos ##################
    old_video_path = os.path.join(input_folder,row[0])
    video = decord.VideoReader(old_video_path,ctx=cpu(0),num_threads=0,height = 1080,width = 1920)
    vlen = len(video)
    
    ################## frames processing ##################
    frame_idx = np.arange(0, vlen,dtype=int)
    frame_idx = np.flip(frame_idx)
    frames = video.get_batch(frame_idx).asnumpy()
    
    fps = video.get_avg_fps()
    new_fps = fps*len(frame_idx)/vlen
    
    ################## write videos ##################
    new_video_path = os.path.join(output_folder,row[0])
    folder_name = os.path.dirname(new_video_path)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    video_format = cv2.VideoWriter(new_video_path, fourcc, new_fps, (1920,1080))
    for t in range(frames.shape[0]):
        video_format.write(frames[t][...,::-1])
    video_format.release()
    cv2.destroyAllWindows()
    
print("Done!")
