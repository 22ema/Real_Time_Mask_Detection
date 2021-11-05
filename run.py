# 2021.11.05 yunsung_han
# webcam and video's inferencing result display
import os
from Video_inference.video_inference import *

if __name__ =="__main__":
    '''run video inference
    weight = inference model file link
    cfg = model configuration file link
    video = video_link(webcam : 0)
    '''
    weights = './model/yolor/new_class_1_2/best_ap50.pt'
    cfg = './Yolor/cfg/yolor_p6.cfg'
    video = 0
    video_inference(video, weights, cfg)