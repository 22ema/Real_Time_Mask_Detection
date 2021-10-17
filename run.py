import os
from Video_inference.video_inference import *
PATH = os.getcwd()
if __name__ =="__main__":
    video = './media/test_video/London_test.mp4'
    video_inference(video)