import cv2
from inference.inference import *

def video_inference(video_file, weigths, cfg):
    '''
    video's frame inference and show
    :param video_file: video file link
    :param weigths: inference model file link
    :param cfg: model configuration file link
    '''
    cap = cv2.VideoCapture(video_file)
    model = setting_model(cfg, weigths)
    if cap.isOpened():
        while True:
            ret, img = cap.read()   # frame read in video
            output, trans_image = inference(img, model)  # inference frame
            result_img = post_processing(output, img, trans_image)    # visualize bounding box and class in image
            cv2.imshow('result', result_img)    # display result image

            if cv2.waitKey(1) & 0xFF == ord('q'):   # to click keyboard's 'q' button for end
                break
    else:
        print("can't open video.")
    cap.release()
    cv2.destroyAllWindows()