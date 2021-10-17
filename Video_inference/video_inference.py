import cv2
from inference.inference import *

def video_inference(video_file):
    cap = cv2.VideoCapture(video_file)
    model = setting_model()
    if cap.isOpened():
        while True:
            ret, img = cap.read()
            result_img = inference(img, model)
            cv2.imshow(video_file, result_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print("can't open video.")
    cap.release()
    cv2.destroyAllWindows()