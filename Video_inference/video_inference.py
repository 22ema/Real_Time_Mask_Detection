import cv2
from inference.inference import *

def video_inference(video_file):
    weights = './model/yolor/epoch_249.pt'
    cfg = './Yolor/cfg/yolor_p6.cfg'
    cap = cv2.VideoCapture(video_file)
    model = setting_model(cfg, weights)
    if cap.isOpened():
        while True:
            ret, img = cap.read()
            output, ori_image, trans_image = inference(img, model)
            result_img = post_processing(output, ori_image, trans_image)
            cv2.imshow(video_file, result_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print("can't open video.")
    cap.release()
    cv2.destroyAllWindows()