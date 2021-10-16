import cv2

def video_inference(video_file):
    cap = cv2.VideoCapture(video_file)
    if cap.isOpened():
        while True:
            ret, img = cap.read()
            if ret:
                cv2.imshow(video_file, img)
                cv2.waitKey()
            else:
                break
    else:
        print("can't open video.")
    cap.release()
    cv2.destroyAllWindows()