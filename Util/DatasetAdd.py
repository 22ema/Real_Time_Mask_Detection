from pycocotools.coco import COCO
from facenet_pytorch import MTCNN, extract_face
import matplotlib.image as mpl
import matplotlib.pyplot as plt
import os
import cv2
import requests

def image_download():
    coco = COCO('./annotation_file/instances_train2014.json')
    catIds = coco.getCatIds(catNms=['person'])

    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    for im in images[:800]:
        img_data = requests.get(im['coco_url']).content
        with open('../media/dataset/coco_pserson' + im['file_name'], 'wb') as handler:
            handler.write(img_data)

def convert_to_darknet(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    if x>1:
        print(size,box)
    return (x,y,w,h)

def make_without_mask_dataset():
    path = '../media/dataset'
    labels_path = '../media/dataset/annotation'
    images = os.listdir(path)
    ignored_ones = []
    mtcnn=MTCNN()
    for img in images:
        if '.jpg' in img:
            img_path = os.path.join(path,img)
            image = cv2.imread(img_path)
            try:
                boxes, probs, points = mtcnn.detect(img=image, landmarks=True)
            except RuntimeError as e:
                print("Couln't detect {}".format(img_path))
                continue
            if boxes is not None:
                # for box, prob in zip(boxes, probs):
                #     startX, startY, endX, endY = box.astype(int)
                #
                #     color = (0, 255, 0)
                #     cv2.putText(image,
                #                 f'{prob:.1%}',
                #                 (startX, startY - 10),
                #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #                 fontScale=.5,
                #                 color=color,
                #                 thickness=2)
                #     cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
                #

                w = int(image.shape[1])
                h = int(image.shape[0])

                with open('{}/{}.txt'.format(labels_path,img[:-4]), 'w') as f:
                    for box in boxes:
                        startX, startY, endX, endY = box.astype(int)
                        if 'coco_psersonCOCO_train2014_000000132888.jpg' in img_path:
                            print(w,h)
                            print(startX, startY, endX, endY)
                        b = (startX, endX, startY, endY)
                        # convert_to_darknet at
                        # https://gist.github.com/AlexanderNixon/fb741fa2e950c7e0228394027ff9dffc
                        bb = convert_to_darknet((w, h), b)
                        box = ' '.join(box.astype(str))
                        f.write(f"1 ")
                        for x in bb:
                            if 'coco_psersonCOCO_train2014_000000132888.jpg' in img_path:
                                print(x)
                            f.write(f"{x} ")
                        f.write("\n")
                # plt.imshow(image)
                # plt.show()
            else:
                ignored_ones.append(img_path)
        else:
            pass

if __name__ == "__main__":
    make_without_mask_dataset()
    # image download
    # image_download()