from pycocotools.coco import COCO
from facenet_pytorch import MTCNN, extract_face
import matplotlib.image as mpl
import matplotlib.pyplot as plt
import os
import cv2
import shutil
import requests

# def image_download():
#     coco = COCO('./annotation_file/instances_train2014.json')
#     catIds = coco.getCatIds(catNms=['person'])
#
#     imgIds = coco.getImgIds(catIds=catIds)
#     images = coco.loadImgs(imgIds)
#
#     for im in images[:800]:
#         img_data = requests.get(im['coco_url']).content
#         with open('../media/dataset/coco_pserson' + im['file_name'], 'wb') as handler:
#             handler.write(img_data)

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
    if w >= 1:
        w =0.9
    if x >= 1:
        x =0.9
    if y >= 1:
        y = 0.9
    if h >= 1:
        h = 0.9
    return (x,y,w,h)

def deconvert_to_darknet(size, box):
    dw = size[0]
    dh = size[1]
    x = float(box[0])*dw
    w = float(box[2])*dw
    y = float(box[1])*dh
    h = float(box[3])*dh

    x_1 = x-(w/2)
    x_2 = x+(w/2)
    y_1 = y-(h/2)
    y_2 = y+(h/2)
    return x_1, y_1, x_2, y_2

def make_masked_image():
    ori_path = '../media/dataset/AFDB_masked_face_dataset'
    move_dir = '../media/dataset/AFDB_dataset'
    ori_dir = os.listdir(ori_path)
    for ori in ori_dir:
        path = os.path.join(ori_path, ori)
        image_dir = os.listdir(path)
        for image in image_dir :
            shutil.move(path+'/'+image,move_dir+'/'+ori+'_'+image)



def make_without_mask_dataset():
    '''
    Make new dataset
    '''
    # path = '../media/dataset/AFDB_face_dataset/aidai'
    # labels_path = '../media/dataset/AFDB_face_dataset/aida_annotation'
    path = '../media/dataset/AFDB_dataset/'
    labels_path = '../media/dataset/AFDB_annotation'
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


                w = int(image.shape[1])
                h = int(image.shape[0])
                with open('{}/{}.txt'.format(labels_path,img[:-4]), 'w') as f:
                    for box in boxes:
                        startX, startY, endX, endY = box.astype(int)
                        b = (startX, endX, startY, endY)
                        # convert_to_darknet at
                        # https://gist.github.com/AlexanderNixon/fb741fa2e950c7e0228394027ff9dffc
                        bb = convert_to_darknet((w, h), b)
                        box = ' '.join(box.astype(str))
                        f.write(f"0 ")
                        for x in bb:
                            f.write(f"{x} ")
                        f.write("\n")
                # plt.imshow(image)
                # plt.show()
            else:
                ignored_ones.append(img_path)
        else:
            pass

if __name__ == "__main__":
    # make_masked_image()
    make_without_mask_dataset()
    # image download
    # image_download()