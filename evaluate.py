from inference.inference import *
from Util.Compute import *
from Util.DatasetAdd import *
import matplotlib.pyplot as plt
import os

def annotation_inform(image_name, annotation_txt_path, annotation_list, w ,h):
    with open(annotation_txt_path, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            result = deconvert_to_darknet((w, h), line.split(" ")[1:])
            annotation_list.append([image_name, int(line.split(" ")[0]), result])

def prediction_inform(image_name, prediction_list, prediction, trans_image, ori_image):
    for si, pred in enumerate(prediction):
        x = pred.clone()
        x[:, :4] = scale_coords(trans_image[si].shape[1:], x[:, :4], ori_image.shape)
        for *xyxy, conf, cls in x:
            prediction_list.append([image_name, int(cls), conf, xyxy])

def make_pr_graph(precision_list, recall_list):

    plt.plot(recall_list, precision_list)
    plt.scatter(recall_list, precision_list)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve 2D')
    # plt.show()
    plt.savefig('./result/Precision_Recall_graph.png',dpi=300)

def evaluate_make(dataset_path, annotation_path, weights,cfg):
    test_annotation_list = os.listdir(annotation_path)
    annotation_list = []
    prediction_list = []
    precision_list = []
    recall_list = []
    classes = [0, 1]
    model = setting_model(cfg, weights)
    iou_thr = 0.5
    precision = 0
    recall = 0
    count = 0
    for annotation_name in test_annotation_list:
        if '.txt' in annotation_name:
            image_name = annotation_name[:-3]+'jpg'
            image_path = os.path.join(dataset_path, image_name)
            annotation_txt_path = os.path.join(annotation_path, annotation_name)
            image = cv2.imread(image_path)
            output, ori_image, trans_image = inference(image, model)
            w = int(image.shape[1])
            h = int(image.shape[0])
            annotation_inform(image_name, annotation_txt_path, annotation_list, w ,h)
            prediction_inform(image_name, prediction_list, output, trans_image, ori_image)
    result = computeAP(prediction_list, annotation_list, classes, iou_thr)
    mAP = computeMap(result)


# def sklearn_base_evaluate(dataset_path, annotation_path, weights,cfg):
#     test_annotation_list = os.listdir(annotation_path)
#     precision_list = []
#     recall_list = []
#     model = setting_model(cfg, weights)
#     for i in range(10, 100, 5):
#         iou_thr = 0.5
#         thr = i / 100
#         precision = 0
#         recall = 0
#         print("---------Threshold : {}% -----------\n\n".format(thr * 100))
#         for annotation_name in test_annotation_list:
#             if '.txt' in annotation_name:
#                 annotation_list = []
#                 index_list = []
#                 test_image = annotation_name[:-3] + 'jpg'
#                 image_path = os.path.join(dataset_path, test_image)
#                 annotation_txt_path = os.path.join(annotation_path, annotation_name)
#
#                 image = cv2.imread(image_path)
#                 output, ori_image, trans_image = inference(image, model)
#                 w = int(image.shape[1])
#                 h = int(image.shape[0])

if __name__ == "__main__":
    dataset_path = "./media/dataset/test/images"
    annotation_path = "./media/dataset/test/labels"
    weights = './model/yolor/best_ap50.pt'
    cfg = './Yolor/cfg/yolor_p6.cfg'
    evaluate_make(dataset_path, annotation_path, weights,cfg)
