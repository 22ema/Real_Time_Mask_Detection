from inference.inference import *
from Util.Compute import *
from Util.DatasetAdd import *
import matplotlib.pyplot as plt
import os

def annotation_inform(image_name, annotation_txt_path, annotation_list, w ,h):
    '''
    To save annotation result in annotation list
    :param image_name: image name "~~.jpg"
    :param annotation_txt_path: annotation information file path
    :param annotation_list: for saving about prediction
    :param w: image width
    :param h: image height
    '''
    with open(annotation_txt_path, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            result = deconvert_to_darknet((w, h), line.split(" ")[1:])
            annotation_list.append([image_name, int(line.split(" ")[0]), result])

def prediction_inform(image_name, prediction_list, prediction, trans_image, ori_image):
    '''
    To save predict result in prediction list
    :param image_name: image name "~~.jpg"
    :param prediction_list: for saving about prediction
    :param prediction: data to predict model
    :param trans_image: transfer image
    :param ori_image: original image
    '''
    for si, pred in enumerate(prediction):
        x = pred.clone()
        x[:, :4] = scale_coords(trans_image[si].shape[1:], x[:, :4], ori_image.shape)
        for *xyxy, conf, cls in x:
            prediction_list.append([image_name, int(cls), conf, xyxy])

def make_pr_graph(precision_list, recall_list, flag):
    '''
    To make the graph
    :param precision_list: precision
    :param recall_list:
    :param flag: classes inform (0 or 1)
    '''
    plt.plot(recall_list, precision_list)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('{}_Precision-Recall Curve 2D'.format(flag))
    # plt.show()
    plt.savefig('./result/{}_Precision_Recall_graph.png'.format(flag),dpi=300)
    plt.close()

def evaluate_make(dataset_path, annotation_path, weights,cfg):
    '''
    To evaluate object detection model
    :param dataset_path: test dataset path
    :param annotation_path: annotations path
    :param weights: inference model path
    :param cfg: model config file path
    '''
    test_annotation_list = os.listdir(annotation_path)
    annotation_list = []
    prediction_list = []
    classes = [0, 1]
    # model setting
    model = setting_model(cfg, weights)
    iou_thr = 0.5
    for annotation_name in test_annotation_list:
        if '.txt' in annotation_name:
            image_name = annotation_name[:-3]+'jpg'
            image_path = os.path.join(dataset_path, image_name)
            annotation_txt_path = os.path.join(annotation_path, annotation_name)
            image = cv2.imread(image_path)
            output, trans_image = inference(image, model)   # inference model
            w = int(image.shape[1])
            h = int(image.shape[0])
            annotation_inform(image_name, annotation_txt_path, annotation_list, w ,h)
            prediction_inform(image_name, prediction_list, output, trans_image, image)
    # compute AP, prcesion, recall
    result = computeAP(prediction_list, annotation_list, classes, iou_thr)
    # make graph
    make_pr_graph(result[0]['precision'], result[0]['recall'], 0)
    make_pr_graph(result[1]['precision'], result[1]['recall'], 1)
    # make_pr_graph(result[2]['precision'], result[2]['recall'], 2)
    # make_pr_graph(result[0]['interpolated precision'], result[0]['interpolated recall'], 0)
    # make_pr_graph(result[1]['interpolated precision'], result[1]['interpolated recall'], 1)
    # make_pr_graph(result[2]['interpolated precision'], result[2]['interpolated recall'],2)
    mAP = computeMap(result)
    print(mAP)

if __name__ == "__main__":
    '''
    evaluate object detection model
    '''
    dataset_path = "./media/dataset/test/images"
    annotation_path = "./media/dataset/test/labels"
    weights = './model/yolor/new_class_1_2/best_ap50.pt'
    cfg = './Yolor/cfg/yolor_p6.cfg'
    evaluate_make(dataset_path, annotation_path, weights,cfg)
