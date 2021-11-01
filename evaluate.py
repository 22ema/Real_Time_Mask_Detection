from inference.inference import *
from Util.Compute import *
from Util.DatasetAdd import *
import matplotlib.pyplot as plt
import os

def annotation_inform(annotation_txt_path, annotation_list, index_list):
    with open(annotation_txt_path, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            result = deconvert_to_darknet((w, h), line.split(" ")[1:])
            annotation_list.append(result)
            index_list.append(line.split(" ")[0])

def make_pr_graph(precision_list, recall_list):

    plt.plot(recall_list, precision_list)
    plt.scatter(recall_list, precision_list)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve 2D')
    plt.show()

if __name__ == "__main__":
    dataset_path = "./media/dataset/test/images"
    annotation_path = "./media/dataset/test/labels"
    weights = './model/yolor/epoch_249.pt'
    cfg = './Yolor/cfg/yolor_p6.cfg'
    test_annotation_list = os.listdir(annotation_path)
    precision_list = []
    recall_list = []
    count = 0
    model = setting_model(cfg, weights)
    for i in range(10, 100, 10):
        iou_thr = 0.5
        thr = i/100
        precision = 0
        recall = 0
        print("---------Threshold : {}% -----------\n\n".format(thr * 100))
        for annotation_name in test_annotation_list:
            if '.txt' in annotation_name:
                annotation_list = []
                index_list = []
                test_image = annotation_name[:-3]+'jpg'
                image_path = os.path.join(dataset_path, test_image)
                annotation_txt_path = os.path.join(annotation_path, annotation_name)

                image = cv2.imread(image_path)
                output, ori_image, trans_image = inference(image, model)
                w = int(image.shape[1])
                h = int(image.shape[0])
                annotation_inform(annotation_txt_path, annotation_list, index_list)
                TP, FP, FN = compute_conf_matrix(output, annotation_list, index_list, trans_image, ori_image, iou_thr, thr)
                precision_value, recall_value = compute_pr_rec(TP, FP, FN)
                precision += precision_value
                recall += recall_value
                print("Precision:", precision_value)
                print("Recall:", recall_value)
        precision_list.append(precision/len(test_annotation_list))
        recall_list.append(recall/len(test_annotation_list))
        print("Precision:", precision/len(test_annotation_list))
        print("Recall:", recall/len(test_annotation_list))
    make_pr_graph(precision_list, recall_list)