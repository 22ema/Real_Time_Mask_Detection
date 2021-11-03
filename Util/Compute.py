from inference.inference import *
import collections
import numpy as np
def compute_intersect_area(rect1, rect2):
    x1, y1 = rect1[0], rect1[1]
    x2, y2 = rect1[2], rect1[3]
    x3, y3 = rect2[0], rect2[1]
    x4, y4 = rect2[2], rect2[3]
    ## case1 오른쪽으로 벗어나 있는 경우
    if x2 < x3:
        return 0
    ## case2 왼쪽으로 벗어나 있는 경우
    if x1 > x4:
        return 0
    ## case3 위쪽으로 벗어나 있는 경우
    if y2 < y3:
        return 0
    ## case4 아래쪽으로 벗어나 있는 경우
    if y1 > y4:
        return 0
    left_up_x = max(x1, x3)
    left_up_y = max(y1, y3)
    right_down_x = min(x2, x4)
    right_down_y = min(y2, y4)
    width = right_down_x - left_up_x
    height = right_down_y - left_up_y
    return width * height

def compute_union(rect1, rect2, intersection_area):
    x1, y1 = rect1[0], rect1[1]
    x2, y2 = rect1[2], rect1[3]
    x3, y3 = rect2[0], rect2[1]
    x4, y4 = rect2[2], rect2[3]

    a_width = x2-x1
    a_height = y2-y1
    b_width = x4-x3
    b_height = y4-y3

    a_area = a_width * a_height
    b_area = b_width * b_height

    union_area = a_area + b_area - intersection_area
    return union_area

def compute_iou(rect1, rect2):
    intersection_area = compute_intersect_area(rect1, rect2)
    union_area = compute_union(rect1, rect2, intersection_area)
    iou_value = intersection_area/union_area
    return iou_value

# def compute_conf_matrix(prediction, annotation_list, index_list, trans_image, ori_image, iou_thr, thr):
#     TP = 0
#     FP = 0
#     With_Mask_confuse_mat = [0, 0, 0]
#     Without_Mask_confuse_mat = [0, 0, 0]
#     With_Mask_Count = 0
#     Without_Mask_Count = 0
#     for x in index_list:
#         if x == 0:
#             With_Mask_Count += 1
#         elif x == 1:
#             Without_Mask_Count += 1
#     for si, pred in enumerate(prediction):
#         x = pred.clone()
#         x[:, :4] = scale_coords(trans_image[si].shape[1:], x[:, :4], ori_image.shape)
#         for *xyxy, conf, cls in x:
#             for i in range(0, len(annotation_list)):
#                 annotation_bbox = annotation_list[i]
#                 annotation_index = index_list[i]
#                 intersection_area = compute_intersect_area(xyxy, annotation_bbox)
#                 if intersection_area != 0 and int(cls) == int(annotation_index) and conf >= thr:
#                     iou_value = compute_iou(xyxy, annotation_bbox, intersection_area)
#                     if iou_value >= iou_thr and int(annotation_index) == 0:
#                         With_Mask_confuse_mat[0] += 1
#                     elif iou_value >= iou_thr and int(annotation_index) == 1:
#                         Without_Mask_confuse_mat[0] += 1
#                 elif intersection_area != 0 and int(cls) != int(annotation_index) and conf >= thr:
#                     iou_value = compute_iou(xyxy, annotation_bbox, intersection_area)
#                     if iou_value >= iou_thr and int(annotation_index) == 0:
#                         With_Mask_confuse_mat[1] +=1
#                     elif iou_value >= iou_thr and int(annotation_index) == 1:
#                         Without_Mask_confuse_mat[1] +=1
#     With_Mask_confuse_mat[2] = With_Mask_Count-With_Mask_confuse_mat[0]
#     Without_Mask_confuse_mat[2] = Without_Mask_Count - Without_Mask_confuse_mat[0]
#     return With_Mask_confuse_mat, Without_Mask_confuse_mat

def computeAP(prediction_list, annotation_list, classes, iou_thr):
    result = []
    for c in classes:
        detection = [d for d in prediction_list if d[1] == c]
        ground_truth = [gt for gt in annotation_list if gt[1] == c]
        gt_num = len(ground_truth)
        detection = sorted(detection, key = lambda conf : detection[2], reverse=True)

        TP = np.zeros(len(detection))
        FP = np.zeros(len(detection))

        image_gt_num_dict = collections.Counter(cc[0] for cc in ground_truth)

        for key, val in image_gt_num_dict.items():
            image_gt_num_dict[key] = np.zeros(val)

        for d in range(len(detection)):
            gt = [gt for gt in ground_truth if gt[0] == detection[d][0]]

            iouMax = 0
            for j in range(len(gt)):
                iou = compute_iou(detection[d][3],gt[j][2])
                if iou> iouMax:
                    iouMax = iou
                    jmax = j

            if iouMax >= iou_thr:
                if image_gt_num_dict[detection[d][0]][jmax] == 0:
                    TP[d] = 1
                    image_gt_num_dict[detection[d][0]][jmax] =1
                else:
                    FP[d] = 1
            else:
                FP[d] = 1
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / gt_num
        prec = np.divide(acc_TP, (acc_FP + acc_TP))

        [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)
        r = {
            'class': c,
            'precision': prec,
            'recall': rec,
            'AP': ap,
            'interpolated precision': mpre,
            'interpolated recall': mrec,
            'total positives': gt_num,
            'total TP': np.sum(TP),
            'total FP': np.sum(FP)
        }

        result.append(r)

    return result


def ElevenPointInterpolatedAP(rec, prec):
    mrec = [e for e in rec]
    mpre = [e for e in prec]

    # recallValues = [1.0, 0.9, ..., 0.0]
    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp, recallValid = [], []

    for r in recallValues:
        # r : recall값의 구간
        # argGreaterRecalls : r보다 큰 값의 index
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0

        # precision 값 중에서 r 구간의 recall 값에 해당하는 최댓값
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])

        recallValid.append(r)
        rhoInterp.append(pmax)

    ap = sum(rhoInterp) / 11

    return [ap, rhoInterp, recallValues, None]

def computeMap(result):

    ap = 0
    for r in result:
        ap += r['AP']
    mAP = ap / len(result)

    return mAP


def compute_pr_rec(TP, FP, FN):
    if TP+FP == 0 and TP+FN == 0:
        precision = 1
        recall = 1
    elif TP+FP == 0:
        precision = 1
        recall = TP / (TP + FN)
    elif TP+FN == 0:
        precision = TP / (TP + FP)
        recall = 1
    else:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
    pr_list = [precision, recall]
    return pr_list