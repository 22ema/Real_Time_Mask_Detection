from inference.inference import *
import collections
import numpy as np

def compute_intersect_area(rect1, rect2):
    '''
    To compute intersect area between rect1 and rect2
    :param rect1: [x1, y1, x2, y2]
    :param rect2: [x1, y1, x2, y2]
    :return: area intersection
    '''
    x1, y1 = rect1[0], rect1[1]
    x2, y2 = rect1[2], rect1[3]
    x3, y3 = rect2[0], rect2[1]
    x4, y4 = rect2[2], rect2[3]

    if x2 < x3:
        return 0

    if x1 > x4:
        return 0

    if y2 < y3:
        return 0

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
    '''
    To compute union area between rect1 and rect2
    :param rect1: [x1, y1, x2, y2]
    :param rect2: [x1, y1, x2, y2]
    :param intersection_area: intersection area
    :return: union area
    '''
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
    '''
    compute IoU between rect1 and rect2
    :param rect1: [x1, y1, x2, y2]
    :param rect2: [x1, y1, x2, y2]
    :return:
    '''
    intersection_area = compute_intersect_area(rect1, rect2)
    union_area = compute_union(rect1, rect2, intersection_area)
    iou_value = intersection_area/union_area
    return iou_value

def computeAP(prediction_list, annotation_list, classes, iou_thr):
    '''
    compute AP, Precision, Recall in each class
    :param prediction_list: prediction list by deep learning
    :param annotation_list: ground truth list
    :param classes: classes
    :param iou_thr: iou threshold
    :return: each class result
    '''
    result = []
    for c in classes:
        detection = [d for d in prediction_list if d[1] == c]
        ground_truth = [gt for gt in annotation_list if gt[1] == c]
        gt_num = len(ground_truth)
        detection = sorted(detection, key = lambda conf : conf[2], reverse=True)

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
                    image_gt_num_dict[detection[d][0]][jmax] = 1
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
    '''
    using 11 interpolation compute mMap, mRecall, mPrecisi
    :param rec: recall
    :param prec: precision
    :return: [ap, mpre, mrec, _]
    '''
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