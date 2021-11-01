from inference.inference import *
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

def compute_iou(rect1, rect2, intersection_area):
    union_area = compute_union(rect1, rect2, intersection_area)
    iou_value = intersection_area/union_area
    return iou_value

def compute_conf_matrix(prediction, annotation_list, index_list, trans_image, ori_image, iou_thr, thr):
    TP = 0
    FP = 0
    GT_num = len(annotation_list)
    for si, pred in enumerate(prediction):
        x = pred.clone()
        x[:, :4] = scale_coords(trans_image[si].shape[1:], x[:, :4], ori_image.shape)
        for *xyxy, conf, cls in x:
            for i in range(0, len(annotation_list)):
                annotation_bbox = annotation_list[i]
                annotation_index = index_list[i]
                intersection_area = compute_intersect_area(xyxy, annotation_bbox)
                if intersection_area != 0 and int(cls) == int(annotation_index) and conf >= thr:
                    iou_value = compute_iou(xyxy, annotation_bbox, intersection_area)
                    if iou_value >= iou_thr:
                        TP += 1
                elif intersection_area != 0 and int(cls) != int(annotation_index) and conf >= thr:
                    FP += 1
    FN = GT_num-TP
    return TP, FP, FN

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
    return precision, recall